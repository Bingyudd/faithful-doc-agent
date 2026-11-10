"""CFB equations (ACL Findings 2026), independently implemented in PyTorch."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import time
import torch


@dataclass(frozen=True)
class DecodeConfig:
    method: str = 'token'
    variant: str = 'author_reference'
    delta: float = 5.0
    delta_min: float = 1.0
    delta_max: float = 10.0
    attention_weight: float = 0.6
    cad_alpha: float = 1.0
    max_source_tokens: int = 1024
    max_new_tokens: int = 64

    def __post_init__(self):
        if self.variant not in ('paper_equations','author_reference'):
            raise ValueError('Unknown CFB implementation variant')
        if self.method not in ('greedy','cad','static','context','token','no_attention','no_semantic'):
            raise ValueError('Unknown decoding method')
        if not 0 <= self.attention_weight <= 1 or self.delta_min > self.delta_max:
            raise ValueError('Invalid weighting/range')
        if min(self.delta,self.delta_min,self.delta_max,self.cad_alpha) < 0:
            raise ValueError('Negative bias')
        if min(self.max_source_tokens,self.max_new_tokens) < 1:
            raise ValueError('Token budgets must be positive')


def js_divergence(logits_p, logits_q):
    """Base-2 JSD in [0,1], evaluated in FP32 without log(0)."""
    logp=torch.log_softmax(logits_p.float(),dim=-1)
    logq=torch.log_softmax(logits_q.float(),dim=-1)
    logm=torch.logaddexp(logp,logq)-math.log(2)
    return ((logp.exp()*(logp-logm)).sum(-1)+(logq.exp()*(logq-logm)).sum(-1))/(2*math.log(2))


def occurrence_attention(attention, source_ids):
    unique,inverse=torch.unique(source_ids,sorted=True,return_inverse=True)
    summed=torch.zeros(unique.shape,device=attention.device,dtype=torch.float32)
    summed.scatter_add_(0,inverse,attention.float())
    return unique,summed


def semantic_scores(embedding, source_ids, variant='paper_equations'):
    """cos(e_w,e_c) mean over source OCCURRENCES; avoid an O(S²) matrix."""
    unique=torch.unique(source_ids,sorted=True)
    unit=torch.nn.functional.normalize(embedding[source_ids].float(),dim=-1)
    candidates=torch.nn.functional.normalize(embedding[unique].float(),dim=-1)
    scores=candidates @ unit.mean(dim=0)
    # Upstream f7e5f38 lines 454-462 shifts cosine before combining it with attention.
    # Equation (6) alone does not include this mapping. Keep both variants explicit.
    if variant=='author_reference':scores=(scores+1)/2
    return unique,scores


def boost_logits(logits, unique_ids, delta, attention=None, semantic=None, weight=0.6):
    output=logits.float().clone()
    if unique_ids.numel()==0 or float(delta)==0:
        return output
    if attention is None:
        bias=torch.full(unique_ids.shape,float(delta),device=logits.device,dtype=torch.float32)
    else:
        relevance=weight*attention+(1-weight)*semantic
        mean=relevance.mean()
        # Degenerate/nonpositive normalization is explicit, never swapped to a hidden model.
        if not torch.isfinite(relevance).all() or mean <= 1e-8:
            raise ValueError('Nonpositive or nonfinite CFB relevance mean')
        bias=float(delta)*relevance/mean
    output[...,unique_ids]+=bias
    return output


def make_prompt(tokenizer, source, task, source_limit):
    raw=tokenizer(source,add_special_tokens=False)['input_ids']
    truncated=len(raw)>source_limit
    source=tokenizer.decode(raw[:source_limit],skip_special_tokens=True)
    prefix='Source material:\n<source>\n'
    user=prefix+source+'\n</source>\nTask: '+task
    prompt=tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
    begin=prompt.index(prefix)+len(prefix)
    end=begin+len(source)
    encoded=tokenizer(prompt,return_offsets_mapping=True,add_special_tokens=False)
    positions=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>a and a>=begin and b<=end
               and encoded['input_ids'][i] not in tokenizer.all_special_ids]
    return encoded['input_ids'],positions,source,truncated


def _forward(model, ids, cache=None, attentions=False):
    return model(input_ids=ids,past_key_values=cache,use_cache=True,
                 output_attentions=attentions,logits_to_keep=1)


def prefill(model, ids, attention=False):
    # Prefill without materializing every layer's S×S attention, then last query.
    if attention and ids.shape[1]>1:
        prefix=_forward(model,ids[:,:-1])
        cache=prefix.past_key_values
        del prefix
        return _forward(model,ids[:,-1:],cache,True)
    return _forward(model,ids,attentions=attention)


@torch.inference_mode()
def generate(model,tokenizer,source,task,config=DecodeConfig()):
    device=next(model.parameters()).device
    ids,positions,used_source,truncated=make_prompt(tokenizer,source,task,config.max_source_tokens)
    input_ids=torch.tensor([ids],device=device)
    pos=torch.tensor(positions,device=device,dtype=torch.long)
    source_ids=input_ids[0,pos]
    unique=torch.unique(source_ids,sorted=True)
    tokenwise=config.method in ('token','no_attention','no_semantic')
    need_attention=tokenwise and config.method!='no_attention'
    if device.type=='cuda':
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    current=prefill(model,input_ids,need_attention)
    contextual=current.logits[0,-1].float()
    cache=current.past_key_values
    jsd=0.0; prior_cache=None;prior=None
    if config.method in ('cad','context','token','no_attention','no_semantic'):
        prior_ids=tokenizer.apply_chat_template([{'role':'user','content':'Task: '+task}],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
        prior_out=prefill(model,prior_ids)
        prior=prior_out.logits[0,-1].float()
        jsd=float(js_divergence(contextual,prior).item())
        if config.method=='cad':prior_cache=prior_out.past_key_values
        del prior_out
    semantic=None
    if tokenwise and source_ids.numel():
        _,semantic=semantic_scores(model.get_input_embeddings().weight,source_ids,config.variant)
    adaptive=config.delta_min+(config.delta_max-config.delta_min)*jsd
    generated=[];diagnostics=[]
    eos=tokenizer.eos_token_id
    stop=set(eos if isinstance(eos,list) else [eos])
    extra=getattr(model.generation_config,'eos_token_id',None)
    if extra is not None:stop.update(extra if isinstance(extra,list) else [extra])
    for step in range(config.max_new_tokens):
        base=current.logits[0,-1].float()
        if config.method=='greedy': adjusted=base
        elif config.method=='cad': adjusted=(1+config.cad_alpha)*torch.log_softmax(base,-1)-config.cad_alpha*torch.log_softmax(prior,-1)
        elif config.method in ('static','context'):
            adjusted=boost_logits(base,unique,config.delta if config.method=='static' else adaptive)
        elif not source_ids.numel(): adjusted=base
        else:
            if need_attention:
                if current.attentions is None or current.attentions[-1] is None:
                    raise RuntimeError('Actual attention unavailable: use eager attention; no hidden fallback')
                weights=current.attentions[-1][0,:,-1,pos].float().mean(0)
                if config.variant=='author_reference':
                    # Upstream get_attention_scores softmaxes the source-only attention.
                    weights=torch.softmax(weights,dim=-1)
                unique,attention=occurrence_attention(weights,source_ids)
            else:attention=torch.zeros_like(semantic)
            weight=0.0 if config.method=='no_attention' else 1.0 if config.method=='no_semantic' else config.attention_weight
            adjusted=boost_logits(base,unique,adaptive,attention,semantic,weight)
        token=int(adjusted.argmax())
        diagnostics.append({'step':step,'token_id':token,'original_argmax':int(base.argmax()),
                            'bias':float((adjusted[token]-base[token]).item()) if config.method!='cad' else None,
                            'source_token':bool((unique==token).any())})
        if token in stop:break
        generated.append(token)
        if step+1==config.max_new_tokens:break
        next_ids=torch.tensor([[token]],device=device)
        del current
        current=_forward(model,next_ids,cache,need_attention)
        cache=current.past_key_values
        if config.method=='cad':
            prior_out=_forward(model,next_ids,prior_cache)
            prior_cache=prior_out.past_key_values;prior=prior_out.logits[0,-1].float()
            del prior_out
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return {'text':tokenizer.decode(generated,skip_special_tokens=True),'config':asdict(config),
            'input_tokens':len(ids),'output_tokens':len(generated),'source_truncated':truncated,
            'used_source':used_source,'jsd':jsd,'seconds':elapsed,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
            'diagnostics':diagnostics,'stopped_by_eos':bool(diagnostics and diagnostics[-1]['token_id'] in stop)}
