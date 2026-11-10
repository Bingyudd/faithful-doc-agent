"""Record numerical checks on real mask positions with bounded synthetic logits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from post_training import dpo_loss,sequence_log_probs,sft_shift_loss,simpo_loss,losses_from_logits
from preference_data import run as export_pairs
from token_masking import encode_supervised,load_tokenizer,collate_right
from transformer_whitebox import kv_cache_bytes,training_state_bytes

ROOT=Path(__file__).resolve().parent


def run():
    torch.manual_seed(19)
    torch.set_num_threads(2)
    output=ROOT/'artifacts'
    preferences=export_pairs()
    pairs=_read_pairs()
    tokenizer=load_tokenizer()
    chosen_rows,rejected_rows=[],[]
    with (output/'preference_tokenized.jsonl').open('w',encoding='utf-8') as stream:
        for pair in pairs:
            start=len(pair['prompt'])
            chosen=encode_supervised(tokenizer,pair['prompt']+pair['chosen'],tools=pair['tools'],supervise_from_message=start)
            rejected=encode_supervised(tokenizer,pair['prompt']+pair['rejected'],tools=pair['tools'],supervise_from_message=start)
            if any(owner is not None and owner<start for row in [chosen,rejected] for owner in row['assistant_message_indices']):
                raise AssertionError('Preference prompt must not be supervised')
            chosen_rows.append(chosen);rejected_rows.append(rejected)
            stream.write(json.dumps({'id':pair['id'],
                'chosen':{key:chosen[key] for key in ['input_ids','attention_mask','labels']},
                'rejected':{key:rejected[key] for key in ['input_ids','attention_mask','labels']}},ensure_ascii=False)+'\n')
    original=np.load(output/'sft_training_batch.npz')['labels']
    # Preserve actual positions and supervision, but keep validation logits small.
    # These are not Qwen model logits or a trained-model quality experiment.
    vocabulary=sorted(set(int(value) for value in original.flat if value!=-100))
    mapping={value:i+1 for i,value in enumerate(vocabulary)}
    labels=torch.tensor([[mapping.get(int(value),-100) for value in row] for row in original],dtype=torch.long)
    logits=torch.randn(*labels.shape,len(mapping)+1,dtype=torch.float64,requires_grad=True)
    token_loss=sft_shift_loss(logits,labels,reduction='token')
    sample_loss=sft_shift_loss(logits,labels,reduction='sample')
    sums,lengths=sequence_log_probs(logits,labels)
    token_loss.backward()
    mask=labels[:,1:].eq(-100)
    zero_gradient=bool(torch.equal(logits.grad[:,:-1][mask],torch.zeros_like(logits.grad[:,:-1][mask])))
    if not zero_gradient:
        raise AssertionError('Ignored prediction position has nonzero gradient')
    chosen=torch.tensor([-2.,-3.],requires_grad=True)
    rejected=torch.tensor([-3.,-5.],requires_grad=True)
    reference_chosen=chosen.detach().clone()
    reference_rejected=rejected.detach().clone()
    dpo=dpo_loss(chosen,rejected,reference_chosen,reference_rejected,beta=0.1)
    simpo=simpo_loss(chosen,rejected,torch.tensor([2,3]),torch.tensor([3,5]),beta=2,gamma=0.5)
    (dpo+simpo).backward()
    # Exercise unequal chosen/rejected sequence lengths using actual preference
    # masks and small synthetic logits, not pretrained-model probabilities.
    cb=collate_right(tokenizer,chosen_rows[:8])
    rb=collate_right(tokenizer,rejected_rows[:8])
    preference_vocabulary=sorted({int(value) for batch in [cb,rb] for value in batch['labels'].flat if value!=-100})
    preference_map={value:i+1 for i,value in enumerate(preference_vocabulary)}
    def remap(batch):
        return torch.tensor([[preference_map.get(int(value),-100) for value in row] for row in batch['labels']],dtype=torch.long)
    chosen_labels,rejected_labels=remap(cb),remap(rb)
    pc=torch.randn(*chosen_labels.shape,len(preference_map)+1,dtype=torch.float64,requires_grad=True)
    pr=torch.randn(*rejected_labels.shape,len(preference_map)+1,dtype=torch.float64,requires_grad=True)
    integrated=losses_from_logits(pc,pr,pc.detach(),pr.detach(),chosen_labels,rejected_labels)
    (integrated['dpo']+integrated['simpo']).backward()
    if not torch.isfinite(pc.grad).all() or not torch.isfinite(pr.grad).all():
        raise AssertionError('Preference gradients must be finite')
    config=json.loads((ROOT/'data/tokenizer_qwen25_7b/config.json').read_text(encoding='utf-8'))
    head_dim=config['hidden_size']//config['num_attention_heads']
    cache=kv_cache_bytes(batch=1,tokens=4096,layers=config['num_hidden_layers'],
                         kv_heads=config['num_key_value_heads'],head_dim=head_dim,element_bytes=2)
    state=training_state_bytes(7_000_000_000)
    report={
        'torch_version':torch.__version__,'validation_device':str(logits.device),
        'actual_mask_batch_shape':list(labels.shape),'synthetic_vocabulary_size':len(mapping)+1,
        'actual_supervised_target_positions':int(labels[:,1:].ne(-100).sum()),
        'sft_token_mean_example':token_loss.item(),'sft_sample_mean_example':sample_loss.item(),
        'masked_prediction_logits_gradient_exactly_zero':zero_gradient,
        'sequence_logprob_reduction_matches_ce':bool(torch.allclose(-sums.sum()/lengths.sum(),token_loss.detach())),
        'dpo_equal_policy_reference_example':dpo.item(),'simpo_example':simpo.item(),
        'preference_pairs':preferences['preference_pairs'],'hard_negative_pairs':preferences['same_answer_invalid_evidence_pairs'],
        'preference_pairs_native_encoded':len(chosen_rows),
        'preference_common_prompt_masked':True,
        'preference_tensor_validation_pairs':8,
        'preference_chosen_batch_shape':list(chosen_labels.shape),
        'preference_rejected_batch_shape':list(rejected_labels.shape),
        'preference_dpo_tensor_example':integrated['dpo'].item(),
        'preference_simpo_tensor_example':integrated['simpo'].item(),
        'preference_gradients_finite':True,
        'qwen_config_dense_kv_cache_bytes_at_B1_T4096_fp16':cache,
        'qwen_config_dense_kv_cache_mib_at_B1_T4096_fp16':cache/1024**2,
        'illustrative_7b_training_state':state,
        'training_state_assumptions':{'parameters':7_000_000_000,'weight_bytes':2,'gradient_bytes':2,'moment_bytes':4,'master_weight_bytes':4},
        'pretrained_model_weights_loaded':False,'pretrained_model_training_runs':0,
        'notes':['Loss values are seeded tensor examples over a remapped target-token subset, not model evaluation scores.',
                 'Zero gradient refers to ignored next-token prediction logits, not all embeddings or context activations.',
                 'Memory figures exclude activation, allocator, paging, communication and quantization overhead.']}
    (output/'algorithm_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.save({'chosen_reward':torch.tensor([p['chosen_reward']['total'] for p in pairs]),
                'rejected_reward':torch.tensor([p['rejected_reward']['total'] for p in pairs])},output/'rule_reward_tensors.pt')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def _read_pairs():
    return [json.loads(line) for line in (ROOT/'artifacts/preference_pairs.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


if __name__=='__main__':
    run()
