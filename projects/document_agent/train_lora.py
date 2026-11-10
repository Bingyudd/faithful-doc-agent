"""Reproducible 0.5B LoRA SFT -> DPO pilot on field-grouped development data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from evidence_core import Corpus, Task, audit_claim
from evaluate_live import score_answer
from post_training import dpo_loss
from token_masking import encode_supervised
from tool_protocol import qwen_messages

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
HOLDOUT_FIELDS = frozenset(('working_length', 'light_source'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def field_for(pair):
    return json.loads(next(m['content'] for m in pair['prompt'] if m['role'] == 'user'))['field']


def prepare_data(tokenizer):
    source = ROOT / 'artifacts/preference_pairs.jsonl'
    pairs = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    unique = {}
    for pair in pairs:
        pair['split'] = 'holdout' if field_for(pair) in HOLDOUT_FIELDS else 'train'
        for side in ('chosen', 'rejected'):
            row = encode_supervised(tokenizer, pair['prompt'] + pair[side], tools=pair['tools'],
                                    max_length=2048, supervise_from_message=len(pair['prompt']))
            if row['truncated']:
                raise ValueError('Pilot requires untruncated final responses')
            pair[side + '_encoded'] = row
        if pair['task_id'] not in unique:
            row = encode_supervised(tokenizer, pair['prompt'] + pair['chosen'], tools=pair['tools'],
                                    max_length=2048)
            if row['truncated']:
                raise ValueError('Pilot requires untruncated SFT trajectories')
            unique[pair['task_id']] = {'task_id': pair['task_id'], 'split': pair['split'],
                                       'encoded': row, 'pair': pair}
    train_ids = {p['task_id'] for p in pairs if p['split'] == 'train'}
    held_ids = {p['task_id'] for p in pairs if p['split'] == 'holdout'}
    assert train_ids.isdisjoint(held_ids)
    return pairs, list(unique.values())


def target_logp(model, row):
    """Project only supervised prediction positions to the full 151,936-word vocabulary.

    All context tokens still traverse the Transformer. This is the same causal shift
    as full-logit CE, while avoiding a [1,1200,vocab] output tensor on an 8GB GPU.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([row['input_ids']], device=device)
    labels = torch.tensor(row['labels'], device=device)
    target_positions = torch.where(labels[1:] != -100)[0] + 1
    if target_positions.numel() == 0:
        raise ValueError('Empty response supervision')
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=target_positions - 1)
    logits = output.logits[0].float()
    loss_sum = F.cross_entropy(logits, labels[target_positions], reduction='sum')
    return -loss_sum, target_positions.numel()


@torch.no_grad()
def evaluate_pairs(model, pairs, reference=None):
    model.eval()
    rows = []
    for pair in pairs:
        chosen, clen = target_logp(model, pair['chosen_encoded'])
        rejected, rlen = target_logp(model, pair['rejected_encoded'])
        row = {'id': pair['id'], 'split': pair['split'], 'negative_category': pair['negative_category'],
               'chosen_logp': chosen.item(), 'rejected_logp': rejected.item(),
               'chosen_tokens': clen, 'rejected_tokens': rlen,
               'margin': (chosen - rejected).item(),
               'mean_token_margin': (chosen / clen - rejected / rlen).item()}
        if reference is not None:
            ref = reference[pair['id']]
            row['reference_adjusted_margin'] = row['margin'] - ref['margin']
        rows.append(row)
    summary = {}
    for split in ('train', 'holdout'):
        subset = [r for r in rows if r['split'] == split]
        summary[split] = {
            'pairs': len(subset), 'chosen_nll_per_token': -sum(r['chosen_logp'] for r in subset) / sum(r['chosen_tokens'] for r in subset),
            'mean_margin': sum(r['margin'] for r in subset) / len(subset),
            'mean_token_margin': sum(r['mean_token_margin'] for r in subset) / len(subset),
            'preference_accuracy': sum(r['margin'] > 0 for r in subset) / len(subset)}
        if reference is not None:
            adjusted = [r['reference_adjusted_margin'] for r in subset]
            summary[split]['reference_adjusted_margin'] = sum(adjusted) / len(adjusted)
            summary[split]['dpo_loss'] = sum(math.log1p(math.exp(-0.1 * v)) for v in adjusted) / len(adjusted)
    return {'summary': summary, 'rows': rows}


@torch.no_grad()
def evaluate_generation(model, tokenizer, examples, corpus):
    """Final answer given fixed tool evidence; NOT autonomous tool-use evaluation."""
    model.eval()
    rows = []
    for example in examples:
        if example['split'] != 'holdout':
            continue
        pair = example['pair']
        prompt = tokenizer.apply_chat_template(qwen_messages(pair['prompt']), tools=pair['tools'],
                                               tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
        output = model.generate(**encoded, max_new_tokens=192, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
        text = tokenizer.decode(output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)
        task = Task.model_validate_json(next(m['content'] for m in pair['prompt'] if m['role'] == 'user'))
        try:
            verdict = audit_claim(corpus, task, json.loads(text))
        except (ValueError, TypeError):
            verdict = {'accepted': False, 'errors': ['invalid_final_json']}
        result = {'status': 'validated' if verdict['accepted'] else 'needs_review',
                  'messages': [{'role': 'assistant', 'content': text}]}
        rows.append({'task_id': pair['task_id'], 'text': text, 'verdict': verdict,
                     **score_answer(result, json.loads(pair['chosen'][0]['content']))})
    return {'setting': 'fixed_gold_tool_observation_final_completion', 'tasks': rows,
            'answer_and_provenance_pass': sum(r['e2e_pass'] for r in rows), 'total': len(rows)}


def train_phase(model, examples, *, steps, accumulation, learning_rate, phase, log_path, reference=None):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    rng = random.Random(20261106 + (phase == 'dpo'))
    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = []
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        sample_ids = []
        for _ in range(accumulation):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            item = examples[order[cursor]]
            cursor += 1
            sample_ids.append(item.get('id', item.get('task_id')))
            if phase == 'sft':
                logp, count = target_logp(model, item['encoded'])
                loss = -logp / count
            else:
                chosen, _ = target_logp(model, item['chosen_encoded'])
                rejected, _ = target_logp(model, item['rejected_encoded'])
                ref = reference[item['id']]
                loss = dpo_loss(chosen, rejected, chosen.new_tensor(ref['chosen_logp']),
                                rejected.new_tensor(ref['rejected_logp']), beta=0.1)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            total += loss.item() / accumulation
            (loss / accumulation).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError('Non-finite gradients')
        optimizer.step()
        row = {'phase': phase, 'step': step, 'loss': total, 'grad_norm': grad_norm.item(),
               'sample_ids': sample_ids, 'elapsed_seconds': round(time.perf_counter() - start, 3),
               'max_memory_allocated': torch.cuda.max_memory_allocated(),
               'max_memory_reserved': torch.cuda.max_memory_reserved()}
        history.append(row)
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        if step == 1 or step % 5 == 0:
            print(json.dumps(row), flush=True)
    update_l2 = math.sqrt(sum((p.detach() - before[name]).float().square().sum().item()
                              for name, p in model.named_parameters() if p.requires_grad))
    model.gradient_checkpointing_disable()
    model.disable_input_require_grads()
    optimizer.zero_grad(set_to_none=True)
    return {'steps': steps, 'accumulation': accumulation, 'learning_rate': learning_rate,
            'adapter_update_l2': update_l2, 'elapsed_seconds': time.perf_counter() - start,
            'max_memory_allocated': torch.cuda.max_memory_allocated(),
            'max_memory_reserved': torch.cuda.max_memory_reserved(), 'history': history}


def main():
    from peft import LoraConfig, get_peft_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sft-steps', type=int, default=40)
    parser.add_argument('--dpo-steps', type=int, default=40)
    args = parser.parse_args()
    if min(args.sft_steps, args.dpo_steps) <= 0:
        parser.error('Both stages require positive optimizer steps')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261106)
    random.seed(20261106)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This pilot requires the verified WSL CUDA environment')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    pairs, examples = prepare_data(tokenizer)
    report = {'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'complete': False,
              'model': MODEL, 'revision': REVISION, 'seed': 20261106, 'dtype': 'bfloat16',
              'gpu': torch.cuda.get_device_name(0), 'python': sys.version, 'executable': sys.executable,
              'packages': {p: importlib.metadata.version(p) for p in
                           ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers', 'pdfplumber', 'pydantic')},
              'training_runs': 0, 'optimizer_steps': 0,
              'split': {'unit': 'field_family_in_one_document', 'holdout_fields': sorted(HOLDOUT_FIELDS),
                        'train_task_ids': [e['task_id'] for e in examples if e['split'] == 'train'],
                        'holdout_task_ids': [e['task_id'] for e in examples if e['split'] == 'holdout'],
                        'train_pairs': sum(p['split'] == 'train' for p in pairs),
                        'holdout_pairs': sum(p['split'] == 'holdout' for p in pairs)},
              'source_sha256': hashlib.sha256((ROOT / 'artifacts/preference_pairs.jsonl').read_bytes()).hexdigest(),
              'code_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                              ('train_lora.py', 'token_masking.py', 'post_training.py')},
              'tokenization': {'max_length': max(len(e['encoded']['input_ids']) for e in examples),
                               'sft_supervision': 'all assistant turns including tool calls and final JSON',
                               'dpo_supervision': 'final assistant answer only', 'truncated': False},
              'notes': ['Pilot on 16 training tasks/96 controlled preference pairs; one source document.',
                        '4 tasks/24 pairs held out by field family, including both subject/predicate together.',
                        'Fresh tokenizer encoding for 0.5B; no reuse of 7B NPZ or remapped vocabulary.',
                        'DPO reference is frozen post-SFT policy, cached before any DPO update.',
                        'Generation uses fixed gold tool evidence; not autonomous Agent success or cross-document generalization.']}
    save(args.output / 'report.json', report)
    print('Loading frozen base weights', flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                                              torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                          target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                                          task_type='CAUSAL_LM', bias='none'))
    report['lora'] = {'r': 8, 'alpha': 16, 'dropout': 0.0, 'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'all_parameters': sum(p.numel() for p in model.parameters())}
    corpus = Corpus.load(ROOT / 'artifacts/corpus.json')
    print('Evaluating base policy', flush=True)
    report['base'] = evaluate_pairs(model, pairs)
    report['base']['generation'] = evaluate_generation(model, tokenizer, examples, corpus)
    save(args.output / 'report.json', report)
    report['sft_training'] = train_phase(model, [e for e in examples if e['split'] == 'train'],
                                        steps=args.sft_steps, accumulation=4, learning_rate=2e-4,
                                        phase='sft', log_path=args.output / 'training_log.jsonl')
    model.save_pretrained(args.output / 'sft_adapter')
    tokenizer.save_pretrained(args.output / 'sft_adapter')
    report['training_runs'] = 1
    report['optimizer_steps'] = args.sft_steps
    save(args.output / 'report.json', report)
    print('Evaluating SFT and caching fixed DPO reference', flush=True)
    report['sft'] = evaluate_pairs(model, pairs)
    report['sft']['generation'] = evaluate_generation(model, tokenizer, examples, corpus)
    reference = {row['id']: row for row in report['sft']['rows']}
    save(args.output / 'reference_logps.json', reference)
    save(args.output / 'report.json', report)
    report['dpo_training'] = train_phase(model, [p for p in pairs if p['split'] == 'train'],
                                        steps=args.dpo_steps, accumulation=2, learning_rate=5e-5,
                                        phase='dpo', log_path=args.output / 'training_log.jsonl', reference=reference)
    model.save_pretrained(args.output / 'dpo_adapter')
    tokenizer.save_pretrained(args.output / 'dpo_adapter')
    report['training_runs'] = 2
    report['optimizer_steps'] += args.dpo_steps
    save(args.output / 'report.json', report)
    print('Evaluating DPO policy', flush=True)
    report['dpo'] = evaluate_pairs(model, pairs, reference)
    report['dpo']['generation'] = evaluate_generation(model, tokenizer, examples, corpus)
    report['complete'] = True
    save(args.output / 'report.json', report)
    print(json.dumps({stage: report[stage]['summary'] for stage in ('base', 'sft', 'dpo')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
