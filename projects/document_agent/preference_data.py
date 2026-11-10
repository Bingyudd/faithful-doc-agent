"""Build shared-context preference pairs and transparent, hand-designed rewards."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from evidence_core import Claim, Corpus, Task, audit_claim

ROOT = Path(__file__).resolve().parent
WEIGHTS = {'format':0.2, 'answer':0.3, 'evidence':0.5}


def rule_reward(corpus, task, claim, expected_value, observed_ids):
    try:
        Claim.model_validate(claim)
        format_valid = True
    except ValidationError:
        format_valid = False
    answer_correct = claim.get('value') == expected_value
    verdict = audit_claim(corpus, task, claim)
    observed = all(ref.get('element_id') in observed_ids for ref in claim.get('evidence', []))
    evidence_valid = verdict['accepted'] and observed
    components = {'format':float(format_valid), 'answer':float(answer_correct), 'evidence':float(evidence_valid)}
    reward = sum(WEIGHTS[key] * value for key, value in components.items())
    return {'total':round(reward, 8), 'components':components,
            'errors':sorted(set(verdict['errors'] + ([] if observed else ['unobserved_evidence'])))}


def build_pairs(corpus, validation_cases, source_conversations):
    clean = {row['task']['task_id']:row for row in validation_cases if row['perturbation']=='clean'}
    conversations = {row['id']:row for row in source_conversations}
    pairs = []
    for negative in validation_cases:
        if negative['perturbation']=='clean':
            continue
        task_id = negative['task']['task_id']
        positive = clean[task_id]
        task = Task(**positive['task'])
        source = conversations[task_id]
        # All external tool observations remain identical. Only the final
        # assistant claim differs; no environmental failure is blamed on policy.
        prompt = copy.deepcopy(source['messages'][:-1])
        observed = {positive['claim']['evidence'][0]['element_id']}
        chosen_reward = rule_reward(corpus,task,positive['claim'],positive['claim']['value'],observed)
        rejected_reward = rule_reward(corpus,task,negative['claim'],positive['claim']['value'],observed)
        if not chosen_reward['total'] > rejected_reward['total']:
            raise ValueError('Preference does not have a verified reward gap')
        hard = rejected_reward['components']['answer']==1.0 and rejected_reward['components']['evidence']==0.0
        pairs.append({'id':negative['id'], 'task_id':task_id,
            'source_document_id':task.document_id, 'split_group':task.document_id,
            'pair_type':'final_claim_under_identical_observations',
            'origin':'controlled_perturbation_not_model_rollout',
            'negative_category':negative['perturbation'], 'hard_negative':hard,
            'prompt':prompt,
            'chosen':[{'role':'assistant','content':json.dumps(positive['claim'],ensure_ascii=False)}],
            'rejected':[{'role':'assistant','content':json.dumps(negative['claim'],ensure_ascii=False)}],
            'tools':copy.deepcopy(source['tools']),
            'chosen_reward':chosen_reward, 'rejected_reward':rejected_reward})
    return pairs


def run():
    output = ROOT/'artifacts'
    corpus = Corpus.load(output/'corpus.json')
    cases = json.loads((output/'controlled_validation_cases.json').read_text(encoding='utf-8'))
    conversations = [json.loads(line) for line in (output/'sft_messages_function_calling.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    pairs = build_pairs(corpus,cases,conversations)
    with (output/'preference_pairs.jsonl').open('w',encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair,ensure_ascii=False)+'\n')
    hard_pairs = [pair for pair in pairs if pair['hard_negative']]
    with (output/'hard_negative_pairs.jsonl').open('w',encoding='utf-8') as f:
        for pair in hard_pairs:
            f.write(json.dumps(pair,ensure_ascii=False)+'\n')
    report = {'source_fact_families':len({pair['task_id'] for pair in pairs}),
        'preference_pairs':len(pairs), 'same_answer_invalid_evidence_pairs':len(hard_pairs),
        'categories':dict(Counter(pair['negative_category'] for pair in pairs)),
        'reward_weights':WEIGHTS, 'weights_are_hand_designed':True,
        'shared_prompt_and_tool_observations':True,
        'chosen_reward_min':min(pair['chosen_reward']['total'] for pair in pairs),
        'rejected_reward_values':sorted({pair['rejected_reward']['total'] for pair in pairs}),
        'minimum_reward_gap':min(pair['chosen_reward']['total']-pair['rejected_reward']['total'] for pair in pairs),
        'model_training_runs':0,
        'notes':['120 pairs are controlled variations from 20 families, not independent user tasks.',
                 'Only one source document is annotated; these artifacts do not support a source-held-out evaluation.',
                 'Group future train/test splits by source document and keep paired variations together.',
                 'Rules and weights are not a learned reward model or an executed RL algorithm.']}
    (output/'preference_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    run()
