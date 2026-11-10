"""Reproduce the offline evidence checks and export inspectable artifacts."""
from __future__ import annotations

import argparse
import copy
import json
import platform
from collections import Counter
from pathlib import Path

import pdfplumber
import pydantic

from evidence_core import (Claim, Corpus, SearchArgs, Task, TraceStore, audit_claim,
                           filter_trajectories, replay_offline, supervision_roles, tool_schemas)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'

# Manually transcribed after inspecting the rendered source PDF pages 7 and 8.
# These are source facts, not model-generated ground truth or regulatory advice.
SOURCE_FACTS = [
    ('outer_diameter', 7, 7, '5.0', '5.85', 'mm', 'max'),
    ('inner_diameter', 7, 8, '2.0', '2.0', 'mm', 'min'),
    ('working_length', 8, 4, '380', '380', 'mm', None),
    ('deflection', 8, 5, '210°up, 180°down', '160° up, 130° down', 'degrees', None),
    ('image_sensor', 8, 7, 'CMOS', 'Color CCD', None, None),
    ('field_of_view', 8, 8, '110', '120', 'degrees', None),
    ('direction_of_view', 8, 9, '0', '0', 'degrees', None),
    ('depth_of_field', 8, 10, '5mm~100mm', '3mm~50mm', 'mm', None),
    ('light_source', 8, 11, 'Internal LEDs', 'External light source', None, None),
    ('number_of_uses', 8, 16, 'Single-Use', 'Reusable', None, None),
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def gold_cases(corpus):
    cases = []
    for field, page, row, subject_value, predicate_value, unit, qualifier in SOURCE_FACTS:
        for scope, column, value in [('subject', 1, subject_value), ('predicate', 4, predicate_value)]:
            element_id = f'K231105:p{page}:t1:r{row}:c{column}'
            element = corpus.by_id[element_id]
            case_id = field + '_' + scope
            task = Task(task_id=case_id, document_id='K231105', field=field,
                        query=field.replace('_', ' '), scope=scope)
            claim = Claim(field=field, status='supported', value=value, unit=unit, qualifier=qualifier,
                          evidence=[{'element_id': element_id, 'pdf_page': page, 'quote': element.text}])
            cases.append({'id': case_id, 'task': task.model_dump(), 'claim': claim.model_dump(),
                          'source_note': 'manual transcription from rendered FDA PDF p7/p8'})
    return cases


def mutate_cases(gold):
    variants = []
    by_id = {r['id']: r for r in gold}
    for record in gold:
        for change in ['clean', 'wrong_value', 'wrong_page', 'unknown_anchor', 'fabricated_quote', 'wrong_scope', 'extra_schema_field']:
            r = copy.deepcopy(record)
            r['id'] += ':' + change
            r['perturbation'] = change
            r['expected_accept'] = change == 'clean'
            if change == 'wrong_value':
                r['claim']['value'] = 'UNSUPPORTED_VALUE_999'
            elif change == 'wrong_page':
                r['claim']['evidence'][0]['pdf_page'] = 1
            elif change == 'unknown_anchor':
                r['claim']['evidence'][0]['element_id'] = 'K231105:p99:t999:r1:c1'
            elif change == 'fabricated_quote':
                r['claim']['evidence'][0]['quote'] = 'FABRICATED source says ' + r['claim']['value']
            elif change == 'wrong_scope':
                other = record['id'].replace('_subject', '_predicate') if record['task']['scope'] == 'subject' else record['id'].replace('_predicate', '_subject')
                r['claim'] = copy.deepcopy(by_id[other]['claim'])
            elif change == 'extra_schema_field':
                r['claim']['unverified_approval'] = True
            variants.append(r)
    return variants


def messages_for(record):
    task = record['task']
    return [
        {'role': 'system', 'content': 'Answer the requested field with cited source evidence.'},
        {'role': 'user', 'content': task['query'] + ' | scope=' + task['scope']},
        {'role': 'assistant', 'content': json.dumps({'tool': 'read_element', 'arguments': {'element_id': record['claim']['evidence'][0]['element_id']}})},
        {'role': 'tool', 'content': record['claim']['evidence'][0]['quote']},
        {'role': 'assistant', 'content': json.dumps(record['claim'], ensure_ascii=False)},
    ]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = Corpus.parse(ROOT / 'data' / 'public')
    corpus.save(OUT / 'corpus.json')
    gold = gold_cases(corpus)
    variants = mutate_cases(gold)
    write_json(OUT / 'gold_source_facts.json', gold)
    write_json(OUT / 'controlled_validation_cases.json', variants)
    verdicts = []
    for r in variants:
        verdict = audit_claim(corpus, Task.model_validate(r['task']), r['claim'])
        verdicts.append({'id': r['id'], 'perturbation': r['perturbation'], 'expected_accept': r['expected_accept'], **verdict})
    write_json(OUT / 'validation_results.json', verdicts)
    retrieval = []
    for r in gold:
        task = Task.model_validate(r['task'])
        hits = corpus.search(SearchArgs(query=task.query, document_id=task.document_id,
                                       kind='table_cell', scope=task.scope, top_k=5))
        anchor = r['claim']['evidence'][0]['element_id']
        rank = next((i for i, hit in enumerate(hits, 1) if hit['element_id'] == anchor), None)
        retrieval.append({'id': r['id'], 'gold_anchor_rank_at_5': rank, 'hits': hits})
    write_json(OUT / 'retrieval_smoke_results.json', retrieval)
    # Controlled replay records. No LLM generated these trajectories.
    records = []
    gold_by_id = {record['id']: record for record in gold}
    for r in variants:
        copied = copy.deepcopy(r)
        # Compare the actual answer string, including scope mutations where
        # some subject/predicate values happen to be numerically identical.
        original = gold_by_id[r['id'].split(':')[0]]
        copied['answer_matches_expected'] = r['claim']['value'] == original['claim']['value']
        copied['messages'] = messages_for(r)
        records.append(copied)
    kept, rejected = filter_trajectories(corpus, records)
    with (OUT / 'sft_messages_controlled_replay.jsonl').open('w', encoding='utf-8') as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(OUT / 'rejected_replay_records.json', rejected)
    write_json(OUT / 'supervision_roles_example.json', supervision_roles(records[0]['messages']))
    write_json(OUT / 'tool_schemas.json', tool_schemas())
    store = TraceStore(OUT / 'traces.sqlite')
    demo = replay_offline(corpus, Task.model_validate(gold[0]['task']), gold[0]['claim'], store)
    write_json(OUT / 'demo_tool_replay.json', {'mode': 'deterministic_offline_replay_no_llm',
                                             **demo, 'events': store.read(demo['run_id'])})
    summary = {
        'mode': 'offline_engineering_smoke_and_controlled_perturbations',
        'documents': len(corpus.documents), 'pdf_pages': len(corpus.pages),
        'elements': len(corpus.elements), 'table_cells': sum(e.kind == 'table_cell' for e in corpus.elements),
        'pages_requiring_ocr_review': sum(p['needs_ocr_review'] for p in corpus.pages),
        'source_fact_records': len(gold), 'unique_field_types': len(SOURCE_FACTS),
        'source_pages_for_fact_records': [7, 8], 'source_document_for_fact_records': 'K231105',
        'controlled_validation_cases': len(verdicts),
        'validation_expected_matches': sum(r['accepted'] == r['expected_accept'] for r in verdicts),
        'accepted_clean': sum(r['accepted'] and r['perturbation'] == 'clean' for r in verdicts),
        'accepted_corrupt': sum(r['accepted'] and r['perturbation'] != 'clean' for r in verdicts),
        'per_category': {category: {'cases': sum(r['perturbation'] == category for r in verdicts),
                                  'expected_matches': sum(r['perturbation'] == category and r['accepted'] == r['expected_accept'] for r in verdicts)}
                         for category in sorted({r['perturbation'] for r in verdicts})},
        'retrieval_gold_anchor_at_1': sum(r['gold_anchor_rank_at_5'] == 1 for r in retrieval),
        'retrieval_gold_anchor_at_5': sum(r['gold_anchor_rank_at_5'] is not None for r in retrieval),
        'outcome_only_accepted_controlled_replays': sum(r['answer_matches_expected'] for r in records),
        'evidence_validated_replays_kept': len(kept),
        'llm_inference_calls': 0, 'model_training_runs': 0,
        'packages': {'python': platform.python_version(), 'pdfplumber': pdfplumber.__version__, 'pydantic': pydantic.__version__},
        'limitations': ['Development smoke set, not a held-out benchmark.',
                       '20 fact records are paired subject/predicate facts for 10 fields from one source document.',
                       '140 cases are controlled mutations, not 140 independent user tasks.',
                       'No semantic model evaluation, OCR, generative agent, or SFT/DPO has been executed.',
                       'Mechanical source checks cannot prove complete semantic or regulatory correctness.']
    }
    write_json(OUT / 'metrics.json', summary)
    lines = ['# Offline engineering smoke results', '', 'This is not an LLM quality or training benchmark.', '',
             '| Item | Measured count |', '|---|---|']
    for key in ['documents', 'pdf_pages', 'elements', 'table_cells', 'pages_requiring_ocr_review',
                'source_fact_records', 'controlled_validation_cases', 'validation_expected_matches',
                'accepted_clean', 'accepted_corrupt', 'retrieval_gold_anchor_at_1', 'retrieval_gold_anchor_at_5',
                'outcome_only_accepted_controlled_replays', 'evidence_validated_replays_kept',
                'llm_inference_calls', 'model_training_runs']:
        lines.append(f'| {key} | {summary[key]} |')
    lines += ['', '## Limits', ''] + ['- ' + text for text in summary['limitations']]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = [r for r in verdicts if r['accepted'] != r['expected_accept']]
    if failures:
        print('Mismatches:', json.dumps(failures[:3], ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
