"""Exercise tool-call protocol using scripted providers; no model requests."""
from __future__ import annotations

import json
from pathlib import Path

from evidence_core import Corpus, Task, TraceStore
from run_project import gold_cases
from tool_protocol import check_messages, function_tools, reexecute_trace, run_tool_loop

ROOT = Path(__file__).resolve().parent


def scripted_provider(record):
    element_id = record['claim']['evidence'][0]['element_id']
    identifier = 'call_' + record['id']

    def provide(messages, tools):
        if messages[-1]['role'] != 'tool':
            return {'role':'assistant','content':None,'tool_calls':[{
                'id':identifier,'type':'function','function':{
                    'name':'read_element','arguments':json.dumps({'element_id':element_id})}}]}
        return {'role':'assistant','content':json.dumps(record['claim'],ensure_ascii=False)}
    return provide


def run():
    output = ROOT / 'artifacts'
    corpus = Corpus.load(output / 'corpus.json')
    store = TraceStore(output / 'traces.sqlite')
    rows = []
    replay_checks = []
    for record in gold_cases(corpus):
        result = run_tool_loop(corpus, Task(**record['task']), scripted_provider(record), store,
            provider_label='scripted_source_fixture_no_llm', max_steps=4, max_tool_calls=4)
        check_messages(result['messages'])
        if result['status'] != 'validated':
            raise AssertionError(result)
        replay = reexecute_trace(corpus, store, result['run_id'])
        if not replay or not all(row['matches_recorded'] for row in replay):
            raise AssertionError('Reexecution differs from recorded source tools')
        replay_checks += replay
        rows.append({'id':record['id'], 'run_id':result['run_id'],
                     'origin':'scripted_protocol_fixture_no_llm',
                     'tools':function_tools(), 'messages':result['messages']})
    with (output / 'sft_messages_function_calling.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False) + '\n')
    (output / 'function_calling_tools.json').write_text(json.dumps(function_tools(),ensure_ascii=False,indent=2),encoding='utf-8')
    report = {'wire_format':'chat_completions_function_tools_text_subset',
        'function_tools':len(function_tools()), 'protocol_fixture_conversations':len(rows),
        'validated_fixture_conversations':len(rows),
        'reexecuted_tool_events':len(replay_checks),
        'reexecution_matches':sum(row['matches_recorded'] for row in replay_checks),
        'provider_origin':'scripted_source_fixture', 'live_model_requests':0,
        'notes':['Local protocol integration checks; not a model task-success benchmark.',
                 'Task-bound tool dispatch, call-id correlation, budgets and final evidence checks are exercised.']}
    (output / 'protocol_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output / 'function_calling_preview.json').write_text(json.dumps(rows[0],ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    run()
