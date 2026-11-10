"""Chat Completions function-call envelopes and a bounded provider-agnostic loop.

No network client or API key is created here. Providers are injected callables.
The verification run uses explicit scripted fixtures, not model inference.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Callable

from evidence_core import Corpus, Task, TraceStore, audit_claim, call_tool, tool_schemas


class ProtocolError(ValueError):
    pass


DESCRIPTIONS = {
    'list_pages': 'List source pages and flags requiring OCR review for the task document.',
    'search_evidence': 'Search source elements inside the task document.',
    'read_element': 'Read one source element with its physical PDF page and coordinates.',
    'audit_claim': 'Check a structured field claim against source evidence and task scope.',
}


def function_tools():
    return [{'type':'function', 'function':{'name':name, 'description':DESCRIPTIONS[name],
             'parameters':schema, 'strict':False}} for name, schema in tool_schemas().items()]


def decode_arguments(raw):
    if not isinstance(raw, str):
        raise ProtocolError('Chat Completions function.arguments must be a JSON string')
    try:
        result = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ProtocolError('Malformed function arguments') from exc
    if not isinstance(result, dict):
        raise ProtocolError('Function arguments must decode to an object')
    return result


def check_messages(messages):
    pending, seen = set(), set()
    for message in messages:
        role = message.get('role')
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ProtocolError('Unsupported role')
        if role == 'tool':
            identifier = message.get('tool_call_id')
            if identifier not in pending:
                raise ProtocolError('Tool response does not match a pending call')
            if not isinstance(message.get('content'), str):
                raise ProtocolError('Tool response content must be a string')
            pending.remove(identifier)
            continue
        if pending:
            raise ProtocolError('Pending tool calls must be resolved before the next message')
        if role == 'assistant' and message.get('tool_calls'):
            for call in message['tool_calls']:
                identifier = call.get('id')
                if not isinstance(identifier, str) or not identifier or identifier in seen:
                    raise ProtocolError('Missing or duplicate tool-call identifier')
                if call.get('type') != 'function' or not isinstance(call.get('function', {}).get('name'), str):
                    raise ProtocolError('Invalid function tool call')
                if not isinstance(call['function'].get('arguments'), str):
                    raise ProtocolError('Function arguments must be a wire-format string')
                seen.add(identifier)
                pending.add(identifier)
        elif not isinstance(message.get('content'), str):
            raise ProtocolError('Text message content must be a string')
    if pending:
        raise ProtocolError('Unresolved tool calls at end of conversation')
    return True


def qwen_messages(messages):
    """HF Qwen expects argument objects; API wire envelopes use JSON strings."""
    check_messages(messages)
    result = copy.deepcopy(messages)
    for message in result:
        for call in message.get('tool_calls', []):
            call['function']['arguments'] = decode_arguments(call['function']['arguments'])
    return result


def run_tool_loop(corpus: Corpus, task: Task, provider: Callable, store: TraceStore,
                  *, provider_label='injected_provider', max_steps=6, max_tool_calls=12):
    if not 1 <= max_steps <= 64 or not 1 <= max_tool_calls <= 128:
        raise ValueError('Invalid execution budget')
    messages = [
        {'role':'system', 'content':'Use source tools within the task document. Return a structured field claim as JSON; do not invent missing evidence.'},
        {'role':'user', 'content':task.model_dump_json()},
    ]
    run_id = store.start(task)
    fingerprints = {d['document_id']:d['sha256'] for d in corpus.documents if d['document_id'] == task.document_id}
    store.append(run_id, {'event_type':'start', 'provider':provider_label,
                         'source_fingerprints':fingerprints, 'max_steps':max_steps,'max_tool_calls':max_tool_calls})
    seen_ids = set()
    observed_elements = set()
    used = 0
    for step in range(max_steps):
        response = provider(copy.deepcopy(messages), copy.deepcopy(function_tools()))
        if 'choices' in response:
            response = response['choices'][0]['message']
        message = copy.deepcopy(response)
        if message.get('role', 'assistant') != 'assistant':
            raise ProtocolError('Provider must return an assistant message')
        message['role'] = 'assistant'
        calls = message.get('tool_calls') or []
        if not isinstance(calls, list):
            raise ProtocolError('tool_calls must be a list')
        if calls and used + len(calls) > max_tool_calls:
            store.append(run_id, {'event_type':'stop', 'reason':'tool_budget', 'step':step})
            return {'run_id':run_id, 'status':'tool_budget', 'messages':messages, 'tool_calls':used}
        for call in calls:
            identifier = call.get('id')
            if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                raise ProtocolError('Missing or reused tool_call_id')
            if call.get('type') != 'function' or not isinstance(call.get('function', {}).get('name'), str):
                raise ProtocolError('Invalid function call')
            seen_ids.add(identifier)
        messages.append(message)
        store.append(run_id, {'event_type':'assistant_message', 'step':step, 'message':message})
        if not calls:
            try:
                claim = json.loads(message.get('content') or '')
                verdict = audit_claim(corpus, task, claim)
                if isinstance(claim, dict) and any(ref.get('element_id') not in observed_elements for ref in claim.get('evidence', [])):
                    verdict['accepted'] = False
                    verdict['errors'] = sorted(set(verdict['errors'] + ['unobserved_evidence']))
            except (ValueError, TypeError):
                verdict = {'accepted':False, 'errors':['invalid_final_json']}
            status = 'validated' if verdict['accepted'] else 'needs_review'
            store.append(run_id, {'event_type':'stop','reason':status,'verdict':verdict})
            check_messages(messages)
            return {'run_id':run_id, 'status':status, 'messages':messages, 'verdict':verdict, 'tool_calls':used}
        for call in calls:
            name = call['function']['name']
            try:
                arguments = decode_arguments(call['function'].get('arguments'))
                result = call_tool(corpus, task, name, arguments)
            except ProtocolError:
                arguments = None
                result = {'ok':False,'error':'invalid_json_arguments'}
            used += 1
            if name == 'read_element' and result.get('ok'):
                observed_elements.add(result['data']['element_id'])
            tool_message = {'role':'tool', 'tool_call_id':call['id'], 'content':json.dumps(result, ensure_ascii=False)}
            messages.append(tool_message)
            store.append(run_id, {'event_type':'tool_result','step':step, 'tool_call_id':call['id'],
                                 'name':name,'arguments':arguments,'result':result})
    store.append(run_id, {'event_type':'stop','reason':'step_budget'})
    return {'run_id':run_id, 'status':'step_budget','messages':messages,'tool_calls':used}


def reexecute_trace(corpus, store, run_id):
    row = store.db.execute('SELECT task FROM runs WHERE run_id=?', (run_id,)).fetchone()
    if row is None:
        raise ValueError('Unknown run')
    task = Task.model_validate_json(row[0])
    events = store.read(run_id)
    header = next((event for event in events if event.get('event_type') == 'start'), None)
    if header is None:
        raise ValueError('Trace missing source fingerprints')
    actual = {d['document_id']:d['sha256'] for d in corpus.documents if d['document_id'] == task.document_id}
    if header['source_fingerprints'] != actual:
        raise ValueError('Source document fingerprint changed')
    rows = []
    for event in events:
        if event.get('event_type') != 'tool_result':
            continue
        if event['arguments'] is None:
            fresh = {'ok':False,'error':'invalid_json_arguments'}
        else:
            fresh = call_tool(corpus, task, event['name'], event['arguments'])
        # SQLite JSON round-tripping turns coordinate tuples into arrays.
        matches = json.dumps(fresh,sort_keys=True) == json.dumps(event['result'],sort_keys=True)
        rows.append({'tool_call_id':event['tool_call_id'], 'matches_recorded':matches})
    return rows
