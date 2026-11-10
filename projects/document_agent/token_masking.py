"""Qwen native-template encoding and assistant-only token labels, without model weights."""
from __future__ import annotations

from local_dependencies import enable
enable()

import json
from pathlib import Path

import numpy as np
import transformers
from transformers import AutoTokenizer

from tool_protocol import function_tools, qwen_messages

ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = ROOT / 'data' / 'tokenizer_qwen25_7b'


class EmptySupervision(ValueError):
    pass


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True,
        local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('Fast offsets are required')
    return tokenizer


def render_with_spans(tokenizer, api_messages, tools=None):
    messages = qwen_messages(api_messages)
    options = {'tokenize':False, 'tools':tools or function_tools()}
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    spans = []
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        if i == 0:
            raise ValueError('Assistant requires preceding context')
        prefix = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, **options)
        turn = tokenizer.apply_chat_template(messages[:i + 1], add_generation_prompt=False, **options)
        if not turn.startswith(prefix) or not full.startswith(turn):
            raise ValueError('Native template is not prefix-stable for this conversation')
        # Native Qwen end-of-turn is a prediction target; trailing formatting LF is not.
        end = len(turn.removesuffix('\n'))
        if not turn[:end].endswith(tokenizer.eos_token):
            raise ValueError('Unexpected native assistant turn ending')
        spans.append({'message_index':i, 'start':len(prefix), 'end':end})
    return full, spans


def encode_supervised(tokenizer, messages, *, tools=None, max_length=4096, supervise_from_message=0):
    if max_length <= 0:
        raise ValueError('max_length must be positive')
    text, spans = render_with_spans(tokenizer, messages, tools)
    spans = [span for span in spans if span['message_index'] >= supervise_from_message]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    native_ids = tokenizer.apply_chat_template(qwen_messages(messages), tools=tools or function_tools(),
        tokenize=True, add_generation_prompt=False)
    if ids != native_ids:
        raise ValueError('Encoding differs from native tokenizer chat template')
    owners = []
    cross_boundary = 0
    for start, end in offsets:
        owner = next((span['message_index'] for span in spans
                      if end > start and start >= span['start'] and end <= span['end']), None)
        if owner is None and end > start and any(start < span['end'] and end > span['start'] for span in spans):
            cross_boundary += 1
        owners.append(owner)
    labels = [token if owner is not None else -100 for token, owner in zip(ids, owners)]
    original_length = len(ids)
    if len(ids) > max_length:
        # Never cut a structured tool call or an assistant JSON response in half.
        complete_ends = []
        for span in spans:
            indices = [i for i, (start, end) in enumerate(offsets)
                       if end > start and start >= span['start'] and end <= span['end']]
            if indices and indices[-1] + 1 <= max_length:
                complete_ends.append(indices[-1] + 1)
        if not complete_ends:
            raise EmptySupervision('Budget contains no complete assistant turn')
        cutoff = max(complete_ends)
        ids, labels, offsets, owners = ids[:cutoff], labels[:cutoff], offsets[:cutoff], owners[:cutoff]
    if not any(value != -100 for value in labels):
        raise EmptySupervision('No assistant target tokens')
    return {'input_ids':ids, 'attention_mask':[1] * len(ids), 'labels':labels,
            'assistant_message_indices':owners, 'offset_mapping':offsets,
            'text':text, 'assistant_spans':spans, 'original_length':original_length,
            'truncated':len(ids) < original_length, 'ignored_boundary_tokens':cross_boundary}


def collate_right(tokenizer, rows, *, pad_to=None):
    size = max(len(r['input_ids']) for r in rows) if pad_to is None else pad_to
    if any(len(r['input_ids']) > size for r in rows):
        raise ValueError('Padding size shorter than an encoded example')
    batch = {'input_ids':[], 'attention_mask':[], 'labels':[]}
    for row in rows:
        padding = size - len(row['input_ids'])
        batch['input_ids'].append(row['input_ids'] + [tokenizer.pad_token_id] * padding)
        batch['attention_mask'].append(row['attention_mask'] + [0] * padding)
        batch['labels'].append(row['labels'] + [-100] * padding)
    return {key:np.asarray(value, dtype=np.int64) for key, value in batch.items()}


def export_tokenized(input_path=None):
    path = input_path or ROOT / 'artifacts' / 'sft_messages_function_calling.jsonl'
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    tokenizer = load_tokenizer()
    rows = [encode_supervised(tokenizer, record['messages'], tools=record['tools']) for record in records]
    output = ROOT / 'artifacts'
    with (output / 'sft_tokenized.jsonl').open('w', encoding='utf-8') as stream:
        for record, row in zip(records, rows):
            stream.write(json.dumps({'id':record['id'], 'input_ids':row['input_ids'],
                'attention_mask':row['attention_mask'], 'labels':row['labels']}) + '\n')
    batch = collate_right(tokenizer, rows)
    np.savez_compressed(output / 'sft_training_batch.npz', **batch)
    if not np.all(batch['labels'][batch['attention_mask'] == 0] == -100):
        raise AssertionError('Padding must not be supervised')
    if not np.all((batch['labels'] == -100) | (batch['labels'] == batch['input_ids'])):
        raise AssertionError('Every supervised target must equal its input token')
    manifest = json.loads((TOKENIZER_DIR / 'manifest.json').read_text(encoding='utf-8'))
    report = {'model_tokenizer':manifest['model_id'], 'revision':manifest['revision'],
        'transformers_version':transformers.__version__, 'examples':len(rows),
        'batch_shape':list(batch['input_ids'].shape),
        'tokens_before_padding':sum(len(r['input_ids']) for r in rows),
        'supervised_tokens':int(np.sum(batch['labels'] != -100)),
        'masked_context_tokens':sum(sum(x == -100 for x in r['labels']) for r in rows),
        'padding_tokens':int(np.sum(batch['attention_mask'] == 0)),
        'truncated_examples':sum(r['truncated'] for r in rows),
        'conservatively_ignored_boundary_tokens':sum(r['ignored_boundary_tokens'] for r in rows),
        'assistant_end_token_is_supervised':True, 'system_user_tool_and_padding_are_masked':True,
        'tokenizer_only':True, 'model_weights_loaded':False, 'training_runs':0}
    (output / 'tokenization_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    first = rows[0]
    preview = {'id':records[0]['id'], 'rendered_native_template':first['text'],
        'supervised_text':tokenizer.decode([token for token,label in zip(first['input_ids'],first['labels']) if label != -100]),
        'assistant_spans':first['assistant_spans']}
    (output / 'token_mask_preview.json').write_text(json.dumps(preview,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    export_tokenized()
