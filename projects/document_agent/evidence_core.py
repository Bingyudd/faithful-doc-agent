"""CPU-only document evidence tools. No model inference or training is claimed."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import pdfplumber
from pydantic import BaseModel, ConfigDict, Field, model_validator


CATALOG = {
    'outer_diameter': ['outer diameter'],
    'inner_diameter': ['inner diameter'],
    'working_length': ['working length'],
    'field_of_view': ['field of view'],
    'direction_of_view': ['direction of view'],
    'depth_of_field': ['depth of field'],
    'deflection': ['deflection'],
    'image_sensor': ['image sensor'],
    'light_source': ['light source'],
    'number_of_uses': ['number of users'],
    'regulatory_class': ['regulatory class', 'device class'],
    'product_code': ['product code'],
    'device_name': ['device name', 'trade/device name', 'device trade name'],
}


def norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().casefold()


def tokens(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', text.casefold())


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Element(StrictModel):
    element_id: str
    document_id: str
    pdf_page: int = Field(ge=1)
    kind: Literal['line', 'table_cell']
    text: str = Field(min_length=1)
    context: str = ''
    bbox: tuple[float, float, float, float]
    scope: Literal['subject', 'predicate', 'unknown'] = 'unknown'

    @model_validator(mode='after')
    def ordered_box(self):
        if self.bbox[0] > self.bbox[2] or self.bbox[1] > self.bbox[3]:
            raise ValueError('Invalid source coordinates')
        return self


class EvidenceRef(StrictModel):
    element_id: str
    pdf_page: int = Field(ge=1)
    quote: str = Field(min_length=1)


class Task(StrictModel):
    task_id: str
    document_id: str
    field: str
    query: str = Field(min_length=1)
    scope: Literal['subject', 'predicate', 'any'] = 'subject'

    @model_validator(mode='after')
    def allowed_field(self):
        if self.field not in CATALOG:
            raise ValueError('Field not in server-side catalog')
        return self


class Claim(StrictModel):
    field: str
    status: Literal['supported', 'missing', 'conflict']
    value: str | None = None
    unit: str | None = None
    qualifier: Literal['max', 'min'] | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode='after')
    def state_consistency(self):
        if self.status == 'supported' and (not self.value or not self.evidence):
            raise ValueError('Supported claim needs a value and evidence')
        if self.status in ('missing', 'conflict') and self.value is not None:
            raise ValueError('Unresolved claim must not contain an invented value')
        if self.status == 'conflict' and len(self.evidence) < 2:
            raise ValueError('Conflict needs at least two references')
        return self


class SearchArgs(StrictModel):
    query: str = Field(min_length=1)
    document_id: str | None = None
    kind: Literal['line', 'table_cell'] | None = None
    scope: Literal['subject', 'predicate', 'any'] = 'any'
    top_k: int = Field(default=5, ge=1, le=20)


class ReadArgs(StrictModel):
    element_id: str = Field(min_length=1)


class EmptyArgs(StrictModel):
    pass


class Corpus:
    def __init__(self, elements: list[Element], pages: list[dict], documents: list[dict]):
        self.elements = elements
        self.by_id = {e.element_id: e for e in elements}
        self.pages = pages
        self.documents = documents
        if len(self.by_id) != len(elements):
            raise ValueError('Duplicate element identifiers')

    @classmethod
    def parse(cls, folder: Path):
        elements, pages, documents = [], [], []
        for path in sorted(folder.glob('*.pdf')):
            document_id = path.stem
            raw = path.read_bytes()
            with pdfplumber.open(path) as pdf:
                documents.append({'document_id': document_id, 'pages': len(pdf.pages),
                                  'sha256': hashlib.sha256(raw).hexdigest()})
                for page_no, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ''
                    sparse = len(re.sub(r'\s+', '', text)) < 50
                    pages.append({'document_id': document_id, 'pdf_page': page_no,
                                  'characters': len(page.chars), 'needs_ocr_review': sparse})
                    # Lines keep exact page coordinates; no fabricated OCR fallback.
                    for line_no, line in enumerate(page.extract_text_lines(return_chars=False), 1):
                        if not line['text'].strip():
                            continue
                        elements.append(Element(element_id=f'{document_id}:p{page_no}:l{line_no}',
                            document_id=document_id, pdf_page=page_no, kind='line', text=line['text'],
                            bbox=(line['x0'], line['top'], line['x1'], line['bottom'])))
                    for table_no, table in enumerate(page.find_tables(), 1):
                        matrix = table.extract()
                        # Merged headers and body rows can have different logical
                        # column indices. Associate them by horizontal geometry.
                        header_regions = []
                        for hi, header_row in enumerate(table.rows[:3]):
                            for hj, hb in enumerate(header_row.cells):
                                if hb is None or hi >= len(matrix) or hj >= len(matrix[hi]):
                                    continue
                                ht = matrix[hi][hj] or ''
                                for keyword, hs in [('subject device', 'subject'), ('predicate device', 'predicate')]:
                                    if keyword in norm(ht):
                                        header_regions.append((hb, ht, hs))
                        for row_no, row in enumerate(table.rows):
                            if row_no >= len(matrix):
                                continue
                            row_values = matrix[row_no]
                            row_label = row_values[0] or '' if row_values else ''
                            for col_no, box in enumerate(row.cells):
                                if box is None or col_no >= len(row_values):
                                    continue
                                cell = row_values[col_no] or ''
                                if not cell.strip():
                                    continue
                                matches = []
                                for hb, ht, hs in header_regions:
                                    overlap = max(0, min(box[2], hb[2]) - max(box[0], hb[0]))
                                    if overlap / max(1, hb[2] - hb[0]) >= 0.8:
                                        matches.append((ht, hs))
                                scopes = {m[1] for m in matches}
                                scope = next(iter(scopes)) if len(scopes) == 1 else 'unknown'
                                header = ' '.join(dict.fromkeys(m[0] for m in matches))
                                elements.append(Element(element_id=f'{document_id}:p{page_no}:t{table_no}:r{row_no}:c{col_no}',
                                    document_id=document_id, pdf_page=page_no, kind='table_cell', text=cell,
                                    context=f'column: {header} | row: {row_label}', bbox=box, scope=scope))
        return cls(elements, pages, documents)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'documents': self.documents, 'pages': self.pages,
                         'elements': [e.model_dump() for e in self.elements]}, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls([Element.model_validate(e) for e in data['elements']], data['pages'], data['documents'])

    def search(self, args: SearchArgs) -> list[dict]:
        pool = [e for e in self.elements if
                (args.document_id is None or e.document_id == args.document_id) and
                (args.kind is None or e.kind == args.kind) and
                (args.scope == 'any' or e.scope == args.scope)]
        if not pool:
            return []
        vectors = [Counter(tokens(e.context + ' ' + e.text)) for e in pool]
        lengths = [sum(c.values()) for c in vectors]
        mean_len = sum(lengths) / len(lengths) or 1
        df = Counter(t for v in vectors for t in v)
        query = set(tokens(args.query))
        scores = []
        for e, tf, length in zip(pool, vectors, lengths):
            score = 0.0
            for t in query:
                frequency = tf[t]
                if not frequency:
                    continue
                idf = math.log(1 + (len(pool) - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / mean_len))
            if score > 0:
                scores.append({'element_id': e.element_id, 'score': round(score, 6),
                               'pdf_page': e.pdf_page, 'kind': e.kind, 'scope': e.scope,
                               'text': e.text, 'context': e.context})
        return sorted(scores, key=lambda r: (-r['score'], r['element_id']))[:args.top_k]

    def read(self, element_id: str):
        if element_id not in self.by_id:
            raise KeyError('Unknown source element')
        return self.by_id[element_id].model_dump()


def contains_value(value: str, text: str):
    # Exact phrase first; token-bound numeric fallback prevents 5 matching 5.85.
    if norm(value) in norm(text):
        if not re.fullmatch(r'\d+(?:\.\d+)?', value.strip()):
            return True
    numeric = re.fullmatch(r'\d+(?:\.\d+)?', value.strip())
    if numeric:
        return float(value) in [float(s) for s in re.findall(
            r'(?<![\w.])\d+(?:\.\d+)?(?=$|[^\w.]|mm\b|cm\b|m\b|fr\b|deg\b)', text.casefold())]
    return norm(value) in norm(text)


def audit_claim(corpus: Corpus, task: Task, payload: dict):
    try:
        claim = Claim.model_validate(payload)
    except ValueError as exc:
        return {'accepted': False, 'errors': ['schema_error'], 'details': str(exc)}
    errors = []
    if claim.field != task.field:
        errors.append('wrong_field')
    if claim.status != 'supported':
        # An abstention is not a validated answer or an automatic conflict diagnosis.
        return {'accepted': False, 'errors': ['unresolved_' + claim.status], 'claim': claim.model_dump()}
    for ref in claim.evidence:
        element = corpus.by_id.get(ref.element_id)
        if element is None:
            errors.append('unknown_anchor')
            continue
        if element.document_id != task.document_id:
            errors.append('wrong_document')
        if element.pdf_page != ref.pdf_page:
            errors.append('wrong_page')
        if task.scope != 'any' and element.scope != task.scope:
            errors.append('wrong_or_unknown_scope')
        if norm(ref.quote) not in norm(element.text):
            errors.append('quote_not_in_source')
        if not contains_value(claim.value or '', ref.quote):
            errors.append('value_not_in_quote')
        # A candidate must not evade a bound by quoting only the bare number.
        source_bounds = [bound for bound in ('max', 'min') if re.search(r'\b' + bound + r'\b', norm(element.text))]
        if source_bounds and claim.qualifier not in source_bounds:
            errors.append('missing_or_wrong_bound_qualifier')
        if claim.qualifier and claim.qualifier not in source_bounds:
            errors.append('qualifier_not_in_source')
        field_context = norm(element.context + ' ' + element.text)
        if not any(alias in field_context for alias in CATALOG.get(task.field, [])):
            errors.append('field_context_mismatch')
        if claim.unit:
            unit = '°' if norm(claim.unit) in ('degree', 'degrees', 'deg') else norm(claim.unit)
            if unit not in field_context:
                errors.append('unit_not_in_source_context')
    return {'accepted': not errors, 'errors': sorted(set(errors)), 'claim': claim.model_dump(),
            'scope': 'mechanical_provenance_check_not_semantic_or_regulatory_certification'}


class TraceStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, task TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (run_id TEXT, step INTEGER, payload TEXT NOT NULL, PRIMARY KEY(run_id,step))')
        self.db.commit()

    def start(self, task: Task):
        run_id = uuid.uuid4().hex
        self.db.execute('INSERT INTO runs VALUES (?,?)', (run_id, task.model_dump_json()))
        self.db.commit()
        return run_id

    def append(self, run_id: str, event: dict):
        with self.db:
            if not self.db.execute('SELECT 1 FROM runs WHERE run_id=?', (run_id,)).fetchone():
                raise ValueError('Unknown run')
            step = self.db.execute('SELECT COALESCE(MAX(step),0)+1 FROM events WHERE run_id=?', (run_id,)).fetchone()[0]
            self.db.execute('INSERT INTO events VALUES (?,?,?)', (run_id, step, json.dumps(event, ensure_ascii=False)))

    def read(self, run_id: str):
        return [json.loads(row[0]) for row in self.db.execute('SELECT payload FROM events WHERE run_id=? ORDER BY step', (run_id,))]


def tool_schemas():
    return {'list_pages': EmptyArgs.model_json_schema(), 'search_evidence': SearchArgs.model_json_schema(),
            'read_element': ReadArgs.model_json_schema(), 'audit_claim': Claim.model_json_schema()}


def call_tool(corpus: Corpus, task: Task, name: str, arguments: dict):
    """Typed tool dispatch with task-bound document scope; no shell execution."""
    try:
        if name == 'list_pages':
            EmptyArgs.model_validate(arguments)
            data = [p for p in corpus.pages if p['document_id'] == task.document_id]
        elif name == 'search_evidence':
            args = SearchArgs.model_validate(arguments)
            if args.document_id not in (None, task.document_id):
                return {'ok': False, 'error': 'document_scope_violation'}
            args.document_id = task.document_id
            data = corpus.search(args)
        elif name == 'read_element':
            args = ReadArgs.model_validate(arguments)
            element = corpus.by_id.get(args.element_id)
            if element is None:
                return {'ok': False, 'error': 'unknown_anchor'}
            if element.document_id != task.document_id:
                return {'ok': False, 'error': 'document_scope_violation'}
            data = element.model_dump()
        elif name == 'audit_claim':
            data = audit_claim(corpus, task, arguments)
        else:
            return {'ok': False, 'error': 'unknown_tool'}
        return {'ok': True, 'data': data}
    except ValueError as exc:
        return {'ok': False, 'error': 'invalid_arguments', 'details': str(exc)}


def replay_offline(corpus: Corpus, task: Task, candidate: dict, store: TraceStore):
    """A deterministic tool replay, deliberately not represented as an LLM agent."""
    run_id = store.start(task)
    search_args = {'query':task.query, 'document_id':task.document_id,
                   'kind':'table_cell', 'scope':task.scope, 'top_k':5}
    store.append(run_id, {'tool': 'search_evidence', 'arguments':search_args,
                         'result': call_tool(corpus, task, 'search_evidence', search_args)})
    for ref in candidate.get('evidence', []):
        read_args = {'element_id':ref['element_id']}
        store.append(run_id, {'tool': 'read_element', 'arguments':read_args,
                             'result':call_tool(corpus, task, 'read_element', read_args)})
    verdict = audit_claim(corpus, task, candidate)
    store.append(run_id, {'tool': 'audit_claim', 'result': verdict})
    return {'run_id': run_id, 'verdict': verdict}


def filter_trajectories(corpus: Corpus, records: list[dict]):
    kept, rejected = [], []
    for record in records:
        task = Task.model_validate(record['task'])
        verdict = audit_claim(corpus, task, record['claim'])
        trace_errors = []
        observed = set()
        messages = record.get('messages', [])
        for i, message in enumerate(messages):
            if message.get('role') != 'tool':
                continue
            try:
                call = json.loads(messages[i - 1]['content'])
                if i == 0 or messages[i - 1].get('role') != 'assistant' or call['tool'] != 'read_element':
                    raise ValueError('Invalid preceding tool call')
                element_id = call['arguments']['element_id']
                source = corpus.by_id[element_id]
                if norm(message['content']) != norm(source.text):
                    trace_errors.append('observation_mismatch')
                else:
                    observed.add(element_id)
            except (KeyError, ValueError, TypeError, IndexError):
                trace_errors.append('invalid_tool_observation')
        for ref in record['claim'].get('evidence', []):
            if ref['element_id'] not in observed:
                trace_errors.append('unobserved_evidence')
        if not messages or messages[-1].get('role') != 'assistant':
            trace_errors.append('missing_final_answer')
        else:
            try:
                if json.loads(messages[-1]['content']) != record['claim']:
                    trace_errors.append('final_answer_mismatch')
            except (ValueError, TypeError):
                trace_errors.append('final_answer_mismatch')
        if trace_errors:
            verdict['accepted'] = False
            verdict['errors'] = sorted(set(verdict['errors'] + trace_errors))
        if not record.get('answer_matches_expected'):
            verdict['accepted'] = False
            verdict['errors'] = sorted(set(verdict['errors'] + ['answer_mismatch']))
        if verdict['accepted']:
            kept.append({'id': record['id'], 'messages': record['messages']})
        else:
            rejected.append({'id': record['id'], 'errors': verdict['errors']})
    return kept, rejected


def supervision_roles(messages: list[dict]):
    """Message-level metadata; token labels are implemented in token_masking.py."""
    return [{'message_index': i, 'role': m['role'], 'supervise': m['role'] == 'assistant'}
            for i, m in enumerate(messages)]
