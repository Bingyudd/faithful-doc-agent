"""Fetch public source documents; freeze source-family splits before experiments."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .io import ROOT, digest, write_json, read_json, write_jsonl

DATA = ROOT / 'data/paper'


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'AI2Resu-research/1.0'}), timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def extract_pdf(path, doc_id):
    import pdfplumber
    chunks, sparse = [], []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            text = page.extract_text(layout=False) or ''
            if len(text.strip()) < 80:
                sparse.append(page_no)
            lines = text.splitlines()
            buffer = ''
            index = 0
            for line in lines + ['']:
                if (len(buffer) + len(line) > 1100 or not line) and buffer:
                    chunks.append({'id': f'{doc_id}:p{page_no}:c{index}', 'document_id': doc_id,
                                   'page': page_no, 'text': buffer.strip()})
                    index += 1
                    buffer = ''
                buffer += line + '\n' if line else ''
    return chunks, sparse


def fda():
    manifest = DATA / 'sources.json'
    if manifest.exists():
        raw_root=(DATA/'raw').resolve()
        for source in read_json(manifest)['sources']:
            path=(DATA/source['file']).resolve()
            if not path.is_relative_to(raw_root):raise ValueError('Manifest path outside raw source folder')
            payload=path.read_bytes() if path.exists() else fetch(source['url'])
            if digest(payload)!=source['sha256']:raise ValueError('Frozen source SHA mismatch: '+source['id'])
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        print('Frozen source PDFs restored/verified; manifest unchanged.', flush=True)
        return
    candidates = []
    for query in ['device_name:endoscop*', 'device_name:cystoscop*', 'device_name:ureteroscop*', 'device_name:bronchoscop*']:
        url = 'https://api.fda.gov/device/510k.json?' + urlencode({'search': query, 'limit': 100, 'sort': 'decision_date:desc'})
        try:
            candidates.extend(json.loads(fetch(url)).get('results', []))
        except Exception as exc:
            print('FDA query failed:', query, type(exc).__name__, flush=True)
    DATA.joinpath('raw').mkdir(parents=True, exist_ok=True)
    selected, chunks, applicants, seen, failures = [], [], set(), set(), []
    for item in candidates:
        kid = item.get('k_number', '')
        applicant = re.sub(r'[^a-z0-9]', '', item.get('applicant', '').lower())
        if not re.fullmatch(r'K\d{6}', kid) or applicant in applicants or kid in seen or kid in {'K231105','K243708','K252598'}:
            continue
        seen.add(kid)
        url = f'https://www.accessdata.fda.gov/cdrh_docs/pdf{kid[1:3]}/{kid}.pdf'
        path = DATA / 'raw' / f'{kid}.pdf'
        try:
            payload = path.read_bytes() if path.exists() else fetch(url)
            if not payload.startswith(b'%PDF'):
                raise ValueError('Not PDF')
            path.write_bytes(payload)
            elements, sparse = extract_pdf(path, kid)
            if sum(len(x['text']) for x in elements) < 3000:
                raise ValueError('Insufficient machine-readable source')
            selected.append({'id': kid, 'family': applicant, 'applicant': item['applicant'],
                             'device_name': item['device_name'], 'url': url,
                             'sha256': digest(payload), 'file': f'raw/{kid}.pdf',
                             'decision_date': item.get('decision_date'), 'sparse_pages': sparse})
            chunks.extend(elements)
            applicants.add(applicant)
            print('Accepted', len(selected), kid, item['device_name'][:65], flush=True)
        except Exception as exc:
            failures.append({'id': kid, 'url': url, 'error': type(exc).__name__})
            print('Skipped', kid, type(exc).__name__, flush=True)
        if len(selected) == 20:
            break
    if len(selected) != 20:
        write_json(DATA / 'discovery_partial.json', {'selected': selected, 'failures': failures})
        raise RuntimeError(f'Need 20 unique applicant families, got {len(selected)}')
    ranked = sorted(selected, key=lambda x: digest('source-split-v1:' + x['family']))
    for index, source in enumerate(ranked):
        source['split'] = 'dev' if index < 8 else 'test'
    write_json(manifest, {'version': 1, 'sources': ranked, 'split_hash': digest([(x['id'],x['split']) for x in ranked]),
                          'grouping': 'one document per normalized FDA applicant; 8 dev / 12 test', 'failures': failures})
    write_json(DATA / 'corpus.json', {'elements': chunks, 'sha256': digest(chunks)})
    print('Frozen', len(ranked), 'families,', len(chunks), 'chunks', flush=True)


def xsum():
    """Use the official dataset-viewer rows endpoint; fixed official split IDs."""
    manifest_path=DATA/'xsum_manifest.json'
    existing=read_json(manifest_path) if manifest_path.exists() else None
    all_rows = []
    for split, target, count in [('validation','dev',50), ('test','test',200)]:
        rows = []
        for offset in range(0, count, 100):
            url = 'https://datasets-server.huggingface.co/rows?' + urlencode({
                'dataset':'EdinburghNLP/xsum', 'config':'default', 'split':split,
                'offset':offset, 'length':min(100,count-offset)})
            result = json.loads(fetch(url))
            rows.extend({'id': str(x['row']['id']), 'source':x['row']['document'],
                         'reference':x['row']['summary'], 'split':target,
                         'dataset':'EdinburghNLP/xsum', 'official_split':split} for x in result['rows'])
        if len(rows) != count:
            raise ValueError('Dataset row count mismatch')
        all_rows.extend(rows)
    if existing and digest(all_rows)!=existing['sha256']:
        raise ValueError('Remote XSum ordering/content differs from frozen manifest; refuse to replace')
    for target in ('dev','test'):
        write_jsonl(DATA/f'xsum_{target}.jsonl',[r for r in all_rows if r['split']==target])
    write_json(DATA / 'xsum_manifest.json', {'dataset':'EdinburghNLP/xsum',
               'selection':'first 50 validation / first 200 test in official viewer order; frozen by IDs and content hashes',
               'samples':[{'id':x['id'],'split':x['split'],'sha256':digest(x)} for x in all_rows],
               'sha256':digest(all_rows), 'raw_text_in_git':False})
    print('Frozen XSum:',len(all_rows),'examples',flush=True)


def provenance():
    if (DATA/'provenance.json').exists():
        print('Provenance already frozen; use recorded revisions.')
        return
    records=[]
    for repo in ['bryant-nn/Template-Based-Financial-Report-Generation','weixuzhang/CFB']:
        metadata=json.loads(fetch(f'https://api.github.com/repos/{repo}'))
        commit=json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{metadata["default_branch"]}'))
        records.append({'repo':repo,'commit':commit['sha'],'stars':metadata['stargazers_count'],
                        'license':(metadata.get('license') or {}).get('spdx_id'),
                        'usage':'paper-method reimplementation; repository used as reference'})
    model=json.loads(fetch('https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct'))
    write_json(DATA/'provenance.json', {'date':'2026-11-06','papers':records,
              'model':{'id':'Qwen/Qwen2.5-1.5B-Instruct','revision':model['sha']},
              'small_model':{'id':'Qwen/Qwen2.5-0.5B-Instruct','revision':'7ae557604adf67be50417f59c2c2f167def9a775'}})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('task', choices=['fda','xsum','provenance'])
    args=parser.parse_args()
    globals()[args.task]()
