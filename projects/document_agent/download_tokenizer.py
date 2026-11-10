"""Download only allowlisted Qwen tokenizer assets, pinned to an exact revision."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
FILES = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt']


def read_url(url, maximum):
    request = urllib.request.Request(url, headers={'User-Agent':'DocumentEvidenceTokenizer/0.2'})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read(maximum + 1)
    if len(content) > maximum:
        raise ValueError('Unexpectedly large tokenizer asset')
    return content


def main():
    output = ROOT / 'data' / 'tokenizer_qwen25_7b'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        revision = manifest['revision']
    else:
        info = json.loads(read_url('https://huggingface.co/api/models/' + MODEL, 2_000_000))
        revision = info['sha']
        manifest = {'model_id':MODEL, 'revision':revision, 'files':[]}
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Invalid repository revision')
    previous = {row['file']:row for row in manifest.get('files', [])}
    rows = []
    for name in FILES:
        url = f'https://huggingface.co/{MODEL}/resolve/{revision}/{name}'
        target = output / name
        if target.exists() and name in previous:
            content = target.read_bytes()
            if hashlib.sha256(content).hexdigest() != previous[name]['sha256']:
                raise ValueError('Cached tokenizer checksum mismatch: ' + name)
        else:
            content = read_url(url, 30_000_000)
            if content.lstrip().startswith(b'<html'):
                raise ValueError('Unexpected HTML for tokenizer asset')
            if name.endswith('.json'):
                json.loads(content)
            target.write_bytes(content)
        rows.append({'file':name, 'bytes':len(content), 'sha256':hashlib.sha256(content).hexdigest(), 'url':url})
        print(name, len(content), 'bytes', flush=True)
    manifest['files'] = rows
    manifest['model_weights_downloaded'] = False
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Pinned tokenizer revision:', revision, flush=True)


if __name__ == '__main__':
    main()
