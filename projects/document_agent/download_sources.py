"""Download three allowlisted public FDA PDFs; no user documents or credentials."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'K252598': 'https://www.accessdata.fda.gov/cdrh_docs/pdf25/K252598.pdf',
    'K243708': 'https://www.accessdata.fda.gov/cdrh_docs/pdf24/K243708.pdf',
    'K231105': 'https://www.accessdata.fda.gov/cdrh_docs/pdf23/K231105.pdf',
}


def main():
    output = (ROOT / 'data' / 'public').resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Output outside project')
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for identifier, url in SOURCES.items():
        target = output / (identifier + '.pdf')
        if target.exists():
            content = target.read_bytes()
        else:
            request = urllib.request.Request(url, headers={'User-Agent': 'DocumentEvidencePrototype/0.1'})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read(10_000_001)
            if len(content) > 10_000_000 or not content.startswith(b'%PDF-'):
                raise RuntimeError('Invalid or oversized PDF: ' + identifier)
            target.write_bytes(content)
        manifest.append({'document_id': identifier, 'url': url, 'file': target.name,
                         'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
        print(identifier, len(content), 'bytes')
    (output / 'sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
