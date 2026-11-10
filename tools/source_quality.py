"""Document structure inventory; heuristic flags are not human annotations."""
from projects.paper_common.io import ROOT,read_json,write_json,digest

sources=read_json(ROOT/'data/paper/sources.json')['sources']
corpus=read_json(ROOT/'data/paper/corpus.json')['elements']
rows=[]
for doc in sources:
    elements=[x for x in corpus if x['document_id']==doc['id']]
    text=' '.join(e['text'] for e in elements).casefold()
    rows.append({'id':doc['id'],'split':doc['split'],'family':doc['family'],'pages_with_extracted_text':max(e['page'] for e in elements),
                 'characters':len(text),'chunks':len(elements),'has_summary_marker':'510(k) summary' in text,
                 'has_performance_marker':'performance' in text,'sparse_pages':doc['sparse_pages'],
                 'classification':'heuristic only; missing markers may indicate scanned or differently titled technical material'})
write_json(ROOT/'artifacts/paper/source_quality_report.json',{'sources':rows,'raw_pdf_hashes_verified':all(
    digest((ROOT/'data/paper'/d['file']).read_bytes())==d['sha256'] for d in sources),
    'with_performance_marker':sum(r['has_performance_marker'] for r in rows),'manual_annotation':False})
print('Source quality inventory saved; raw hashes checked.')
