"""Source-scoped transfer set. No human annotation or abstractive gold is implied."""
from .io import ROOT,read_json,write_json,write_jsonl,digest
from projects.report_agent.evidence import retrieve


def main():
    manifest=read_json(ROOT/'data/paper/sources.json')
    corpus=read_json(ROOT/'data/paper/corpus.json')['elements']
    rows=[]
    for doc in manifest['sources']:
        selected=retrieve(corpus,'subject device technical characteristics performance testing results length diameter sterilization',doc['id'],5)
        for item in selected:
            rows.append({'id':item['id'].replace(':','_'),'dataset':'fda_technical','document_id':doc['id'],
                         'family':doc['family'],'split':doc['split'],'source_id':item['id'],'source':item['text'],
                         'task':'Summarize the subject device technical characteristics or reported test evidence in at most two sentences. Use only the source. Preserve numbers, units and conditions you mention. Do not assign predicate-device specifications to the subject device.',
                         'reference':'','annotation':'source-backed extraction; no human-written summary reference'})
    for split in ('dev','test'):
        write_jsonl(ROOT/f'data/paper/technical_{split}.jsonl',[x for x in rows if x['split']==split])
    write_json(ROOT/'data/paper/technical_manifest.json',{'count':len(rows),'source_split_hash':manifest['split_hash'],
               'sha256':digest(rows),'selection':'up to 5 BM25 technical chunks per source; split by applicant family',
               'counts':{s:sum(x['split']==s for x in rows) for s in ('dev','test')},
               'reference_summary':'not available; evaluate source consistency and numeric precision separately'})
    print('Technical transfer set:',len(rows))


if __name__=='__main__':main()
