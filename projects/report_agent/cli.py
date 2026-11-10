from __future__ import annotations
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from projects.paper_common.io import ROOT,read_json,write_json,digest
from projects.paper_common.snapshot_provider import SnapshotProvider as Provider
from .engine import ReportEngine,METHODS
from .schema import TEMPLATES


def local_key():
    if os.environ.get('LOCAL_LLM_API_KEY'):return os.environ['LOCAL_LLM_API_KEY']
    # The user explicitly authorized this one gateway configuration; no secrets are logged.
    location='/home/bingyu/WebAI2API/workbuddy2api-hub/accounts/settings.json'
    if os.name=='nt':
        code="import json; print(json.dumps(json.load(open('"+location+"'))))"
        settings=json.loads(subprocess.check_output(['wsl','-d','Ubuntu-22.04','--exec','/usr/bin/python3','-c',code],encoding='utf-8'))
    else:settings=read_json(location)
    candidates=[x['key'] for x in settings.get('api_keys',[]) if isinstance(x,dict) and x.get('enabled') is True and not x.get('deleted_at')]
    if len(candidates)==1:return candidates[0]
    if len(candidates)>1:raise ValueError('Multiple active gateway keys; select LOCAL_LLM_API_KEY explicitly')
    return settings['launcher_key']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['generate','evaluate'])
    parser.add_argument('--split',choices=['dev','test','all'],default='dev')
    parser.add_argument('--document')
    parser.add_argument('--methods',choices=METHODS,nargs='+',default=list(METHODS))
    parser.add_argument('--templates',choices=list(TEMPLATES),nargs='+',default=list(TEMPLATES))
    parser.add_argument('--model',default='deepseek-v4.1-flash')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',default='artifacts/paper/raw/reports_v3')
    args=parser.parse_args()
    if args.split in ('test','all'):
        from tools.freeze_paper import check
        check()
    manifest=read_json(ROOT/'data/paper/sources.json')
    documents=[d for d in manifest['sources'] if (args.split=='all' or d['split']==args.split) and (not args.document or d['id']==args.document)]
    if args.limit:documents=documents[:args.limit]
    if not documents:parser.error('No documents match')
    elements=read_json(ROOT/'data/paper/corpus.json')['elements']
    key=local_key()
    available=Provider(args.model,api_key=key).models()
    if args.model not in available:raise ValueError('Requested model not available from gateway')
    root=ROOT/args.output/args.model
    tasks=[(d,t,m) for d in documents for t in args.templates for m in args.methods]

    def work(task):
        d,t,m=task
        provider=Provider(args.model,api_key=key)
        result=ReportEngine(provider,elements).run(d,t,m,root)
        return result

    results=[]
    with ThreadPoolExecutor(max_workers=max(1,min(args.workers,4))) as pool:
        futures={pool.submit(work,t):t for t in tasks}
        for future in as_completed(futures):
            d,t,m=futures[future]
            result=future.result()
            results.append({'document':d['id'],'split':d['split'],'template':t,'method':m,
                            'run_id':result['run_id'],'status':result['status'],'error':result.get('error'),
                            'usage':result['usage']})
            print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    index_name=digest({'documents':[d['id'] for d in documents],'methods':args.methods,'templates':args.templates})[:12]
    write_json(root/f'batch_{index_name}.json',{'split_hash':manifest['split_hash'],'runs':results})


if __name__=='__main__':main()
