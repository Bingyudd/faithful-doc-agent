from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from projects.paper_common.io import ROOT,digest,read_json,read_jsonl,write_json
from .core import DecodeConfig,generate

METHODS=['greedy','cad','static','context','token','no_attention','no_semantic']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['generate','evaluate'])
    parser.add_argument('--dataset',default='data/paper/xsum_dev.jsonl')
    parser.add_argument('--model',choices=['main','small'],default='main')
    parser.add_argument('--variant',choices=['paper_equations','author_reference'],default='author_reference')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--source-file')
    parser.add_argument('--task',default='Write one concise factual summary sentence using only the source. Do not add introductions or explanations.')
    parser.add_argument('--max-new-tokens',type=int,default=64)
    parser.add_argument('--max-source-tokens',type=int,default=1024)
    parser.add_argument('--output',default='artifacts/paper/raw/decoding_reference')
    args=parser.parse_args()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import torch
    torch.manual_seed(20261107)
    spec=read_json(ROOT/'data/paper/provenance.json')['model' if args.model=='main' else 'small_model']
    tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'])
    model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],torch_dtype=torch.bfloat16,
              attn_implementation='eager',device_map={'':'cuda:0'}).eval()
    if args.command=='generate':
        if not args.source_file:parser.error('--source-file required for generate')
        rows=[{'id':'interactive','source':Path(args.source_file).read_text(encoding='utf-8'),'split':'interactive'}]
    else:rows=read_jsonl(ROOT/args.dataset)
    if args.limit:rows=rows[:args.limit]
    if any(r['split']=='test' for r in rows):
        from tools.freeze_paper import check
        check()
        if args.max_new_tokens!=64 or args.max_source_tokens!=1024 or args.variant!='author_reference':
            raise ValueError('Held-out generation budgets differ from frozen configuration')
    configs=[DecodeConfig(method=m,variant=args.variant,max_new_tokens=args.max_new_tokens,max_source_tokens=args.max_source_tokens) for m in args.methods]
    output=ROOT/args.output/args.model
    # Warmup is excluded from per-case timing.
    generate(model,tokenizer,'A sample device uses visible light.','Summarize the source.',DecodeConfig(method='greedy',max_new_tokens=3))
    count=0
    for row in rows:
        for config in configs:
            identity={'model':spec,'sample_hash':digest(row),'config':asdict(config),'task':row.get('task',args.task),
                      'implementation_sha256':digest(Path(__file__).with_name('core.py').read_bytes())}
            key=digest(identity)
            path=output/f'{row["id"]}_{config.method}_{key[:10]}.json'
            if path.exists():continue
            record={'id':row['id'],'split':row['split'],'dataset':row.get('dataset','technical'),
                    'method':config.method,'identity':identity,'run_id':key,'reference':row.get('reference','')}
            try:
                record.update(generate(model,tokenizer,row['source'],row.get('task',args.task),config))
                record['status']='complete'
            except Exception as exc:
                record.update(status='error',error=type(exc).__name__+': '+str(exc))
                if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
            write_json(path,record)
            count+=1
            print(json.dumps({'completed':count,'id':row['id'],'method':config.method,'status':record['status'],
                              'seconds':record.get('seconds'),'output_tokens':record.get('output_tokens')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
