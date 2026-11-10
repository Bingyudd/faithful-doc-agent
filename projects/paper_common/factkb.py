"""Independent evaluator. Scores are model predictions, not human factual labels."""
import argparse
import json
from pathlib import Path
from projects.paper_common.io import ROOT,digest,read_json,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--kind',choices=['decoding','reports'],default='decoding');args=parser.parse_args()
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from huggingface_hub import HfApi
    import torch
    info=ROOT/'data/paper/factkb_model.json'
    if info.exists():spec=read_json(info)
    else:
        spec={'id':'bunsenfeng/FactKB','revision':HfApi().model_info('bunsenfeng/FactKB').sha,
              'tokenizer':'FacebookAI/roberta-base','tokenizer_revision':HfApi().model_info('FacebookAI/roberta-base').sha}
        write_json(info,spec)
    tokenizer=AutoTokenizer.from_pretrained(spec['tokenizer'],revision=spec['tokenizer_revision'])
    model=AutoModelForSequenceClassification.from_pretrained(spec['id'],revision=spec['revision']).to('cuda').eval()
    output=ROOT/args.output
    scores=read_json(output).get('scores',{}) if output.exists() else {}
    files=list((ROOT/args.input).glob('*/*.json')) if args.kind=='decoding' else list((ROOT/args.input).glob('*/*/result.json'))
    with torch.inference_mode():
        for index,path in enumerate(files):
            row=read_json(path)
            if row.get('status')!='complete' or 'run_id' not in row:continue
            if row['run_id'] in scores:continue
            if args.kind=='decoding':pairs=[(row['text'],row['used_source'])]
            else:
                pairs=[(s['text'],' '.join(c['quote'] for c in s['citations'])) for sec in row['report']['sections'] for s in sec['statements']]
            values=[];truncated=0
            for text,source in pairs:
                if not text.strip():values.append(0.0);continue
                encoded=tokenizer(text,source,return_tensors='pt',max_length=512,truncation=True)
                full_length=len(tokenizer(text,source)['input_ids'])
                truncated+=int(full_length>512)
                probability=model(**{k:v.to('cuda') for k,v in encoded.items()}).logits.float().softmax(-1)[0,1].item()
                values.append(probability)
            scores[row['run_id']]={'score':sum(values)/len(values) if values else None,'claim_scores':values,
                                   'pairs':len(pairs),'truncated_pairs':truncated}
            if index%25==0:
                write_json(output,{'model':spec,'scores':scores,'interpretation':'automatic FactKB score; 512-token pair truncation counted; not human adjudication'})
                print('Scored',len(scores),flush=True)
    write_json(output,{'model':spec,'scores':scores,'interpretation':'automatic FactKB score; 512-token pair truncation counted; not human adjudication'})


if __name__=='__main__':main()
