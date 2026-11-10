"""Recompute transparent mechanical metrics from saved model outputs."""
from __future__ import annotations
import argparse
import math
import re
import statistics
from collections import defaultdict
from .io import ROOT,read_json,write_json,digest


def words(text):return re.findall(r'\w+',text.casefold())


def rouge_l(prediction,reference):
    a,b=words(prediction),words(reference)
    if not a or not b:return 0.0
    last=[0]*(len(b)+1)
    for x in a:
        now=[0]
        for j,y in enumerate(b):now.append(last[j]+1 if x==y else max(last[j+1],now[-1]))
        last=now
    return 2*last[-1]/(len(a)+len(b))


def mean(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def text_metrics(text,source,reference=''):
    from projects.paper_common.text import numbers
    tokens=words(text);grams=[tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
    output_numbers=numbers(text);source_numbers=numbers(source)
    return {'rouge_l':rouge_l(text,reference) if reference else None,
            'numeric_precision':len(output_numbers & source_numbers)/len(output_numbers) if output_numbers else None,
            'source_number_recall':len(output_numbers & source_numbers)/len(source_numbers) if source_numbers else None,
            'repeated_trigram_fraction':1-len(set(grams))/len(grams) if grams else 0,
            'words':len(tokens)}


def decoding(root):
    records=[]
    for path in root.glob('*/*.json'):
        r=read_json(path)
        if 'method' not in r:continue
        row={'id':r['id'],'run_id':r['run_id'],'model':r['identity']['model']['id'],'dataset':r['dataset'],
             'split':r['split'],'method':r['method'],'status':r['status'],'path':str(path.relative_to(ROOT))}
        if r['status']=='complete':
            row.update(text_metrics(r['text'],r['used_source'],r.get('reference','')))
            row.update(seconds=r['seconds'],peak_mib=(r['peak_allocated_bytes'] or 0)/2**20,
                       output_tokens=r['output_tokens'],eos=r['stopped_by_eos'],source_truncated=r['source_truncated'])
        records.append(row)
    groups=defaultdict(list)
    for r in records:groups[(r['model'],r['dataset'],r['split'],r['method'])].append(r)
    summary=[]
    for key,rows in sorted(groups.items()):
        done=[r for r in rows if r['status']=='complete']
        summary.append(dict(zip(['model','dataset','split','method'],key),count=len(rows),complete=len(done),failed=len(rows)-len(done),
                       **{k:mean(done,k) for k in ('rouge_l','numeric_precision','source_number_recall','repeated_trigram_fraction','words','seconds','peak_mib','output_tokens')},
                       eos_rate=mean(done,'eos'),source_truncation_rate=mean(done,'source_truncated')))
    return {'summary':summary,'samples':records,'definitions':{
        'rouge_l':'whitespace/word-token LCS F1; no stemming; not identical to official rouge-score implementation',
        'numeric_precision':'literal output number/unit occurrences supported by source; not semantic factual accuracy',
        'source_number_recall':'coverage proxy only; a summary need not repeat every source number',
        'human_review':'pending; automatic metrics do not substitute for human semantic adjudication'}}


def reports(root):
    rows=[]
    for directory in root.glob('*/*'):
        if not directory.is_dir():continue
        path=directory/'result.json'
        if not path.exists():path=directory/'failure.json'
        if not path.exists():continue
        r=read_json(path);identity=r['identity']
        row={'run_id':r['run_id'],'document':identity['document']['id'],'split':identity['document']['split'],
             'model':identity['model'],'method':identity['method'],'template':identity['template'],'status':r['status'],
             'path':str(path.relative_to(ROOT)),**r.get('usage',{})}
        if r['status']=='complete':
            from projects.report_agent.schema import TEMPLATES
            sections=r['report']['sections'];statements=[s for sec in sections for s in sec['statements']]
            citations=[c for s in statements for c in s['citations']]
            from projects.document_agent.evidence_core import norm
            valid=sum(c['source_id'] in r['evidence'] and norm(c['quote']) in norm(r['evidence'][c['source_id']]['text']) for c in citations)
            row.update(section_coverage=len({s['id'] for s in sections}&{s['id'] for s in TEMPLATES[identity['template']]})/len(TEMPLATES[identity['template']]),
                       statements=len(statements),citations=len(citations),valid_citations=valid,
                       citation_valid_rate=valid/len(citations) if citations else None,
                       mechanical_errors=sum(len(e) for e in r['audits'].values()),
                       unresolved_sections=sum(s['status']!='supported' for s in sections),
                       revision_rounds=len(r['revisions']),
                       revision_new_mechanical_errors=sum(max(0,len(x['after_errors'])-len(x['before_errors'])) for x in r['revisions']))
        rows.append(row)
    groups=defaultdict(list)
    for row in rows:groups[(row['model'],row['split'],row['method'])].append(row)
    summary=[]
    for key,items in sorted(groups.items()):
        done=[r for r in items if r['status']=='complete']
        summary.append(dict(zip(['model','split','method'],key),count=len(items),complete=len(done),failed=len(items)-len(done),
            **{k:mean(done,k) for k in ('section_coverage','statements','citation_valid_rate','mechanical_errors','unresolved_sections','revision_rounds','revision_new_mechanical_errors')},
            **{k:sum(x.get(k,0) for x in items) for k in ('requests','prompt_tokens','completion_tokens','seconds')}))
    return {'summary':summary,'samples':rows,'definitions':{
        'section_coverage':'template section inventory, not semantic coverage of all source facts',
        'citation_valid_rate':'observed source anchor plus exact quote; not entailment',
        'human_review':'pending; fixed review queue is delivered separately'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['decoding','reports']);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=globals()[args.kind](ROOT/args.input)
    report['input']=args.input;report['output_hash']=digest(report['samples'])
    write_json(ROOT/args.output,report)
    print(report['summary'])
