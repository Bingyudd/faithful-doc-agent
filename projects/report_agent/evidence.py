"""Reuse the original BM25 tokenization and formula; no invented bbox metadata."""
import math
import re
from collections import Counter
from projects.document_agent.evidence_core import norm,tokens
from projects.paper_common.text import numbers


def retrieve(elements,query,document_id,top_k=4):
    pool=[e for e in elements if e['document_id']==document_id]
    vectors=[Counter(tokens(e['text'])) for e in pool]
    lengths=[sum(v.values()) for v in vectors]
    mean=sum(lengths)/len(lengths) if lengths else 1
    df=Counter(t for v in vectors for t in v)
    ranked=[]
    for element,tf,length in zip(pool,vectors,lengths):
        score=sum(math.log(1+(len(pool)-df[t]+.5)/(df[t]+.5))*tf[t]*2.5/(tf[t]+1.5*(.25+.75*length/(mean or 1)))
                  for t in set(tokens(query)) if tf[t])
        if score>0:ranked.append((score,element))
    return [e for _,e in sorted(ranked,key=lambda x:(-x[0],x[1]['id']))[:top_k]]


def audit_section(section,observed,target_document):
    issues=[]
    for i,statement in enumerate(section.statements):
        if statement.target_document!=target_document:
            issues.append({'statement':i,'code':'wrong_product_document'})
        quotes=[]
        for citation in statement.citations:
            source=observed.get(citation.source_id)
            if not source:
                issues.append({'statement':i,'code':'unobserved_citation','source_id':citation.source_id})
                continue
            if source['document_id']!=target_document:
                issues.append({'statement':i,'code':'cross_document_citation'})
            if norm(citation.quote) not in norm(source['text']):
                issues.append({'statement':i,'code':'quote_not_in_source','source_id':citation.source_id})
            else:quotes.append(citation.quote)
        combined=' '.join(quotes)
        unsupported=numbers(statement.text)-numbers(combined)
        if unsupported:
            issues.append({'statement':i,'code':'unsupported_number_or_unit','values':sorted(unsupported)})
    return issues


def markdown(report):
    lines=['# '+report.title,'']
    for section in report.sections:
        lines.extend(['## '+section.title,''])
        if section.status!='supported':lines.extend(['状态：'+section.status,''])
        for statement in section.statements:
            refs=' '.join('['+c.source_id+']' for c in statement.citations)
            lines.extend([statement.text+' '+refs,''])
        lines.extend('- '+n for n in section.notes)
        lines.append('')
    return '\n'.join(lines)
