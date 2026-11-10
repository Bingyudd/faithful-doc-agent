"""Machine-checkable final gates; human review remains a separate state."""
import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path
from projects.paper_common.io import ROOT,read_json,write_json,digest
from tools.freeze_paper import check

check()
files=[]
for directory in ['projects/report_agent','projects/faithful_decoding','projects/paper_common','tools']:
    for path in (ROOT/directory).glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'));files.append(str(path.relative_to(ROOT)))
suite=unittest.defaultTestLoader.loadTestsFromNames(['projects.report_agent.test_engine','projects.report_agent.test_ui','projects.paper_common.test_integrity','projects.paper_common.test_provider','tools.test_resume_metrics'])
result=unittest.TextTestRunner(verbosity=1).run(suite)
summary=read_json(ROOT/'artifacts/paper/delivery_report.json')
resume_checks=[]
from pypdf import PdfReader
for role in ['Agent开发','大模型算法']:
    path=ROOT/f'output/pdf/邓任_{role}_简历.pdf'
    if not path.exists():continue
    pdf=PdfReader(path);text='\n'.join(p.extract_text() or '' for p in pdf.pages)
    resume_checks.append({'role':role,'pages':len(pdf.pages),'text_chars':len(text),
                          'one_page':len(pdf.pages)==1,'sha256':digest(path.read_bytes()),
                          'a4':all(abs(float(p.mediabox.width)-595.276)<1 and abs(float(p.mediabox.height)-841.89)<1 for p in pdf.pages),
                          'paper_citations_absent':all(t not in text for t in ['SIGIR','ACL Findings','arxiv.org','论文公式','作者实现']),
                          'new_projects_present':all(t in text for t in ['技术文档','解码'])})
algorithm=read_json(ROOT/'artifacts/paper/algorithm_tests_report.json')
visual=read_json(ROOT/'artifacts/paper/pdf_visual_report.json')
visual_hashes={x['role']:x['pdf_sha256'] for x in visual['documents']}
visual_valid=all(visual_hashes.get(r['role'])==r['sha256'] for r in resume_checks)
checks={'python_files_parsed':len(files),'tests':result.testsRun,'test_failures':len(result.failures),
        'test_errors':len(result.errors),'tests_successful':result.wasSuccessful(),
        'frozen_configuration_valid':True,'campaign':summary,'resume_checks':resume_checks,
        'algorithm_tests':algorithm['tests'],'algorithm_tests_successful':algorithm['successful'],
        'visual_review_matches_final_pdfs':visual_valid,
        'source_hashes':{p:digest((ROOT/p).read_bytes()) for p in files},
        'git_head_at_validation':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
checks['automatic_acceptance_passed']=result.wasSuccessful() and algorithm['successful'] and summary['machine_campaign_complete'] and visual_valid and len(resume_checks)==2 and all(
    r['one_page'] and r['a4'] and r['new_projects_present'] and r['paper_citations_absent'] for r in resume_checks)
checks['human_review_pending']=46-summary['human_review']
write_json(ROOT/'artifacts/paper/acceptance_report.json',checks)
print(json.dumps({k:v for k,v in checks.items() if k not in ('campaign','resume_checks','source_hashes')},ensure_ascii=False))
raise SystemExit(not checks['automatic_acceptance_passed'])
