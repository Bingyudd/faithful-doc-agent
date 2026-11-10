"""Project adapters: tracked-file evidence and strict metric comparison gates."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from projects.paper_common.io import ROOT, digest, read_json, write_json

CONDITIONS=('metric','model','revision','split_hash','config_hash','unit','run_id')


def compare_claim(claim, result, *, abs_tol=0.0, rel_tol=0.0):
    if claim.get('confirmed') is not True:
        return {'status':'unverified','reason':'claim not confirmed'}
    if any(not claim.get(k) or not result.get(k) for k in CONDITIONS):
        return {'status':'unverified','reason':'missing experiment conditions'}
    differing=[k for k in CONDITIONS if claim[k]!=result[k]]
    if differing:
        return {'status':'incomparable','reason':','.join(differing)}
    if 'value' not in claim or 'value' not in result:
        return {'status':'unverified','reason':'missing value'}
    import math
    try:a,b=float(claim['value']),float(result['value'])
    except (TypeError,ValueError):return {'status':'unverified','reason':'nonnumeric value'}
    if not math.isfinite(a) or not math.isfinite(b):
        return {'status':'unverified','reason':'non-finite value'}
    difference=abs(a-b)
    return {'status':'match' if difference<=max(abs_tol,rel_tol*abs(a)) else 'mismatch','difference':difference}


def inventory(root=ROOT):
    output=subprocess.check_output(['git','-c','core.quotepath=false','ls-files','-z'],cwd=root)
    files=[p for p in output.decode('utf-8').split('\0') if p]
    deps=[p for p in files if Path(p).name.startswith(('requirements','environment')) or Path(p).name=='pyproject.toml']
    tracked_python=[p for p in files if p.endswith('.py')]
    return {'git_tracked_python_files':len(tracked_python),'files':tracked_python,
            'git_head_at_collection':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'scope':'Git-tracked files at collection time; untracked pending edits are not silently counted',
            'dependencies':{p:{'sha256':digest((root/p).read_bytes()),'text':(root/p).read_text(encoding='utf-8')} for p in deps},
            'ownership':'Not inferred from Git author share; imported papers and prior code recorded separately.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='artifacts/paper/skill_inventory_report.json')
    args=parser.parse_args()
    write_json(ROOT/args.output,inventory())
