"""Record the actual Conda interpreter and project package versions."""
import importlib.metadata
import json
import sys
import subprocess
from pathlib import Path

from local_dependencies import enable
enable()

ROOT = Path(__file__).resolve().parent
names = ['pdfplumber','pydantic','transformers','tokenizers','Jinja2','numpy',
         'python-docx','reportlab','pypdf','torch']
report = {
    'environment':'ai-job-search',
    'python_version':sys.version.split()[0],
    'python_executable':sys.executable,
    'conda_prefix':sys.prefix,
    'conda_meta_present':(Path(sys.prefix)/'conda-meta').is_dir(),
    'legacy_deps_on_sys_path':str(ROOT/'.deps') in sys.path,
    'packages':{name:importlib.metadata.version(name) for name in names},
}
try:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version',
                                 '--format=csv,noheader,nounits'],text=True,timeout=10)
    report['gpu_hardware']=[{'name':parts[0],'memory_total_mib':int(parts[1]),
                            'memory_free_mib':int(parts[2]),'driver':parts[3]}
                           for parts in [[part.strip() for part in line.split(',')] for line in raw.strip().splitlines()]]
except (OSError,subprocess.SubprocessError):
    report['gpu_hardware']=[]
(ROOT/'artifacts/environment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
