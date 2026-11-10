"""Run the three verification suites and save the measured test result."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {'tests_run':result.testsRun,'failures':len(result.failures),
          'errors':len(result.errors),'skipped':len(result.skipped),'successful':result.wasSuccessful()}
(ROOT/'artifacts/test_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
raise SystemExit(0 if result.wasSuccessful() else 1)
