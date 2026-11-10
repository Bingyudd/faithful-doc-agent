"""Use the active Conda environment; retain legacy local packages as a fallback."""
import sys
from pathlib import Path


def enable():
    if any((Path(prefix) / 'conda-meta').is_dir() for prefix in (sys.prefix, sys.base_prefix)):
        return
    folder = Path(__file__).resolve().parent / '.deps'
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
