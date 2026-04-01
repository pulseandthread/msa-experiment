"""
Wrapper to run MSA training on a GPU server.
Adapts paths for whatever environment you're running on.

Usage:
    python run_training.py          # runs v2 by default
    python run_training.py --v1     # runs v1
"""
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--v1', action='store_true', help='Run v1 training instead of v2')
args = parser.parse_args()

if args.v1:
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_v1", Path(__file__).parent / "05_train_v1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
else:
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_v2", Path(__file__).parent / "06_train_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
