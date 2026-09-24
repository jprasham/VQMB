import sys
from pathlib import Path

# make test_acceptance importable from test_register
sys.path.insert(0, str(Path(__file__).parent))
