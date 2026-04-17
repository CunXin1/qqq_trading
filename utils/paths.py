"""Project path configuration — single source of truth."""
from pathlib import Path
import os

# Allow override via environment variable
_root = os.environ.get("QQQ_PROJECT_ROOT")
PROJECT_ROOT = Path(_root) if _root else Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHART_DIR = OUTPUT_DIR / "charts"
MODEL_DIR = OUTPUT_DIR / "model"

# Create directories on import
for _d in [OUTPUT_DIR, CHART_DIR, MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
