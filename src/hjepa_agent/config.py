"""Project paths. Downloads and outputs go inside the project, not into default caches."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
ASSETS_DIR = PROJECT_DIR / "assets"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

os.environ.setdefault("HF_HOME", str(ASSETS_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ASSETS_DIR / "torch"))
