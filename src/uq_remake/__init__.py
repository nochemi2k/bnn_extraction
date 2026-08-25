"""UQ remake package root.

On Windows, entry scripts must `import torch` before any rdkit import
(WinError 1114 on c10.dll). This package __init__ deliberately does NOT
import torch so lightweight CLIs stay torch-free.

Repo root = REMAKE_ROOT = REPO_ROOT (configs/, data/, results/, checkpoints/).
Package layout: data/, models/, training/, metrics/, downstream/, manuscript/.
"""
from pathlib import Path

# src/uq_remake/__init__.py → parents[2] = repository root
REMAKE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = REMAKE_ROOT
