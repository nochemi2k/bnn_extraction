"""Chemprop CLI with descriptastorus stubbed before import.

Newer RDKit no longer exposes Descriptors.fr_* fragment counts that
descriptastorus RDKit2DNormalized instantiates at import time. This project
always supplies ``--features_path``, so those generators are unused.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402  — before rdkit on Windows


def main(argv: list[str] | None = None) -> int:
    from uq_remake.models.chemprop_dmpnn import _ensure_rdbase, _stub_descriptastorus

    _ensure_rdbase()
    _stub_descriptastorus()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("usage: chemprop_cli.py train|predict|fingerprint <chemprop args>")
    tool, rest = args[0], args[1:]
    sys.argv = [f"chemprop_{tool}", *rest]
    if tool == "train":
        from chemprop.train import chemprop_train

        chemprop_train()
    elif tool == "predict":
        from chemprop.train import chemprop_predict

        chemprop_predict()
    elif tool == "fingerprint":
        from chemprop.train import chemprop_fingerprint

        chemprop_fingerprint()
    else:
        raise SystemExit(f"unknown chemprop tool {tool!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
