"""Import torch before RDKit so Windows unittest discovery does not hit WinError 1114."""
from __future__ import annotations

import torch  # noqa: F401
