"""Species-level metal identifiers for DS2 grouping (oxidation-state aware)."""
from __future__ import annotations

from typing import Any


def species_key(element: Any, oxidation: Any) -> str:
    return f"{str(element).strip()}:{float(oxidation):.1f}"


def canonical_species_pair(el_x: Any, ox_x: Any, el_y: Any, ox_y: Any) -> str:
    species = sorted(
        [
            species_key(el_x, ox_x),
            species_key(el_y, ox_y),
        ]
    )
    return f"{species[0]}--{species[1]}"


def canonical_element_pair(el_x: Any, el_y: Any) -> str:
    """Descriptive element-level pair label (not for grouping/metrics)."""
    a, b = str(el_x).strip(), str(el_y).strip()
    # Preserve the row's x–y orientation; callers should pass already-canonical x,y.
    return f"{a}-{b}"

