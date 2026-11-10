"""Readable reference code for every lab: labkit.reference.lab1, lab2, ...

Each module holds clear, commented implementations of the CORE and OPTIONAL functions of
that lab, written in plain loop style first. Use them to catch up on a lab you missed, to compare
with your own code, or through rescue("C2") when you are stuck.

    from labkit.reference import lab1
    lab1.squared_distance([0, 0], [3, 4])      # 25
"""
from __future__ import annotations

import importlib
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent


def available() -> list[int]:
    """Lab numbers whose reference code is in this kit."""
    labs = []
    for path in REFERENCE_DIR.glob("lab*.py"):
        number = path.stem[3:]
        if number.isdigit():
            labs.append(int(number))
    return sorted(labs)


def module_for(lab: int):
    """Return the module labkit.reference.labN, or None if this kit does not include it."""
    name = f"{__name__}.lab{lab}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            return None
        raise


def get(lab: int, name: str):
    """The reference object called `name` from lab N, or None if it is not in this kit."""
    module = module_for(lab)
    return getattr(module, name, None) if module is not None else None
