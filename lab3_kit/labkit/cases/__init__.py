"""Registry of task specifications: the checks for lab N live in cases/labN.py.

Each cases/labN.py module defines a list called TASKS (see AUTHORING.md). Expected values that
come from the reference are stored as data in cases/labN_data.json by generators/gen_cases.py,
so the cases files contain no solution code.

Optional module attributes:
    DATA_FILES    data files this lab needs (names relative to data/); start() verifies them
    MOMENT_WORK   (number of distances, pixels per image) for the start() speed estimate
    FINALE        the name of the lab's final run, e.g. "Putting It All Together" (used by check_all(),
                  evaluate() and the start() speed line; default: no name)
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..spec import REFERENCE, Case, Fact, Task, decode

CASES_DIR = Path(__file__).resolve().parent


@dataclass
class LabCases:
    """The tasks of one lab, in notebook order, plus their stored expected values."""
    lab: int
    module: Any
    tasks: dict = field(default_factory=dict)     # task id -> Task
    data: dict = field(default_factory=dict)      # contents of labN_data.json
    source: Any = None                            # the module's TASKS list (to notice edits)

    def find(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def core(self) -> list[Task]:
        return [task for task in self.tasks.values() if task.core]

    def optional_part(self) -> list[Task]:
        """The tasks of the OPTIONAL part (an optional C task, ENG1, SCI1, BLD1), in notebook order."""
        return [task for task in self.tasks.values() if task.in_optional_part]

    def minimum_path(self) -> list[Task]:
        return [task for task in self.tasks.values() if task.minimum_path]

    @property
    def finale(self) -> str | None:
        """The name of the lab's final run (cases module attribute FINALE), or None."""
        return getattr(self.module, "FINALE", None)

    def by_name(self, name: str) -> Task | None:
        for task in self.tasks.values():
            if task.name == name:
                return task
        return None

    def expected(self, task: Task, case: Case, index: int) -> Any:
        """The expected value of a case: literal, or the stored REFERENCE value."""
        if case.expected is not REFERENCE:
            return case.expected
        stored = self.data.get("expected", {}).get(task.id, {})
        key = case_key(case, index)
        if key not in stored:
            raise LookupError(f"no stored expected value for {task.id} case {key}")
        return decode(stored[key])

    def facts(self, task: Task, case: Case, index: int) -> dict:
        """Facts for detectors: literal values, plus Fact(...) values stored at build time."""
        stored = self.data.get("facts", {}).get(task.id, {}).get(case_key(case, index), {})
        result = {}
        for name, value in case.facts.items():
            if isinstance(value, Fact):
                result[name] = decode(stored[name]) if name in stored else None
            else:
                result[name] = value
        return result


def case_key(case: Case, index: int) -> str:
    """The name of a case in labN_data.json: its `key`, or its position in the task's list."""
    return case.key if case.key is not None else str(index)


_loaded: dict[int, LabCases] = {}


def module_name(lab: int) -> str:
    return f"{__name__}.lab{lab}"


def available() -> list[int]:
    """Lab numbers that have a cases file in this kit."""
    labs = []
    for path in CASES_DIR.glob("lab*.py"):
        number = path.stem[3:]
        if number.isdigit():
            labs.append(int(number))
    return sorted(labs)


def load(lab: int) -> LabCases | None:
    """Import cases/labN.py and its data file; None if this kit has no checks for that lab."""
    try:
        module = importlib.import_module(module_name(lab))
    except ModuleNotFoundError as error:
        if error.name == module_name(lab):
            return None
        raise
    task_list = getattr(module, "TASKS", [])
    cached = _loaded.get(lab)
    if cached is not None and cached.module is module and cached.source is task_list:
        return cached                      # already loaded (and not edited since)
    tasks = {}
    for task in task_list:
        tasks[task.id] = task
    data_path = CASES_DIR / f"lab{lab}_data.json"
    data = {}
    if data_path.is_file():
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
    result = LabCases(lab, module, tasks, data, source=task_list)
    _loaded[lab] = result
    return result
