"""rescue("C2"): borrow the reference solution of a task so the rest of the lab keeps working.

    predict_nearest = rescue("C2")

The reference code lives in labkit/reference/labN.py: open it to read a clear, commented version.
Come back to the task later; check("C2", your_function) still checks your own version.
"""
from __future__ import annotations

from . import cases, reference
from .env import current_lab
from .text import say


def rescue(task_id: str) -> object:
    """Return the reference function (or class) for a task of this lab, e.g. rescue("C2").

    Raises LookupError with a friendly message for an unknown task or a task without a rescue,
    so a typo never replaces your own function with None.
    """
    lab = current_lab()
    if lab is None:
        raise LookupError("Run the setup cell first (the first code cell): it tells labkit which lab this is.")
    lab_cases = cases.load(lab)
    if lab_cases is None or not lab_cases.tasks:
        raise LookupError(f"This kit has no tasks for Lab {lab}, so there is nothing to rescue.")
    key = str(task_id).strip().upper().replace(" ", "")
    task = lab_cases.find(key)
    if task is None:
        rescuable = [t.id for t in lab_cases.tasks.values()
                     if t.rescue and reference.get(lab, t.reference_name) is not None]
        raise LookupError(f"I don't know a task called \"{task_id}\" in Lab {lab}. "
                          f"Tasks with a rescue: {', '.join(rescuable)}.")
    found = reference.get(lab, task.reference_name) if task.rescue else None
    if found is None:
        raise LookupError(f"{task.id} has no rescue: it is an open task with no single answer to borrow. "
                          "Its hints can help, and so can Javier.")
    say(f"Using the reference for {task.id} {task.name}. Your notebook keeps working; "
        f"come back to {task.id} when you're ready.")
    return found
