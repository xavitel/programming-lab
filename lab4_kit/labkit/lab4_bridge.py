"""Helpers for the provided cells of bridge_4to5.ipynb, the catch-up notebook between Lab 4 and Lab 5.

It runs inside the Lab 4 kit. The views use YOUR class and function and never crash: if one isn't written yet,
or raises, they say what to do. Standard library only; scikit-learn and PyTorch are used only if installed.

    ready("CentroidModel", "score")             # True if both are defined and written (give the NAMES)
    checks_pass(CentroidModel, score)            # one status line; True if every check passes
    show_model(CentroidModel, train_images, train_labels, test_images[0], test_labels[0])
    show_score(score, CentroidModel, train_images, train_labels, test_images[:10], test_labels[:10])
    show_track(my_track)                         # the Lab 5 card you picked, and what to prepare
    need("sklearn", "...")                       # labkit.optional.need, with Lab 5's install commands
    wants("torch")                               # does the track you picked need this library?
    try_network(train_images, train_labels)      # a tiny scikit-learn network: does the install work?
    try_torch()                                  # one PyTorch layer, forward and backward

You never need to change this file; open it if you're curious how the helpers work.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import sys
import time
import warnings

from . import cases
from .check import PASS, is_stub, run_call, run_task, your_line
from .env import current_lab
from .show import passing_note as _passing, rescue_hint as _rescue_hint, show_digit, task_for as _task_for
from .text import say, short

__all__ = ["TRACKS", "ready", "checks_pass", "show_model", "show_score", "track_id", "show_track", "need", "wants",
           "try_network", "try_torch"]

_MISSING = object()


# ============================================================================= small helpers
def _lab_cases():
    lab = current_lab()
    return cases.load(lab) if lab else None


def _help_for(name: str) -> str:
    """'CentroidModel (BR1): run its cell above, or run CentroidModel = rescue("BR1")'."""
    lab_cases = _lab_cases()
    task = lab_cases.by_name(name) if lab_cases is not None else None
    if task is None:
        return f"{name}: run its cell above"
    if task.rescue:
        return f"{name} ({task.id}): run its cell above, or run {name} = rescue(\"{task.id}\") in a new cell"
    return f"{name} ({task.id}): run its cell above"


def _notebook() -> dict:
    main = sys.modules.get("__main__")
    return vars(main) if main is not None else {}


def ready(*objs: object) -> bool:
    """True if every class or function is defined and written (not a stub); otherwise say what's missing.

        if ready("CentroidModel", "score"):
            ...the view...

    Give the NAMES (strings): ready() looks them up in the notebook, so one that was never defined (e.g. after a
    kernel restart) gets a friendly line instead of a NameError. Classes and functions work too.
    """
    caller = inspect.currentframe()
    namespace = caller.f_back.f_globals if caller is not None and caller.f_back is not None else _notebook()
    undefined, stubs = [], []
    for obj in objs:
        if isinstance(obj, str):
            if obj not in namespace:
                undefined.append(_help_for(obj))
                continue
            obj = namespace[obj]
        if not callable(obj):
            stubs.append(f"{short(obj, 30)} (not a class or a function)")
        elif is_stub(obj):
            stubs.append(_help_for(getattr(obj, "__name__", "a class")))
    if not undefined and not stubs:
        return True
    lines = []
    if undefined:
        lines += ["This cell needs something that is not defined yet (after a restart, the kernel forgets "
                  "everything):", *[f"   - {item}" for item in undefined]]
    if stubs:
        lines += ["This cell needs code that is still a stub (write it, then run its cell):",
                  *[f"   - {item}" for item in stubs]]
    lines.append("Then run this cell again.")
    say(*lines)
    return False


def checks_pass(*objs: object) -> bool:
    """Run the checks of these classes/functions quietly and print one status line:

        Your checks: BR1 CentroidModel PASS · BR2 score NOT YET.

    Returns True when every check passes; otherwise names the check to run first.
    """
    lab_cases = _lab_cases()
    if lab_cases is None:
        return True
    parts, not_yet = [], []
    for obj in objs:
        task = lab_cases.by_name(getattr(obj, "__name__", ""))
        if task is None:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                status = run_task(task, obj, lab_cases)[0].status
        except KeyboardInterrupt:
            raise
        except Exception:
            status = "NOT YET"
        parts.append(f"{task.id} {task.name} {status}")
        if status != PASS:
            not_yet.append(task)
    lines = ["Your checks: " + " · ".join(parts) + "."]
    if not_yet:
        first = not_yet[0]
        lines.append(f"Fix {' and '.join(t.id for t in not_yet)} first: check(\"{first.id}\", {first.name}) shows "
                     "what is still wrong. Then run this cell again.")
    say(*lines)
    return not not_yet


def _failed(what: str, attempt, obj: object, fallback: str) -> bool:
    """Print a friendly report if the attempt raised (or was interrupted); True if it did."""
    if attempt.interrupted:
        say("Stopped (you pressed Interrupt).")
        return True
    if attempt.error is None:
        return False
    task, name, _ = _task_for(obj, fallback)
    lines = [f"{what} raised {attempt.error_text}."]
    place = your_line(attempt.error, obj, always_name=True)
    if place:
        lines.append(f"{place[0]} {place[1]}")
    check_text = f"check(\"{task.id}\", {task.name}) shows why, or " if task is not None else ""
    lines.append(f"Next step: {check_text}{_rescue_hint(task, name)}.")
    say(*lines)
    return True


def _note_if_not_passing(obj: object, fallback: str) -> None:
    ok, note = _passing(obj, fallback)
    if not ok:
        say(note)


def _fitted(model_class: type, images: list, labels: list, *, quiet: bool = False):
    """model_class().fit(images, labels), safely: (model, what fit returned), or None after a friendly message."""
    name = getattr(model_class, "__name__", "CentroidModel")
    made = run_call(lambda: model_class(), (), track=False)
    if _failed(f"{name}()", made, model_class, name):
        return None
    model = made.got
    fitted = run_call(lambda: model.fit(images, labels), (), track=False)
    if _failed(f"{name}().fit(...)", fitted, model_class, name):
        return None
    if fitted.got is not model and not quiet:
        say(f"Note: fit() returned {short(fitted.got, 30)}, not the model itself. End fit() with `return self`, "
            f"so that calls can be chained: {name}().fit(images, labels).predict_one(image).", "")
    return model, fitted.got


# ============================================================================= BR1 · the class at work
def _describe_centroids(value: object) -> str:
    if value is _MISSING:
        return "(no attribute called centroids)"
    if isinstance(value, dict) and value and all(isinstance(v, (list, tuple)) for v in value.values()):
        labels = sorted(value, key=str)
        sizes = sorted({len(v) for v in value.values()})
        size = f"{sizes[0]} numbers" if len(sizes) == 1 else f"{sizes[0]}–{sizes[-1]} numbers"
        span = f"labels {labels[0]}–{labels[-1]}" if len(labels) > 2 else "labels " + ", ".join(map(str, labels))
        return f"a dict with {len(value)} entries ({span}), each an average image of {size}"
    return short(value, 60)


def show_model(model_class: type, train_images: list, train_labels: list, image: list[int], truth: int) -> None:
    """What lives inside one object of YOUR class: before fit, after fit, and one prediction."""
    name = getattr(model_class, "__name__", "CentroidModel")
    if not ready(model_class):
        return
    _note_if_not_passing(model_class, "CentroidModel")
    made = run_call(lambda: model_class(), (), track=False)
    if _failed(f"{name}()", made, model_class, name):
        return
    model = made.got
    say(f"model = {name}()",
        f"   model.centroids  ->  {_describe_centroids(getattr(model, 'centroids', _MISSING))}")
    t0 = time.perf_counter()
    fitted = run_call(lambda: model.fit(train_images, train_labels), (), track=False)
    seconds = time.perf_counter() - t0
    if _failed("model.fit(train_images, train_labels)", fitted, model_class, name):
        return
    returned = ("the model itself" if fitted.got is model else
                f"{short(fitted.got, 40)}, not the model (end fit() with `return self`)")
    say(f"model.fit(train_images, train_labels)   # {len(train_labels):,} images, {seconds:.2f} s; it returned "
        f"{returned}",
        f"   model.centroids  ->  {_describe_centroids(getattr(model, 'centroids', _MISSING))}")
    say("", f"A test image (true label {truth}):")
    show_digit(image)
    answer = run_call(lambda: model.predict_one(image), (), track=False)
    if _failed("model.predict_one(test_images[0])", answer, model_class, name):
        return
    verdict = "right" if answer.got == truth else "not right this time"
    say(f"model.predict_one(test_images[0])  ->  {short(answer.got, 30)}   (true label {truth}: {verdict})",
        "The object remembers what fit() learnt: nothing else had to be passed to predict_one.")


# ============================================================================= BR2 · score any model
def show_score(score_fn, model_class: type, train_images: list, train_labels: list, images: list,
               labels: list) -> None:
    """YOUR score() on a model of YOUR class: each image's answer, then the fraction right."""
    if not ready(score_fn, model_class):
        return
    _note_if_not_passing(score_fn, "score")
    found = _fitted(model_class, train_images, train_labels, quiet=True)
    if found is None:
        return
    model = found[0]
    right = 0
    for i, (image, label) in enumerate(zip(images, labels)):
        answer = run_call(lambda: model.predict_one(image), (), track=False)
        if _failed(f"model.predict_one(test_images[{i}])", answer, model_class, "CentroidModel"):
            return
        ok = answer.got == label
        right += ok
        say(f"   test image {i}: read {short(answer.got, 12)}, true label {label}  {'right' if ok else '-'}")
    result = run_call(lambda: score_fn(model, images, labels), (), track=False)
    if _failed("score(model, ...)", result, score_fn, "score"):
        return
    say(f"score(model, test_images[:{len(labels)}], test_labels[:{len(labels)}])  ->  {short(result.got, 30)}"
        f"   ({right} of {len(labels)} right above)")


# ============================================================================= choose a Lab 5 track
# The ADVANCED cards of Lab 5 (labs/lab5_model_arena/kit_extra/cards/track_cards.md, in short), plus the CORE.
TRACKS: dict[str, dict[str, str]] = {
    "SCI2": {
        "title": "The PyTorch track", "part": "ADVANCED", "difficulty": "demanding", "minutes": "25–40",
        "learn": "a neural network in PyTorch with an explicit training loop (forward → loss → backward → step), "
                 "wrapped as a model class; then a small convolutional network that inherits the loop",
        "needs": "PyTorch, a large download: at home, `uv sync --group nn --group torch` in the kit folder (or "
                 "`pip install torch`), or Google Colab, which already has it",
        "before": "install PyTorch at home and run the library cell below until it says it works",
        "libraries": "sklearn torch"},
    "SCI3": {
        "title": "Under the hood", "part": "ADVANCED", "difficulty": "demanding", "minutes": "25–40",
        "learn": "backpropagation by hand with numpy (the chain rule, one layer at a time), checked with a gradient "
                 "check before any training",
        "needs": "numpy, which comes with scikit-learn: `uv sync --group nn` (or `pip install numpy`)",
        "before": "install scikit-learn at home (it brings numpy); the library cell below checks both",
        "libraries": "sklearn numpy"},
    "ENG2": {
        "title": "Pythonic objects", "part": "ADVANCED", "difficulty": "demanding", "minutes": "15–20",
        "learn": "__repr__, __eq__, a read-only @property and a @dataclass, so your objects print, compare and "
                 "protect themselves",
        "needs": "nothing beyond Python",
        "before": "nothing to install: today's class is a good start, because ENG2 builds on it",
        "libraries": "sklearn"},
    "BLD2": {
        "title": "Business rules", "part": "ADVANCED", "difficulty": "demanding", "minutes": "20–25",
        "learn": "constrained decoding (only real postcodes), a reject option for a person to check, and what each "
                 "kind of error costs (€0.35 per review, €7.80 per misroute)",
        "needs": "nothing beyond Python",
        "before": "nothing to install; the Holiday Quest after Lab 2 is a good warm-up if you did it",
        "libraries": "sklearn"},
    "CH": {
        "title": "The Challenge", "part": "ADVANCED", "difficulty": "open-ended", "minutes": "open",
        "learn": "your choice: the League final on a holdout no lab has shown (stdlib or library division), your "
                 "own small tool built with classes, or a human in the loop",
        "needs": "nothing beyond Python; the library division allows scikit-learn and PyTorch",
        "before": "if you want the library division, install what you'll use at home",
        "libraries": "sklearn"},
    "CORE": {
        "title": "The CORE, done well", "part": "CORE", "difficulty": "", "minutes": "the lab",
        "learn": "classes as a common interface: a base class, inheritance, super() and an arena that compares "
                 "every model the same way. Finishing the CORE is a complete lab; OPTIONAL tasks (ENG1, SCI1, "
                 "BLD1) follow if you have time",
        "needs": "nothing beyond Python",
        "before": "today's class: Lab 5's first CORE task starts from it",
        "libraries": "sklearn"},
}
_ALIASES = {"PYTORCH": "SCI2", "TORCH": "SCI2", "UNDER THE HOOD": "SCI3", "NUMPY": "SCI3", "PYTHONIC": "ENG2",
            "PYTHONIC OBJECTS": "ENG2", "BUSINESS RULES": "BLD2", "CHALLENGE": "CH", "LEAGUE": "CH",
            "THE CHALLENGE": "CH", "THE PYTORCH TRACK": "SCI2"}


def track_id(choice: object) -> str | None:
    """'sci2', ' SCI2 ', 'PyTorch' -> 'SCI2'; None if it isn't one of the cards."""
    if not isinstance(choice, str):
        return None
    text = " ".join(choice.strip().upper().split())
    text = _ALIASES.get(text, text)
    return text if text in TRACKS else None


def _menu() -> list[str]:
    return [f"   {key:<5} {card['title']} ({card['part']}" + (f", {card['difficulty']}" if card["difficulty"] else "")
            + ")" for key, card in TRACKS.items()]


def show_track(choice: object) -> None:
    """Print the card you picked and what to prepare before Lab 5 (or the list of ids, if it isn't one of them)."""
    if not isinstance(choice, str) or not choice.strip():
        say("No track picked yet: write an id between the quotes, e.g. my_track = \"SCI2\", run that cell, then "
            "this one. The ids:", *_menu())
        return
    key = track_id(choice)
    if key is None:
        say(f"I don't know the track {short(choice, 30)}. Write one of these ids between the quotes:", *_menu())
        return
    card = TRACKS[key]
    level = card["part"] + (f" · {card['difficulty']}" if card["difficulty"] else "")
    minutes = f" · {card['minutes']} min" if card["minutes"][:1].isdigit() else ""
    say(f"Your plan for Lab 5: {key} · {card['title']} ({level}{minutes})",
        f"   What you'll learn: {card['learn']}.",
        f"   Needs: {card['needs']}.",
        f"   Before 1 Dec: {card['before']}.",
        "   Everyone: the whole class trains a real neural network with scikit-learn (with a pure-Python fallback "
        "if it isn't installed).",
        "A plan, not a promise: you can change your mind in class, and the CORE comes first.")


def wants(library: str) -> bool:
    """True if the track in `my_track` (in the notebook) needs this library ('sklearn', 'numpy', 'torch')."""
    key = track_id(_notebook().get("my_track"))
    return key is not None and library in TRACKS[key]["libraries"].split()


# ============================================================================= optional libraries
_INSTALL = {"sklearn": "`uv sync --group nn`", "numpy": "`uv sync --group nn`",
            "torch": "`uv sync --group nn --group torch`"}


def need(package: str, purpose: str):
    """labkit.optional.need (import an optional library, or ONE friendly line and None), with Lab 5's commands.

    scikit-learn and numpy: `uv sync --group nn` (scikit-learn brings numpy). PyTorch: `uv sync --group nn --group
    torch`, naming both groups, because `uv sync` keeps only the groups you name.
    """
    import re

    from . import optional

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        module = optional.need(package, purpose)
    text = buffer.getvalue()
    if text:
        top = optional._import_name(package).split(".")[0]
        command = _INSTALL.get(top)
        if command:
            warning = " (a large download: at home, not on the classroom Wi-Fi)" if top == "torch" else ""
            text = re.sub(r"`uv sync --group \w+`( in the kit folder)?",
                          lambda found: command + (found.group(1) or "") + warning, text)
        text = text.replace("restart the kernel and run this cell again",
                            "restart the kernel, run the setup cell, the data cell and your track cell, then this "
                            "cell again")
        print(text, end="", flush=True)
    return module


def try_network(images: list, labels: list, n: int = 200) -> bool:
    """Train a tiny scikit-learn network on `n` images: True if the install really works (no score is shown)."""
    try:
        import sklearn
        from sklearn.neural_network import MLPClassifier
    except Exception as error:                        # noqa: BLE001 (a broken install)
        say(f"scikit-learn could not be imported ({type(error).__name__}). Reinstall it, or use Google Colab.")
        return False
    try:
        x = [[pixel / 255 for pixel in image] for image in images[:n]]
        y = list(labels[:n])
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # a 20-step network hasn't converged: that's expected here
            network = MLPClassifier(hidden_layer_sizes=(16,), max_iter=20, random_state=0).fit(x, y)
            network.predict(x[:5])
        seconds = time.perf_counter() - t0
    except KeyboardInterrupt:
        raise
    except Exception as error:                        # noqa: BLE001 (a broken install, a missing DLL, ...)
        text = str(error).strip().split("\n")[0][:160]
        say(f"scikit-learn is installed, but a tiny network failed to train ({type(error).__name__}: {text}). "
            "Reinstall it at home, or plan to use Google Colab in Lab 5.")
        return False
    say(f"scikit-learn {sklearn.__version__} works: a tiny network learnt from {len(y)} images in {seconds:.2f} s. "
        "You're ready for Lab 5's Train a Real Neural Network.")
    return True


def try_torch() -> bool:
    """One PyTorch layer, forward and backward: True if PyTorch really works on this computer."""
    try:
        t0 = time.perf_counter()
        import torch
        layer = torch.nn.Linear(4, 2)
        loss = layer(torch.ones(1, 4)).pow(2).sum()
        loss.backward()
        assert layer.weight.grad is not None
        seconds = time.perf_counter() - t0
    except KeyboardInterrupt:
        raise
    except Exception as error:                        # noqa: BLE001 (a broken install, a missing DLL, ...)
        text = str(error).strip().split("\n")[0][:160]
        say(f"PyTorch is installed, but a tiny layer failed ({type(error).__name__}: {text}). Reinstall it at home, "
            "or plan to use Google Colab for the PyTorch track.")
        return False
    say(f"PyTorch {torch.__version__} works: one layer ran forward and backward ({seconds:.1f} s, import included). "
        "No graphics card needed: the PyTorch track runs on the processor.")
    return True
