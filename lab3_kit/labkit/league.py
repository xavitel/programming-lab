"""The Pure-Python League (opt-in, pseudonymous). The rules are in LEAGUE.md.

    league_run(my_predict)                    # my_predict(image, train_images, train_labels) -> label
    league_run(my_predict, fit_fn=my_fit)     # model = my_fit(train_images, train_labels)
                                              # my_predict(image, model) -> label
    league_run(my_predict, fit_fn=my_fit, on="validation")   # tune here, as often as you like

Training data: the 10k kit subset. Test: the 2k public test set (labels are used only for the
score, never shown to your function). on="validation" learns from the first 8,000 training images
and scores the last 2,000 instead (200 of each digit): use it to choose settings, and run the
public test once per idea. Prediction budget: 60 s on the build laptop, scaled by machine_factor()
so a slower laptop gets more time. The official ranking is the League final in the Lab 5 kit (league_final):
one run on a holdout kept apart for the final (please don't open it; the League works on trust).
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

from .check import is_stub, run_call, your_line
from .data import cached_digits, digit_files, watch_reads
from .text import say, short, thousands, type_name
from .timing import format_duration, machine_factor

__all__ = ["machine_factor", "league_run", "league_final"]

N_VALIDATION = 2000                 # on="validation": the last 2,000 of the 10,000 training images
SUSPICIOUS_ACCURACY = 0.99          # plain-Python readers on these images stay well below this
_FORBIDDEN_FILES = ("mnist_test_2k", "league_holdout")     # what an entry must never read (LEAGUE.md rule 1)


def _error(name: str, error: BaseException, where: str, fn: object = None) -> list[str]:
    text = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
    lines = [f"League: {name} raised {text} on {where}."]
    place = your_line(error, fn, always_name=True)
    if place:
        lines.append(f"{place[0]} {place[1]}")
    return lines


def _forbidden_reads(seen: list[str]) -> list[str]:
    """The names of test or holdout files in `seen` (each once, in order)."""
    names = []
    for path in seen:
        name = Path(path).name
        if any(word in name.lower() for word in _FORBIDDEN_FILES) and name not in names:
            names.append(name)
    return names


def league_run(predict_fn, *, fit_fn=None, budget_s: float = 60.0, size: int = 28, on: str = "test") -> dict:
    """Score predict_fn on 2,000 images within the League's time budget.

    Without fit_fn: predict_fn(image, train_images, train_labels).
    With fit_fn:    model = fit_fn(train_images, train_labels), then predict_fn(image, model).
    on="test" (default): learn from the 10,000 training images, score the 2,000 public test images.
    on="validation": learn from the first 8,000 training images, score the last 2,000 (for tuning).
    Returns {"correct", "total", "accuracy", "predict_s", "normalised_s", "fit_s", "budget_s",
    "machine_factor", "division", "on", "board", "status"}.
    """
    result = {"status": "not run", "correct": None, "total": 2000, "accuracy": None, "predict_s": None,
              "normalised_s": None, "fit_s": 0.0, "budget_s": None, "machine_factor": None,
              "division": None, "size": size, "on": on, "board": False}
    if size not in (28, 14):
        say(f"League: size must be 28 or 14 (got {size!r}).")
        return result
    if on not in ("test", "validation"):
        say(f"League: on must be \"test\" or \"validation\" (got {on!r}).")
        return result
    if isinstance(budget_s, bool) or not isinstance(budget_s, (int, float)) or not 0 < budget_s <= 600:
        say(f"League: budget_s must be a number of seconds up to 600 (got {budget_s!r}).")
        return result
    name = getattr(predict_fn, "__name__", "your function")
    for fn, label in ((predict_fn, "predict_fn"), (fit_fn, "fit_fn")):
        if fn is None and label == "fit_fn":
            continue
        if not callable(fn) or is_stub(fn):
            say(f"League: {label} isn't ready yet ({'a stub' if callable(fn) else type_name(fn)}). "
                "Write it first; the League is waiting.")
            result["status"] = "not ready"
            return result

    factor = machine_factor()
    budget = budget_s * factor
    train_file, test_file = digit_files(size)
    train_images, train_labels = cached_digits(train_file)
    if on == "validation":
        cut = len(train_images) - N_VALIDATION
        test_images, test_labels = train_images[cut:], train_labels[cut:]
        train_images, train_labels = train_images[:cut], train_labels[:cut]
        scored = f"the last {thousands(N_VALIDATION)} training images (validation)"
    else:
        test_images, test_labels = cached_digits(test_file)
        scored = f"{thousands(len(test_images))} public test images"
    train_copy, labels_copy = [image[:] for image in train_images], train_labels[:]
    result.update(budget_s=round(budget, 1), machine_factor=factor, total=len(test_images))
    tag = " (validation)" if on == "validation" else ""

    with watch_reads() as seen:                   # LEAGUE.md rule 1: nothing may read the test file
        model = None
        if fit_fn is not None:
            fitted = run_call(fit_fn, (train_copy, labels_copy), track=False)
            result["fit_s"] = round(fitted.seconds, 2)
            if fitted.interrupted:
                say("League: stopped during fit_fn (you pressed Interrupt).")
                result["status"] = "stopped"
                return result
            if fitted.error is not None:
                say(*_error(getattr(fit_fn, "__name__", "fit_fn"), fitted.error, "the training data", fit_fn))
                result["status"] = "not ready"
                return result
            model = fitted.got

        def predict(image: list[int]):
            return predict_fn(image, model) if fit_fn is not None else predict_fn(image, train_copy, labels_copy)

        say(f"League run{tag}: {scored} · {thousands(len(train_images))} to learn from · size {size} · "
            f"budget {format_duration(budget)} ({budget_s:g} s × machine factor {factor:.2f})")
        screen = sys.stdout
        predictions = []
        status, crash = "done", None
        n = len(test_images)
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for index, image in enumerate(test_images):
                    predictions.append(predict(image[:]))
                    elapsed = time.perf_counter() - start
                    # stop early when the pace can't fit the budget: a rough look after 10 images,
                    # a closer one after 100
                    if (index == 9 and elapsed / 10 * n > budget * 1.5) or \
                            (index == 99 and elapsed / 100 * n > budget * 1.05):
                        status = "too slow"
                        break
                    if elapsed > budget:
                        status = "over budget"
                        break
                    if (index + 1) % 100 == 0:
                        screen.write(".")
                        screen.flush()
        except KeyboardInterrupt:
            status = "stopped"
        except Exception as error:
            status, crash = "error", error
        seconds = time.perf_counter() - start
    screen.write("\n")
    result["predict_s"] = round(seconds, 2)
    done = len(predictions)

    def so_far() -> str:
        right = sum(1 for p, t in zip(predictions, test_labels) if p == t)
        estimate = seconds / max(1, done) * n
        return (f"League: {right} of {done} correct so far ({right / max(1, done):.1%}); at this speed "
                f"{thousands(n)} images would take about {estimate:.0f} s, and your budget here is {budget:.0f} s.")

    if status == "too slow":
        say(so_far(), "Make predict faster (fewer or smaller training images, an early exit, centroids, ...), "
                      "then run again.")
        result["status"] = status
        return result
    if status in ("over budget", "stopped", "error"):
        if crash is not None:
            say(*_error(name, crash, f"image #{done}", predict_fn))
            if isinstance(crash, TypeError) and "argument" in str(crash):
                say("League: without fit_fn, predict is called as predict_fn(image, train_images, train_labels); "
                    "with fit_fn, as predict_fn(image, model).")
        reason = {"over budget": f"over the {budget:.0f} s budget", "stopped": "you pressed Interrupt",
                  "error": "an error"}[status]
        say(f"League: stopped after {done} of {n} images ({reason}). No score this time.")
        if status == "over budget" and done:
            say(so_far())
        result["status"] = status
        return result
    odd = [p for p in predictions if isinstance(p, bool) or not hasattr(p, "__index__")]
    if odd:
        say(f"League: {name} returned {short(odd[0], 30)} ({type_name(odd[0])}); every answer must be an "
            "int label 0-9.")
        result["status"] = "not ready"
        return result
    predictions = [p.__index__() for p in predictions]            # numpy integers become plain ints
    correct = sum(1 for p, t in zip(predictions, test_labels) if p == t)
    accuracy = correct / len(test_labels)
    division = "numpy" if "numpy" in sys.modules else "stdlib"          # numpy loaded: numpy division
    normalised = seconds / factor                                      # seconds on the build laptop
    read = _forbidden_reads(seen)
    board = on == "test" and not read and accuracy < SUSPICIOUS_ACCURACY
    result.update(status="done", correct=correct, accuracy=accuracy, division=division,
                  normalised_s=round(normalised, 1), board=board)
    fit_text = f" · fit {format_duration(result['fit_s'])}" if fit_fn is not None else ""
    say(f"League result{tag}: {correct:,} of {len(test_labels):,} correct ({accuracy:.1%}) · predict "
        f"{format_duration(seconds)} here = {normalised:.1f} s normalised (budget {budget_s:g} s){fit_text} · "
        f"size {size}")
    if read:
        say(f"League: your code read {' and '.join(read)} during the run, so this entry can't go on the board "
            "(LEAGUE.md, rule 1). Learn from the training images only; the League shows your code the test "
            "images one at a time.")
    elif accuracy >= SUSPICIOUS_ACCURACY:
        say(f"League: {accuracy:.1%} is far above what plain-Python readers reach on these images. Before you "
            "write it on the board, make sure nothing in your code uses test labels or the test file "
            "(LEAGUE.md, rules 1 and 2).")
    elif on == "validation" and _holdout_path().is_file():           # the Lab 5 kit: the next step is the final
        say("Validation scores are for choosing settings, not for the board. When your entry is ready, run the "
            f"League final, once: league_final(my_predict, fit_fn=my_fit, size={size}).")
    elif on == "validation":
        say("Validation scores are for choosing settings, not for the board. When an idea is ready, run it "
            "once on the public test: league_run(my_predict, fit_fn=my_fit).")
    else:
        say(f"For the board (opt-in): <your pseudonym> · {division} · {accuracy:.1%} · {normalised:.1f} s",
            "Use a pseudonym, never your real name. The official ranking is the League final in Lab 5.")
    return result


# ============================================================================= the final (Lab 5)
HOLDOUT_FILE = "holdout/league_holdout.csv.gz"     # 1,000 MNIST test images that are not in the public 2,000
LIBRARIES = ("numpy", "sklearn", "torch", "scipy", "pandas", "tensorflow", "jax", "keras")
SUSPICIOUS_LIBRARY_ACCURACY = 0.995                 # strong library entries (a CNN on 60,000 images) get near 99%


def _library_of(obj: object) -> str | None:
    """'numpy', 'sklearn', 'torch', ... if obj is a module, class, function or object from such a library."""
    import types

    if isinstance(obj, types.ModuleType):
        name = obj.__name__
    elif isinstance(obj, (type, types.FunctionType, types.BuiltinFunctionType)):
        name = getattr(obj, "__module__", None) or ""
    else:
        name = getattr(type(obj), "__module__", None) or ""
    root = name.split(".")[0]
    return root if root in LIBRARIES else None


def _uses_library(functions: list, model: object) -> list[str]:
    """The libraries an entry uses: names its functions refer to, and what its model holds (a few levels deep)."""
    from .check import names_used

    found: list[str] = []
    for fn in functions:
        if fn is None:
            continue
        namespace = getattr(getattr(fn, "__func__", fn), "__globals__", {})
        try:
            used = names_used(fn)
        except Exception:
            used = set()
        for name in sorted(used):
            library = _library_of(namespace[name]) if name in namespace else None
            if library and library not in found:
                found.append(library)
    seen: set[int] = set()
    todo = [(model, 0)]
    while todo:
        obj, depth = todo.pop()
        if id(obj) in seen or depth > 3:
            continue
        seen.add(id(obj))
        library = _library_of(obj)
        if library:
            if library not in found:
                found.append(library)
            continue
        if isinstance(obj, dict):
            todo += [(v, depth + 1) for v in list(obj.values())[:50]]
        elif isinstance(obj, (list, tuple)):
            todo += [(v, depth + 1) for v in list(obj)[:5]]
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            todo += [(v, depth + 1) for v in list(vars(obj).values())[:50]]
    return found


def _holdout_path() -> Path:
    from .data import DATA_DIR
    return DATA_DIR / HOLDOUT_FILE


def _final_record_path() -> Path:
    from .data import KIT_DIR
    return KIT_DIR / "output" / "league_final.json"


def _first_final() -> dict | None:
    """The record of the first completed final on this laptop (output/league_final.json), or None."""
    import json

    try:
        with open(_final_record_path(), encoding="utf-8") as f:
            record = json.load(f)
        return record if isinstance(record, dict) and record.get("status") == "done" else None
    except (OSError, ValueError):
        return None


def _save_final(record: dict) -> None:
    import json

    path = _final_record_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=1)
            f.write("\n")
    except OSError:
        say("League final: I couldn't save output/league_final.json (is the kit folder writable?). Note your "
            "result by hand.")


def league_final(predict_fn, *, fit_fn=None, budget_s: float = 60.0, size: int = 28,
                 division: str | None = None) -> dict:
    """The League final (Lab 5): ONE run on the holdout of 1,000 images kept apart for the final (rules in LEAGUE.md).

    Without fit_fn: predict_fn(image, train_images, train_labels).
    With fit_fn:    model = fit_fn(train_images, train_labels), then predict_fn(image, model).
    Training data: the 10,000-image kit subset (size=28, or size=14 for the 14 x 14 version). Your predict
    function sees the 1,000 holdout images one at a time, within the same normalised budget as league_run.

    Divisions: "stdlib" (the standard library only) and "library" (numpy, scikit-learn, PyTorch, ... are allowed,
    and you may train on the full 60,000 MNIST training images inside fit_fn; never on x_test). The division is
    worked out from what your functions and your model use; division="library" puts a stdlib entry there too.

    The first completed run is saved in output/league_final.json: a holdout only tells the truth the first
    time, so a second run is marked as a re-run, not for the board.
    Returns {"correct", "total", "accuracy", "division", "predict_s", "normalised_s", "fit_s", "budget_s",
    "machine_factor", "size", "board", "rerun", "status"}.
    """
    import datetime
    import platform

    from .data import downsample, load_digits

    if not _holdout_path().is_file():           # Labs 1-4: the holdout ships only with the Lab 5 kit
        raise NotImplementedError("league_final() is available in the Lab 5 kit. Until then, practise with "
                                  "league_run(), which scores the public test set.")
    result = {"status": "not run", "correct": None, "total": 1000, "accuracy": None, "division": None,
              "predict_s": None, "normalised_s": None, "fit_s": 0.0, "budget_s": None, "machine_factor": None,
              "size": size, "board": False, "rerun": False}
    if size not in (28, 14):
        say(f"League final: size must be 28 or 14 (got {size!r}).")
        return result
    if division not in (None, "stdlib", "library"):
        say(f"League final: division must be \"stdlib\" or \"library\" (got {division!r}).")
        return result
    if isinstance(budget_s, bool) or not isinstance(budget_s, (int, float)) or not 0 < budget_s <= 600:
        say(f"League final: budget_s must be a number of seconds up to 600 (got {budget_s!r}).")
        return result
    name = getattr(predict_fn, "__name__", "your function")
    for fn, label in ((predict_fn, "predict_fn"), (fit_fn, "fit_fn")):
        if fn is None and label == "fit_fn":
            continue
        if not callable(fn) or is_stub(fn):
            say(f"League final: {label} isn't ready yet ({'a stub' if callable(fn) else type_name(fn)}). "
                "Write it and tune it with league_run(..., on=\"validation\") first: the final is one run.")
            result["status"] = "not ready"
            return result
    first = _first_final()
    if first is not None:
        result["rerun"] = True
        accuracy_text = f"{first['accuracy']:.1%}" if isinstance(first.get("accuracy"), (int, float)) else "?"
        say(f"League final: you already ran the final on this laptop ({first.get('when', '?')}: {accuracy_text}, "
            f"{first.get('division', '?')} division). A holdout only tells the truth the first time, so this run "
            "is a re-run: interesting for you, not for the board.")

    try:
        holdout_images, holdout_labels = load_digits(_holdout_path())
    except (OSError, ValueError) as error:
        say(f"League final: I can't read data/{HOLDOUT_FILE} ({error}). Re-extract the Lab 5 kit.")
        result["status"] = "no data"
        return result
    if size == 14:
        holdout_images = [downsample(image) for image in holdout_images]
    factor = machine_factor()
    budget = budget_s * factor
    train_file, _ = digit_files(size)
    train_images, train_labels = cached_digits(train_file)
    train_copy, labels_copy = [image[:] for image in train_images], train_labels[:]
    result.update(budget_s=round(budget, 1), machine_factor=factor, total=len(holdout_images))

    with watch_reads() as seen:                   # LEAGUE.md rule 1: the entry must not read the test files
        model = None
        if fit_fn is not None:
            fitted = run_call(fit_fn, (train_copy, labels_copy), track=False)
            result["fit_s"] = round(fitted.seconds, 2)
            if fitted.interrupted:
                say("League final: stopped during fit_fn (you pressed Interrupt). Nothing was recorded.")
                result["status"] = "stopped"
                return result
            if fitted.error is not None:
                say(*_error(getattr(fit_fn, "__name__", "fit_fn"), fitted.error, "the training data", fit_fn))
                say("League final: nothing was recorded; fix it and run the final again.")
                result["status"] = "not ready"
                return result
            model = fitted.got

        def predict(image: list[int]):
            return predict_fn(image, model) if fit_fn is not None else predict_fn(image, train_copy, labels_copy)

        say(f"League final: {thousands(len(holdout_images))} holdout images kept apart for the final · "
            f"{thousands(len(train_images))} to learn from · size {size} · budget {format_duration(budget)} "
            f"({budget_s:g} s × machine factor {factor:.2f})")
        screen = sys.stdout
        predictions = []
        status, crash = "done", None
        n = len(holdout_images)
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for index, image in enumerate(holdout_images):
                    predictions.append(predict(image[:]))
                    elapsed = time.perf_counter() - start
                    if (index == 9 and elapsed / 10 * n > budget * 1.5) or \
                            (index == 99 and elapsed / 100 * n > budget * 1.05):
                        status = "too slow"
                        break
                    if elapsed > budget:
                        status = "over budget"
                        break
                    if (index + 1) % 100 == 0:
                        screen.write(".")
                        screen.flush()
        except KeyboardInterrupt:
            status = "stopped"
        except Exception as error:
            status, crash = "error", error
        seconds = time.perf_counter() - start
    screen.write("\n")
    result["predict_s"] = round(seconds, 2)
    done = len(predictions)
    if status != "done":
        if crash is not None:
            say(*_error(name, crash, f"holdout image #{done}", predict_fn))
        reason = {"too slow": "too slow for the budget at this pace", "over budget": f"over the {budget:.0f} s budget",
                  "stopped": "you pressed Interrupt", "error": "an error"}[status]
        say(f"League final: stopped after {done} of {n} images ({reason}). Nothing was recorded, so you can fix it "
            "and run the final again; tune with league_run(..., on=\"validation\") first.")
        result["status"] = status
        return result
    odd = [p for p in predictions if isinstance(p, bool) or not hasattr(p, "__index__")]
    if odd:
        say(f"League final: {name} returned {short(odd[0], 30)} ({type_name(odd[0])}); every answer must be an int "
            "label 0-9. Nothing was recorded.")
        result["status"] = "not ready"
        return result
    predictions = [p.__index__() for p in predictions]            # numpy integers become plain ints
    correct = sum(1 for p, t in zip(predictions, holdout_labels) if p == t)
    accuracy = correct / n
    libraries = _uses_library([predict_fn, fit_fn], model)
    chosen = "library" if (libraries or division == "library") else "stdlib"
    normalised = seconds / factor
    read = _forbidden_reads(seen)
    limit = SUSPICIOUS_ACCURACY if chosen == "stdlib" else SUSPICIOUS_LIBRARY_ACCURACY
    board = not read and accuracy < limit and not result["rerun"]
    result.update(status="done", correct=correct, accuracy=accuracy, division=chosen,
                  normalised_s=round(normalised, 1), board=board)
    fit_text = f" · fit {format_duration(result['fit_s'])}" if fit_fn is not None else ""
    uses = f" (uses {', '.join(libraries)})" if libraries else ""
    say(f"League final, {chosen} division{uses}: {correct:,} of {n:,} correct ({accuracy:.1%}) · predict "
        f"{format_duration(seconds)} here = {normalised:.1f} s normalised (budget {budget_s:g} s){fit_text} · "
        f"size {size}")
    if division == "stdlib" and libraries:
        say(f"League final: you asked for the stdlib division, but your entry uses {', '.join(libraries)}, so it "
            "counts in the library division.")
    if read:
        say(f"League final: your code read {' and '.join(read)} during the run, so this entry can't go on the board "
            "(LEAGUE.md, rule 1).")
    elif accuracy >= limit:
        say(f"League final: {accuracy:.1%} is far above what {chosen} entries reach on these images. Before you "
            "write it on the board, make sure nothing in your code uses test or holdout labels (LEAGUE.md, rules 1 "
            "and 2).")
    elif not result["rerun"]:
        say(f"For the board (opt-in): <your pseudonym> · {chosen} · {accuracy:.1%} · {normalised:.1f} s",
            "Use a pseudonym, never your real name. This was your final: it is saved in output/league_final.json.")
    if not result["rerun"]:
        _save_final({"status": "done", "when": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "accuracy": round(accuracy, 4), "correct": correct, "total": n, "division": chosen,
                     "size": size, "predict_s": round(seconds, 2), "normalised_s": round(normalised, 1),
                     "fit_s": result["fit_s"], "python": platform.python_version(), "board": board})
    return result
