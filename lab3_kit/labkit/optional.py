"""Optional libraries (scikit-learn, PyTorch, numpy, matplotlib): import one only when a task needs it.

    from labkit.optional import need

    sklearn = need("sklearn", "Train a Real Neural Network")
    if sklearn is not None:
        from sklearn.neural_network import MLPClassifier
        ...

The CORE of every lab uses only Python's standard library. need() never raises: if the library is missing
(or installed but broken), it prints ONE friendly line that says what is missing, why that's fine, how to
install it (uv or pip) and that Google Colab already has it, then returns None, so the cell can skip
the optional part instead of crashing.

You never need to change this file; open it if you're curious how it works.
"""
from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

from .text import say

__all__ = ["need", "installed", "PACKAGES"]

# import name -> (what pip calls it, the uv dependency group(s) of the kit's pyproject.toml, on Colab already?)
# `uv sync` keeps only the groups you name, so PyTorch names nn too: installing it must not remove scikit-learn.
PACKAGES = {
    "sklearn": ("scikit-learn", "nn", True),
    "torch": ("torch", ("nn", "torch"), True),
    "numpy": ("numpy", "extras", True),
    "matplotlib": ("matplotlib", "extras", True),
}
# Said after the uv command (PyTorch: hundreds of MB, several GB of CUDA files on Linux).
NOTES = {"torch": " (a large download: not on the classroom Wi-Fi)"}
# Names people type for the same library.
ALIASES = {"scikit-learn": "sklearn", "scikit_learn": "sklearn", "pytorch": "torch"}


def _import_name(package: str) -> str:
    """'scikit-learn' -> 'sklearn'; 'sklearn.neural_network' stays as it is."""
    text = package.strip()
    return ALIASES.get(text.lower(), text)


def installed(package: str) -> bool:
    """True if the library can be found (without importing it, and without printing anything)."""
    try:
        name = _import_name(package)
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except Exception:
        return False


def install_line(package: str, purpose: str, problem: str | None = None) -> str:
    """The one line need() prints when `package` is missing (or `problem` says why it could not be imported)."""
    name = _import_name(package)
    top = name.split(".")[0]
    pip_name, group, on_colab = PACKAGES.get(top, (top, None, False))
    why = f" for {purpose.strip()}" if purpose and purpose.strip() else ""
    if problem:
        what = f"Optional library {pip_name}{why} is installed but could not be imported ({problem})."
    else:
        what = f"Optional library missing: {pip_name}{why}."
    groups = (group,) if isinstance(group, str) else tuple(group or ())
    command = " ".join(f"--group {g}" for g in groups)
    uv = f"`uv sync {command}` in the kit folder{NOTES.get(top, '')}, or " if groups else ""
    colab = ("Or open this notebook in Google Colab, which already has it." if on_colab else
             "Or try Google Colab, which has many libraries already.")
    return (f"{what} It's optional: the CORE never needs it, and the rest of the notebook keeps working. "
            f"To install it at home: {uv}`pip install {pip_name}`; then restart the kernel and run this cell "
            f"again. {colab}")


def need(package: str, purpose: str) -> ModuleType | None:
    """Import an optional library and return the module, or print ONE friendly line and return None.

        torch = need("torch", "the PyTorch track")
        if torch is None:
            ...skip, or use the pure-Python version...

    `package` is the import name ("sklearn", "torch", "numpy", "matplotlib.pyplot"), or the pip name
    ("scikit-learn"). `purpose` says what the library is for, e.g. "Train a Real Neural Network".
    Never raises (except KeyboardInterrupt, so Interrupt still works).
    """
    try:
        if not isinstance(package, str) or not package.strip():
            say(f"need(): give the library's name as text, e.g. need(\"sklearn\", \"...\"), not {package!r}.")
            return None
        name = _import_name(package)
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            missing = (getattr(error, "name", None) or "").split(".")[0]
            if missing and missing != name.split(".")[0]:
                # the library is there, but something it needs is not
                say(install_line(package, purpose, f"ModuleNotFoundError: {error}"))
            else:
                say(install_line(package, purpose))
            return None
        except Exception as error:          # a broken install: ImportError, OSError (a missing DLL), ...
            text = str(error).strip().split("\n")[0][:160]
            say(install_line(package, purpose, f"{type(error).__name__}: {text}"))
            return None
    except KeyboardInterrupt:
        raise
    except BaseException as error:          # last resort: need() must never stop the cell
        try:
            print(f"Optional library {package!r} could not be loaded ({type(error).__name__}). "
                  "It's optional: the CORE never needs it.")
        except Exception:
            pass
        return None
