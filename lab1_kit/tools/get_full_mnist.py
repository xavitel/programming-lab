"""Download the full MNIST dataset (mnist.npz: 70,000 images) for the numpy Challenges.

You only need this for the optional numpy routes (for example the numpy division of the
League). Every Core task uses the small files in data/ instead.

Usage, from the kit folder:
    python tools/get_full_mnist.py                       # saves data/full/mnist.npz
    python tools/get_full_mnist.py --folder D:/datasets  # saves D:/datasets/mnist.npz
    python tools/get_full_mnist.py --verify-only PATH    # only check a file you already have
(Windows: you can type `py` instead of `python`; macOS: `python3`.)

It needs internet (about 11 MB) and only the standard library. No internet? Download the
offline copy from the course page on IE Campus, then run it with --verify-only.

The file is checked twice: its size must be 11,490,434 bytes and its SHA-256 must be
731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1.

Data: MNIST (Yann LeCun, Corinna Cortes, Christopher J. C. Burges), CC BY-SA 3.0
(https://creativecommons.org/licenses/by-sa/3.0/). See LICENSE-DATA.md.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import ssl
except ImportError:  # a Python built without SSL cannot download over HTTPS
    ssl = None

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
FILENAME = "mnist.npz"
EXPECTED_BYTES = 11_490_434
EXPECTED_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
# data/full/ inside the kit folder (this script lives in <kit>/tools/), whatever the current folder is
DEFAULT_FOLDER = Path(__file__).resolve().parent.parent / "data" / "full"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> tuple[bool, str]:
    """Check that `path` is the expected mnist.npz. Returns (ok, a one-line message)."""
    path = Path(path)
    if not path.is_file():
        return False, f"{path} does not exist."
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        return False, (f"{path} has {size:,} bytes, expected {EXPECTED_BYTES:,}. "
                       "The download is incomplete or it is a different file.")
    digest = sha256_of(path)
    if digest != EXPECTED_SHA256:
        return False, (f"{path} has the right size but the wrong SHA-256 ({digest[:12]}...). "
                       "The file is damaged or it is a different file.")
    return True, f"{path} verified: {size:,} bytes, SHA-256 matches."


def download(url: str, target: Path, timeout: float = 60.0) -> None:
    """Download `url` to `target` through a temporary .part file, printing progress."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "garabato-lab-kit"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(part, "wb") as out:
            total = int(response.headers.get("Content-Length") or EXPECTED_BYTES)
            received, next_report = 0, 10
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                received += len(chunk)
                percent = received * 100 // max(total, 1)
                if percent >= next_report:
                    print(f"  {min(percent, 100):3d}%  {received / 1e6:5.1f} MB", flush=True)
                    next_report = percent // 10 * 10 + 10
    except BaseException:
        part.unlink(missing_ok=True)  # never leave half a file behind (also on Ctrl+C)
        raise
    os.replace(part, target)


def network_help(err: BaseException) -> str:
    """One concrete fix for a failed download."""
    reason = getattr(err, "reason", err)
    if (ssl is not None and isinstance(reason, ssl.SSLCertVerificationError)) or "CERTIFICATE_VERIFY_FAILED" in str(err):
        return ("Your Python cannot check HTTPS certificates. On macOS, open your Python folder "
                "in Applications and run 'Install Certificates.command', then try again. "
                "Or use the offline copy from IE Campus with --verify-only.")
    return ("Check your internet connection and try again, or download the offline copy from "
            "IE Campus and run: python tools/get_full_mnist.py --verify-only PATH_TO/mnist.npz")


def how_to_load(path: Path) -> str:
    return (
        "Load it with numpy (numpy is only needed for these optional routes):\n"
        "    import numpy as np\n"
        f"    with np.load(r\"{path}\", allow_pickle=False) as mnist:\n"
        "        x_train, y_train = mnist[\"x_train\"], mnist[\"y_train\"]   # (60000, 28, 28) and (60000,), uint8\n"
        "        x_test, y_test = mnist[\"x_test\"], mnist[\"y_test\"]       # (10000, 28, 28) and (10000,), uint8\n"
        "    x_train = x_train.reshape(len(x_train), 784)                 # one row of 784 pixels per image\n"
        "Note: the 2k test images in data/ are part of x_test, so never train on x_test."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="get_full_mnist",
        description="Download and verify the full MNIST dataset (mnist.npz, about 11 MB).",
    )
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER,
                        help="folder to save mnist.npz in (default: data/full/ inside the kit)")
    parser.add_argument("--url", default=URL, help="download address (default: the Keras mirror)")
    parser.add_argument("--verify-only", type=Path, metavar="PATH",
                        help="do not download; only verify an existing mnist.npz")
    args = parser.parse_args(argv)

    if args.verify_only is not None:
        ok, message = verify(args.verify_only)
        print(("OK: " if ok else "PROBLEM: ") + message, file=sys.stdout if ok else sys.stderr)
        if ok:
            print(how_to_load(args.verify_only.resolve()))
        return 0 if ok else 1

    target = (args.folder / FILENAME).resolve()
    if target.is_file():
        ok, message = verify(target)
        if ok:
            print("Already downloaded. OK: " + message)
            print(how_to_load(target))
            return 0
        print("The existing file failed the check, so it will be downloaded again. " + message)

    print(f"Downloading {args.url}\n  to {target} (about {EXPECTED_BYTES / 1e6:.1f} MB) ...", flush=True)
    try:
        download(args.url, target)
    except (urllib.error.URLError, OSError, ValueError) as err:
        print(f"PROBLEM: the download failed ({err}).\n{network_help(err)}", file=sys.stderr)
        return 1

    ok, message = verify(target)
    if not ok:
        print("PROBLEM: " + message + "\nDelete the file and run this script again, "
              "or use the offline copy from IE Campus.", file=sys.stderr)
        return 1
    print("OK: " + message)
    print(f"Saved to: {target}")
    print(how_to_load(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
