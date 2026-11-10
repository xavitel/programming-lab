"""Timing helpers: measure a call, format durations, and compare this laptop with the build laptop.

    best_time(fn, *args)   seconds per call (the fastest of a few runs)
    format_duration(s)     0.000055 -> '55 µs', 0.057 -> '57 ms', 5.74 -> '5.7 s'
    machine_factor()       how much slower (> 1) or faster (< 1) this laptop is than the build laptop
"""
from __future__ import annotations

import random
import time

# machine_factor() runs the same fixed workload on every laptop. On the build laptop
# (Windows 11, CPython 3.12.10, 10 Oct 2026) its best time was BUILD_BENCHMARK_S seconds.
# Same laptop: CPython 3.10.22 0.0359 s (factor 3.7: `** 2` is slow there), 3.14.8 0.0093 s (0.96).
BUILD_BENCHMARK_S = 0.0097
_FACTOR_LIMITS = (0.5, 4.0)       # keep the League budget sensible on very fast or very slow machines
_factor_cache: list[float] = []


def format_duration(seconds: float) -> str:
    """Return a short human duration: '55 µs', '5.7 ms', '57 ms', '5.0 s', '62 s', '2.5 min'."""
    if seconds != seconds or seconds < 0:          # nan or negative: should not happen
        return "? s"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 0.01:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 120:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} min"


def format_eta(seconds: float) -> str:
    """Round an estimate for humans: 5.7 -> 'about 6 s', 0.4 -> 'under 1 s', 95 -> 'about 95 s'."""
    if seconds < 1:
        return "under 1 s"
    if seconds < 120:
        return f"about {seconds:.0f} s"
    return f"about {seconds / 60:.0f} min"


def best_time(fn, *args, repeat: int = 5, budget_s: float = 0.2, **kwargs) -> float:
    """Return the fastest time, in seconds, of one call fn(*args, **kwargs).

    The first call is timed on its own. If it is quick, the call is repeated in batches
    (at most `repeat` batches, within about `budget_s` seconds) and the best batch wins,
    which hides one-off hiccups from the operating system.
    """
    start = time.perf_counter()
    fn(*args, **kwargs)
    first = time.perf_counter() - start
    if first <= 0:
        first = 1e-7
    if first * 2 > budget_s:
        return first
    loops = max(1, int(budget_s / repeat / first))
    best = first
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(loops):
            fn(*args, **kwargs)
        best = min(best, (time.perf_counter() - start) / loops)
    return best


def plain_distance(a: list[int], b: list[int]) -> int:
    """The squared distance written the way most students write it, used for speed estimates.

    The style matters: on Python 3.10, `x ** 2` is about 3 times slower than on 3.11+, so this
    function measures the interpreter as well as the laptop.
    """
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return total


def sample_images(n_pixels: int = 784, seed: int = 2026) -> tuple[list[int], list[int]]:
    """Two fixed digit-like images for timing (the same on every laptop).

    Like real digits, about 4 in 5 pixels are paper (0); the others are random ink values.
    The mix matters for speed: Python reuses the objects for small numbers.
    """
    rng = random.Random(seed)

    def image() -> list[int]:
        return [rng.randrange(1, 256) if rng.random() < 0.2 else 0 for _ in range(n_pixels)]

    return image(), image()


def distance_time(distance_fn=None, n_pixels: int = 784) -> float:
    """Seconds for one squared distance between two `n_pixels` images on this laptop."""
    a, b = sample_images(n_pixels)
    return best_time(distance_fn or plain_distance, a, b, budget_s=0.1)


def _benchmark_once() -> float:
    a, b = sample_images(784, seed=7)
    start = time.perf_counter()
    for _ in range(150):
        plain_distance(a, b)
    return time.perf_counter() - start


def machine_factor() -> float:
    """How fast is this laptop compared with the build laptop? 1.0 = the same speed.

    2.0 means this laptop needs twice as long for the same pure-Python work, so the League
    gives it twice the time budget. The value is measured once per session (best of 5 runs
    of a fixed distance workload written in the usual student style) and kept between 0.5
    and 4.0. It includes the Python version: 3.10 runs this workload about 3 times slower
    than 3.11 and newer.
    """
    if not _factor_cache:
        best = min(_benchmark_once() for _ in range(5))
        low, high = _FACTOR_LIMITS
        _factor_cache.append(round(min(high, max(low, best / BUILD_BENCHMARK_S)), 2))
    return _factor_cache[0]
