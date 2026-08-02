"""Measure the accuracy of the normal/error functions against mpmath.

The claims in cpp/include/vse/normal.hpp are numbers, so they get checked with
numbers rather than asserted in a comment. This walks each routine over its
domain, compares against a 60-digit mpmath evaluation, and prints the worst
relative error by region.

Relative, never absolute: N(-20) is 2.8e-89, and an absolute tolerance is passed
by an implementation that returns zero. Relative accuracy in the lower tail is
the entire reason this library does not use the Hart/West rational CDF that most
option code uses -- the comparison is printed at the bottom so the size of that
difference is visible rather than asserted.

    python scripts/audit_normal.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import mpmath as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))
import vsepy

mp.mp.dps = 60


def ref_cdf(x):
    return mp.erfc(-mp.mpf(x) / mp.sqrt(2)) / 2


def ref_erfcx(x):
    return mp.exp(mp.mpf(x) ** 2) * mp.erfc(mp.mpf(x))


def ref_mills(z):
    return ref_cdf(-mp.mpf(z)) / (mp.exp(-(mp.mpf(z) ** 2) / 2) / mp.sqrt(2 * mp.pi))


def hart_west_cdf(x: float) -> float:
    """The rational CDF this library replaced, for comparison only.

    Graeme West's implementation of Hart (1968), which is the standard choice in
    option pricing code and is built for ~1e-15 *absolute* accuracy.
    """
    ax = abs(x)
    if ax > 37.0:
        c = 0.0
    else:
        e = math.exp(-0.5 * ax * ax)
        if ax < 7.07106781186547:
            num = 3.52624965998911e-02 * ax + 0.700383064443688
            for k in (
                6.37396220353165,
                33.912866078383,
                112.079291497871,
                221.213596169931,
                220.206867912376,
            ):
                num = num * ax + k
            den = 8.83883476483184e-02 * ax + 1.75566716318264
            for k in (
                16.064177579207,
                86.7807322029461,
                296.564248779674,
                637.333633378831,
                793.826512519948,
                440.413735824752,
            ):
                den = den * ax + k
            c = e * num / den
        else:
            b = ax + 0.65
            for k in (4.0, 3.0, 2.0, 1.0):
                b = ax + k / b
            c = e / (b * 2.5066282746310005)
    return 1.0 - c if x > 0 else c


def worst(fn, ref, points):
    w, wx = 0.0, 0.0
    for x in points:
        r = ref(x)
        if r == 0:
            continue
        rel = float(abs(mp.mpf(fn(x)) / r - 1))
        if rel > w:
            w, wx = rel, x
    return w, wx


def span(lo, hi, n=400):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def main() -> None:
    print(f"vsepy {vsepy.__version__}, reference: mpmath at {mp.mp.dps} digits\n")

    print("norm_cdf, relative error by region")
    print(f"  {'region':>18}  {'worst rel':>10}  {'at x':>10}  {'value there':>13}")
    regions = [(-1, 1), (-3, -1), (-5, -3), (-7.5, -5), (-12, -7.5), (-25, -12), (-37, -25), (1, 8)]
    for lo, hi in regions:
        w, wx = worst(vsepy.norm_cdf, ref_cdf, span(lo, hi))
        print(f"  {f'[{lo:g}, {hi:g}]':>18}  {w:10.2e}  {wx:10.3f}  {float(ref_cdf(wx)):13.3e}")

    print("\nerfcx, relative error by region")
    for lo, hi in [(-5, 0), (0, 0.46875), (0.46875, 4), (4, 30), (30, 1e6)]:
        w, wx = worst(vsepy.erfcx, ref_erfcx, span(lo, hi))
        print(f"  {f'[{lo:g}, {hi:g}]':>18}  {w:10.2e}  {wx:10.3f}")

    print("\nmills_ratio, relative error by region")
    for lo, hi in [(-3, 0), (0, 3), (3, 10), (10, 40), (40, 1000)]:
        w, wx = worst(vsepy.mills_ratio, ref_mills, span(lo, hi))
        print(f"  {f'[{lo:g}, {hi:g}]':>18}  {w:10.2e}  {wx:10.3f}")

    print("\nnorm_inv_cdf, round trip |N(N^-1(p))/p - 1|")
    ps = [10.0**-k for k in range(1, 15)] + [0.25, 0.5, 0.75]
    w = max(abs(float(mp.mpf(vsepy.norm_cdf(vsepy.norm_inv_cdf(p))) / mp.mpf(p) - 1)) for p in ps)
    print(f"  worst over p in [1e-14, 0.75]: {w:.2e}")

    print("\nfor comparison: Hart/West rational CDF, the usual choice")
    print(f"  {'region':>18}  {'worst rel':>10}  {'this library':>13}")
    for lo, hi in regions:
        wh, _ = worst(hart_west_cdf, ref_cdf, span(lo, hi))
        wo, _ = worst(vsepy.norm_cdf, ref_cdf, span(lo, hi))
        print(f"  {f'[{lo:g}, {hi:g}]':>18}  {wh:10.2e}  {wo:13.2e}")
    print("\n  Hart/West is accurate to ~1e-15 absolute, which is all it claims.")
    print("  In the tail that is not the same thing as relative accuracy, and the")
    print("  implied-vol round trip in the wings needs the relative kind.")


if __name__ == "__main__":
    main()
