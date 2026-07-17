"""Surface calibration driver.

One entry point, `fit`, that takes a cleaned `Chain` and returns something that
can be asked for a volatility at any (k, T) along with an honest account of
whether it is arbitrage-free.

The three parameterisations are not interchangeable and the choice is not
stylistic:

  svi    Per-slice, five parameters each. Fits the tightest because it has the
         most freedom, and gives no calendar guarantee whatsoever -- adjacent
         slices are fitted independently and nothing stops their total variances
         from crossing.
  ssvi   One rho and one phi(theta) for the whole surface. Butterfly and
         calendar conditions become closed-form checks on the parameters
         (Gatheral-Jacquier Theorem 4.2), so the surface is arbitrage-free by
         construction rather than by inspection. The price is a single skew
         shape for every expiry, which real boards violate.
  essvi  Per-expiry rho and psi, with the calendar condition built into the
         parameterisation rather than penalised. Recovers most of the fit
         quality of per-slice SVI while keeping the guarantee. This is the
         default.

A fit that is 0.02 vol points tighter and admits arbitrage is worse than one
that is looser and does not, because the arbitrage is what a hedging strategy
finds first. `fit` therefore reports the arbitrage diagnostics next to the RMSE
and does not let the caller see one without the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _vse

METHODS = ("essvi", "ssvi", "svi")


@dataclass
class SliceDiagnostics:
    expiry: float
    n: int
    rmse_vol_points: float
    max_error_vol_points: float
    rmse_in_spreads: float
    butterfly_free: bool
    min_g: float
    k_at_min: float
    min_density: float
    density_integral: float
    butterfly_violations: int


@dataclass
class FittedSurface:
    """A calibrated surface plus everything needed to distrust it."""

    method: str
    model: object = field(repr=False, default=None)
    expiries: list = field(default_factory=list)
    slices: list = field(repr=False, default_factory=list)      # raw SVI per expiry
    diagnostics: list = field(repr=False, default_factory=list)
    rmse_vol_points: float = float("nan")
    max_error_vol_points: float = float("nan")
    rmse_in_spreads: float = float("nan")
    quotes: int = 0
    calendar_free: bool = False
    butterfly_free: bool = False
    calendar_report: object = field(repr=False, default=None)
    converged: bool = False
    message: str = ""
    seconds: float = float("nan")

    # -- evaluation ------------------------------------------------------
    def total_variance(self, k, expiry):
        """w(k, T). Broadcasts over k."""
        k = np.asarray(k, dtype=float)
        if self.method in ("ssvi", "essvi"):
            f = np.vectorize(lambda kk: self.model.total_variance(float(kk),
                                                                  float(expiry)))
            return f(k) if k.ndim else float(f(k))
        # Per-slice SVI: interpolate in total variance between bracketing
        # expiries, which is the only interpolation that cannot manufacture a
        # calendar arbitrage out of two slices that individually have none.
        return self._interpolate_slices(k, expiry)

    def implied_vol(self, k, expiry):
        w = self.total_variance(k, expiry)
        return np.sqrt(np.asarray(w) / expiry)

    def vol_at_strike(self, strike, forward, expiry):
        return self.implied_vol(np.log(np.asarray(strike, float) / forward), expiry)

    def slice_at(self, expiry):
        """The raw-SVI form of one slice, whatever the parameterisation."""
        i = int(np.argmin([abs(e - expiry) for e in self.expiries]))
        return self.slices[i]

    def _interpolate_slices(self, k, expiry):
        times = np.asarray(self.expiries, float)
        if expiry <= times[0]:
            base = self.slices[0]
            scale = expiry / times[0]
            return scale * _eval_svi(base, k)
        if expiry >= times[-1]:
            base = self.slices[-1]
            return (expiry / times[-1]) * _eval_svi(base, k)
        j = int(np.searchsorted(times, expiry))
        lo, hi = times[j - 1], times[j]
        t = (expiry - lo) / (hi - lo)
        return (1 - t) * _eval_svi(self.slices[j - 1], k) + t * _eval_svi(self.slices[j], k)

    # -- reporting -------------------------------------------------------
    def summary(self) -> str:
        arb = ("arbitrage-free" if (self.calendar_free and self.butterfly_free)
               else "ARBITRAGE PRESENT")
        lines = [
            f"{self.method} fit: {self.rmse_vol_points:.4f} vol points RMSE, "
            f"worst {self.max_error_vol_points:.4f}, "
            f"{self.rmse_in_spreads:.2f} bid-ask spreads, "
            f"{self.quotes} quotes, {self.seconds * 1e3:.0f} ms -- {arb}",
            f"  {'T':>8} {'pts':>5} {'rmse':>8} {'worst':>8} {'spreads':>8} "
            f"{'min g':>10} {'min dens':>10} {'integral':>9} {'bf':>4}",
        ]
        for d in self.diagnostics:
            lines.append(
                f"  {d.expiry:8.4f} {d.n:5d} {d.rmse_vol_points:8.4f} "
                f"{d.max_error_vol_points:8.4f} {d.rmse_in_spreads:8.2f} "
                f"{d.min_g:10.3e} {d.min_density:10.3e} "
                f"{d.density_integral:9.6f} "
                f"{'ok' if d.butterfly_free else 'BAD':>4}")
        if self.calendar_report is not None:
            c = self.calendar_report
            lines.append(f"  calendar: {c.violations} violations, worst decrease "
                         f"{c.worst_decrease:.3e} at k={c.k_at_worst:.3f}, "
                         f"T={c.t_at_worst:.4f}")
        if self.message:
            lines.append(f"  {self.message}")
        return "\n".join(lines)


def _eval_svi(raw, k):
    k = np.asarray(k, dtype=float)
    km = k - raw.m
    return raw.a + raw.b * (raw.rho * km + np.sqrt(km * km + raw.sigma * raw.sigma))


def fit(chain, method: str = "essvi", *, density_points: int = 2001) -> FittedSurface:
    """Calibrate a surface to a cleaned chain.

    The returned object carries the arbitrage diagnostics whether or not anyone
    asks for them, because the failure mode this library exists to avoid is a
    surface that quotes a good RMSE and a negative density in the same breath.
    """
    import time

    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if len(chain) == 0:
        raise ValueError("chain has no usable expiries")

    slices = [list(s.points) for s in chain]
    expiries = list(chain.expiries)
    theta = list(chain.theta)

    start = time.perf_counter()
    if method == "essvi":
        result = _vse.fit_essvi(slices, expiries, theta)
        model = result.surface
        raws = [model.slice_at_index(i).to_raw() for i in range(len(expiries))]
        calendar_free = result.calendar_conditions_hold
        message = result.message
        converged = True
    elif method == "ssvi":
        result = _vse.fit_ssvi(slices, expiries, theta)
        model = result.surface
        raws = [model.slice_at(t).to_raw() for t in expiries]
        calendar_free = (result.calendar.free
                         and all(c.calendar_free for c in result.conditions))
        message = result.message
        converged = result.converged
    else:
        fits = [_vse.fit_svi_slice(list(s.points), s.expiry) for s in chain]
        model = None
        raws = [f.params for f in fits]
        converged = all(f.converged for f in fits)
        message = "; ".join(f.message for f in fits if f.message) or ""
        calendar_free = _calendar_holds(raws, expiries)
    seconds = time.perf_counter() - start

    diagnostics = []
    sq_total, sq_spread_total, worst, count = 0.0, 0.0, 0.0, 0
    for raw, s in zip(raws, chain):
        cols = s.arrays()
        w = _eval_svi(raw, cols["log_moneyness"])
        model_vol = np.sqrt(np.maximum(w, 0.0) / s.expiry)
        err = (model_vol - cols["implied_vol"]) * 100.0
        spreads = err / np.maximum(cols["spread_vol"] * 100.0, 1e-12)
        bf = _vse.check_butterfly(raw, s.expiry, 0.0, density_points)
        diagnostics.append(SliceDiagnostics(
            expiry=s.expiry, n=s.n,
            rmse_vol_points=float(np.sqrt(np.mean(err ** 2))),
            max_error_vol_points=float(np.max(np.abs(err))),
            rmse_in_spreads=float(np.sqrt(np.mean(spreads ** 2))),
            butterfly_free=bf.free, min_g=bf.min_g, k_at_min=bf.k_at_min,
            min_density=bf.min_density,
            density_integral=bf.density_integral, butterfly_violations=bf.violations))
        sq_total += float(np.sum(err ** 2))
        sq_spread_total += float(np.sum(spreads ** 2))
        worst = max(worst, float(np.max(np.abs(err))))
        count += len(err)

    calendar_report = None
    if method == "ssvi":
        calendar_report = result.calendar
    elif method in ("essvi", "svi"):
        calendar_report = _calendar_scan(raws, expiries)
        calendar_free = calendar_report.free if method == "svi" else calendar_free

    return FittedSurface(
        method=method, model=model, expiries=expiries, slices=raws,
        diagnostics=diagnostics,
        rmse_vol_points=math.sqrt(sq_total / count),
        max_error_vol_points=worst,
        rmse_in_spreads=math.sqrt(sq_spread_total / count),
        quotes=count,
        calendar_free=calendar_free,
        butterfly_free=all(d.butterfly_free for d in diagnostics),
        calendar_report=calendar_report, converged=converged, message=message,
        seconds=seconds)


@dataclass
class _ScanReport:
    free: bool
    worst_decrease: float
    k_at_worst: float
    t_at_worst: float
    violations: int


def _calendar_scan(raws, expiries, half_width: float = 1.5, points: int = 401):
    """Scan w(k, T) for decreases in T on a grid of k.

    Per-slice SVI has no structural calendar guarantee, so the only way to know
    is to look. Reported as the worst decrease in total variance, which is the
    quantity a calendar spread monetises.
    """
    ks = np.linspace(-half_width, half_width, points)
    worst, k_at, t_at, violations = 0.0, 0.0, 0.0, 0
    for i in range(1, len(raws)):
        w_lo = _eval_svi(raws[i - 1], ks)
        w_hi = _eval_svi(raws[i], ks)
        drop = w_lo - w_hi
        bad = drop > 1e-12
        violations += int(np.count_nonzero(bad))
        j = int(np.argmax(drop))
        if drop[j] > worst:
            worst, k_at, t_at = float(drop[j]), float(ks[j]), float(expiries[i])
    return _ScanReport(free=violations == 0, worst_decrease=worst, k_at_worst=k_at,
                       t_at_worst=t_at, violations=violations)


def _calendar_holds(raws, expiries) -> bool:
    return _calendar_scan(raws, expiries).free


def spline_control(chain, slice_index: int = -1, points: int = 2001):
    """The unconstrained alternative, for comparison.

    A cubic spline through the same quotes in total variance -- what most
    surfaces are actually built from. It interpolates the quotes essentially
    exactly, which looks like the better fit until the second derivative is
    examined: an interpolant has no reason to produce a positive density and
    generally does not.

    Returns a dict with the spline's RMSE and its butterfly diagnostics, on the
    same grid the SVI diagnostics use.
    """
    from scipy.interpolate import CubicSpline

    s = chain[slice_index]
    cols = s.arrays()
    k, w = cols["log_moneyness"], cols["total_variance"]
    order = np.argsort(k)
    k, w = k[order], w[order]
    keep = np.concatenate(([True], np.diff(k) > 1e-12))
    spline = CubicSpline(k[keep], w[keep], bc_type="natural")

    grid = np.linspace(k[0], k[-1], points)
    wv = spline(grid)
    dw = spline(grid, 1)
    d2w = spline(grid, 2)
    # Durrleman's g, the same expression check_butterfly uses in the core.
    g = ((1.0 - 0.5 * grid * dw / np.maximum(wv, 1e-300)) ** 2
         - 0.25 * dw ** 2 * (0.25 + 1.0 / np.maximum(wv, 1e-300))
         + 0.5 * d2w)
    fitted_vol = np.sqrt(np.maximum(spline(cols["log_moneyness"]), 0.0) / s.expiry)
    err = (fitted_vol - cols["implied_vol"]) * 100.0
    return {
        "expiry": s.expiry,
        "rmse_vol_points": float(np.sqrt(np.mean(err ** 2))),
        "min_g": float(np.min(g)),
        "violations": int(np.count_nonzero(g < 0.0)),
        "points": points,
        "k": grid,
        "g": g,
        "total_variance": wv,
    }
