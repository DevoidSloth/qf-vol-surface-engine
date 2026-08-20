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
    slices: list = field(repr=False, default_factory=list)  # raw SVI per expiry
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
            f = np.vectorize(lambda kk: self.model.total_variance(float(kk), float(expiry)))
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
        arb = (
            "arbitrage-free"
            if (self.calendar_free and self.butterfly_free)
            else "ARBITRAGE PRESENT"
        )
        lines = [
            (
                f"{self.method} fit: {self.rmse_vol_points:.4f} vol points RMSE, "
                f"worst {self.max_error_vol_points:.4f}, "
                f"{self.rmse_in_spreads:.2f} bid-ask spreads, "
                f"{self.quotes} quotes, {self.seconds * 1e3:.0f} ms -- {arb}"
            ),
            (
                f"  {'T':>8} {'pts':>5} {'rmse':>8} {'worst':>8} {'spreads':>8} "
                f"{'min g':>10} {'min dens':>10} {'integral':>9} {'bf':>4}"
            ),
        ]
        for d in self.diagnostics:
            lines.append(
                f"  {d.expiry:8.4f} {d.n:5d} {d.rmse_vol_points:8.4f} "
                f"{d.max_error_vol_points:8.4f} {d.rmse_in_spreads:8.2f} "
                f"{d.min_g:10.3e} {d.min_density:10.3e} "
                f"{d.density_integral:9.6f} "
                f"{'ok' if d.butterfly_free else 'BAD':>4}"
            )
        if self.calendar_report is not None:
            c = self.calendar_report
            lines.append(
                f"  calendar: {c.violations} violations, worst decrease "
                f"{c.worst_decrease:.3e} at k={c.k_at_worst:.3f}, "
                f"T={c.t_at_worst:.4f}"
            )
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
        calendar_free = result.calendar.free and all(c.calendar_free for c in result.conditions)
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
        diagnostics.append(
            SliceDiagnostics(
                expiry=s.expiry,
                n=s.n,
                rmse_vol_points=float(np.sqrt(np.mean(err**2))),
                max_error_vol_points=float(np.max(np.abs(err))),
                rmse_in_spreads=float(np.sqrt(np.mean(spreads**2))),
                butterfly_free=bf.free,
                min_g=bf.min_g,
                k_at_min=bf.k_at_min,
                min_density=bf.min_density,
                density_integral=bf.density_integral,
                butterfly_violations=bf.violations,
            )
        )
        sq_total += float(np.sum(err**2))
        sq_spread_total += float(np.sum(spreads**2))
        worst = max(worst, float(np.max(np.abs(err))))
        count += len(err)

    calendar_report = None
    if method == "ssvi":
        calendar_report = result.calendar
    elif method in ("essvi", "svi"):
        calendar_report = _calendar_scan(raws, expiries)
        calendar_free = calendar_report.free if method == "svi" else calendar_free

    return FittedSurface(
        method=method,
        model=model,
        expiries=expiries,
        slices=raws,
        diagnostics=diagnostics,
        rmse_vol_points=math.sqrt(sq_total / count),
        max_error_vol_points=worst,
        rmse_in_spreads=math.sqrt(sq_spread_total / count),
        quotes=count,
        calendar_free=calendar_free,
        butterfly_free=all(d.butterfly_free for d in diagnostics),
        calendar_report=calendar_report,
        converged=converged,
        message=message,
        seconds=seconds,
    )


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
    return _ScanReport(
        free=violations == 0,
        worst_decrease=worst,
        k_at_worst=k_at,
        t_at_worst=t_at,
        violations=violations,
    )


def _calendar_holds(raws, expiries) -> bool:
    return _calendar_scan(raws, expiries).free


def spline_control(chain, slice_index: int = -1, points: int = 2001, space: str = "delta"):
    """The unconstrained alternative, and what it is actually failing at.

    A cubic spline through the same quotes -- what most surfaces are really
    built from. It interpolates them exactly, so its RMSE is zero, and its
    implied density is negative over roughly half the range. That much is the
    usual argument for an arbitrage-free parameterisation.

    The usual argument gets the reason wrong, and the measurement says so. On
    one slice of a generated board, varying only what is done to the quotes:

        quotes                     delta space          log-moneyness space
        exact                    min g  +0.18, 0       min g  +0.33, 0
        tick rounding only       min g  -2099, 423     min g  -36.5, 429
        quote noise only         min g  -5127, 983     min g  -686, 986
        both                     min g  -8530, 983     min g  -666, 985

    A spline through EXACT quotes is arbitrage-free. The interpolant is not the
    problem and neither is the coordinate; the problem is that an interpolant
    has as many degrees of freedom as it has quotes and is required to honour
    every one of them, including the part that is a rounding to the nearest
    tick. A second derivative then turns half a tick into a density of -8530.
    A parameterisation with two free parameters per slice cannot chase that
    noise, which is not a limitation of it -- it is the entire mechanism by
    which it stays a distribution.

    So the honest claim is not "splines admit arbitrage". It is that
    interpolation and noisy data cannot be combined, and market data is noisy.

    `space` is the x-coordinate:

      delta          Volatility against the Black-Scholes call delta, which is
                     what desks quote and interpolate in. The default, because
                     comparing against an interpolant nobody would defend proves
                     nothing. Note that it is NOT better behaved here -- it
                     compresses the wings, which concentrates the knots and
                     makes the worst g several times more negative -- so this
                     default costs the comparison rather than flattering it.
      log_moneyness  Total variance against k. Simpler, and the violation counts
                     come out much the same.

    Returns a dict with the spline's RMSE and its butterfly diagnostics on the
    same grid the SVI diagnostics use.
    """
    from scipy.interpolate import CubicSpline

    if space not in ("delta", "log_moneyness"):
        raise ValueError(f"space must be 'delta' or 'log_moneyness', got {space!r}")

    s = chain[slice_index]
    cols = s.arrays()
    k_in, vol_in = cols["log_moneyness"], cols["implied_vol"]
    grid = np.linspace(k_in.min(), k_in.max(), points)

    if space == "log_moneyness":
        w_in = cols["total_variance"]
        order = np.argsort(k_in)
        x, y = k_in[order], w_in[order]
        keep = np.concatenate(([True], np.diff(x) > 1e-12))
        spline = CubicSpline(x[keep], y[keep], bc_type="natural")
        wv = spline(grid)
        dw = spline(grid, 1)
        d2w = spline(grid, 2)
        fitted_vol = np.sqrt(np.maximum(spline(k_in), 0.0) / s.expiry)
    else:
        # Forward call delta, N(d1), which is monotone decreasing in strike and
        # so gives a well-ordered knot sequence without any sorting subtlety.
        def call_delta(k, vol):
            sd = vol * math.sqrt(s.expiry)
            return _vse.norm_cdf((-k + 0.5 * sd * sd) / sd)

        d_in = call_delta(k_in, vol_in)
        order = np.argsort(d_in)
        x, y = d_in[order], vol_in[order]
        keep = np.concatenate(([True], np.diff(x) > 1e-12))
        spline = CubicSpline(x[keep], y[keep], bc_type="natural")

        # Evaluating at a strike needs a fixed point, because the delta the
        # spline is indexed by depends on the volatility it returns. Damped, and
        # seeded from the nearest quote, it converges in a handful of passes;
        # this is the cost of quoting in delta and is why the convention is
        # awkward to implement even though it is the right one to use.
        def vol_at(k):
            v = float(np.interp(k, k_in, vol_in))
            for _ in range(60):
                nxt = float(spline(call_delta(k, v)))
                if not math.isfinite(nxt) or nxt <= 1e-6:
                    return v
                if abs(nxt - v) < 1e-13:
                    return nxt
                v = 0.5 * (v + nxt)
            return v

        smile = np.array([vol_at(float(kk)) for kk in grid])
        wv = smile**2 * s.expiry
        # Differentiate the reconstructed total variance numerically: the spline
        # is a function of delta, not of k, so its own derivatives are in the
        # wrong variable and the chain rule through the fixed point is not worth
        # deriving for a control.
        h = grid[1] - grid[0]
        dw = np.gradient(wv, h, edge_order=2)
        d2w = np.gradient(dw, h, edge_order=2)
        fitted_vol = np.array([vol_at(float(kk)) for kk in k_in])

    # Durrleman's g, the same expression check_butterfly uses in the core.
    g = (
        (1.0 - 0.5 * grid * dw / np.maximum(wv, 1e-300)) ** 2
        - 0.25 * dw**2 * (0.25 + 1.0 / np.maximum(wv, 1e-300))
        + 0.5 * d2w
    )
    err = (fitted_vol - vol_in) * 100.0
    return {
        "expiry": s.expiry,
        "space": space,
        "rmse_vol_points": float(np.sqrt(np.mean(err**2))),
        "min_g": float(np.min(g)),
        "violations": int(np.count_nonzero(g < 0.0)),
        "points": points,
        "k": grid,
        "g": g,
        "total_variance": wv,
    }
