"""Stochastic-volatility model calibration against a cleaned chain.

The surface layer answers "what is the market saying". This layer answers "what
process is consistent with it", which is a different and strictly harder
question: an SVI surface can fit anything, and a Heston fit that misses by a
whole vol point in the short-dated wings is telling you something true about the
model rather than about the optimiser.

That distinction is the reason the report here breaks the error down BY EXPIRY
rather than quoting one number. A single RMSE hides exactly the failure everyone
already knows Heston has -- it cannot make a short-dated smile steep enough,
because diffusive volatility needs time to generate skew -- and hiding it makes
the calibration look better than the model is.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import _vse


@dataclass
class ExpiryError:
    expiry: float
    n: int
    rmse_vol_points: float
    max_error_vol_points: float
    atm_error_vol_points: float


@dataclass
class ModelFit:
    name: str
    params: object
    rmse_vol_points: float
    max_error_vol_points: float
    quotes: int
    seconds: float
    by_expiry: list = field(repr=False, default_factory=list)
    iterations: int = 0
    converged: bool = False
    message: str = ""

    def summary(self) -> str:
        lines = [
            (
                f"{self.name}: {self.rmse_vol_points:.3f} vol points RMSE, "
                f"worst {self.max_error_vol_points:.3f}, {self.quotes} quotes, "
                f"{self.iterations} iterations, {self.seconds * 1e3:.0f} ms"
            ),
            f"  {self.params}",
            f"  {'T':>8} {'pts':>5} {'rmse':>8} {'worst':>8} {'atm':>8}",
        ]
        for e in self.by_expiry:
            lines.append(
                f"  {e.expiry:8.4f} {e.n:5d} {e.rmse_vol_points:8.3f} "
                f"{e.max_error_vol_points:8.3f} "
                f"{e.atm_error_vol_points:+8.3f}"
            )
        if self.message:
            lines.append(f"  {self.message}")
        return "\n".join(lines)


def _calibration_quotes(chain):
    quotes = []
    for s in chain:
        for p in s.points:
            quotes.append(
                _vse.CalibrationQuote(s.forward, p.strike, s.expiry, p.implied_vol, p.weight)
            )
    return quotes


def _model_vol(price_fn, forward, strike, expiry, option_type):
    """Invert a model price back to a Black vol, on the OTM side.

    Comparing in vol rather than in price is not cosmetic: price errors are
    dominated by vega, so a price RMSE is really a weighted vol RMSE with
    weights nobody chose, and it makes a model look best wherever options are
    cheapest.
    """
    price = price_fn(forward, strike, expiry, option_type)
    if price <= 0.0:
        return float("nan")
    return _vse.implied_volatility(price, forward, strike, expiry, 1.0, option_type)


def fit_heston(chain, start=None, *, quadrature_order: int = 32, panels: int = 16) -> ModelFit:
    """Calibrate Heston to the whole board at once.

    Whole board, not slice by slice. Heston has five parameters and one process:
    fitting each expiry separately would produce five different processes and
    would not be a model of anything. The term structure of the smile is
    precisely the constraint that makes the calibration informative.
    """
    quotes = _calibration_quotes(chain)
    start = start if start is not None else _vse.HestonParams(0.04, 1.5, 0.04, 0.5, -0.7)
    t0 = time.perf_counter()
    result = _vse.calibrate_heston(quotes, start)
    seconds = time.perf_counter() - t0

    p = result.params

    def price(forward, strike, expiry, option_type):
        return _vse.heston_price(
            p, forward, strike, expiry, 1.0, option_type, quadrature_order, panels
        )

    by_expiry = _error_breakdown(chain, price)
    return ModelFit(
        name="Heston",
        params=p,
        # The core reports absolute vol; vol points are what a
        # desk reads, and the conversion happens once, here.
        rmse_vol_points=result.rmse_vol * 100.0,
        max_error_vol_points=result.max_error_vol * 100.0,
        quotes=result.quotes,
        seconds=seconds,
        by_expiry=by_expiry,
        iterations=result.iterations,
        converged=result.converged,
        message=result.message,
    )


def fit_sabr_slice(chain_slice, beta: float = 0.5, *, start=None):
    """Calibrate SABR to one expiry.

    Per-slice by design, unlike Heston. SABR is an asymptotic expansion of a
    single-expiry smile, not a term-structure model; fitting it across expiries
    would be using it for something it does not claim to do.

    beta is fixed rather than fitted. It is close to unidentifiable from vanilla
    quotes -- beta and rho trade off against each other along a valley the
    optimiser will wander down -- so it is a modelling choice (0.5 for rates,
    1.0 for equity-style lognormal dynamics) and pretending to fit it produces a
    number that moves day to day for no economic reason.
    """
    from scipy.optimize import least_squares

    cols = chain_slice.arrays()
    strikes, vols, weights = cols["strike"], cols["implied_vol"], cols["weight"]
    forward, expiry = chain_slice.forward, chain_slice.expiry
    atm = chain_slice.atm_vol
    w = np.sqrt(weights / np.mean(weights))

    def residual(x):
        rho, nu = np.tanh(x[0]), math.exp(x[1])
        alpha = _vse.sabr_alpha_from_atm(atm, forward, expiry, beta, rho, nu, 0.0)
        model = _vse.sabr_lognormal_vol(alpha, beta, rho, nu, 0.0, forward, strikes, expiry)
        return w * (model - vols) * 100.0

    x0 = np.array([np.arctanh(-0.3), math.log(0.4)]) if start is None else np.asarray(start)
    t0 = time.perf_counter()
    sol = least_squares(residual, x0, method="lm", xtol=1e-12, ftol=1e-12)
    seconds = time.perf_counter() - t0

    rho, nu = float(np.tanh(sol.x[0])), float(math.exp(sol.x[1]))
    alpha = _vse.sabr_alpha_from_atm(atm, forward, expiry, beta, rho, nu, 0.0)
    params = _vse.SABRParams(alpha, beta, rho, nu, 0.0)
    err = residual(sol.x) / w
    scan = _vse.sabr_density_scan(params, forward, expiry, 0.02, 3.0, 1500, False)
    return ModelFit(
        name=f"SABR(beta={beta})",
        params=params,
        rmse_vol_points=float(np.sqrt(np.mean(err**2))),
        max_error_vol_points=float(np.max(np.abs(err))),
        quotes=len(strikes),
        seconds=seconds,
        by_expiry=[
            ExpiryError(
                expiry,
                len(strikes),
                float(np.sqrt(np.mean(err**2))),
                float(np.max(np.abs(err))),
                0.0,
            )
        ],
        iterations=int(sol.nfev),
        converged=sol.success,
        message=(
            "arbitrage-free over the scanned range"
            if scan.free
            else f"Hagan density goes negative below K={scan.arbitrage_boundary:.2f} "
            f"({scan.violations} of {scan.points} grid points)"
        ),
    )


def _error_breakdown(chain, price_fn):
    out = []
    for s in chain:
        cols = s.arrays()
        errors = []
        atm_error = 0.0
        best_atm = float("inf")
        for k, strike, vol in zip(cols["log_moneyness"], cols["strike"], cols["implied_vol"]):
            option_type = _vse.OptionType.Call if strike >= s.forward else _vse.OptionType.Put
            model = _model_vol(price_fn, s.forward, float(strike), s.expiry, option_type)
            if not math.isfinite(model):
                continue
            e = (model - vol) * 100.0
            errors.append(e)
            if abs(k) < best_atm:
                best_atm, atm_error = abs(k), e
        errors = np.array(errors)
        out.append(
            ExpiryError(
                s.expiry,
                len(errors),
                float(np.sqrt(np.mean(errors**2))),
                float(np.max(np.abs(errors))),
                atm_error,
            )
        )
    return out


def heston_surface_grid(
    params, forwards, expiries, log_moneyness, *, order: int = 32, panels: int = 16
):
    """Model implied vols on a (k, T) grid, for plotting against the market.

    Prices the OTM side at every point and inverts, for the reason given in
    `_model_vol`.
    """
    k = np.asarray(log_moneyness, dtype=float)
    grid = np.empty((len(expiries), len(k)))
    for i, (T, F) in enumerate(zip(expiries, forwards)):
        for j, kk in enumerate(k):
            strike = F * math.exp(float(kk))
            option_type = _vse.OptionType.Call if kk >= 0.0 else _vse.OptionType.Put
            grid[i, j] = _model_vol(
                lambda f, s, t, o: _vse.heston_price(params, f, s, t, 1.0, o, order, panels),
                F,
                strike,
                T,
                option_type,
            )
    return grid
