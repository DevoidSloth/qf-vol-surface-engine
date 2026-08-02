"""Figures.

Every plot here exists to make a specific claim checkable, and each one is
documented with the claim it supports. A surface plot that looks smooth is not
evidence of anything -- an arbitrageable surface looks smooth too, because the
violation lives in the second derivative where the eye cannot go. So the figures
that matter are the residual plot (does the fit sit inside the bid-ask?) and the
density plot (is it a probability distribution?), and the pretty 3D one is last.

matplotlib is imported lazily so that importing vsepy in a script that only
prices does not pay for it.
"""

from __future__ import annotations

import math

import numpy as np

from . import _vse
from .surface import _eval_svi

_STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def _plt():
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE)
    return plt


def smiles(chain, fitted, *, max_panels: int = 8, path=None):
    """Market quotes, the bid-ask band, and the fitted smile, per expiry.

    Plotted against log-moneyness rather than strike so the panels are
    comparable across expiries, and with the bid-ask drawn as a band because
    that is the only scale on which a residual means anything: a fit that misses
    by 0.3 vol points is excellent on a 1-point-wide market and unacceptable on
    a 0.05-point one.
    """
    plt = _plt()
    n = min(len(chain), max_panels)
    idx = np.linspace(0, len(chain) - 1, n).round().astype(int)
    cols_n = min(4, n)
    rows = math.ceil(n / cols_n)
    fig, axes = plt.subplots(rows, cols_n, figsize=(3.4 * cols_n, 2.8 * rows), squeeze=False)
    for ax, i in zip(axes.ravel(), idx):
        s = chain[i]
        c = s.arrays()
        k = c["log_moneyness"]
        ax.fill_between(
            k,
            (c["implied_vol"] - c["spread_vol"]) * 100,
            (c["implied_vol"] + c["spread_vol"]) * 100,
            color="0.85",
            label="bid-ask",
        )
        is_call = np.array([p.type == _vse.OptionType.Call for p in s.points])
        ax.plot(
            k[is_call],
            c["implied_vol"][is_call] * 100,
            ".",
            ms=3,
            color="#1f77b4",
            label="OTM calls",
        )
        ax.plot(
            k[~is_call],
            c["implied_vol"][~is_call] * 100,
            ".",
            ms=3,
            color="#2ca02c",
            label="OTM puts",
        )
        grid = np.linspace(k.min(), k.max(), 400)
        ax.plot(
            grid,
            np.sqrt(np.maximum(_eval_svi(fitted.slices[i], grid), 0) / s.expiry) * 100,
            "-",
            lw=1.2,
            color="#d62728",
            label=fitted.method,
        )
        ax.set_title(
            f"T = {s.expiry:.3f}y ({s.expiry * 365:.0f}d), F = {s.forward:.1f}", fontsize=8
        )
        ax.set_xlabel("k = ln(K/F)")
        ax.set_ylabel("implied vol (%)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{fitted.method.upper()} fit -- {fitted.rmse_vol_points:.3f} vol "
        f"points RMSE, {fitted.rmse_in_spreads:.2f} bid-ask spreads",
        fontsize=10,
    )
    fig.tight_layout()
    return _finish(fig, path)


def residuals(chain, fitted, *, path=None):
    """Fit error in units of the bid-ask spread, against moneyness and expiry.

    The claim: the residuals are noise, not shape. A residual plot with
    structure -- a smile in the residuals, a systematic sign in the wings --
    means the parameterisation cannot represent what the market is quoting, and
    no amount of reweighting fixes that.

    The +/-1 band is where a fit is indistinguishable from the quote itself.
    """
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    all_k, all_r, all_t = [], [], []
    for i, s in enumerate(chain):
        c = s.arrays()
        model = np.sqrt(np.maximum(_eval_svi(fitted.slices[i], c["log_moneyness"]), 0) / s.expiry)
        r = (model - c["implied_vol"]) / np.maximum(c["spread_vol"], 1e-12)
        all_k.append(c["log_moneyness"])
        all_r.append(r)
        all_t.append(np.full(len(r), s.expiry))
    k = np.concatenate(all_k)
    r = np.concatenate(all_r)
    t = np.concatenate(all_t)

    sc = ax1.scatter(k, r, c=t, s=5, cmap="viridis", norm="log")
    ax1.axhspan(-1, 1, color="0.9", zorder=0)
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.set_xlabel("k = ln(K/F)")
    ax1.set_ylabel("residual / bid-ask half-spread")
    ax1.set_title("by moneyness")
    fig.colorbar(sc, ax=ax1, label="expiry (y)")

    for s, d in zip(chain, fitted.diagnostics):
        ax2.plot(s.expiry, d.rmse_in_spreads, "o", ms=5, color="#1f77b4")
    ax2.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("expiry (y)")
    ax2.set_ylabel("RMSE / half-spread")
    ax2.set_title("by expiry")
    fig.suptitle(f"{fitted.method.upper()} residuals", fontsize=10)
    fig.tight_layout()
    return _finish(fig, path)


def density(chain, fitted, *, slice_index: int = -1, path=None, compare_spline: bool = True):
    """THE figure. Durrleman's g and the risk-neutral density.

    A surface is arbitrage-free iff g(k) >= 0 everywhere, and g is built from
    the first and second derivatives of total variance -- which is why looking
    at the fitted smile tells you nothing. Two smiles that overlay to the width
    of a line can have densities of opposite sign.

    The spline overlay is the point being made. A cubic spline through the same
    quotes has zero fit error, because it interpolates them, and produces a
    density that is negative over a large fraction of the range. It is the
    better fit by every measure except the one that decides whether the surface
    is a probability distribution.
    """
    plt = _plt()
    from .surface import spline_control

    i = slice_index if slice_index >= 0 else len(chain) + slice_index
    s = chain[i]
    raw = fitted.slices[i]
    c = s.arrays()
    k_lo, k_hi = c["log_moneyness"].min(), c["log_moneyness"].max()
    # Scan wider than the quotes, using the same adaptive wing window the core
    # uses. Plotting only where quotes exist is how a violation gets missed: the
    # 7-day slice below reports min g < 0 from a region no quote reaches, and a
    # plot cropped to the quote range shows a perfectly well-behaved curve. The
    # quoted region is shaded so the extrapolation is visible AS extrapolation.
    half = 6.0 * math.sqrt(max(raw.total_variance(0.0), 1e-10))
    for _ in range(3):
        half = 6.0 * math.sqrt(max(raw.total_variance(half), raw.total_variance(-half), 1e-10))
    k = np.linspace(-half, half, 1200)

    g = np.array([_vse.durrleman_g(raw, float(x)) for x in k])
    p = np.array([_vse.risk_neutral_density(raw, float(x)) for x in k])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(k, g, lw=1.4, color="#d62728", label=f"{fitted.method}")
    ax2.plot(k, p, lw=1.4, color="#d62728", label=f"{fitted.method}")
    if compare_spline:
        # scipy is an optional dependency and the overlay is a comparison, not
        # the figure. Narrow to ImportError deliberately: a numerical failure in
        # the spline is a bug worth seeing, and swallowing it here would leave
        # the panel silently missing the line that makes the whole argument.
        try:
            sp = spline_control(chain, i, points=1200)
            ax1.plot(
                sp["k"],
                sp["g"],
                lw=1.0,
                color="#1f77b4",
                alpha=0.85,
                label=f"cubic spline ({sp['violations']} violations)",
            )
        except ImportError:
            compare_spline = False
    d = fitted.diagnostics[i]
    for ax in (ax1, ax2):
        ax.axvspan(k_lo, k_hi, color="#fff3cd", zorder=0, label="quoted range")
        ax.axhline(0, color="0.3", lw=0.9)
        ax.set_xlabel("k = ln(K/F)")
    ax1.set_ylabel("g(k)")
    ax1.set_title("Durrleman's g -- negative anywhere means butterfly arbitrage")
    if d.min_g < 0:
        ax1.plot([d.k_at_min], [d.min_g], "v", ms=7, color="#d62728")
        ax1.annotate(
            f"min g = {d.min_g:.3f}",
            (d.k_at_min, d.min_g),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=7,
            color="#d62728",
        )
    ax1.legend(fontsize=7)
    ax1.set_ylim(min(-0.5, float(np.min(g)) * 1.2), max(2.0, float(np.max(g)) * 1.2))

    ax2.set_ylabel("density in k")
    ax2.set_title(f"risk-neutral density (integrates to {d.density_integral:.6f})")
    fig.suptitle(f"T = {s.expiry:.3f}y, scan +/-{half:.2f} in k", fontsize=10)
    fig.tight_layout()
    return _finish(fig, path)


def total_variance_term_structure(chain, fitted, *, path=None):
    """w(k, T) against T at fixed k. Every line must be non-decreasing.

    That is the calendar condition, and it is a statement about the SURFACE, not
    about any one slice: each slice can be a perfectly good probability
    distribution while the pair of them admits a calendar spread that is free
    money. Per-slice SVI, fitted independently, fails this routinely.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    times = np.array(fitted.expiries)
    for k in (-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6):
        w = np.array([_eval_svi(raw, k) for raw in fitted.slices])
        bad = np.diff(w) < 0
        ax.plot(times, w, "o-", ms=3, lw=1.1, label=f"k = {k:+.1f}")
        if bad.any():
            for j in np.flatnonzero(bad):
                ax.plot(
                    times[j : j + 2], w[j : j + 2], "-", lw=3, color="#d62728", alpha=0.6, zorder=0
                )
    ax.set_xscale("log")
    ax.set_xlabel("expiry (y)")
    ax.set_ylabel("total variance w = sigma^2 T")
    state = "no crossings" if fitted.calendar_free else "CALENDAR ARBITRAGE (thick red)"
    ax.set_title(f"{fitted.method.upper()} term structure -- {state}", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _finish(fig, path)


def surface_3d(chain, fitted, *, path=None):
    """The obligatory surface. Deliberately last.

    It is the least informative figure in this module: every surface in it,
    arbitrage-free or not, looks like a smooth sheet. Included because it shows
    the term structure and the skew together, and because the quotes are drawn
    on it so the extrapolation beyond them is visible as extrapolation.
    """
    plt = _plt()
    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    k = np.linspace(-1.0, 0.6, 90)
    t = np.geomspace(max(min(fitted.expiries), 1e-3), max(fitted.expiries), 60)
    K, T = np.meshgrid(k, t)
    V = np.array([[float(fitted.implied_vol(kk, tt)) for kk in k] for tt in t]) * 100

    ax.plot_surface(
        K, T, V, cmap="viridis", alpha=0.85, linewidth=0, rstride=1, cstride=1, antialiased=True
    )
    for s in chain:
        c = s.arrays()
        ax.scatter(
            c["log_moneyness"],
            np.full(s.n, s.expiry),
            c["implied_vol"] * 100,
            s=1.5,
            color="k",
            alpha=0.35,
            depthshade=False,
        )
    ax.set_xlabel("k = ln(K/F)")
    ax.set_ylabel("expiry (y)")
    ax.set_zlabel("implied vol (%)")
    ax.view_init(elev=24, azim=-128)
    ax.set_title(f"{fitted.method.upper()} surface, quotes in black", fontsize=10)
    fig.tight_layout()
    return _finish(fig, path)


def model_vs_market(chain, model_fit, *, max_panels: int = 8, path=None):
    """A calibrated model's smile against the market, expiry by expiry.

    Kept separate from `smiles` because the interesting content is different.
    A surface fit that misses is a fitting failure; a MODEL fit that misses is
    telling you the process cannot produce that smile, and the place it misses
    is the diagnosis. For Heston that is the short end, every time.
    """
    plt = _plt()
    from .models import _model_vol

    n = min(len(chain), max_panels)
    idx = np.linspace(0, len(chain) - 1, n).round().astype(int)
    cols_n = min(4, n)
    rows = math.ceil(n / cols_n)
    fig, axes = plt.subplots(rows, cols_n, figsize=(3.4 * cols_n, 2.8 * rows), squeeze=False)
    p = model_fit.params
    for ax, i in zip(axes.ravel(), idx):
        s = chain[i]
        c = s.arrays()
        ax.plot(c["log_moneyness"], c["implied_vol"] * 100, ".", ms=3, color="0.35", label="market")
        grid = np.linspace(c["log_moneyness"].min(), c["log_moneyness"].max(), 120)
        vols = []
        for kk in grid:
            strike = s.forward * math.exp(float(kk))
            ot = _vse.OptionType.Call if kk >= 0 else _vse.OptionType.Put
            vols.append(
                _model_vol(
                    lambda f, k_, t_, o: _vse.heston_price(p, f, k_, t_, 1.0, o, 32, 16),
                    s.forward,
                    strike,
                    s.expiry,
                    ot,
                )
            )
        ax.plot(grid, np.array(vols) * 100, "-", lw=1.2, color="#ff7f0e", label=model_fit.name)
        e = model_fit.by_expiry[i]
        ax.set_title(f"T = {s.expiry:.3f}y -- {e.rmse_vol_points:.2f} pts RMSE", fontsize=8)
        ax.set_xlabel("k = ln(K/F)")
        ax.set_ylabel("implied vol (%)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(
        f"{model_fit.name} calibrated to the whole board -- "
        f"{model_fit.rmse_vol_points:.3f} vol points RMSE",
        fontsize=10,
    )
    fig.tight_layout()
    return _finish(fig, path)


def convergence(
    results,
    *,
    xlabel="steps",
    ylabel="absolute error",
    title="",
    path=None,
    reference_slopes=(1, 2),
):
    """Log-log error against work, with reference slopes.

    A convergence plot without reference slopes is decoration. The claim being
    made is an ORDER -- that halving the step size divides the error by four --
    and the only way to read that off a curve is against a line of known slope.
    `results` maps a label to (x array, error array).
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, (x, err) in results.items():
        ax.loglog(x, err, "o-", ms=4, lw=1.2, label=label)
    x0 = np.array(
        [min(min(x) for x, _ in results.values()), max(max(x) for x, _ in results.values())],
        dtype=float,
    )
    y0 = max(max(e) for _, e in results.values())
    for slope in reference_slopes:
        ax.loglog(x0, y0 * (x0 / x0[0]) ** (-slope), "--", lw=0.8, color="0.5")
        ax.annotate(
            f"slope -{slope}",
            (x0[-1], y0 * (x0[-1] / x0[0]) ** (-slope)),
            fontsize=7,
            color="0.4",
            ha="right",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _finish(fig, path)


def _finish(fig, path):
    if path is not None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig
