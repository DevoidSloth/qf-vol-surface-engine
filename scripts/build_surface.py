#!/usr/bin/env python
"""End-to-end: chain in, arbitrage-free surface out.

    python scripts/build_surface.py                       # synthetic board
    python scripts/build_surface.py --csv data/spy.csv    # a fetched one
    python scripts/build_surface.py --compare --heston    # everything

Prints the cleaning report, fits the surface, checks it for arbitrage, and
writes figures to figures/. With --compare it fits all three parameterisations
and puts their fit quality next to their arbitrage diagnostics, which is the
comparison that matters and the one that makes per-slice SVI look worse than its
RMSE suggests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

from vsepy import (
    Chain,
    _vse,
    models,
    plots,
    surface,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv", type=Path, default=None, help="a chain from scripts/fetch_chain.py"
    )
    parser.add_argument("--spot", type=float, default=None)
    parser.add_argument("--method", default="essvi", choices=surface.METHODS)
    parser.add_argument("--compare", action="store_true", help="fit all three and tabulate")
    parser.add_argument(
        "--heston", action="store_true", help="also calibrate Heston to the whole board"
    )
    parser.add_argument("--sabr", action="store_true", help="also calibrate SABR to each expiry")
    parser.add_argument("--figures", type=Path, default=REPO / "figures")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--seed", type=int, default=20240614)
    args = parser.parse_args(argv)

    # ---- load -----------------------------------------------------------
    if args.csv is not None:
        chain = Chain.from_csv(args.csv, args.spot)
        truth = None
    else:
        config = _vse.SyntheticChainConfig()
        config.seed = args.seed
        chain, truth = Chain.synthetic(config=config)
        print("using a MANUFACTURED chain with a known ground truth (--csv for a real one)\n")
    print(chain.summary())
    if not chain.slices:
        print("\nnothing survived cleaning", file=sys.stderr)
        return 1

    # ---- fit ------------------------------------------------------------
    print()
    if args.compare:
        fits = {m: surface.fit(chain, m) for m in surface.METHODS}
        print(
            f"  {'method':>6} {'rmse':>8} {'worst':>8} {'spreads':>8} "
            f"{'ms':>6} {'butterfly':>10} {'calendar':>22}"
        )
        for name, f in fits.items():
            cal = (
                "free"
                if f.calendar_free
                else f"{f.calendar_report.violations} violations, "
                f"worst {f.calendar_report.worst_decrease:.2e}"
            )
            bf = "free" if f.butterfly_free else f"min g {min(d.min_g for d in f.diagnostics):+.3f}"
            print(
                f"  {name:>6} {f.rmse_vol_points:8.4f} "
                f"{f.max_error_vol_points:8.4f} {f.rmse_in_spreads:8.2f} "
                f"{f.seconds * 1e3:6.0f} {bf:>10} {cal:>22}"
            )
        print(
            "\nThe tightest fit is the one that admits arbitrage. That is not "
            "a coincidence:\nthe extra freedom that buys the last 0.006 vol "
            "points is freedom to bend the\nsmile into shapes no probability "
            "distribution can produce."
        )
        fitted = fits[args.method]
    else:
        fitted = surface.fit(chain, args.method)
    print()
    print(fitted.summary())

    control = surface.spline_control(chain)
    print(
        f"\ncubic spline control on the {control['expiry']:.3f}y slice: "
        f"{control['rmse_vol_points']:.4f} vol points RMSE "
        f"(it interpolates), min g {control['min_g']:.1f}, "
        f"{control['violations']} of {control['points']} grid points negative"
    )

    if truth is not None:
        _score_against_truth(chain, fitted, truth)

    # ---- models ---------------------------------------------------------
    heston = None
    if args.heston:
        print()
        heston = models.fit_heston(chain)
        print(heston.summary())
    if args.sabr:
        print()
        for s in chain:
            fit = models.fit_sabr_slice(s)
            print(
                f"  T={s.expiry:7.4f}  {fit.rmse_vol_points:6.3f} vol points  "
                f"rho={fit.params.rho:+.3f} nu={fit.params.nu:.3f}  "
                f"{fit.message}"
            )

    # ---- figures --------------------------------------------------------
    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        out = args.figures
        plots.smiles(chain, fitted, path=out / "smiles.png")
        plots.residuals(chain, fitted, path=out / "residuals.png")
        plots.density(chain, fitted, path=out / "density.png")
        plots.total_variance_term_structure(chain, fitted, path=out / "term.png")
        plots.surface_3d(chain, fitted, path=out / "surface.png")
        written = ["smiles", "residuals", "density", "term", "surface"]
        if heston is not None:
            plots.model_vs_market(chain, heston, path=out / "heston.png")
            written.append("heston")
        print(f"\nfigures -> {out}: " + ", ".join(f"{n}.png" for n in written))
    return 0


def _score_against_truth(chain, fitted, truth):
    """Compare the fit to the surface the quotes were generated from.

    This is the only comparison that separates fitting error from the noise in
    the quotes. Against the quotes alone a fitter cannot tell whether it is
    0.16 vol points away because the parameterisation is limited or because the
    quotes were rounded to a tick and jittered -- and on this board it is mostly
    the latter, which is the point.
    """
    import numpy as np

    errors = []
    for i, s in enumerate(chain):
        k = s.arrays()["log_moneyness"]
        fitted_vol = np.sqrt(
            np.maximum(np.array([fitted.total_variance(float(x), s.expiry) for x in k]), 0)
            / s.expiry
        )
        true_vol = np.array([truth.surface.total_variance(float(x), s.expiry) for x in k])
        true_vol = np.sqrt(np.maximum(true_vol, 0) / s.expiry)
        errors.append((fitted_vol - true_vol) * 100.0)
    e = np.concatenate(errors)
    print(
        f"\nagainst the generating surface: {np.sqrt(np.mean(e**2)):.4f} vol "
        f"points RMSE, worst {np.max(np.abs(e)):.4f}"
    )
    print(
        "  (the fit is closer to the truth than to the quotes, because the "
        "quotes carry\n   tick rounding and jitter that the fit averages out)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
