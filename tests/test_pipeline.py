"""Tests for the Python pipeline: chain cleaning, calibration, diagnostics.

The C++ suite tests the numerics against 60-digit references. What is tested
here is the layer above: that a board of quotes goes in and a surface that is
actually arbitrage-free comes out, that the things which are supposed to be
recovered are recovered, and that the failure modes announce themselves instead
of producing a plausible number.

The synthetic generator is what makes most of this possible. It manufactures
quotes from a known eSSVI surface at known forwards and discount factors, so
every stage can be checked against the thing it is trying to recover rather than
against its own output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import vsepy
from vsepy import Chain, _vse, models, surface


@pytest.fixture(scope="module")
def board():
    config = _vse.SyntheticChainConfig()
    config.seed = 909091
    chain, truth = Chain.synthetic(config=config)
    return chain, truth


# ---------------------------------------------------------------------------
# Chain cleaning
# ---------------------------------------------------------------------------


def test_the_forward_is_recovered_from_parity_not_assumed(board):
    """The parity regression must find the true forward and discount factor.

    This is the stage everything else rests on. An error here is not noise: it
    shifts every log-moneyness on the slice in the same direction, which tilts
    the smile and biases the skew, and no fitter downstream can undo it.
    """
    chain, truth = board
    for s, true_f, true_df in zip(chain, truth.forwards, truth.discounts):
        assert s.fit.ok
        assert s.forward == pytest.approx(true_f, rel=2e-5)
        assert s.discount == pytest.approx(true_df, rel=2e-5)


def test_cleaning_keeps_the_out_of_the_money_side(board):
    """Calls above the forward, puts below, always.

    Both members of a strike pair carry the same information, but the in-the-
    money one carries it behind an intrinsic value that can dwarf the time
    value. Inverting that throws away the digits that distinguish one volatility
    from another.
    """
    chain, _ = board
    for s in chain:
        for p in s.points:
            if p.strike >= s.forward:
                assert p.type == vsepy.OptionType.Call
            else:
                assert p.type == vsepy.OptionType.Put


def test_the_cleaning_report_accounts_for_every_quote(board):
    chain, _ = board
    for s in chain:
        r = s.report
        assert r.balances(), (
            f"{r.input_quotes} in, {r.kept} kept, {r.dropped_total()} accounted for"
        )
        assert r.kept == s.n
        # The in-the-money side is roughly half of every board and has to be
        # counted, not merely skipped.
        assert r.dropped_in_the_money > 0


def test_an_expiry_that_cannot_be_fitted_is_named_not_dropped_silently():
    """A board that quietly loses its front month is worse than one that says so."""
    quotes = vsepy.quotes_from_arrays(
        strikes=[100.0, 110.0], bids=[1.0, 0.5], asks=[1.2, 0.7], types=["C", "C"]
    )  # calls only: no parity pairs
    chain = Chain.build({0.25: quotes}, spot=100.0)
    assert len(chain) == 0
    assert len(chain.rejected) == 1
    expiry, why = chain.rejected[0]
    assert expiry == 0.25
    assert "parity" in why


# ---------------------------------------------------------------------------
# Surface calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["essvi", "ssvi"])
def test_the_structural_parameterisations_are_arbitrage_free(board, method):
    """The claim the library exists to support, checked rather than asserted."""
    chain, _ = board
    fitted = surface.fit(chain, method)
    assert fitted.butterfly_free
    assert fitted.calendar_free
    for d in fitted.diagnostics:
        assert d.min_g > 0.0
        assert d.min_density >= 0.0
        assert d.density_integral == pytest.approx(1.0, abs=1e-4)


def test_essvi_beats_ssvi_on_fit_and_matches_it_on_arbitrage(board):
    """Per-expiry rho and psi buy fit quality without giving up the guarantee.

    That is the entire argument for the extended form. If it did not beat SSVI
    there would be no reason to carry the extra parameters.
    """
    chain, _ = board
    essvi = surface.fit(chain, "essvi")
    ssvi = surface.fit(chain, "ssvi")
    assert essvi.rmse_vol_points < ssvi.rmse_vol_points
    assert essvi.butterfly_free and essvi.calendar_free


def test_per_slice_svi_gives_up_the_guarantee(board):
    """The trade, measured -- and it is not the trade it looks like.

    Independent slices have five free parameters each against eSSVI's two, so at
    the optimum they cannot fit worse. What they give up is consistency: nothing
    in a per-slice fit stops adjacent total variances from crossing, and on this
    board they do.

    The assertion deliberately does NOT claim SVI fits tighter. It often does
    not, and the reason is worth stating: 35 free parameters over 7 slices, each
    slice optimised alone with no neighbour to stabilise it, is a harder
    landscape than 14 parameters tied together. The per-slice optimiser reports
    "normal equations singular at maximum damping" on this board and lands at
    0.188 vol points against eSSVI's 0.172. More freedom is not more fit when
    the freedom makes the problem harder to solve.
    """
    chain, _ = board
    svi = surface.fit(chain, "svi")
    assert not (svi.calendar_free and svi.butterfly_free), (
        "per-slice SVI came out arbitrage-free on this board. The seed may need "
        "changing, but do not weaken the assertion: the claim is that nothing "
        "in the parameterisation prevents the failure, not that it always fails"
    )
    assert svi.calendar_report.violations > 0 or not svi.butterfly_free


def test_the_fit_is_closer_to_the_generating_surface_than_to_the_quotes(board):
    """Quote noise is being averaged out, not fitted.

    A parameterisation that got closer to the quotes than to the surface behind
    them would be fitting the tick rounding, and its RMSE would be a measure of
    how well it memorised the noise.
    """
    chain, truth = board
    fitted = surface.fit(chain, "essvi")
    errors = []
    for s in chain:
        k = s.arrays()["log_moneyness"]
        model = np.array([fitted.total_variance(float(x), s.expiry) for x in k])
        true = np.array([truth.surface.total_variance(float(x), s.expiry) for x in k])
        errors.append((np.sqrt(model / s.expiry) - np.sqrt(true / s.expiry)) * 100)
    to_truth = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
    assert to_truth < fitted.rmse_vol_points


def test_an_unconstrained_spline_interpolates_and_is_not_a_distribution(board):
    """The control the whole library is arguing against.

    Zero fit error, because it passes through every quote, and a density that is
    negative over a large part of the range. Every measure except the one that
    decides whether the surface is a probability distribution says it is better.
    """
    chain, _ = board
    control = surface.spline_control(chain)
    fitted = surface.fit(chain, "essvi")
    assert control["rmse_vol_points"] < 1e-9
    assert control["rmse_vol_points"] < fitted.rmse_vol_points
    assert control["violations"] > 0.1 * control["points"]
    assert control["min_g"] < -1.0


def test_fit_rejects_an_unknown_method(board):
    chain, _ = board
    with pytest.raises(ValueError, match="method must be one of"):
        surface.fit(chain, "quintic-splines-and-hope")


# ---------------------------------------------------------------------------
# Model calibration
# ---------------------------------------------------------------------------


def test_heston_misses_the_short_end_and_the_report_shows_it(board):
    """A model limitation, reported rather than averaged away.

    Diffusive volatility needs time to generate skew, so Heston cannot make a
    week-long smile steep enough. A driver that quoted one RMSE would hide its
    most useful output; this asserts the breakdown exposes it.
    """
    chain, _ = board
    fit = models.fit_heston(chain)
    assert len(fit.by_expiry) == len(chain)
    shortest = fit.by_expiry[0]
    longest = fit.by_expiry[-1]
    assert shortest.rmse_vol_points > 2.0 * longest.rmse_vol_points
    assert fit.rmse_vol_points < shortest.rmse_vol_points


def test_the_reported_aggregate_matches_the_per_expiry_breakdown(board):
    """Units, checked.

    The core reports absolute vol and the driver reports vol points. This is the
    assertion that would have caught the conversion being missed, which it once
    was -- an aggregate a hundred times smaller than every one of its parts.
    """
    chain, _ = board
    fit = models.fit_heston(chain)
    weighted = math.sqrt(
        sum(e.n * e.rmse_vol_points**2 for e in fit.by_expiry) / sum(e.n for e in fit.by_expiry)
    )
    assert fit.rmse_vol_points == pytest.approx(weighted, rel=0.35)


def test_sabr_fits_one_slice_and_reports_its_arbitrage_boundary(board):
    chain, _ = board
    fit = models.fit_sabr_slice(chain[2])
    assert fit.rmse_vol_points < 2.0
    assert -1.0 < fit.params.rho < 1.0
    assert fit.params.nu > 0.0
    assert fit.message


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_every_figure_renders(board, tmp_path):
    """Not a check that the plots are right -- a check that they run.

    Plotting code is the part of a project that rots first, because nothing
    imports it and nobody notices until a demo.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from vsepy import plots

    chain, _ = board
    fitted = surface.fit(chain, "essvi")
    for name, fn in (
        ("smiles", plots.smiles),
        ("residuals", plots.residuals),
        ("density", plots.density),
        ("term", plots.total_variance_term_structure),
        ("surface", plots.surface_3d),
    ):
        path = tmp_path / f"{name}.png"
        fn(chain, fitted, path=path)
        assert path.exists() and path.stat().st_size > 5000
