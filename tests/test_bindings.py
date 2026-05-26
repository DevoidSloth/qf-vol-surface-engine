"""Tests for the binding layer.

These do not re-test the numerics -- cpp/tests does that against 60-digit
references. What is tested here is everything that can go wrong *between* the two
languages, which in practice is argument order, broadcasting, and exceptions
silently becoming return values.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import vsepy
from vsepy import OptionType


def test_module_loads_and_reports_a_version():
    assert vsepy.__version__


def test_scalar_functions_broadcast_like_ufuncs():
    x = np.linspace(-4.0, 4.0, 9)
    cdf = vsepy.norm_cdf(x)
    assert isinstance(cdf, np.ndarray)
    assert cdf.shape == x.shape
    np.testing.assert_allclose(cdf + vsepy.norm_cdf(-x), 1.0, atol=1e-15)

    # Two-dimensional broadcasting: strikes across columns, expiries down rows.
    K = np.array([[90.0, 100.0, 110.0]])
    T = np.array([[0.25], [1.0], [2.0]])
    px = vsepy.black76(100.0, K, T, 0.2, 1.0, OptionType.Call)
    assert px.shape == (3, 3)
    # Longer expiry is worth more at every strike.
    assert np.all(np.diff(px, axis=0) > 0)
    # Higher strike is worth less at every expiry.
    assert np.all(np.diff(px, axis=1) < 0)


def test_argument_order_matches_the_cpp_signature():
    # A transposed forward and strike would still produce a plausible number, so
    # pin it against a case where the two are not interchangeable.
    itm = vsepy.black76(100.0, 50.0, 1.0, 0.2, 1.0, OptionType.Call)
    otm = vsepy.black76(50.0, 100.0, 1.0, 0.2, 1.0, OptionType.Call)
    assert itm > 49.0
    assert otm < 1.0


def test_greeks_struct_exposes_every_field():
    g = vsepy.bs_greeks(100.0, 105.0, 0.7, 0.03, 0.012, 0.28, OptionType.Call)
    for field in ("price", "delta", "gamma", "vega", "theta", "rho",
                  "vanna", "volga", "dual_delta", "dual_gamma"):
        assert isinstance(getattr(g, field), float)
    assert 0.0 < g.delta < 1.0
    assert g.gamma > 0.0
    assert g.vega > 0.0
    assert g.theta < 0.0
    assert "Greeks(" in repr(g)


def test_implied_volatility_result_carries_diagnostics():
    px = vsepy.black76(4275.0, 4600.0, 0.37, 0.18, 0.982, OptionType.Call)
    r = vsepy.implied_volatility_ex(px, 4275.0, 4600.0, 0.37, 0.982, OptionType.Call)
    assert r.converged
    assert 0 < r.iterations <= 8
    assert r.sigma == pytest.approx(0.18, rel=1e-12)


def test_domain_errors_surface_as_python_exceptions():
    # A C++ exception that fails to cross the boundary either aborts the process
    # or, worse, returns a garbage double.
    with pytest.raises(ValueError):
        vsepy.implied_volatility(-1.0, 100.0, 100.0, 1.0, 1.0, OptionType.Call)
    with pytest.raises(ValueError):
        vsepy.implied_volatility(101.0, 100.0, 100.0, 1.0, 1.0, OptionType.Call)
    with pytest.raises(ValueError):
        vsepy.norm_inv_cdf(0.0)


def test_a_whole_chain_inverts_in_one_call():
    F, T, df = 4275.0, 0.37, 0.982
    K = np.linspace(2600.0, 6800.0, 400)
    sigma = 0.13 + 0.35 * (np.log(F / K) ** 2)          # a crude smile
    otm_call = K >= F

    px = np.where(otm_call,
                  vsepy.black76(F, K, T, sigma, df, OptionType.Call),
                  vsepy.black76(F, K, T, sigma, df, OptionType.Put))

    # Index, do not np.where. np.where evaluates *both* arguments before
    # selecting, so the call branch would be handed the put prices of every
    # low strike -- prices far below a call's intrinsic value, which the solver
    # correctly rejects as arbitrage. The mask has to be applied to the inputs.
    iv = np.empty_like(K)
    iv[otm_call] = vsepy.implied_volatility(
        px[otm_call], F, K[otm_call], T, df, OptionType.Call)
    iv[~otm_call] = vsepy.implied_volatility(
        px[~otm_call], F, K[~otm_call], T, df, OptionType.Put)
    np.testing.assert_allclose(iv, sigma, rtol=1e-12)


def test_inverting_the_wrong_side_of_the_pair_raises():
    """The failure mode the test above sidesteps, pinned so it stays loud."""
    F, K, T, df = 4275.0, 2600.0, 0.37, 0.982
    put = vsepy.black76(F, K, T, 0.4, df, OptionType.Put)
    with pytest.raises(ValueError, match="below intrinsic"):
        vsepy.implied_volatility(put, F, K, T, df, OptionType.Call)


@settings(max_examples=300, deadline=None)
@given(
    z=st.floats(min_value=-4.0, max_value=4.0),
    sigma=st.floats(min_value=0.02, max_value=2.0),
    T=st.floats(min_value=1.0 / 365.0, max_value=5.0),
)
def test_round_trip_holds_for_arbitrary_inputs(z, sigma, T):
    """Property test over the binding, not just the fixed grid in cpp/tests."""
    F = 100.0
    s = sigma * np.sqrt(T)
    K = F * np.exp(z * s)
    otm = OptionType.Call if z >= 0 else OptionType.Put
    px = vsepy.black76(F, K, T, sigma, 1.0, otm)
    if px <= 0.0:
        return  # below the smallest representable double; nothing to invert
    iv = vsepy.implied_volatility(px, F, K, T, 1.0, otm)
    assert iv == pytest.approx(sigma, rel=1e-11)


@settings(max_examples=200, deadline=None)
@given(x=st.floats(min_value=-26.0, max_value=40.0))
def test_mills_ratio_satisfies_its_defining_ode(x):
    """R'(z) = z R(z) - 1, checked by central difference.

    The lower bound is not arbitrary. R(z) = N(-z)/phi(z) grows like
    e^{z^2/2} sqrt(2 pi) as z goes negative and passes the largest double at
    about z = -26.6, so below that the identity is between two infinities.
    """
    h = 1e-6
    fd = (vsepy.mills_ratio(x + h) - vsepy.mills_ratio(x - h)) / (2 * h)
    assert fd == pytest.approx(x * vsepy.mills_ratio(x) - 1.0, rel=1e-6, abs=1e-12)


def test_mills_ratio_overflows_to_infinity_not_nan():
    """R(z) ~ sqrt(2 pi) e^{z^2/2} going negative, so it passes the largest
    double near z = -37.6. Below that the answer is +inf, which is the IEEE
    result for an overflowing computation and the one that stays honest: a
    finite sentinel such as DBL_MAX is indistinguishable from data downstream,
    and NaN propagates silently and surfaces somewhere unrelated.
    """
    assert np.isfinite(vsepy.mills_ratio(-30.0))
    assert vsepy.mills_ratio(-30.0) == pytest.approx(6.79e195, rel=1e-2)
    assert np.isfinite(vsepy.mills_ratio(-37.0))
    for z in (-38.0, -50.0, -1e3):
        assert vsepy.mills_ratio(z) == np.inf
        assert not np.isnan(vsepy.mills_ratio(z))
