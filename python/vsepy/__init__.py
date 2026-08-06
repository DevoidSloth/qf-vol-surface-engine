"""vsepy -- Python interface to the qf-vol-surface-engine C++ core.

The compiled extension `_vse` carries every pricing routine. This package adds
the parts that are better expressed in Python: option-chain cleaning, the
implied-forward regression, calibration drivers and plots. Nothing here
reimplements a pricer.

Scalar functions from the core are NumPy ufuncs and broadcast, so a whole chain
inverts in one call:

    >>> import numpy as np, vsepy
    >>> K = np.array([4000.0, 4275.0, 4600.0])
    >>> px = vsepy.black76(4275.0, K, 0.37, 0.18, 0.982, vsepy.OptionType.Call)
    >>> np.round(vsepy.implied_volatility(px, 4275.0, K, 0.37, 0.982,
    ...                                   vsepy.OptionType.Call), 12)
    array([0.18, 0.18, 0.18])

The pipeline is three calls:

    >>> from vsepy import Chain, surface
    >>> chain, truth = Chain.synthetic()          # or Chain.from_csv(path, spot)
    >>> fitted = surface.fit(chain, "essvi")
    >>> fitted.calendar_free and fitted.butterfly_free
    True

`fitted` will tell you its RMSE and whether it admits arbitrage, and it will not
let you have one without the other. That is the whole point of the library.
"""

from __future__ import annotations

try:
    from . import _vse
except ImportError as exc:  # pragma: no cover - only hit before a build
    raise ImportError(
        "the compiled core (vsepy._vse) is not built. From the repository root:\n"
        "    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S . -B build\n"
        "    cmake --build build\n"
        f"(original error: {exc})"
    ) from exc

from . import chain as _chain_module  # noqa: F401
from . import models, plots, surface
from ._vse import (
    ConvergenceError,
    DomainError,
    Exercise,
    Greeks,
    HestonParams,
    HestonScheme,
    ImpliedVolResult,
    LSMCConfig,
    MCConfig,
    OptionType,
    PDEConfig,
    SABRParams,
    Sampling,
    SVIRaw,
    bates_price,
    binomial_crr,
    binomial_leisen_reimer,
    black76,
    bs_greeks,
    bs_price,
    calibrate_heston,
    check_butterfly,
    check_calendar,
    durrleman_g,
    erf,
    erfc,
    erfcx,
    heston_call_and_gradient,
    heston_call_lewis,
    heston_carr_madan,
    heston_mc,
    heston_mc_greeks_aad,
    heston_mc_greeks_bump,
    heston_pde,
    heston_price,
    heston_risk_factor_names,
    implied_total_volatility,
    implied_volatility,
    implied_volatility_ex,
    isotonic_increasing,
    lsmc_american,
    mills_ratio,
    norm_cdf,
    norm_inv_cdf,
    norm_pdf,
    normalised_black,
    normalised_black_inflection,
    normalised_vega,
    pde_vanilla,
    repair_sabr,
    repair_smile,
    risk_neutral_density,
    sabr_alpha_from_atm,
    sabr_density_scan,
    sabr_lognormal_vol,
    sabr_normal_vol,
)
from .chain import Chain, Slice, quotes_from_arrays, year_fraction

__version__ = _vse.__version__

# Grouped by what a reader would look for, not sorted, because an alphabetical
# list of 60 names is a list nobody reads.
__all__ = [  # noqa: RUF022
    # pipeline
    "Chain",
    "Slice",
    "surface",
    "models",
    "plots",
    "quotes_from_arrays",
    "year_fraction",
    # core types
    "ConvergenceError",
    "DomainError",
    "Exercise",
    "Greeks",
    "HestonParams",
    "HestonScheme",
    "ImpliedVolResult",
    "LSMCConfig",
    "MCConfig",
    "OptionType",
    "PDEConfig",
    "SABRParams",
    "SVIRaw",
    "Sampling",
    # pricing
    "bates_price",
    "binomial_crr",
    "binomial_leisen_reimer",
    "black76",
    "bs_greeks",
    "bs_price",
    "heston_carr_madan",
    "heston_call_and_gradient",
    "heston_call_lewis",
    "heston_mc",
    "heston_mc_greeks_aad",
    "heston_mc_greeks_bump",
    "heston_pde",
    "heston_price",
    "heston_risk_factor_names",
    "lsmc_american",
    "pde_vanilla",
    "isotonic_increasing",
    "repair_sabr",
    "repair_smile",
    "sabr_alpha_from_atm",
    "sabr_density_scan",
    "sabr_lognormal_vol",
    "sabr_normal_vol",
    # inversion and surface diagnostics
    "calibrate_heston",
    "check_butterfly",
    "check_calendar",
    "durrleman_g",
    "erf",
    "erfc",
    "erfcx",
    "implied_total_volatility",
    "implied_volatility",
    "implied_volatility_ex",
    "mills_ratio",
    "norm_cdf",
    "norm_inv_cdf",
    "norm_pdf",
    "normalised_black",
    "normalised_black_inflection",
    "normalised_vega",
    "risk_neutral_density",
]
