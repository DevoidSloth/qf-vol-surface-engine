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

from ._vse import (  # noqa: F401
    DomainError,
    Greeks,
    ImpliedVolResult,
    OptionType,
    black76,
    bs_greeks,
    bs_price,
    erf,
    erfc,
    erfcx,
    implied_total_volatility,
    implied_volatility,
    implied_volatility_ex,
    mills_ratio,
    norm_cdf,
    norm_inv_cdf,
    norm_pdf,
    normalised_black,
    normalised_black_inflection,
    normalised_vega,
)

__version__ = _vse.__version__

__all__ = [
    "DomainError",
    "Greeks",
    "ImpliedVolResult",
    "OptionType",
    "black76",
    "bs_greeks",
    "bs_price",
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
]
