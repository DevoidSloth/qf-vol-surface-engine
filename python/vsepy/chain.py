"""Option chains: loading, cleaning, and reduction to fittable points.

The C++ core does the numerics. What lives here is everything that is really
data handling -- reading a vendor file, deciding what an expiry is in years,
grouping by expiry, and calling the core once per slice. That work is a poor fit
for C++ and a natural one for pandas, and keeping it on this side means the core
never has to know what a CSV column is called.

The one thing this module is opinionated about is that the forward is FITTED,
not assumed. See `Chain.build`.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _vse

# yfinance and a few other vendors spell things differently. Anything not in
# this map has to be renamed by the caller, which is deliberate: silently
# guessing which column is the bid is not a service.
_COLUMN_ALIASES = {
    "strike": "strike",
    "bid": "bid",
    "ask": "ask",
    "volume": "volume",
    "openinterest": "open_interest",
    "open_interest": "open_interest",
    "opt_type": "type",
    "type": "type",
    "right": "type",
    "cp": "type",
    "callput": "type",
    "expiry": "expiry",
    "expiration": "expiry",
    "expirationdate": "expiry",
    "lastprice": "last",
    "last": "last",
}

_YEAR = 365.0


def year_fraction(expiry, asof=None) -> float:
    """Calendar-day year fraction to an expiry.

    ACT/365 to the 4pm close, which is the convention listed equity options are
    quoted against and therefore the one the quoted vols were computed with.
    A business-day count would be defensible in isolation and would put this
    library's vols on a different footing from every screen the numbers get
    compared against.
    """
    if isinstance(expiry, str):
        expiry = _dt.datetime.fromisoformat(expiry)
    if isinstance(expiry, _dt.date) and not isinstance(expiry, _dt.datetime):
        expiry = _dt.datetime.combine(expiry, _dt.time(16, 0))
    if asof is None:
        # Naive local time on purpose. An expiry is a local exchange close, and
        # a tz-aware "now" against a naive expiry raises; making both aware
        # would need the exchange's zone, which the caller has and this function
        # does not. Pass an aware `asof` with an aware `expiry` and it works.
        asof = _dt.datetime.now()  # noqa: DTZ005
    if isinstance(asof, _dt.date) and not isinstance(asof, _dt.datetime):
        asof = _dt.datetime.combine(asof, _dt.time(16, 0))
    return (expiry - asof).total_seconds() / 86400.0 / _YEAR


@dataclass
class Slice:
    """One expiry, after cleaning."""

    expiry: float
    forward: float
    discount: float
    points: list = field(repr=False)
    fit: object = field(repr=False, default=None)  # _vse.ForwardFit
    report: object = field(repr=False, default=None)  # _vse.SliceBuildReport
    label: str = ""

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def atm_total_variance(self) -> float:
        return _vse.atm_total_variance(self.points)

    @property
    def atm_vol(self) -> float:
        return math.sqrt(self.atm_total_variance / self.expiry)

    def arrays(self):
        """Columns as NumPy arrays, for fitting and plotting."""
        names = (
            "log_moneyness",
            "strike",
            "implied_vol",
            "total_variance",
            "vega",
            "spread_vol",
            "weight",
        )
        return {name: np.array([getattr(p, name) for p in self.points]) for name in names}

    def __repr__(self) -> str:
        return (
            f"Slice(T={self.expiry:.4f}y, F={self.forward:.2f}, "
            f"df={self.discount:.6f}, n={self.n}, atm={self.atm_vol:.2%})"
        )


@dataclass
class Chain:
    """A board of expiries with a common as-of time."""

    slices: list
    spot: float
    asof: object = None
    source: str = ""
    rejected: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.slices)

    def __iter__(self):
        return iter(self.slices)

    def __getitem__(self, i):
        return self.slices[i]

    @property
    def expiries(self):
        return [s.expiry for s in self.slices]

    @property
    def theta(self):
        """ATM total variance per expiry -- what SSVI hangs its term structure on."""
        return [s.atm_total_variance for s in self.slices]

    @property
    def forwards(self):
        return [s.forward for s in self.slices]

    @property
    def discounts(self):
        return [s.discount for s in self.slices]

    @property
    def total_points(self) -> int:
        return sum(s.n for s in self.slices)

    @classmethod
    def build(
        cls,
        quotes_by_expiry,
        spot,
        *,
        filters=None,
        asof=None,
        source="",
        moneyness_window=0.10,
        min_points=5,
    ):
        """Clean a board and reduce it to fittable points.

        `quotes_by_expiry` maps an expiry in years to a sequence of RawQuote.

        THE FORWARD IS FITTED PER EXPIRY, not derived from a rate curve and a
        dividend assumption. An equity option board prices off a forward the
        market agrees on, and that forward embeds a borrow cost and a dividend
        forecast that no published curve knows. Assume them instead and the
        error goes straight into the log-moneyness, which tilts the entire
        smile: an ATM vol a fraction of a point off, and a skew that is wrong in
        a way no amount of fitting can absorb, because the fitter is being shown
        the wrong x-coordinates.

        Expiries that cannot produce a usable regression, or that keep too few
        points, are dropped and NAMED in `rejected` rather than silently
        included with a bad forward.
        """
        filters = filters if filters is not None else _vse.FilterConfig()
        built, rejected = [], []
        for expiry in sorted(quotes_by_expiry):
            quotes = list(quotes_by_expiry[expiry])
            if expiry <= 0.0:
                rejected.append((expiry, "expired"))
                continue
            calls, puts = _vse.split_by_type(quotes)
            fit = _vse.implied_forward_from_parity(calls, puts, expiry, spot, moneyness_window)
            if not fit.ok:
                rejected.append((expiry, (f"parity regression failed on {fit.pairs_used} pairs")))
                continue
            points, report = _vse.build_slice(quotes, fit.forward, expiry, fit.discount, filters)
            if len(points) < min_points:
                rejected.append(
                    (expiry, (f"{len(points)} points survived filtering of {report.input_quotes}"))
                )
                continue
            built.append(
                Slice(
                    expiry=expiry,
                    forward=fit.forward,
                    discount=fit.discount,
                    points=points,
                    fit=fit,
                    report=report,
                    label=f"{expiry * _YEAR:.0f}d",
                )
            )
        return cls(slices=built, spot=spot, asof=asof, source=source, rejected=rejected)

    @classmethod
    def from_frame(cls, df, spot, *, asof=None, **kwargs):
        """Build from a DataFrame with one row per quoted option.

        Required columns after alias resolution: strike, bid, ask, type, and
        expiry either as a year fraction or as a date. volume and open_interest
        are used when present.
        """
        renamed = {c: _COLUMN_ALIASES.get(str(c).lower().replace(" ", ""), c) for c in df.columns}
        df = df.rename(columns=renamed)
        missing = {"strike", "bid", "ask", "type", "expiry"} - set(df.columns)
        if missing:
            raise ValueError(
                f"chain is missing columns {sorted(missing)}; "
                f"present: {sorted(map(str, df.columns))}"
            )

        column = df["expiry"]
        if np.issubdtype(column.dtype, np.number):
            years = column.to_numpy(dtype=float)
        else:
            years = np.array([year_fraction(e, asof) for e in column])

        by_expiry = {}
        for row, T in zip(df.itertuples(index=False), years):
            q = _vse.RawQuote()
            q.strike = float(row.strike)
            q.bid = float(row.bid)
            q.ask = float(row.ask)
            q.volume = float(getattr(row, "volume", 0.0) or 0.0)
            q.open_interest = float(getattr(row, "open_interest", 0.0) or 0.0)
            q.type = parse_type(row.type)
            by_expiry.setdefault(round(float(T), 10), []).append(q)
        return cls.build(by_expiry, spot, asof=asof, **kwargs)

    @classmethod
    def from_csv(cls, path, spot=None, **kwargs):
        import pandas as pd

        df = pd.read_csv(path)
        if spot is None:
            if "spot" not in df.columns:
                raise ValueError("pass spot=, or include a 'spot' column")
            spot = float(df["spot"].iloc[0])
        asof = kwargs.pop("asof", None)
        if asof is None and "asof" in df.columns:
            asof = _dt.datetime.fromisoformat(str(df["asof"].iloc[0]))
        return cls.from_frame(df, spot, asof=asof, source=str(Path(path).name), **kwargs)

    @classmethod
    def synthetic(cls, expiries=None, config=None, **kwargs):
        """A MANUFACTURED board with a known ground truth.

        Returns (chain, truth). The truth carries the eSSVI surface the quotes
        were generated from, so a calibration can be scored against the thing it
        is trying to recover rather than against itself. This is not market data
        and nothing derived from it should be presented as though it were.
        """
        if expiries is None:
            expiries = [7 / 365, 30 / 365, 60 / 365, 91 / 365, 182 / 365, 1.0, 2.0]
        config = config if config is not None else _vse.SyntheticChainConfig()
        generated = _vse.generate_synthetic_chain(list(expiries), config)
        by_expiry = {e.expiry: list(e.quotes) for e in generated.expiries}
        chain = cls.build(by_expiry, config.spot, source="synthetic", **kwargs)
        return chain, generated.truth

    def summary(self) -> str:
        head = (
            f"chain {self.source or 'unnamed'}: spot {self.spot:.2f}, "
            f"{len(self.slices)} expiries, {self.total_points} points"
        )
        lines = [
            head,
            (
                f"  {'T':>8} {'days':>5} {'forward':>10} {'discount':>9} "
                f"{'rate':>7} {'pts':>5} {'atm':>7} {'kept/in':>10} {'rms':>8}"
            ),
        ]
        for s in self.slices:
            r = s.report
            lines.append(
                f"  {s.expiry:8.4f} {s.expiry * _YEAR:5.0f} {s.forward:10.2f} "
                f"{s.discount:9.6f} {s.fit.implied_rate:6.2%} {s.n:5d} "
                f"{s.atm_vol:7.2%} {r.kept:5d}/{r.input_quotes:<4d} "
                f"{s.fit.residual_rms:8.2e}"
            )
        for expiry, why in self.rejected:
            lines.append(f"  dropped {expiry * _YEAR:.0f}d: {why}")
        return "\n".join(lines)


def parse_type(value):
    """Read a call/put marker in whatever spelling the vendor used."""
    if isinstance(value, _vse.OptionType):
        return value
    text = str(value).strip().lower()
    if text.startswith("c"):
        return _vse.OptionType.Call
    if text.startswith("p"):
        return _vse.OptionType.Put
    raise ValueError(f"cannot read {value!r} as an option type")


def quotes_from_arrays(strikes, bids, asks, types, volumes=None, open_interest=None):
    """Assemble RawQuotes without going through pandas."""
    strikes = np.asarray(list(strikes), dtype=float)
    n = len(strikes)
    volumes = np.zeros(n) if volumes is None else np.asarray(list(volumes), float)
    open_interest = np.zeros(n) if open_interest is None else np.asarray(list(open_interest), float)
    out = []
    for k, b, a, t, v, oi in zip(strikes, bids, asks, types, volumes, open_interest):
        q = _vse.RawQuote()
        q.strike, q.bid, q.ask = float(k), float(b), float(a)
        q.volume, q.open_interest = float(v), float(oi)
        q.type = parse_type(t)
        out.append(q)
    return out
