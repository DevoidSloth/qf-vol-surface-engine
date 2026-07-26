#!/usr/bin/env python
"""Download a live option chain to data/.

    python scripts/fetch_chain.py SPY --max-expiries 12

Writes data/<ticker>_<date>.csv with one row per quoted option and the columns
the loader expects. data/ is gitignored, so this is the step that has to be
re-run rather than a file that ships.

WHAT THIS DATA IS. Free delayed quotes from Yahoo, which is fine for exercising
a pipeline and not fine for anything else. Specifically:

  * The quotes are delayed and are not from a single instant. Different strikes
    were last updated at different times, so the board is not a snapshot and the
    put-call parity regression will show it in the residual.
  * Illiquid strikes carry stale bids that never moved, which is why the
    cleaning stage drops one-sided and zero-bid quotes rather than trusting a
    mid.
  * There is no borrow or dividend data, which is a reason to fit the forward
    from parity rather than to construct it -- see chain.py.

The numbers in benchmarks/RESULTS.md are measured on synthetic chains with a
known ground truth for exactly this reason: a benchmark whose reference is
itself delayed and asynchronous measures the data, not the code.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ticker")
    parser.add_argument("--max-expiries", type=int, default=12,
                        help="nearest N expiries (default 12)")
    parser.add_argument("--min-days", type=float, default=3.0,
                        help="skip expiries closer than this; the last days of "
                             "an option's life are dominated by pin risk and "
                             "the vols are not comparable to anything")
    parser.add_argument("--max-days", type=float, default=760.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        import pandas as pd
        import yfinance as yf
    except ImportError:
        print("needs yfinance and pandas:  pip install yfinance pandas",
              file=sys.stderr)
        return 2

    ticker = yf.Ticker(args.ticker)
    try:
        spot = float(ticker.fast_info["last_price"])
    except Exception:
        history = ticker.history(period="1d")
        if history.empty:
            print(f"no price data for {args.ticker}", file=sys.stderr)
            return 1
        spot = float(history["Close"].iloc[-1])

    asof = _dt.datetime.now().replace(microsecond=0)
    frames = []
    kept = 0
    for date_string in ticker.options:
        expiry = _dt.datetime.fromisoformat(date_string).replace(hour=16)
        days = (expiry - asof).total_seconds() / 86400.0
        if days < args.min_days or days > args.max_days:
            continue
        if kept >= args.max_expiries:
            break
        try:
            board = ticker.option_chain(date_string)
        except Exception as exc:                      # a single bad expiry
            print(f"  {date_string}: {exc}", file=sys.stderr)
            continue
        for side, marker in ((board.calls, "C"), (board.puts, "P")):
            frame = side.loc[:, ["strike", "bid", "ask", "volume", "openInterest"]].copy()
            frame["type"] = marker
            frame["expiry"] = date_string
            frames.append(frame)
        kept += 1
        print(f"  {date_string}  {days:6.1f}d  "
              f"{len(board.calls):4d} calls  {len(board.puts):4d} puts")

    if not frames:
        print("no expiries in range", file=sys.stderr)
        return 1

    out = pd.concat(frames, ignore_index=True)
    out["openInterest"] = out["openInterest"].fillna(0.0)
    out["volume"] = out["volume"].fillna(0.0)
    out["spot"] = spot
    out["asof"] = asof.isoformat()
    out["ticker"] = args.ticker.upper()

    path = args.out or (REPO / "data" /
                        f"{args.ticker.lower()}_{asof:%Y%m%d}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\n{len(out)} rows from {kept} expiries, spot {spot:.2f}")
    print(f"wrote {path}")
    print("\nnote: delayed, asynchronous quotes. Fine for exercising the "
          "pipeline;\nthe benchmark numbers are measured on synthetic chains "
          "with a known truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
