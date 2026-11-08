"""Pot 1 - income pot sleeve tracker (the draw-paying pot).

Fully-buyable 3-leg sleeve, tested in lojiktrade research/sleeve3.py on the
same 5y window as the old 4-leg (beats it: crude was a net drag): 50% Nifty
ETF + 25% gold ETF + 25% S&P 500 held in rupees via an Indian index
fund-of-fund (leg valued as ^GSPC x USDINR). No USD leg, no oil. Monthly
rebalance back to target weights on the first trading day of each month.
NAV starts at 1.0 on start_date.

Pot sizes are deliberately kept OUT of this public repo (see
LIVE_FORWARD_STATUS.md). The Nifty leg is valued on the index (^NSEI) exactly
as in the research backtest; the tradable ETF (NIFTYBEES) tracks it closely.

State lives in ledger.json under "pot1". The sleeve_update() function is pure
so tests/test_pot1.py can exercise it without network.
"""
import json
from datetime import date

import pandas as pd
import yfinance as yf

LEDGER_PATH = "ledger.json"

# Tested in research/sleeve3.py: +10.9%/yr, maxDD -13.1% (beats the old 4-leg's
# +7.6%/yr on the same window; the S&P 500 leg more than replaces crude).
YF_TICKS = ["^NSEI", "GOLDBEES.NS", "^GSPC", "INR=X"]
TARGET_WEIGHTS = {"^NSEI": 0.5, "GOLDBEES.NS": 0.25, "SPX_INR": 0.25}


def next_rebalance_date(d):
    """First day of the month after d."""
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(y, m, 1)


def sleeve_update(state, prices, today):
    """Pure state update. state: dict or None. prices: {leg: close}. today: date."""
    prices = {k: float(v) for k, v in prices.items() if v is not None and float(v) > 0}
    missing = [k for k in TARGET_WEIGHTS if k not in prices]
    if missing:
        raise ValueError("missing prices for legs: %s" % ", ".join(missing))
    today = today if isinstance(today, date) else date.fromisoformat(str(today))

    if state is None:
        units = {leg: w / prices[leg] for leg, w in TARGET_WEIGHTS.items()}
        return {
            "start_date": str(today),
            "target_weights": TARGET_WEIGHTS,
            "units": units,
            "nav": 1.0,
            "total_return_pct": 0.0,
            "last_rebalance": str(today),
            "next_rebalance": str(next_rebalance_date(today)),
            "current_weights": {leg: round(w, 4) for leg, w in TARGET_WEIGHTS.items()},
            "nav_history": [{"date": str(today), "nav": 1.0}],
        }

    units = dict(state["units"])
    nav = sum(units[leg] * prices[leg] for leg in units)
    rebalanced = False
    if today >= date.fromisoformat(state["next_rebalance"]):
        units = {leg: nav * w / prices[leg] for leg, w in TARGET_WEIGHTS.items()}
        state["last_rebalance"] = str(today)
        state["next_rebalance"] = str(next_rebalance_date(today))
        rebalanced = True
    state["units"] = units
    state["nav"] = round(nav, 6)
    state["total_return_pct"] = round((nav - 1.0) * 100, 2)
    state["current_weights"] = {leg: round(units[leg] * prices[leg] / nav, 4) for leg in units}
    state["prices"] = prices
    hist = state.get("nav_history", [])
    if not hist or hist[-1]["date"] != str(today):
        hist.append({"date": str(today), "nav": round(nav, 6)})
    state["nav_history"] = hist[-400:]
    state["last_updated"] = str(today)
    state["rebalanced_today"] = rebalanced
    return state


def fetch_prices():
    """Latest closes per leg; SPX_INR is the composite ^GSPC x USDINR leg."""
    out = {}
    try:
        data = yf.download(YF_TICKS, period="10d", interval="1d",
                           progress=False, auto_adjust=False)["Close"]
    except Exception:
        return out

    def last(tick):
        try:
            s = data[tick].dropna() if tick in getattr(data, "columns", []) else pd.Series(dtype=float)
        except Exception:
            s = pd.Series(dtype=float)
        return float(s.iloc[-1]) if len(s) else None

    nifty, gold, spx, fx = last("^NSEI"), last("GOLDBEES.NS"), last("^GSPC"), last("INR=X")
    if nifty:
        out["^NSEI"] = nifty
    if gold:
        out["GOLDBEES.NS"] = gold
    if spx and fx:
        out["SPX_INR"] = spx * fx  # S&P 500 in INR terms (Indian FOF valuation)
    return out


def main():
    ledger = json.load(open(LEDGER_PATH))
    prices = fetch_prices()
    state = sleeve_update(ledger.get("pot1"), prices, date.today())
    ledger["pot1"] = state
    json.dump(ledger, open(LEDGER_PATH, "w"), indent=2)
    print("pot1 nav %.6f (%+.2f%%) next_rebalance %s weights %s" % (
        state["nav"], state["total_return_pct"], state["next_rebalance"],
        state.get("current_weights")))


if __name__ == "__main__":
    main()
