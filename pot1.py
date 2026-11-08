"""Pot 1 - Cover constant-mix sleeve tracker (the income pot).

Paper valuation of the tested sleeve: 50% Nifty, 1/6 gold (INR), 1/6 USDINR,
1/6 oil. Monthly rebalance back to target weights on the first trading day of
each month. NAV starts at 1.0 on start_date.

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

# Tested in research/cover_test.py: ~+12.8-13.1%/yr, maxDD ~-8%.
TARGET_WEIGHTS = {"^NSEI": 0.5, "GOLDBEES.NS": 1 / 6.0, "INR=X": 1 / 6.0, "CL=F": 1 / 6.0}


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
    out = {}
    data = yf.download(list(TARGET_WEIGHTS), period="10d", interval="1d",
                       progress=False, auto_adjust=False)
    close = data["Close"]
    for leg in TARGET_WEIGHTS:
        s = close[leg].dropna() if leg in close else pd.Series(dtype=float)
        if len(s):
            out[leg] = float(s.iloc[-1])
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
