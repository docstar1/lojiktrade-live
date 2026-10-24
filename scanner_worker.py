"""Standalone live forward scanner.

Imports only the standard library, numpy, pandas and yfinance so it runs in
<15s locally and in GitHub Actions.
"""
import os
import sys
import json
import math
import time
import hashlib
import pathlib
import datetime
import concurrent.futures

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parent
LEDGER_PATH = ROOT / "ledger.json"
STATUS_MD = ROOT / "LIVE_FORWARD_STATUS.md"
DASHBOARD_HTML = None

WATCHLIST = {
    "IndiaNSE": {
        "symbols": ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "INFY.NS",
                    "TCS.NS", "BAJFINANCE.NS", "LT.NS", "M&M.NS", "HAL.NS", "BEL.NS",
                    "TRENT.NS", "MARUTI.NS", "PNB.NS"],
        "tfs": ["1h", "1d"],
        "cost": 0.0011,
        "slip": 0.0005,
    },
    "US": {
        "symbols": ["NVDA", "TSLA", "AMD", "META", "AMZN", "AVGO", "AMAT", "PLTR",
                    "COIN", "QQQ", "SPY", "XLE"],
        "tfs": ["15m", "1h", "1d"],
        "cost": 0.0004,
        "slip": 0.0005,
    },
    "Crypto": {
        "symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "LTC-USD",
                    "NEAR-USD", "TRX-USD"],
        "tfs": ["15m", "1h", "1d"],
        "cost": 0.0014,
        "slip": 0.0005,
    },
    "FXCommod": {
        "symbols": ["GC=F", "SI=F", "CL=F", "EURUSD=X", "GBPUSD=X", "USDJPY=X"],
        "tfs": ["1h", "1d"],
        "cost": 0.00045,
        "slip": 0.0005,
    },
}

TF_PERIOD = {"15m": "15d", "1h": "60d", "1d": "2y"}
TF_SEC = {"15m": 900, "1h": 3600, "1d": 86400}


def market_info(market):
    return WATCHLIST.get(market, {})


def get_costs(market):
    info = market_info(market)
    return float(info.get("cost", 0.0)), float(info.get("slip", 0.0))


def fetch_one(symbol, tf):
    period = TF_PERIOD[tf]
    try:
        df = yf.download(symbol, period=period, interval=tf, progress=False, auto_adjust=True)
    except Exception:
        return symbol, tf, None
    if df is None or (hasattr(df, "empty") and df.empty):
        return symbol, tf, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    # normalize time column
    tcol = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("datetime", "date") or "date" in cl or "time" in cl:
            tcol = c
            break
    if tcol is None:
        tcol = df.columns[0]
    if tcol != "Time":
        df = df.rename(columns={tcol: "Time"})
    # ensure standard columns
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            # try lowercase
            for c in df.columns:
                if str(c).lower() == col.lower():
                    df = df.rename(columns={c: col})
    df = df.dropna(subset=["Close"])
    return symbol, tf, df


def compute_indicators(df):
    df = df.copy()
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    o = df["Open"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0, index=df.index)

    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA30"] = c.ewm(span=30, adjust=False).mean()
    df["SMA200"] = c.rolling(200, min_periods=50).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=5).mean()
    vm = v.rolling(20, min_periods=5).mean()
    vs = v.rolling(20, min_periods=5).std(ddof=0)
    df["Z_V"] = (v - vm) / vs.replace(0, np.nan)
    df["H20"] = h.rolling(20, min_periods=5).max().shift(1)
    df["L20"] = l.rolling(20, min_periods=5).min().shift(1)
    df["H10"] = h.rolling(10, min_periods=3).max().shift(1)
    df["L10"] = l.rolling(10, min_periods=3).min().shift(1)
    df["_O"] = o
    return df


def time_col():
    return "Time"


def build_signal(symbol, market, tf, arch, direction, placed_bar_time, limit, stop, target, score):
    try:
        limit = float(limit); stop = float(stop); target = float(target)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(limit) or not np.isfinite(stop) or not np.isfinite(target):
        return None
    R = abs(limit - stop)
    if R <= 0:
        return None
    delta = R / abs(limit)
    if delta < 0.012:
        return None
    if score < 2.4:
        return None
    oid = f"{symbol}|{tf}|{arch}|{direction}|{placed_bar_time}"
    seal = hashlib.sha256(
        f"{symbol}|{tf}|{arch}|{direction}|{placed_bar_time}|{limit:.6f}|{stop:.6f}|{target:.6f}".encode()
    ).hexdigest()
    return {
        "order_id": oid,
        "symbol": symbol,
        "market": market,
        "tf": tf,
        "arch": arch,
        "direction": direction,
        "placed_bar_time": str(placed_bar_time),
        "limit_price": limit,
        "stop_price": stop,
        "target_price": target,
        "score": float(score),
        "delta": float(delta),
        "seal_sha256": seal,
    }


def run_detectors(symbol, market, tf, df):
    out = []
    n = len(df)
    start = max(0, n - 5)
    is_equity = market in ("IndiaNSE", "US")
    for i in range(start, n):
        row = df.iloc[i]
        try:
            close = float(row["Close"]); high = float(row["High"]); low = float(row["Low"]); openp = float(row["Open"])
            ema9 = float(row["EMA9"]); ema30 = float(row["EMA30"]); sma200 = float(row["SMA200"])
            atr = float(row["ATR"]); zv = float(row["Z_V"])
        except Exception:
            continue
        if not np.isfinite(atr) or atr <= 0:
            continue
        if not np.isfinite(ema30) or not np.isfinite(sma200):
            continue
        ts = str(row["Time"])
        h20 = row.get("H20", np.nan); l20 = row.get("L20", np.nan)
        h10 = row.get("H10", np.nan); l10 = row.get("L10", np.nan)
        try:
            h20 = float(h20); l20 = float(l20); h10 = float(h10); l10 = float(l10)
        except (TypeError, ValueError):
            h20 = l20 = h10 = l10 = np.nan

        # MA7
        if ema9 > ema30 > sma200 and close > sma200:
            limit = ema30
            stop = ema30 - 1.5 * atr
            R = limit - stop
            target = limit + 3.2 * R
            s = build_signal(symbol, market, tf, "MA7", "long", ts, limit, stop, target, 2.6)
            if s: out.append(s)
        elif ema9 < ema30 < sma200 and close < sma200:
            limit = ema30
            stop = ema30 + 1.5 * atr
            R = stop - limit
            target = limit - 3.2 * R
            s = build_signal(symbol, market, tf, "MA7", "short", ts, limit, stop, target, 2.6)
            if s: out.append(s)

        # F3 (1h/1d only)
        if tf in ("1h", "1d"):
            if np.isfinite(h20) and close > h20 and ema9 > ema30 > sma200:
                limit = h20
                ten_low = l10 if np.isfinite(l10) else limit - 1.6 * atr
                stop = max(ten_low, limit - 1.6 * atr)
                R = limit - stop
                if R > 0:
                    target = limit + 3.5 * R
                    s = build_signal(symbol, market, tf, "F3", "long", ts, limit, stop, target, 2.7)
                    if s: out.append(s)
            if np.isfinite(l20) and close < l20 and ema9 < ema30 < sma200:
                limit = l20
                ten_high = h10 if np.isfinite(h10) else limit + 1.6 * atr
                stop = min(ten_high, limit + 1.6 * atr)
                R = stop - limit
                if R > 0:
                    target = limit - 3.5 * R
                    s = build_signal(symbol, market, tf, "F3", "short", ts, limit, stop, target, 2.7)
                    if s: out.append(s)

        # F2 climax exhaustion
        rng = high - low
        if rng > 0:
            stretch_down = (ema30 - close) / atr
            if stretch_down > 2.6 and (not np.isnan(zv)) and zv > 2.2:
                lower_wick = min(openp, close) - low
                if lower_wick / rng >= 0.25:
                    limit = close
                    stop = low - 0.25 * atr
                    R = limit - stop
                    if R > 0:
                        d = ema30 - limit
                        d_clipped = min(max(d, 2.5 * R), 4.5 * R)
                        target = limit + d_clipped
                        s = build_signal(symbol, market, tf, "F2", "long", ts, limit, stop, target, 2.8)
                        if s: out.append(s)
            stretch_up = (close - ema30) / atr
            if stretch_up > 2.6 and (not np.isnan(zv)) and zv > 2.2:
                upper_wick = high - max(openp, close)
                if upper_wick / rng >= 0.25:
                    limit = close
                    stop = high + 0.25 * atr
                    R = stop - limit
                    if R > 0:
                        d = limit - ema30
                        d_clipped = min(max(d, 2.5 * R), 4.5 * R)
                        target = limit - d_clipped
                        s = build_signal(symbol, market, tf, "F2", "short", ts, limit, stop, target, 2.8)
                        if s: out.append(s)

        # P3_v2
        if np.isfinite(l20):
            sweep = (l20 - low) / atr
            if 0.1 <= sweep <= 1.2 and close > l20 and (not np.isnan(zv)) and zv > 1.2:
                limit = l20
                stop = low - 0.20 * atr
                R = limit - stop
                if R > 0:
                    target = limit + 3.5 * R
                    s = build_signal(symbol, market, tf, "P3_v2", "long", ts, limit, stop, target, 2.9)
                    if s: out.append(s)
        if np.isfinite(h20):
            sweep = (high - h20) / atr
            if 0.1 <= sweep <= 1.2 and close < h20 and (not np.isnan(zv)) and zv > 1.2:
                if is_equity and close > sma200:
                    pass
                else:
                    limit = h20
                    stop = high + 0.20 * atr
                    R = stop - limit
                    if R > 0:
                        target = limit - 3.5 * R
                        s = build_signal(symbol, market, tf, "P3_v2", "short", ts, limit, stop, target, 2.9)
                        if s: out.append(s)

        # E1 impulse
        if rng >= 1.6 * atr and (not np.isnan(zv)) and zv > 1.6:
            if close > openp:
                limit = (high + low) / 2.0
                stop = low - 0.15 * atr
                R = limit - stop
                if R > 0:
                    target = limit + 3.2 * R
                    s = build_signal(symbol, market, tf, "E1", "long", ts, limit, stop, target, 2.5)
                    if s: out.append(s)
            else:
                limit = (high + low) / 2.0
                stop = high + 0.15 * atr
                R = stop - limit
                if R > 0:
                    target = limit - 3.2 * R
                    s = build_signal(symbol, market, tf, "E1", "short", ts, limit, stop, target, 2.5)
                    if s: out.append(s)

    return out


def load_ledger():
    if LEDGER_PATH.exists():
        try:
            return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "account": {"equity": 100.0, "risk_pct": 0.05, "target": 1000.0},
        "open_positions": [],
        "pending_orders": [],
        "closed_trades": [],
        "block_stats": {"consecutive_losses": 0, "blocked": False},
        "last_run": None,
    }


def bars_after(df, time_str):
    try:
        mask = df["Time"].astype(str) > str(time_str)
        return df[mask]
    except Exception:
        return df.iloc[0:0]


def try_fill(order, df):
    tf_sec = TF_SEC[order["tf"]]
    bars = bars_after(df, order["placed_bar_time"])
    for _, bar in bars.iterrows():
        H = float(bar["High"]); L = float(bar["Low"])
        buf = 0.15 * (H - L) * math.sqrt(2.0 / tf_sec)
        if order["direction"] == "long":
            if L <= order["limit_price"] - buf:
                return True, str(bar["Time"]), float(order["limit_price"])
        else:
            if H >= order["limit_price"] + buf:
                return True, str(bar["Time"]), float(order["limit_price"])
    return False, None, None


def is_rejected(order_id, entry_time):
    h = int(hashlib.sha256((order_id + str(entry_time)).encode()).hexdigest(), 16) / (2 ** 256)
    return h < 0.02


def step_a_update_positions(ledger, data):
    open_positions = ledger["open_positions"]
    for pos in list(open_positions):
        df = data.get((pos["symbol"], pos["tf"]))
        if df is None:
            # mark to market with nothing; skip
            continue
        bars = bars_after(df, pos["entry_time"])
        exited = False
        cost_pct, slip_pct = get_costs(pos["market"])
        for _, bar in bars.iterrows():
            H = float(bar["High"]); L = float(bar["Low"]); C = float(bar["Close"])
            if pos["direction"] == "long":
                risk_price = pos["entry_price"] - pos["initial_stop"]
                if risk_price <= 0:
                    break
                unreal_r = (C - pos["entry_price"]) / risk_price
                if unreal_r >= 1.2:
                    new_stop = pos["entry_price"] + 0.25 * risk_price
                    if new_stop > pos["stop_price"]:
                        pos["stop_price"] = new_stop
                if L <= pos["stop_price"]:
                    exit_price = pos["stop_price"]; reason = "stop"; exited = True
                elif H >= pos["target_price"]:
                    exit_price = pos["target_price"]; reason = "target"; exited = True
            else:
                risk_price = pos["initial_stop"] - pos["entry_price"]
                if risk_price <= 0:
                    break
                unreal_r = (pos["entry_price"] - C) / risk_price
                if unreal_r >= 1.2:
                    new_stop = pos["entry_price"] - 0.25 * risk_price
                    if new_stop < pos["stop_price"]:
                        pos["stop_price"] = new_stop
                if H >= pos["stop_price"]:
                    exit_price = pos["stop_price"]; reason = "stop"; exited = True
                elif L <= pos["target_price"]:
                    exit_price = pos["target_price"]; reason = "target"; exited = True

            if exited:
                if pos["direction"] == "long":
                    pnl_r = (exit_price - pos["entry_price"]) / risk_price
                else:
                    pnl_r = (pos["entry_price"] - exit_price) / risk_price
                # deduct costs
                extra_slip = slip_pct if reason == "stop" else 0.0
                cost_r = (cost_pct + extra_slip) * pos["entry_price"] / risk_price
                pnl_r -= cost_r
                ledger["account"]["equity"] += pnl_r * pos["risk_capital"]
                bs = ledger["block_stats"]
                if pnl_r > 0:
                    bs["consecutive_losses"] = 0
                    bs["blocked"] = False
                else:
                    bs["consecutive_losses"] = bs.get("consecutive_losses", 0) + 1
                    if bs["consecutive_losses"] >= 4:
                        bs["blocked"] = True
                ledger["closed_trades"].append({
                    "order_id": pos["order_id"],
                    "symbol": pos["symbol"],
                    "market": pos["market"],
                    "tf": pos["tf"],
                    "arch": pos["arch"],
                    "direction": pos["direction"],
                    "entry_time": pos["entry_time"],
                    "exit_time": str(bar["Time"]),
                    "entry_price": pos["entry_price"],
                    "exit_price": float(exit_price),
                    "pnl_r": round(float(pnl_r), 4),
                    "reason": reason,
                })
                open_positions.remove(pos)
                break

        if exited:
            continue
        # mark to market
        if len(bars) > 0:
            last = bars.iloc[-1]
        else:
            last = df.iloc[-1]
        C = float(last["Close"])
        if pos["direction"] == "long":
            risk_price = pos["entry_price"] - pos["initial_stop"]
            if risk_price > 0:
                pos["unrealized_r"] = (C - pos["entry_price"]) / risk_price
        else:
            risk_price = pos["initial_stop"] - pos["entry_price"]
            if risk_price > 0:
                pos["unrealized_r"] = (pos["entry_price"] - C) / risk_price


def step_b_pending(ledger, data, open_positions):
    still_pending = []
    for order in ledger["pending_orders"]:
        df = data.get((order["symbol"], order["tf"]))
        if df is None:
            still_pending.append(order)
            continue
        bars = bars_after(df, order["placed_bar_time"])
        # expire older than 10 bars
        if len(bars) > 10:
            continue
        filled, entry_time, entry_price = try_fill(order, df)
        if not filled:
            still_pending.append(order)
            continue
        # rejection check
        if is_rejected(order["order_id"], entry_time):
            continue
        # market space
        market_positions = [p for p in open_positions if p["market"] == order["market"]]
        if len(market_positions) >= 1:
            still_pending.append(order)
            continue
        if ledger["block_stats"].get("blocked"):
            still_pending.append(order)
            continue
        risk_price = abs(entry_price - order["stop_price"])
        if risk_price <= 0:
            still_pending.append(order)
            continue
        equity = ledger["account"]["equity"]
        pos = {
            "order_id": order["order_id"],
            "symbol": order["symbol"],
            "market": order["market"],
            "tf": order["tf"],
            "arch": order["arch"],
            "direction": order["direction"],
            "entry_time": entry_time,
            "entry_price": float(entry_price),
            "stop_price": float(order["stop_price"]),
            "initial_stop": float(order["stop_price"]),
            "target_price": float(order["target_price"]),
            "risk_capital": equity * ledger["account"]["risk_pct"],
            "unrealized_r": 0.0,
            "seal_sha256": order.get("seal_sha256"),
        }
        open_positions.append(pos)
    ledger["pending_orders"] = still_pending


def main():
    t0 = time.time()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "dashboard").mkdir(parents=True, exist_ok=True)

    # fetch data
    data = {}
    tasks = []
    for market, info in WATCHLIST.items():
        for symbol in info["symbols"]:
            for tf in info["tfs"]:
                tasks.append((symbol, tf))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, s, t): (s, t) for s, t in tasks}
        for f in concurrent.futures.as_completed(futs):
            s, t, df = f.result()
            if df is not None:
                try:
                    data[(s, t)] = compute_indicators(df)
                except Exception:
                    pass

    ledger = load_ledger()
    open_positions = ledger["open_positions"]
    # ensure list
    if not isinstance(open_positions, list):
        open_positions = []
        ledger["open_positions"] = open_positions

    # Step A
    step_a_update_positions(ledger, data)

    # Step B
    step_b_pending(ledger, data, open_positions)

    # Step A (re-run after Step B fills: check stops/targets/ratchet on all bars after each fill bar)
    step_a_update_positions(ledger, data)

    # Step C: scan signals
    existing_ids = set()
    for o in ledger["pending_orders"]:
        existing_ids.add(o["order_id"])
    for p in open_positions:
        existing_ids.add(p["order_id"])
    for c in ledger["closed_trades"]:
        existing_ids.add(c["order_id"])

    all_signals = []
    for market, info in WATCHLIST.items():
        for symbol in info["symbols"]:
            for tf in info["tfs"]:
                df = data.get((symbol, tf))
                if df is None:
                    continue
                sigs = run_detectors(symbol, market, tf, df)
                all_signals.extend(sigs)

    # dedupe by order_id preserving first
    seen = set()
    uniq = []
    for s in all_signals:
        if s["order_id"] in seen:
            continue
        seen.add(s["order_id"])
        uniq.append(s)

    blocked = ledger["block_stats"].get("blocked", False)
    for sig in uniq:
        if sig["order_id"] in existing_ids:
            continue
        if blocked:
            continue
        df = data.get((sig["symbol"], sig["tf"]))
        if df is None:
            continue
        market_positions = [p for p in open_positions if p["market"] == sig["market"]]
        # try immediate fill
        filled, entry_time, entry_price = try_fill(sig, df)
        if filled and not is_rejected(sig["order_id"], entry_time) and len(market_positions) < 1:
            risk_price = abs(entry_price - sig["stop_price"])
            if risk_price > 0:
                equity = ledger["account"]["equity"]
                pos = {
                    "order_id": sig["order_id"],
                    "symbol": sig["symbol"],
                    "market": sig["market"],
                    "tf": sig["tf"],
                    "arch": sig["arch"],
                    "direction": sig["direction"],
                    "entry_time": entry_time,
                    "entry_price": float(entry_price),
                    "stop_price": float(sig["stop_price"]),
                    "initial_stop": float(sig["stop_price"]),
                    "target_price": float(sig["target_price"]),
                    "risk_capital": equity * ledger["account"]["risk_pct"],
                    "unrealized_r": 0.0,
                    "seal_sha256": sig["seal_sha256"],
                }
                open_positions.append(pos)
                existing_ids.add(sig["order_id"])
                continue
        # else pending
        ledger["pending_orders"].append(sig)
        existing_ids.add(sig["order_id"])

    # Step A (re-run after Step C fills: check stops/targets/ratchet on all bars after each fill bar)
    step_a_update_positions(ledger, data)

    # mark-to-market unrealized R for all open positions
    for pos in open_positions:
        df = data.get((pos["symbol"], pos["tf"]))
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        C = float(last["Close"])
        if pos["direction"] == "long":
            risk_price = pos["entry_price"] - pos["initial_stop"]
            if risk_price > 0:
                pos["unrealized_r"] = (C - pos["entry_price"]) / risk_price
        else:
            risk_price = pos["initial_stop"] - pos["entry_price"]
            if risk_price > 0:
                pos["unrealized_r"] = (pos["entry_price"] - C) / risk_price
    ledger["open_positions"] = open_positions
    ledger["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # save ledger
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")

    # write markdown + html
    write_status(ledger)
    print(f"done in {time.time()-t0:.1f}s | open={len(open_positions)} pending={len(ledger['pending_orders'])} closed={len(ledger['closed_trades'])}")


def write_status(ledger):
    acct = ledger["account"]
    lines = []
    lines.append("# Live Forward Scanner Status")
    lines.append("")
    lines.append(f"Last run: {ledger['last_run']}")
    lines.append("")
    lines.append("## Account")
    lines.append(f"- Equity: ${acct['equity']:.2f} (target ${acct['target']:.2f})")
    lines.append(f"- Risk %: {acct['risk_pct']*100:.1f}%")
    lines.append(f"- Open positions: {len(ledger['open_positions'])}")
    lines.append(f"- Pending orders: {len(ledger['pending_orders'])}")
    lines.append(f"- Closed trades: {len(ledger['closed_trades'])}")
    lines.append(f"- Block stats: {ledger['block_stats']}")
    lines.append("")
    lines.append("## Open Positions")
    if ledger["open_positions"]:
        lines.append("| Symbol | Market | TF | Arch | Dir | Entry | Stop | Target | Unreal R |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for p in ledger["open_positions"]:
            lines.append(f"| {p['symbol']} | {p['market']} | {p['tf']} | {p['arch']} | {p['direction']} | "
                         f"{p['entry_price']:.4f} | {p['stop_price']:.4f} | {p['target_price']:.4f} | "
                         f"{p.get('unrealized_r',0):.2f} |")
    else:
        lines.append("_None_")
    lines.append("")
    lines.append("## Pending Orders")
    if ledger["pending_orders"]:
        lines.append("| Symbol | Market | TF | Arch | Dir | Limit | Stop | Target | Score |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for o in ledger["pending_orders"]:
            lines.append(f"| {o['symbol']} | {o['market']} | {o['tf']} | {o['arch']} | {o['direction']} | "
                         f"{o['limit_price']:.4f} | {o['stop_price']:.4f} | {o['target_price']:.4f} | "
                         f"{o['score']:.2f} |")
    else:
        lines.append("_None_")
    lines.append("")
    lines.append("## Recent Closed Trades")
    recent = ledger["closed_trades"][-15:]
    if recent:
        lines.append("| Symbol | Market | TF | Arch | Dir | PnL R | Reason |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in recent:
            lines.append(f"| {c['symbol']} | {c['market']} | {c['tf']} | {c['arch']} | {c['direction']} | "
                         f"{c['pnl_r']} | {c['reason']} |")
    else:
        lines.append("_None_")
    lines.append("")
    STATUS_MD.write_text("\n".join(lines), encoding="utf-8")

    # html
    html = ["<html><head><title>Live Forward Scanner</title></head><body>"]
    html.append(f"<h1>Live Forward Scanner</h1><p>Last run: {ledger['last_run']}</p>")
    html.append(f"<p>Equity: ${acct['equity']:.2f} | Open: {len(ledger['open_positions'])} | "
                f"Pending: {len(ledger['pending_orders'])} | Closed: {len(ledger['closed_trades'])}</p>")
    html.append("<h2>Open Positions</h2><table border='1'><tr><th>Symbol</th><th>Market</th><th>TF</th><th>Arch</th><th>Dir</th><th>Entry</th><th>Stop</th><th>Target</th><th>Unreal R</th></tr>")
    for p in ledger["open_positions"]:
        html.append(f"<tr><td>{p['symbol']}</td><td>{p['market']}</td><td>{p['tf']}</td><td>{p['arch']}</td>"
                    f"<td>{p['direction']}</td><td>{p['entry_price']:.4f}</td><td>{p['stop_price']:.4f}</td>"
                    f"<td>{p['target_price']:.4f}</td><td>{p.get('unrealized_r',0):.2f}</td></tr>")
    html.append("</table>")
    html.append("<h2>Pending Orders</h2><table border='1'><tr><th>Symbol</th><th>Market</th><th>TF</th><th>Arch</th><th>Dir</th><th>Limit</th><th>Stop</th><th>Target</th><th>Score</th></tr>")
    for o in ledger["pending_orders"]:
        html.append(f"<tr><td>{o['symbol']}</td><td>{o['market']}</td><td>{o['tf']}</td><td>{o['arch']}</td>"
                    f"<td>{o['direction']}</td><td>{o['limit_price']:.4f}</td><td>{o['stop_price']:.4f}</td>"
                    f"<td>{o['target_price']:.4f}</td><td>{o['score']:.2f}</td></tr>")
    html.append("</table>")
    html.append("<h2>Recent Closed Trades</h2><table border='1'><tr><th>Symbol</th><th>Market</th><th>TF</th><th>Arch</th><th>Dir</th><th>PnL R</th><th>Reason</th></tr>")
    for c in ledger["closed_trades"][-15:]:
        html.append(f"<tr><td>{c['symbol']}</td><td>{c['market']}</td><td>{c['tf']}</td><td>{c['arch']}</td>"
                    f"<td>{c['direction']}</td><td>{c['pnl_r']}</td><td>{c['reason']}</td></tr>")
    html.append("</table></body></html>")
    if DASHBOARD_HTML:
        DASHBOARD_HTML.write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
