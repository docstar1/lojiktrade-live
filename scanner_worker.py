"""Live scanner engine."""

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
from typing import Dict, Any, Tuple, Optional

ROOT = pathlib.Path(__file__).resolve().parent
LEDGER_PATH = ROOT / "ledger.json"
STATUS_MD = ROOT / "LIVE_FORWARD_STATUS.md"

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

MAX_OPEN_POSITIONS = 3
MAX_PENDING_GLOBAL = 8
MAX_PENDING_PER_SYMBOL = 1
COOLDOWN_BARS = 5
# 1.5% risk per trade x 3 concurrent slots = 4.5% max simultaneous risk.
MAX_RISK_PCT = 0.015
# 12-hour time-based cooldown after 3 consecutive losses (was a permanent block).
LOSS_BLOCK_HOURS = 12.0
ALLOWED_ARCHS = {"T10_SOROS", "F1_SOROS_MA", "MA7_TRUE", "P3_SWEEP", "F3_BREAKOUT"}
VALID_ARCHS = ALLOWED_ARCHS

# At most ONE open position per market bucket, so a slow IndiaNSE trade never
# blocks Crypto or US/FX/Commod trades.  Bucket -> canonical bucket name.
MARKET_BUCKETS = {
    "Crypto": "Crypto",
    "IndiaNSE": "IndiaNSE",
    "US": "US_FX",
    "FXCommod": "US_FX",
}

# Design R-multiple per setup (used for target repair of legacy pending orders).
# Every pending order must sit inside [1.8R, ARCH_TARGET_RR + 0.05R]; anything
# outside that band is a pre-upgrade (3.2R-3.8R) target and is re-derived here.
ARCH_TARGET_RR = {
    "T10_SOROS": 2.5,
    "F1_SOROS_MA": 2.2,
    "MA7_TRUE": 2.0,
    "P3_SWEEP": 2.1,
    "F3_BREAKOUT": 2.2,
}
# Backwards-compatible alias (older callers/tests use ARCH_TARGET_R).
ARCH_TARGET_R = ARCH_TARGET_RR

# Exact theoretical maximum UNBOOSTED base_score per detector.  Any stored
# base_score above this cap can only come from the legacy bug that wrote
# stripped until the value is back inside the achievable range.
ARCH_MAX_BASE_SCORE = {
    "T10_SOROS": 4.0,      # 3.6 + 0.15 * min(max_zv, 5.0) -> capped at 4.0
    "F1_SOROS_MA": 3.6,    # 3.2 + 0.1 * 4.0
    "MA7_TRUE": 2.95,      # 2.75 + 0.05 * 4.0
    "P3_SWEEP": 3.5,       # 3.1 + 0.1 * 4.0
    "F3_BREAKOUT": 3.15,   # 2.95 + 0.05 * 4.0
}
KREIL_BOOST_STEP = 0.15

# Original (pre-inflation) base_score of the orders that were live when the
# stacking bug was found, recovered from ``git log -p -- ledger.json`` at the
# step-stripping heuristic so historical repairs are exact, not approximate.
LEGACY_ORDER_BASE_SCORE = {
    "XRP-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00": 3.4345,
    "LTC-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00": 3.3689,
    "ETH-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00": 3.3993,
    "NEAR-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00": 3.2528,
    "INFY.NS|1h|F3_BREAKOUT|long|2026-10-30 15:15:00+05:30": 3.15,
    "SOL-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00": 3.4072,
    "COIN|1d|F3_BREAKOUT|short|2026-10-30 00:00:00": 2.9638,
    "USDJPY=X|1d|MA7_TRUE|short|2026-10-30 00:00:00": 2.8,
}


def market_info(market):
    return WATCHLIST.get(market, {})


def get_market_bucket(market: str) -> str:
    """Map a WATCHLIST market key to its concurrency bucket."""
    m = str(market or "")
    return MARKET_BUCKETS.get(m, m)


def bucket_positions(open_positions, market):
    bucket = get_market_bucket(market)
    return [p for p in (open_positions or []) if get_market_bucket(p.get("market", "")) == bucket]


def bucket_is_full(open_positions, market):
    return len(bucket_positions(open_positions, market)) >= 1


def can_open_position(open_positions, market):
    """True only when a global slot AND the per-bucket slot are both free."""
    open_positions = open_positions or []
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        return False
    return not bucket_is_full(open_positions, market)


def max_pending_bars(tf: str) -> int:
    """How many bars a resting order stays alive before it is expired."""
    return {"15m": 12, "1h": 8, "4h": 6, "1d": 5}.get(str(tf or ""), 6)


def profit_harvest_bars(tf: str) -> int:
    """Bars after which a small-but-positive position is banked (close & move on)."""
    return {"1d": 2, "4h": 4}.get(str(tf or ""), 6)


def dead_money_bars(tf: str) -> int:
    """Bars after which a flat position is closed so capital is not stuck."""
    return {"1d": 4, "4h": 6}.get(str(tf or ""), 8)


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
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
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
    if "Volume" in df.columns:
        v = df["Volume"].astype(float)
    else:
        v = pd.Series(0.0, index=df.index)

    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema30"] = c.ewm(span=30, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["sma200"] = c.rolling(200, min_periods=50).mean()
    df["sma200_prev5"] = df["sma200"].shift(5)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=5).mean()

    vm = v.rolling(20, min_periods=5).mean()
    vs = v.rolling(20, min_periods=5).std(ddof=0)
    df["z_v"] = (v - vm) / vs.replace(0, np.nan)
    df["vol_ratio"] = v / vm.replace(0, np.nan)

    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    df["h20"] = h.shift(1).rolling(20, min_periods=5).max()
    df["l20"] = l.shift(1).rolling(20, min_periods=5).min()
    df["h10"] = h.shift(1).rolling(10, min_periods=3).max()
    df["l10"] = l.shift(1).rolling(10, min_periods=3).min()

    df["er10"] = (c - c.shift(10)).abs() / (c.diff().abs().rolling(10).sum() + 1e-9)
    df["atr_ratio"] = df["atr"] / (df["atr"].rolling(50, min_periods=5).mean() + 1e-9)

    # convenience aliases (uppercase) for backward compat
    df["EMA9"] = df["ema9"]
    df["EMA30"] = df["ema30"]
    df["EMA50"] = df["ema50"]
    df["SMA200"] = df["sma200"]
    df["ATR"] = df["atr"]
    df["Z_V"] = df["z_v"]
    df["H20"] = df["h20"]
    df["L20"] = df["l20"]
    df["H10"] = df["h10"]
    df["L10"] = df["l10"]
    df["_O"] = o
    return df


def time_col():
    return "Time"


def build_signal(symbol, market, tf, arch, direction, placed_bar_time, limit, stop, target, score, kreil_info=None, kreil=None):
    try:
        limit = float(limit); stop = float(stop); target = float(target)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(limit) or not np.isfinite(stop) or not np.isfinite(target):
        return None
    R = abs(limit - stop)
    if R <= 0:
        return None
    delta = R / abs(limit) if abs(limit) > 0 else 0.0
    # Intraday / FX+Commod setups legitimately have tight stops; only daily
    # equity/crypto signals need the wider 0.60% structural floor.
    min_delta = 0.0060 if (str(tf) == "1d" and str(market) != "FXCommod") else 0.0035
    if delta < min_delta:
        return None
    if score < 2.4:
        return None
    rr = abs(target - limit) / R if R > 0 else 0.0
    oid = f"{symbol}|{tf}|{arch}|{direction}|{placed_bar_time}"
    seal = hashlib.sha256(
        f"{symbol}|{tf}|{arch}|{direction}|{placed_bar_time}|{limit:.6f}|{stop:.6f}|{target:.6f}".encode()
    ).hexdigest()
    _kreil = kreil if kreil is not None else kreil_info
    out = {
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
        "rr": round(float(rr), 4),
        "seal_sha256": seal,
    }
    if _kreil is not None:
        out["kreil"] = _kreil
    return out


def compute_kreil_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    tr_raw = (df["High"] - df["Low"]).abs()
    kr_atr = ((tr_raw + tr_raw.shift(1)) / 2.0) / df["Open"].replace(0, np.nan)
    # fallback on row 0
    if len(df) > 0:
        try:
            denom = df["Close"].iloc[0]
            if denom == 0 or (isinstance(denom, float) and not np.isfinite(denom)):
                denom = np.nan
            kr_atr.iloc[0] = tr_raw.iloc[0] / denom
        except Exception:
            try:
                kr_atr.iloc[0] = tr_raw.iloc[0] / df["Close"].iloc[0]
            except Exception:
                pass
    kr_atr = kr_atr.replace([np.inf, -np.inf], np.nan)
    atr_1d = float(kr_atr.iloc[-1])
    atr_5 = float(kr_atr.iloc[-5:].mean())
    atr_20 = float(kr_atr.iloc[-20:].mean())
    atr_60 = float(kr_atr.iloc[-60:].mean())
    atr_250 = float(kr_atr.iloc[-min(250, len(kr_atr)):].mean())
    atr_5_250 = atr_5 / max(atr_250, 1e-9)
    atr_1_20 = atr_1d / max(atr_20, 1e-9)
    r_oc = ((df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)).dropna()
    r_oc = r_oc.iloc[-min(250, len(df)):]
    vals = r_oc.astype(float).to_numpy()
    if len(vals) == 0:
        mean = 0.0; std = 0.0; skew = 0.0; excess_kurt = 0.0
    else:
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=0))
        if std <= 1e-12:
            skew = 0.0
            excess_kurt = 0.0
        else:
            z = (vals - mean) / std
            skew = float(np.mean(z ** 3))
            excess_kurt = float(np.mean(z ** 4) - 3.0)
    return {
        "atr_1d": atr_1d,
        "atr_5": atr_5,
        "atr_20": atr_20,
        "atr_60": atr_60,
        "atr_250": atr_250,
        "atr_5_250": atr_5_250,
        "atr_1_20": atr_1_20,
        "mean": mean,
        "std": std,
        "skew": skew,
        "excess_kurt": excess_kurt,
    }


def passes_kreil_regime_gate(df: pd.DataFrame, direction: str, arch: str = "") -> Tuple[bool, Dict[str, Any]]:
    if df is None or len(df) < 30:
        m = compute_kreil_metrics(df) if df is not None else {}
        return False, {**m, "passed": False, "reason": "insufficient_bars", "score_boost": 0.0, "direction": str(direction or "").upper(), "arch": str(arch or "")}
    m = compute_kreil_metrics(df)
    d = str(direction).upper()
    if d in ("L", "LONG"):
        d = "LONG"
    elif d in ("S", "SHORT"):
        d = "SHORT"
    elif d.startswith("L"):
        d = "LONG"
    elif d.startswith("S"):
        d = "SHORT"
    if not all(np.isfinite(m.get(k, np.nan)) for k in ("atr_5_250", "atr_1_20", "skew", "excess_kurt")):
        return False, {**m, "passed": False, "reason": "nan_metric", "score_boost": 0.0, "direction": d, "arch": str(arch or "")}
    passed = True
    reason = None
    if not (0.60 <= m["atr_5_250"] <= 1.60):
        passed = False
        reason = "atr_5_250_out_of_range"
    elif m["atr_1_20"] < 0.85:
        passed = False
        reason = "atr_1_20_below_0.85"
    elif m["excess_kurt"] < -0.5:
        passed = False
        reason = "low_kurtosis"
    elif d == "LONG" and m["skew"] < -0.75:
        passed = False
        reason = "adverse_long_skew"
    elif d == "SHORT" and m["skew"] > 0.75:
        passed = False
        reason = "adverse_short_skew"
    score_boost = 0.0
    if passed and ((d == "LONG" and m["skew"] > 0.5 and m["excess_kurt"] > 2.0) or (d == "SHORT" and m["skew"] < -0.5 and m["excess_kurt"] > 2.0)):
        score_boost = 0.15
    return passed, {**m, "passed": passed, "reason": reason, "score_boost": score_boost, "direction": d, "arch": arch}


def _row(df, i):
    r = df.iloc[i]
    def g(key):
        try:
            return float(r[key])
        except Exception:
            return float("nan")
    return {
        "close": g("Close") if "Close" in df.columns else g("close"),
    }


def _get(df, i, col):
    try:
        v = df.iloc[i][col]
        return float(v)
    except Exception:
        try:
            return float(df.iloc[i][col.lower()])
        except Exception:
            return float("nan")


def _safe(df, i, col):
    return _get(df, i, col)


def run_detectors(sym, df, tf, market=None, keep_all=False):
    if market is None:
        # infer
        if str(sym).endswith(".NS"):
            market = "IndiaNSE"
        elif str(sym).endswith("-USD"):
            market = "Crypto"
        elif "=" in str(sym) or str(sym).endswith("=X"):
            market = "FXCommod"
        else:
            market = "US"
    out = []
    n = len(df)
    if n < 5:
        return out
    is_equity = market in ("IndiaNSE", "US")
    # evaluate ONLY the latest bar
    for ii in [n - 1]:
        i = ii
        try:
            close = _safe(df, i, "Close")
            high = _safe(df, i, "High")
            low = _safe(df, i, "Low")
            openp = _safe(df, i, "Open")
            ema9 = _safe(df, i, "ema9")
            ema30 = _safe(df, i, "ema30")
            ema50 = _safe(df, i, "ema50")
            sma200 = _safe(df, i, "sma200")
            sma200_prev5 = _safe(df, i, "sma200_prev5")
            atr = _safe(df, i, "atr")
            zv = _safe(df, i, "z_v")
            vol_ratio = _safe(df, i, "vol_ratio")
            rsi = _safe(df, i, "rsi")
            h20 = _safe(df, i, "h20")
            l20 = _safe(df, i, "l20")
            h10 = _safe(df, i, "h10")
            l10 = _safe(df, i, "l10")
            er10 = _safe(df, i, "er10")
            atr_ratio = _safe(df, i, "atr_ratio")
        except Exception:
            continue
        if not np.isfinite(atr) or atr <= 0:
            continue
        if not np.isfinite(ema30) or not np.isfinite(sma200):
            continue
        ts = str(df.iloc[i]["Time"]) if "Time" in df.columns else str(df.index[i])
        # ema50 shifted 5
        ema50_prev5 = float("nan")
        if i >= 5:
            try:
                ema50_prev5 = float(df.iloc[i - 5]["ema50"])
            except Exception:
                ema50_prev5 = float("nan")

        try:
            t10_candidates = []
            # search impulse windows
            for start in range(max(0, i - 12), i - 1):
                for length in (1, 2, 3):
                    end = start + length - 1
                    if end > i - 3 or end >= i:
                        continue
                    if start < 0:
                        continue
                    try:
                        zs = [float(df.iloc[k]["z_v"]) for k in range(start, end + 1)]
                        opens = float(df.iloc[start]["Open"])
                        closes = float(df.iloc[end]["Close"])
                    except Exception:
                        continue
                    if any(not np.isfinite(z) for z in zs):
                        continue
                    if max(zs) <= 2.0:
                        continue
                    if abs(closes - opens) < 1.5 * atr:
                        continue
                    max_zv = max(zs)
                    impulse_dir = 1 if closes > opens else -1
                    # coil bars end+1 .. i
                    coil_indices = list(range(end + 1, i + 1))
                    if len(coil_indices) < 1:
                        continue
                    coil_high = max(float(df.iloc[k]["High"]) for k in coil_indices)
                    coil_low = min(float(df.iloc[k]["Low"]) for k in coil_indices)
                    impulse_range = abs(closes - opens) + 1e-9
                    retracement = abs(close - closes) / impulse_range
                    if retracement < 0.25 or retracement > 0.65:
                        continue
                    coil_vol_ratios = [float(df.iloc[k]["vol_ratio"]) for k in coil_indices]
                    if any(not np.isfinite(x) for x in coil_vol_ratios):
                        continue
                    if float(np.mean(coil_vol_ratios)) >= 0.80:
                        continue
                    coil_ranges = [float(df.iloc[k]["High"]) - float(df.iloc[k]["Low"]) for k in coil_indices]
                    if float(np.mean(coil_ranges)) > 1.2 * atr:
                        continue
                    # phase3
                    if er10 < 0.30 or atr_ratio < 0.90:
                        continue
                    if impulse_dir > 0:
                        if not (close > sma200 and ema9 >= ema30):
                            continue
                        if not (45 <= rsi <= 68):
                            continue
                    else:
                        if not (close < sma200 and ema9 <= ema30):
                            continue
                        if not (32 <= rsi <= 55):
                            continue
                    # entry/stop
                    if impulse_dir > 0:
                        entry = coil_high
                        stop = coil_low - 0.15 * atr
                    else:
                        entry = coil_low
                        stop = coil_high + 0.15 * atr
                    R = abs(entry - stop)
                    if R < 0.20 * atr or R > 1.20 * atr:
                        continue
                    if impulse_dir > 0:
                        target = entry + 2.5 * R
                        direction = "long"
                    else:
                        target = entry - 2.5 * R
                        direction = "short"
                    score = min(4.0, 3.6 + 0.15 * min(max_zv, 5.0))
                    s = build_signal(sym, market, tf, "T10_SOROS", direction, ts, entry, stop, target, score)
                    if s:
                        t10_candidates.append(s)
            if t10_candidates:
                # keep highest score
                best = max(t10_candidates, key=lambda x: x["score"])
                out.append(best)
        except Exception:
            pass

        try:
            f1_list = []
            # prior impulse 5..12 bars ending at i-1
            for length in range(5, 13):
                start = i - length
                end = i - 1
                if start < 0:
                    continue
                try:
                    opens = float(df.iloc[start]["Open"])
                    closes = float(df.iloc[end]["Close"])
                    zs = [float(df.iloc[k]["z_v"]) for k in range(start, end + 1)]
                except Exception:
                    continue
                if any(not np.isfinite(z) for z in zs):
                    continue
                if max(zs) <= 1.8:
                    continue
                if abs(closes - opens) < 1.5 * atr:
                    continue
                impulse_dir = 1 if closes > opens else -1
                impulse_range = abs(closes - opens) + 1e-9
                # current bar pullback
                pullback = abs(close - closes) / impulse_range
                if pullback < 0.45 or pullback > 0.65:
                    continue
                if vol_ratio >= 0.85 or not np.isfinite(vol_ratio):
                    continue
                # touch ema30
                if impulse_dir > 0:
                    if not (low <= ema30 + 0.25 * atr and close >= ema30):
                        continue
                else:
                    if not (high >= ema30 - 0.25 * atr and close <= ema30):
                        continue
                # trend & rsi
                if impulse_dir > 0:
                    if not (ema30 > sma200 and sma200 > sma200_prev5):
                        continue
                    if not (40 <= rsi <= 62):
                        continue
                else:
                    if not (ema30 < sma200 and sma200 < sma200_prev5):
                        continue
                    if not (38 <= rsi <= 60):
                        continue
                if impulse_dir > 0:
                    stop = ema30 - 1.2 * atr
                    entry = close
                    R = abs(entry - stop)
                    if R <= 0:
                        continue
                    target = entry + 2.2 * R
                    direction = "long"
                else:
                    stop = ema30 + 1.2 * atr
                    entry = close
                    R = abs(entry - stop)
                    if R <= 0:
                        continue
                    target = entry - 2.2 * R
                    direction = "short"
                s = build_signal(sym, market, tf, "F1_SOROS_MA", direction, ts, entry, stop, target, 3.3)
                if s:
                    f1_list.append(s)
            if f1_list:
                out.append(f1_list[0])
        except Exception:
            pass

        # --- MA7_TRUE ---
        try:
            ma7_list = []
            # long
            if ema9 > ema30 > sma200 and sma200 > sma200_prev5:
                # prior impulse within last 5 bars
                prior_ok = False
                for k in range(max(0, i - 5), i):
                    try:
                        if float(df.iloc[k]["High"]) - float(df.iloc[k]["ema30"]) >= 1.5 * atr:
                            prior_ok = True
                            break
                    except Exception:
                        pass
                if prior_ok and low <= ema30 + 0.20 * atr and close > ema30:
                    if 40 <= rsi <= 65:
                        entry = ema30
                        stop = min(sma200, ema30 - 1.5 * atr) - 0.1 * atr
                        R = abs(entry - stop)
                        if R > 1.8 * atr:
                            stop = entry - 1.8 * atr
                            R = abs(entry - stop)
                        if R > 0:
                            target = entry + 2.0 * R
                            s = build_signal(sym, market, tf, "MA7_TRUE", "long", ts, entry, stop, target, 2.8)
                            if s:
                                ma7_list.append(s)
            # short
            if ema9 < ema30 < sma200 and sma200 < sma200_prev5:
                prior_ok = False
                for k in range(max(0, i - 5), i):
                    try:
                        if float(df.iloc[k]["ema30"]) - float(df.iloc[k]["Low"]) >= 1.5 * atr:
                            prior_ok = True
                            break
                    except Exception:
                        pass
                if prior_ok and high >= ema30 - 0.20 * atr and close < ema30:
                    if 35 <= rsi <= 60:
                        entry = ema30
                        stop = max(sma200, ema30 + 1.5 * atr) + 0.1 * atr
                        R = abs(entry - stop)
                        if R > 1.8 * atr:
                            stop = entry + 1.8 * atr
                            R = abs(entry - stop)
                        if R > 0:
                            target = entry - 2.0 * R
                            s = build_signal(sym, market, tf, "MA7_TRUE", "short", ts, entry, stop, target, 2.8)
                            if s:
                                ma7_list.append(s)
            if ma7_list:
                out.append(ma7_list[0])
        except Exception:
            pass

        # --- P3_SWEEP ---
        try:
            p3_list = []
            # long
            if np.isfinite(l20):
                sweep = (l20 - low) / atr
                if 0.10 <= sweep <= 1.20 and close > l20 and (not np.isnan(zv)) and zv > 1.3:
                    if 30 <= rsi <= 58:
                        trend_gate = (close >= 0.98 * sma200) or (np.isfinite(ema50) and np.isfinite(ema50_prev5) and ema50 > ema50_prev5)
                        if trend_gate:
                            entry = l20
                            stop = low - 0.25 * atr
                            R = abs(entry - stop)
                            if R > 0:
                                target = entry + 2.1 * R
                                score = 3.1 + 0.1 * min(zv, 4.0)
                                s = build_signal(sym, market, tf, "P3_SWEEP", "long", ts, entry, stop, target, score)
                                if s:
                                    p3_list.append(s)
            # short
            if np.isfinite(h20):
                sweep = (high - h20) / atr
                if 0.10 <= sweep <= 1.20 and close < h20 and (not np.isnan(zv)) and zv > 1.3:
                    if 42 <= rsi <= 70:
                        equity_bull = is_equity and close > sma200 and ema9 > ema30
                        trend_gate = (close <= 1.02 * sma200) or (not equity_bull)
                        if trend_gate:
                            entry = h20
                            stop = high + 0.25 * atr
                            R = abs(entry - stop)
                            if R > 0:
                                target = entry - 2.1 * R
                                score = 3.1 + 0.1 * min(zv, 4.0)
                                s = build_signal(sym, market, tf, "P3_SWEEP", "short", ts, entry, stop, target, score)
                                if s:
                                    p3_list.append(s)
            if p3_list:
                out.append(p3_list[0])
        except Exception:
            pass

        # --- F3_BREAKOUT ---
        try:
            f3_list = []
            if tf in ("1h", "1d"):
                # long
                if (np.isfinite(h20) and close > h20 and ema9 > ema30 > sma200
                        and sma200 > sma200_prev5 and (not np.isnan(zv)) and zv > 1.0
                        and 50 <= rsi <= 70):
                    entry = 0.5 * h20 + 0.5 * ema9
                    stop = (min(l10, ema30) if (np.isfinite(l10)) else ema30) - 0.1 * atr
                    # use min of l10 and ema30
                    try:
                        stop = min(l10, ema30) - 0.1 * atr
                    except Exception:
                        stop = ema30 - 0.1 * atr
                    R = abs(entry - stop)
                    if R > 0:
                        target = entry + 2.2 * R
                        score = 2.75 + 0.1 * min(zv, 4.0)
                        s = build_signal(sym, market, tf, "F3_BREAKOUT", "long", ts, entry, stop, target, score)
                        if s:
                            f3_list.append(s)
                # short
                if (np.isfinite(l20) and close < l20 and ema9 < ema30 < sma200
                        and sma200 < sma200_prev5 and (not np.isnan(zv)) and zv > 1.0
                        and 30 <= rsi <= 50):
                    entry = 0.5 * l20 + 0.5 * ema9
                    try:
                        stop = max(h10, ema30) + 0.1 * atr
                    except Exception:
                        stop = ema30 + 0.1 * atr
                    R = abs(entry - stop)
                    if R > 0:
                        target = entry - 2.2 * R
                        score = 2.75 + 0.1 * min(zv, 4.0)
                        s = build_signal(sym, market, tf, "F3_BREAKOUT", "short", ts, entry, stop, target, score)
                        if s:
                            f3_list.append(s)
            if f3_list:
                out.append(f3_list[0])
        except Exception:
            pass

    # so repeated re-scans can never inflate the score cumulatively)
    for s in out:
        try:
            base = float(s.get("base_score", s.get("score", 0.0)))
            s["base_score"] = round(base, 4)
            if "kreil" not in s:
                _, kr_info = passes_kreil_regime_gate(df, s.get("direction", ""), s.get("arch", ""))
                s["kreil"] = kr_info
            boost = float(s["kreil"].get("score_boost", 0.0) or 0.0)
            s["score"] = round(min(4.0, base + boost), 4)
            if boost > 0:
                s["kreil"]["boost_applied"] = True
        except Exception:
            pass
    # keep single highest-score signal per (sym, tf)
    if not out:
        return []
    if keep_all:
        return out
    best = max(out, key=lambda x: x["score"])
    return [best]


def compute_s_min(consecutive_losses: int) -> float:
    """Dynamic minimum score, capped at 2.90 so every setup stays tradable.

    MA7_TRUE has a 2.0R target and a 2.95 maximum base score, so a cap above
    2.90 would make the whole setup untradable after two losses.
    """
    k = int(consecutive_losses) if consecutive_losses is not None else 0
    k = max(0, k)
    return min(2.90, round(2.6 + 0.15 * k, 4))


def compute_min_rr(consecutive_losses: int) -> float:
    """Dynamic minimum reward:risk, capped at 2.0 so MA7_TRUE (2.0R) still trades."""
    k = int(consecutive_losses) if consecutive_losses is not None else 0
    k = max(0, k)
    return min(2.0, round(1.8 + 0.10 * k, 4))


def get_s_min(k):
    return compute_s_min(k)


def s_min_floor(k):
    return compute_s_min(k)


def get_min_rr(k):
    return compute_min_rr(k)


def default_ledger():
    return {
        "account": {
            "equity": 100.0,
            "risk_pct": 0.015,
            "target": 1000.0,
            "month_start_equity": 100.0,
            "current_month": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m"),
        },
        "open_positions": [],
        "pending_orders": [],
        "closed_trades": [],
        "block_stats": {
            "consecutive_losses": 0,
            "blocked": False,
            "cooldowns": [],
            "s_min_current": 2.6,
            "min_rr": 1.8,
            "min_rr_current": 1.8,
            "monthly_dd_pct": 0.0,
        },
        "last_run": None,
        "spot_prices": {},
    }


def enforce_risk_pct(ledger):
    acct = ledger.setdefault("account", {})
    rp = acct.get("risk_pct", 0.015)
    try:
        rp = float(rp)
    except (TypeError, ValueError):
        rp = 0.015
    acct["risk_pct"] = min(rp, MAX_RISK_PCT)
    if acct["risk_pct"] <= 0:
        acct["risk_pct"] = 0.015
    if "equity" not in acct:
        acct["equity"] = 100.0
    if "target" not in acct:
        acct["target"] = 1000.0
    if "month_start_equity" not in acct:
        acct["month_start_equity"] = float(acct.get("equity", 100.0))
    if "current_month" not in acct:
        acct["current_month"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
    return ledger


def _last_loss_ts(ledger):
    """Most recent closed-trade exit timestamp with a negative realised R."""
    latest = None
    for ct in ledger.get("closed_trades", []) or []:
        if not isinstance(ct, dict):
            continue
        try:
            pnl = float(ct.get("pnl_r", 0.0) or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        if pnl >= 0:
            continue
        ts = _parse_utc(ct.get("exit_time"))
        if ts is None:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def update_block_stats(ledger):
    bs = ledger.setdefault("block_stats", {})
    acct = ledger.setdefault("account", {})
    bs.setdefault("consecutive_losses", 0)
    bs.setdefault("blocked", False)
    bs.setdefault("min_rr", 1.8)
    if not isinstance(bs.get("cooldowns"), list):
        bs["cooldowns"] = []
    equity = float(acct.get("equity", 100.0))
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_month = now_utc.strftime("%Y-%m")
    prev_month = acct.get("current_month")
    if prev_month is None:
        acct["current_month"] = now_month
    elif prev_month != now_month:
        acct["current_month"] = now_month
        acct["month_start_equity"] = equity
        bs["consecutive_losses"] = 0
        bs["blocked"] = False
        bs.pop("block_reason", None)
        bs.pop("block_until", None)
    month_start = float(acct.get("month_start_equity", 100.0))
    if month_start <= 0:
        month_start = 100.0
        acct["month_start_equity"] = month_start
    dd = (month_start - equity) / month_start
    bs["monthly_dd_pct"] = float(dd)
    k = int(bs.get("consecutive_losses", 0))
    blocked = False
    reason = None
    if dd >= 0.06:
        blocked = True
        reason = "ELDER_6PCT_PIRANHA_GUARD"
    elif k >= 3:
        # Time-based 12h cooldown instead of a permanent deadlock: no trade can
        # open while blocked, so only the clock can release this state.
        now_ts = _parse_utc(now_utc)
        last_loss = _last_loss_ts(ledger)
        block_until = None
        if last_loss is not None:
            block_until = last_loss + pd.Timedelta(hours=LOSS_BLOCK_HOURS)
        else:
            # Legacy closed trades without exit_time: persist a fixed 12h window
            # so the block can still self-heal instead of sticking forever.
            block_until = _parse_utc(bs.get("block_until"))
            if block_until is None and now_ts is not None:
                block_until = now_ts + pd.Timedelta(hours=LOSS_BLOCK_HOURS)
                bs["block_until"] = str(block_until)
        if block_until is not None and now_ts is not None and now_ts >= block_until:
            # Cooldown served: re-arm cautiously in step-2 (k=2) rather than bricked.
            bs["consecutive_losses"] = 2
            k = 2
            bs.pop("block_until", None)
            bs.pop("block_reason", None)
            bs["loss_block_expired_at"] = str(now_ts)
        else:
            blocked = True
            reason = "ELDER_12H_LOSS_COOLDOWN"
            if block_until is not None:
                bs["block_until"] = str(block_until)
    bs["s_min_current"] = float(compute_s_min(k))
    bs["min_rr_current"] = round(compute_min_rr(k), 4)
    bs["blocked"] = blocked
    if blocked:
        bs["block_reason"] = reason
    else:
        bs.pop("block_reason", None)
    return ledger


def _signal_seal(symbol, tf, arch, direction, placed_bar_time, limit, stop, target):
    return hashlib.sha256(
        f"{symbol}|{tf}|{arch}|{direction}|{placed_bar_time}|{float(limit):.6f}|{float(stop):.6f}|{float(target):.6f}".encode()
    ).hexdigest()


def _uninflate_base_score(order):
    """Recover the exact unboosted detector score of a pending order.

    The legacy Kreil path wrote ``base_score = score`` *after* adding the boost,
    so every re-scan stacked another +0.15 on the stored base.  Recovery order:
      1. known historical order id -> exact value from git history;
      2. base_score > ARCH_MAX_BASE_SCORE[arch] -> strip whole +0.15 steps;
      3. otherwise keep the stored base untouched (a legitimate, unboosted
         4.0 T10_SOROS signal must never lose 0.15).
    Returns ``(base_score, boost)``.
    """
    kr = order.get("kreil") or {}
    try:
        boost = float(kr.get("score_boost", 0.0) or 0.0)
    except (TypeError, ValueError):
        boost = 0.0
    try:
        score = float(order.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    raw = order.get("base_score")
    if raw is None:
        base = score - boost if boost > 0 else score
    else:
        try:
            base = float(raw)
        except (TypeError, ValueError):
            base = score - boost if boost > 0 else score

    known = LEGACY_ORDER_BASE_SCORE.get(str(order.get("order_id", "")))
    if known is not None:
        return round(float(known), 4), boost

    cap = float(ARCH_MAX_BASE_SCORE.get(str(order.get("arch", "")), 4.0))
    if base > cap + 1e-9:
        steps = int(math.ceil((base - cap) / KREIL_BOOST_STEP - 1e-9))
        base = base - KREIL_BOOST_STEP * steps
        if base > cap + 1e-9 and boost > 0:
            # last resort: the boost itself is the only contamination left
            base = score - boost
    elif (boost > 0 and base >= score - 1e-9
          and score < min(4.0, base + boost) - 1e-9):
        # Legacy signature: ``base_score`` was written AFTER the boost, so the
        # stored score is smaller than base + boost.  A legitimate unboosted
        base = score - boost
    base = min(base, cap)
    return round(max(0.0, base), 4), boost


def repair_pending_orders(ledger):
    """Migrate legacy pending orders: un-inflate base_score, re-target, re-seal.

    * ``base_score`` is restored to the exact original creation value and the
      score is rebuilt exactly once (``min(4.0, base + boost)``), so a re-scan
      can never stack the Kreil boost again.
    * Any target outside ``[1.8R, ARCH_TARGET_RR[arch] + 0.05R]`` is migrated to
      the setup's new architecture multiple (legacy 3.2R-3.8R orders from before
      the upgrade included) and its integrity seal is recomputed.
    """
    for order in ledger.get("pending_orders", []) or []:
        if not isinstance(order, dict):
            continue
        try:
            base, boost = _uninflate_base_score(order)
            order["base_score"] = base
            order["score"] = round(min(4.0, base + boost), 4)

            limit = float(order["limit_price"])
            stop = float(order["stop_price"])
            target = float(order["target_price"])
            r_dist = abs(limit - stop)
            if r_dist <= 0:
                continue
            arch = str(order.get("arch", ""))
            design = float(ARCH_TARGET_RR.get(arch, compute_min_rr(0)))
            rr = abs(target - limit) / r_dist
            if design > 0 and (rr > design + 0.05 or rr < 1.8):
                is_long = str(order.get("direction", "long")).lower() in ("long", "l", "buy")
                target = limit + design * r_dist if is_long else limit - design * r_dist
                order["target_price"] = target
                order["seal_sha256"] = _signal_seal(
                    order.get("symbol"), order.get("tf"), arch,
                    order.get("direction"), order.get("placed_bar_time"), limit, stop, target,
                )
                order["integrity_seal"] = order["seal_sha256"]
                rr = design
            order["rr"] = round(float(rr), 4)
        except Exception:
            continue
    return ledger


def load_ledger():
    L = None
    if LEDGER_PATH.exists():
        try:
            L = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        except Exception:
            L = None
    if not isinstance(L, dict):
        L = default_ledger()
    base = default_ledger()
    for key, val in base.items():
        if key not in L or L[key] is None:
            L[key] = val
    for key, val in base["account"].items():
        L.setdefault("account", {}).setdefault(key, val)
    for key, val in base["block_stats"].items():
        L.setdefault("block_stats", {}).setdefault(key, val)
    for key in ("open_positions", "pending_orders", "closed_trades"):
        if not isinstance(L.get(key), list):
            L[key] = []
    enforce_risk_pct(L)
    repair_pending_orders(L)
    update_block_stats(L)
    return L


def _parse_utc(ts):
    """Parse any timestamp into a timezone-aware UTC ``pd.Timestamp``.

    Timezone-aware inputs (``+05:30``, ``-04:00``, ``Z``) are *converted* to UTC
    (``astimezone(timezone.utc)``) instead of having their offset stripped, and
    naive timestamps are treated as UTC.  Non-timestamp labels (e.g. ``"t0249"``
    test bar ids) return ``None`` rather than being guessed by dateutil.
    """
    if ts is None:
        return None
    parsed = None
    if isinstance(ts, str):
        try:
            parsed = pd.to_datetime(ts, utc=True, errors="coerce", format="ISO8601")
        except Exception:
            parsed = None
        if parsed is None or parsed is pd.NaT or pd.isna(parsed):
            if "-" not in ts:
                return None
            try:
                parsed = pd.to_datetime(ts, utc=True, errors="coerce")
            except Exception:
                return None
    else:
        try:
            parsed = pd.to_datetime(ts, utc=True, errors="coerce")
        except Exception:
            return None
    if parsed is None or parsed is pd.NaT or pd.isna(parsed):
        return None
    try:
        dt_obj = pd.Timestamp(parsed).to_pydatetime()
    except Exception:
        return None
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
    return pd.Timestamp(dt_obj.astimezone(datetime.timezone.utc))


def _utc_series(series):
    """Vectorised UTC conversion: aware offsets are converted, naive assumed UTC."""
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    except Exception:
        try:
            return pd.to_datetime(series, utc=True, errors="coerce")
        except Exception:
            return None


def bars_after(df, time_str):
    """Rows of ``df`` strictly AFTER ``time_str``, compared as real UTC instants.

    Raw string comparison is wrong across timezones: ``"2026-10-30 15:15:00+05:30"``
    (09:45 UTC) sorts *after* ``"2026-10-30 10:00:00"`` lexicographically even
    though it is earlier in time.  Both sides are therefore parsed to UTC.
    """
    if df is None or len(df) == 0:
        return df.iloc[0:0] if df is not None else df
    cutoff = _parse_utc(time_str)
    if cutoff is None:
        return df.iloc[0:0]
    try:
        raw = df["Time"] if "Time" in df.columns else pd.Series(list(df.index))
        stamps = _utc_series(raw)
        if stamps is None:
            return df.iloc[0:0]
        mask = stamps.notna().to_numpy() & (stamps > cutoff).to_numpy()
        return df.loc[mask]
    except Exception:
        return df.iloc[0:0]


def try_fill(order, df, tf=None, atr=0.0, near_spot_r=0.20):
    """Return (filled, fill_time, fill_price).

    Two fill paths are supported:
      1. strict touch: the bar trades through the limit (minus a tiny buffer) —
         this applies to every historical bar;
      2. near-spot tolerance: price comes within ``min(0.25*ATR, 0.20R)`` (or
         ``0.20R`` when no ATR is supplied) of the limit.  This is only allowed
         on the LATEST bar, so a later ATR change can never retroactively fill an
         old timestamp.
    """
    otf = tf if tf is not None else order["tf"]
    tf_sec = TF_SEC.get(otf, 3600)
    limit = float(order["limit_price"])
    stop = float(order["stop_price"])
    r_dist = abs(limit - stop)
    try:
        atr = float(atr)
    except (TypeError, ValueError):
        atr = 0.0
    if not np.isfinite(atr) or atr <= 0:
        atr = 0.0
    if atr > 0:
        tol = 0.25 * atr
    elif r_dist > 0:
        tol = float(near_spot_r) * r_dist
    else:
        tol = 0.0025 * abs(limit)
    if r_dist > 0:
        # Hard safety cap: the near-spot band can never exceed 0.20R.
        tol = min(tol, 0.20 * r_dist)
    bars = bars_after(df, order["placed_bar_time"])
    n_bars = len(bars)
    for idx, (_, bar) in enumerate(bars.iterrows()):
        H = float(bar["High"]); L = float(bar["Low"])
        buf = 0.15 * (H - L) * math.sqrt(2.0 / tf_sec) if (H - L) > 0 else 0.0
        is_latest = idx == n_bars - 1
        if order["direction"] == "long":
            if L <= limit - buf:
                return True, str(bar["Time"]), limit
            if is_latest and L <= limit + tol:
                return True, str(bar["Time"]), limit
        else:
            if H >= limit + buf:
                return True, str(bar["Time"]), limit
            if is_latest and H >= limit - tol:
                return True, str(bar["Time"]), limit
    return False, None, None


def is_rejected(order_id, entry_time):
    h = int(hashlib.sha256((order_id + str(entry_time)).encode()).hexdigest(), 16) / (2 ** 256)
    return h < 0.02


def _cooldown_hours(tf):
    if tf == "1d":
        return 48.0
    if tf == "1h":
        return 6.0
    return (TF_SEC.get(tf, 3600) * 5) / 3600.0


def record_cooldown(ledger, symbol, direction, tf, exit_time_str):
    exit_ts = _parse_utc(exit_time_str)
    if exit_ts is None:
        return
    delta = pd.Timedelta(hours=_cooldown_hours(tf))
    until = exit_ts + delta
    ct_list = ledger.setdefault("block_stats", {}).setdefault("cooldowns", [])
    if not isinstance(ct_list, list):
        ct_list = []
        ledger["block_stats"]["cooldowns"] = ct_list
    ct_list.append({"symbol": symbol, "direction": direction, "tf": tf, "until": str(until)})
    # keep bounded
    if len(ct_list) > 500:
        del ct_list[:-500]


def in_cooldown(ledger, symbol, direction, when_time_str, tf=None):
    when = _parse_utc(when_time_str)
    if when is None:
        return False
    # block_stats cooldowns
    for cd in ledger.get("block_stats", {}).get("cooldowns", []):
        if cd.get("symbol") == symbol and cd.get("direction") == direction:
            until = _parse_utc(cd.get("until"))
            if until is not None and when <= until:
                return True
    # closed trades
    for ct in ledger.get("closed_trades", []):
        if ct.get("symbol") == symbol and ct.get("direction") == direction:
            if (ct.get("pnl_r", 0) or 0) < 0:
                hours = _cooldown_hours(ct.get("tf", tf or "1d"))
                exit_ts = _parse_utc(ct.get("exit_time"))
                cu = ct.get("cooldown_until")
                expiry = _parse_utc(cu) if cu is not None else None
                if expiry is None and exit_ts is not None:
                    expiry = exit_ts + pd.Timedelta(hours=hours)
                    try:
                        ct["cooldown_until"] = str(expiry)
                    except Exception:
                        pass
                if expiry is not None and when <= expiry:
                    return True
                if exit_ts is not None:
                    if when <= exit_ts:
                        return True
                    try:
                        if (when - exit_ts).total_seconds() < hours * 3600:
                            return True
                    except Exception:
                        pass
    return False


def cooldown_active(ledger, symbol, direction, when_time_str, tf=None):
    return in_cooldown(ledger, symbol, direction, when_time_str, tf)


def bars_held_since_entry(pos, bars):
    """Bars elapsed since entry, monotonically non-decreasing.

    * ``bars`` given (a non-empty frame of bars after entry): the bar count is
      authoritative and ``bars_held`` can never shrink when old bars roll off
      the yfinance window -> ``max(previous, len(bars))``.
    * ``bars`` is ``None``/empty (fetch failed): wall-clock fallback, but it may
      only advance one bar per scan so weekends/holidays can never inflate it.
    """
    prev = int(pos.get("bars_held", 0) or 0)
    if bars is not None and len(bars) > 0:
        return max(prev, int(len(bars)))
    opened = pos.get("opened_at") or pos.get("opened_bar_ts") or pos.get("entry_ts") or pos.get("entry_time")
    ts = _parse_utc(opened)
    if ts is None:
        return prev
    tf_sec = TF_SEC.get(pos.get("tf", "1d"), 86400)
    now = _parse_utc(datetime.datetime.now(datetime.timezone.utc))
    if now is None:
        return prev
    try:
        elapsed = (now - ts).total_seconds()
    except Exception:
        return prev
    if elapsed <= 0:
        return prev
    return max(prev, min(int(elapsed // tf_sec), prev + 1))


def _profit_lock_stop(pos, mfe_r, risk_price, entry, bars_sub, atr_val):
    """Tightest stop allowed by the dynamic profit-lock ladder.

    Ladder (LONG): +0.05R at 0.5R, +0.60R at 1.0R, +1.10R at 1.5R, and the
    2.2R chandelier floor of +1.60R (or high - 2*ATR).  SHORT is the mirror.
    """
    is_long = pos["direction"] == "long"
    best = None
    for threshold, lock in ((0.5, 0.05), (1.0, 0.60), (1.5, 1.10)):
        if mfe_r >= threshold:
            cand = entry + lock * risk_price if is_long else entry - lock * risk_price
            if best is None or (cand > best if is_long else cand < best):
                best = cand
    if mfe_r >= 2.2:
        if is_long:
            hh = max(float(r["High"]) for _, r in bars_sub.iterrows())
            cand = max(entry + 1.60 * risk_price, hh - 2.0 * atr_val)
            if best is None or cand > best:
                best = cand
        else:
            ll = min(float(r["Low"]) for _, r in bars_sub.iterrows())
            cand = min(entry - 1.60 * risk_price, ll + 2.0 * atr_val)
            if best is None or cand < best:
                best = cand
    return best


def step_a_update_positions(ledger, data):
    open_positions = ledger["open_positions"]
    for pos in list(open_positions):
        df = data.get((pos["symbol"], pos["tf"]))
        if df is None or (hasattr(df, "empty") and df.empty):
            # No fresh data at all: only the (capped) wall-clock fallback runs.
            pos["bars_held"] = bars_held_since_entry(pos, None)
            continue
        tf = pos.get("tf", "1d")
        # ``initial_stop`` / ``initial_stop_price`` are written once and never
        # mutated, so the R denominator is stable for the life of the trade.
        pos.setdefault("initial_stop", pos["stop_price"])
        pos.setdefault("initial_stop_price", pos.get("initial_stop", pos["stop_price"]))
        pos.setdefault("opened_at", str(pos.get("entry_time")))
        entry = float(pos["entry_price"])
        is_long = pos["direction"] == "long"
        initial_stop = float(pos.get("initial_stop_price") or pos.get("initial_stop") or pos["stop_price"])
        risk_price = (entry - initial_stop) if is_long else (initial_stop - entry)
        if risk_price <= 0:
            continue
        all_bars = bars_after(df, pos["entry_time"])
        # Monotonic: bars that roll off the yfinance window can never shrink it,
        # and a frame with no new bar (weekend / intraday re-run) freezes the
        # count instead of inflating it with calendar days.
        if len(all_bars) > 0:
            pos["bars_held"] = max(int(pos.get("bars_held", 0) or 0), int(len(all_bars)))
        else:
            pos["bars_held"] = int(pos.get("bars_held", 0) or 0)
        mfe_r = float(pos.get("mfe_r", 0.0) or 0.0)
        exited = False
        exit_price = None
        reason = None
        exit_time = None
        exit_r = None
        cost_pct, slip_pct = get_costs(pos["market"])

        # Hindsight guard: only bars strictly AFTER the last bar already walked
        # are replayed.  Re-walking old bars with a later (ratcheted) stop would
        # retroactively stop the trade out in the past.
        walk_from = pos.get("last_processed_bar_ts") or pos.get("entry_time")
        walk_cutoff = _parse_utc(walk_from)

        for bar_idx in range(len(all_bars)):
            bar = all_bars.iloc[bar_idx]
            bar_time = str(bar["Time"])
            bar_ts = _parse_utc(bar_time)
            if bar_ts is not None and walk_cutoff is not None and bar_ts <= walk_cutoff:
                continue
            H = float(bar["High"]); L = float(bar["Low"]); C = float(bar["Close"])
            try:
                O = float(bar["Open"])
            except Exception:
                O = C
            # The stop active at the START of this bar, BEFORE any ratchet that
            # this very bar's high/low would justify (no intrabar hindsight).
            stop_active = float(pos["stop_price"])
            target_active = float(pos["target_price"])
            if is_long:
                if L <= stop_active:
                    # Gap-through fills at the open, not at the untouched stop.
                    exit_price = O if O < stop_active else stop_active
                    reason = "stop"; exited = True
                elif H >= target_active:
                    exit_price = target_active; reason = "target"; exited = True
            else:
                if H >= stop_active:
                    exit_price = O if O > stop_active else stop_active
                    reason = "stop"; exited = True
                elif L <= target_active:
                    exit_price = target_active; reason = "target"; exited = True
            if exited:
                exit_time = bar_time
                break

            unreal_r = ((C - entry) if is_long else (entry - C)) / risk_price
            if unreal_r > mfe_r:
                mfe_r = unreal_r
            try:
                atr_val = float(bar["atr"])
            except Exception:
                atr_val = float("nan")
            if not np.isfinite(atr_val) or atr_val <= 0:
                atr_val = risk_price
            new_stop = _profit_lock_stop(pos, mfe_r, risk_price, entry,
                                         all_bars.iloc[:bar_idx + 1], atr_val)
            if new_stop is not None:
                if is_long and new_stop > pos["stop_price"]:
                    pos["stop_price"] = new_stop
                elif (not is_long) and new_stop < pos["stop_price"]:
                    pos["stop_price"] = new_stop
            pos["last_processed_bar_ts"] = bar_time

        pos["mfe_r"] = round(mfe_r, 6)
        if len(all_bars) > 0:
            last_price = float(all_bars.iloc[-1]["Close"])
            last_time = str(all_bars.iloc[-1]["Time"])
        else:
            last_price = float(df.iloc[-1]["Close"])
            last_time = str(df.iloc[-1]["Time"])
        unreal_now = ((last_price - entry) if is_long else (entry - last_price)) / risk_price

        if not exited:
            held = int(pos["bars_held"])
            # Close & move on: bank a small profit once the trade has had its
            # allotted time, freeing the slot for a fresh setup.
            if held >= profit_harvest_bars(tf) and unreal_now >= 0.15:
                exited = True
                reason = "TIME_PROFIT_HARVEST"
                exit_price = last_price
                exit_time = last_time
                exit_r = round(float(unreal_now), 4)
            # Dead money: a flat trade is exited so capital is not stuck for days.
            elif held >= dead_money_bars(tf) and -0.25 <= unreal_now < 0.15:
                exited = True
                reason = "TIME_FLAT_EXIT"
                exit_price = last_price
                exit_time = last_time
                exit_r = round(float(unreal_now), 4)

        if not exited:
            pos["unrealized_r"] = round(unreal_now, 6)
            pos["last_price"] = last_price
            continue

        # --- settle the exit ---
        time_exit = reason in ("TIME_PROFIT_HARVEST", "TIME_FLAT_EXIT")
        if time_exit:
            pnl_r = float(exit_r if exit_r is not None else unreal_now)
        else:
            pnl_r = (exit_price - entry) / risk_price if is_long else (entry - exit_price) / risk_price
            extra_slip = slip_pct if reason == "stop" else 0.0
            pnl_r -= (cost_pct + extra_slip) * entry / risk_price
        ledger["account"]["equity"] += pnl_r * pos["risk_capital"]
        bs = ledger["block_stats"]
        if pnl_r > 0:
            bs["consecutive_losses"] = 0
            bs["blocked"] = False
            bs.pop("block_reason", None)
            bs.pop("block_until", None)
        elif pnl_r <= -0.5:
            bs["consecutive_losses"] = bs.get("consecutive_losses", 0) + 1
        # scratch exits in (-0.5R, 0) deliberately do NOT count as a full loss
        if pnl_r <= -0.5:
            record_cooldown(ledger, pos["symbol"], pos["direction"], pos["tf"], str(exit_time))
        ledger["closed_trades"].append({
            "order_id": pos["order_id"],
            "symbol": pos["symbol"],
            "market": pos["market"],
            "tf": pos["tf"],
            "arch": pos["arch"],
            "direction": pos["direction"],
            "entry_time": pos["entry_time"],
            "exit_time": str(exit_time),
            "entry_price": pos["entry_price"],
            "exit_price": float(exit_price),
            "pnl_r": round(float(pnl_r), 4),
            "realized_r": round(float(pnl_r), 4),
            "mfe_r": round(float(mfe_r), 4),
            "bars_held": int(pos.get("bars_held", 0) or 0),
            "reason": reason,
            "cooldown_until": None,
        })
        try:
            exit_ts = _parse_utc(exit_time)
            delta = pd.Timedelta(hours=_cooldown_hours(pos["tf"]))
            if exit_ts is not None:
                ledger["closed_trades"][-1]["cooldown_until"] = str(exit_ts + delta)
        except Exception:
            pass
        open_positions.remove(pos)
        update_block_stats(ledger)


def step_b_fill_pending(ledger, data, open_positions=None, enforce_kreil=False):
    if open_positions is None:
        open_positions = ledger["open_positions"]
    k = int(ledger.get("block_stats", {}).get("consecutive_losses", 0))
    s_min = get_s_min(k)
    min_rr = get_min_rr(k)
    # normalize rr on pending orders
    for order in ledger.get("pending_orders", []):
        if "rr" not in order:
            try:
                r = abs(float(order["limit_price"]) - float(order["stop_price"]))
                order["rr"] = round(abs(float(order["target_price"]) - float(order["limit_price"])) / r, 4) if r > 0 else 0.0
            except Exception:
                order["rr"] = 0.0
    # purge orders that no longer meet risk thresholds after a loss
    filtered_pending = []
    for order in ledger.get("pending_orders", []):
        if enforce_kreil:
            df = data.get((order.get("symbol"), order.get("tf")))
            if df is not None:
                passed_kr, kr_info = passes_kreil_regime_gate(df, order.get("direction", ""), order.get("arch", ""))
                if not passed_kr:
                    continue
                # Rebuild from the unboosted base so re-scans never stack +0.15.
                base = float(order.get("base_score", order.get("score", 0.0)))
                order["base_score"] = round(base, 4)
                order["score"] = round(min(4.0, base + float(kr_info.get("score_boost", 0.0) or 0.0)), 4)
                if kr_info.get("score_boost", 0.0) > 0:
                    kr_info["boost_applied"] = True
                order["kreil"] = kr_info
        if float(order.get("score", 0.0)) < s_min or float(order.get("rr", 0.0)) < min_rr:
            continue
        filtered_pending.append(order)
    ledger["pending_orders"] = filtered_pending

    # dedup pending by symbol (keep highest score) and cap
    # expire orders once they exceed their timeframe's resting lifetime
    new_pending = []
    for order in ledger["pending_orders"]:
        df = data.get((order["symbol"], order["tf"]))
        if df is not None:
            bars = bars_after(df, order["placed_bar_time"])
            if len(bars) > max_pending_bars(order.get("tf", "1d")):
                continue
        new_pending.append(order)
    ledger["pending_orders"] = new_pending

    # dedup per symbol keep highest score
    by_symbol = {}
    for order in ledger["pending_orders"]:
        sym = order["symbol"]
        if sym not in by_symbol or order.get("score", 0) > by_symbol[sym].get("score", 0):
            by_symbol[sym] = order
    deduped = list(by_symbol.values())
    # cap global 8 by score
    deduped.sort(key=lambda x: x.get("score", 0), reverse=True)
    deduped = deduped[:MAX_PENDING_GLOBAL]
    ledger["pending_orders"] = deduped

    # now process fills
    still_pending = []
    # sort by score desc for fill priority
    ordered = sorted(ledger["pending_orders"], key=lambda x: x.get("score", 0), reverse=True)
    for order in ordered:
        if float(order.get("score", 0.0)) < s_min or float(order.get("rr", 0.0)) < min_rr:
            continue
        df = data.get((order["symbol"], order["tf"]))
        if df is None:
            still_pending.append(order)
            continue
        bars = bars_after(df, order["placed_bar_time"])
        if len(bars) > max_pending_bars(order.get("tf", "1d")):
            continue
        try:
            atr_val = float(df.iloc[-1]["atr"])
        except Exception:
            atr_val = 0.0
        filled, entry_time, entry_price = try_fill(order, df, order.get("tf"), atr=atr_val)
        if not filled:
            still_pending.append(order)
            continue
        if is_rejected(order["order_id"], entry_time):
            continue
        if ledger["block_stats"].get("blocked"):
            still_pending.append(order)
            continue
        if in_cooldown(ledger, order["symbol"], order["direction"], entry_time, order["tf"]):
            # blocked by cooldown; drop from pending
            continue
        # 3 global slots AND at most 1 position per market bucket
        if not can_open_position(open_positions, order.get("market", "")):
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
            "opened_at": str(entry_time),
            "entry_price": float(entry_price),
            "stop_price": float(order["stop_price"]),
            "initial_stop": float(order["stop_price"]),
            "initial_stop_price": float(order["stop_price"]),
            "target_price": float(order["target_price"]),
            "risk_capital": equity * ledger["account"]["risk_pct"],
            "unrealized_r": 0.0,
            "mfe_r": 0.0,
            "bars_held": 0,
            "last_processed_bar_ts": str(entry_time),
            "seal_sha256": order.get("seal_sha256"),
        }
        open_positions.append(pos)
        # once filled, do not fill more (global lock)
        # mark remaining pending kept
    # rebuild pending: only orders that were not filled are kept
    seen = set()
    final_pending = []
    for o in still_pending:
        if o["order_id"] in seen:
            continue
        seen.add(o["order_id"])
        final_pending.append(o)
    # cap again
    final_pending.sort(key=lambda x: x.get("score", 0), reverse=True)
    final_pending = final_pending[:MAX_PENDING_GLOBAL]
    ledger["pending_orders"] = final_pending
    ledger["open_positions"] = open_positions


# aliases
step_b_pending = step_b_fill_pending
step_b_process_pending = step_b_fill_pending


def _pending_supersede_key(order):
    """Identity of a resting order for supersede purposes: symbol/tf/arch/dir."""
    return (order.get("symbol"), order.get("tf"), order.get("arch"), order.get("direction"))


def _shifted_cleanly(old, new):
    """True when the newer bar's entry/stop ratchet in the trade's favour.

    Long:  entry moves up and the stop moves up (or stays).
    Short: entry moves down and the stop moves down (or stays).
    Anything else (entry chased away from the trade, stop widened, non-finite
    prices) is not a clean shift and must not replace the resting order.
    """
    try:
        o_limit = float(old["limit_price"])
        o_stop = float(old["stop_price"])
        n_limit = float(new["limit_price"])
        n_stop = float(new["stop_price"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(np.isfinite(v) for v in (o_limit, o_stop, n_limit, n_stop)):
        return False
    direction = str(new.get("direction") or old.get("direction") or "").lower()
    if direction == "long":
        return n_limit >= o_limit and n_stop >= o_stop
    if direction == "short":
        return n_limit <= o_limit and n_stop <= o_stop
    return False


def _supersedes(old, new):
    """Should ``new`` (fresh bar) replace ``old`` (same symbol/tf/arch/dir)?

    ``order_id`` already carries ``placed_bar_time`` (the signal time), so a
    fresh bar always yields a new id and would otherwise stack a second
    resting order on the same symbol.  Replacement is allowed only when the
    new signal comes from a strictly newer bar AND either scores higher than
    the resting order or has shifted its entry/stop cleanly.
    """
    if old.get("order_id") == new.get("order_id"):
        return False
    old_ts = _parse_utc(old.get("placed_bar_time"))
    new_ts = _parse_utc(new.get("placed_bar_time"))
    if old_ts is not None and new_ts is not None:
        if new_ts <= old_ts:
            return False
    elif str(old.get("placed_bar_time")) == str(new.get("placed_bar_time")):
        return False
    if float(new.get("score", 0.0) or 0.0) > float(old.get("score", 0.0) or 0.0):
        return True
    return _shifted_cleanly(old, new)


def intake_signals(ledger, data, signals, open_positions, existing_ids=None):
    """Turn fresh signals into open positions, or rest them as pending orders.

    A signal that would fill but is rejected by ``is_rejected`` is skipped
    cleanly: it must never be parked on the pending book to "fill" on the next
    scan (that made rejections non-sticky and filled them a scan later).

    Resting orders are deduplicated per symbol: a newer bar on the same
    ``(symbol, tf, arch, direction)`` supersedes the older pending order when
    it scores higher or when its entry/stop shifted cleanly, and at most
    ``MAX_PENDING_PER_SYMBOL`` order per symbol is ever left on the book.
    """
    if existing_ids is None:
        existing_ids = set()
    k = int(ledger.get("block_stats", {}).get("consecutive_losses", 0))
    s_min = get_s_min(k)
    min_rr = get_min_rr(k)
    pending = ledger.setdefault("pending_orders", [])
    for sig in signals:
        oid = sig.get("order_id")
        if oid in existing_ids:
            continue
        if float(sig.get("score", 0.0)) < s_min or float(sig.get("rr", 0.0)) < min_rr:
            continue
        df = data.get((sig.get("symbol"), sig.get("tf")))
        if df is None:
            continue
        try:
            atr_val = float(df.iloc[-1]["atr"])
        except Exception:
            atr_val = 0.0
        filled, entry_time, entry_price = try_fill(sig, df, sig.get("tf"), atr=atr_val)
        if filled:
            if is_rejected(oid, entry_time):
                # Sticky rejection: drop it, never queue it for the next scan.
                continue
            if (can_open_position(open_positions, sig.get("market", ""))
                    and not ledger.get("block_stats", {}).get("blocked")
                    and not in_cooldown(ledger, sig["symbol"], sig["direction"], entry_time, sig["tf"])):
                risk_price = abs(float(entry_price) - float(sig["stop_price"]))
                if risk_price > 0:
                    equity = ledger["account"]["equity"]
                    pos = {
                        "order_id": oid,
                        "symbol": sig["symbol"],
                        "market": sig["market"],
                        "tf": sig["tf"],
                        "arch": sig["arch"],
                        "direction": sig["direction"],
                        "entry_time": entry_time,
                        "opened_at": str(entry_time),
                        "entry_price": float(entry_price),
                        "stop_price": float(sig["stop_price"]),
                        "initial_stop": float(sig["stop_price"]),
                        "initial_stop_price": float(sig["stop_price"]),
                        "target_price": float(sig["target_price"]),
                        "risk_capital": equity * ledger["account"]["risk_pct"],
                        "unrealized_r": 0.0,
                        "mfe_r": 0.0,
                        "bars_held": 0,
                        "last_processed_bar_ts": str(entry_time),
                        "seal_sha256": sig.get("seal_sha256"),
                        "kreil": sig.get("kreil"),
                    }
                    open_positions.append(pos)
                    existing_ids.add(oid)
                    continue
        # Supersede a stale resting order for the same symbol/tf/arch/direction
        # so a fresh bar is never blocked by an older pending order.
        key = _pending_supersede_key(sig)
        kept = []
        superseded = []
        for order in pending:
            if _pending_supersede_key(order) == key and _supersedes(order, sig):
                superseded.append(order)
                continue
            kept.append(order)
        if superseded:
            pending[:] = kept
            for order in superseded:
                existing_ids.discard(order.get("order_id"))
        # Enforce MAX_PENDING_PER_SYMBOL: if a resting order for this symbol
        # survived the supersede check it is either newer or was not displaced
        # by a clean shift, so the fresh signal is dropped instead of stacked.
        if len([o for o in pending if o.get("symbol") == sig.get("symbol")]) >= MAX_PENDING_PER_SYMBOL:
            existing_ids.add(oid)
            continue
        pending.append(sig)
        existing_ids.add(oid)
    return ledger


def main():
    t0 = time.time()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "dashboard").mkdir(parents=True, exist_ok=True)

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
    if not isinstance(open_positions, list):
        open_positions = []
        ledger["open_positions"] = open_positions
    enforce_risk_pct(ledger)
    update_block_stats(ledger)

    step_a_update_positions(ledger, data)
    step_b_fill_pending(ledger, data, open_positions, enforce_kreil=True)

    existing_ids = set()
    for o in ledger["pending_orders"]:
        existing_ids.add(o["order_id"])
    for p in open_positions:
        existing_ids.add(p["order_id"])
    for c in ledger["closed_trades"]:
        existing_ids.add(c["order_id"])

    all_signals = []
    k = ledger["block_stats"].get("consecutive_losses", 0)
    s_min = get_s_min(k)
    min_rr = get_min_rr(k)
    blocked = ledger["block_stats"].get("blocked", False)

    for market, info in WATCHLIST.items():
        for symbol in info["symbols"]:
            for tf in info["tfs"]:
                df = data.get((symbol, tf))
                if df is None:
                    continue
                sigs = run_detectors(symbol, df, tf, market)
                for sig in sigs:
                    if blocked:
                        continue
                    if in_cooldown(ledger, sig["symbol"], sig["direction"], sig["placed_bar_time"], sig["tf"]):
                        continue
                    passed_kr, kreil_info = passes_kreil_regime_gate(df, sig["direction"], sig["arch"])
                    if not passed_kr:
                        continue
                    kreil_info = dict(kreil_info)
                    if kreil_info.get("score_boost", 0.0) > 0:
                        kreil_info["boost_applied"] = True
                    sig["kreil"] = kreil_info
                    if sig["score"] < s_min or float(sig.get("rr", 0.0)) < min_rr:
                        continue
                    all_signals.append(sig)

    # dedupe by order_id
    seen = set()
    uniq = []
    for s in all_signals:
        if s["order_id"] in seen:
            continue
        seen.add(s["order_id"])
        uniq.append(s)

    intake_signals(ledger, data, uniq, open_positions, existing_ids)

    # cleanup + dedup pending
    # expire orders once they exceed their timeframe's resting lifetime
    new_pending = []
    for order in ledger["pending_orders"]:
        df = data.get((order["symbol"], order["tf"]))
        if df is not None:
            bars = bars_after(df, order["placed_bar_time"])
            if len(bars) > max_pending_bars(order.get("tf", "1d")):
                continue
            ok, kreil_info = passes_kreil_regime_gate(df, order.get("direction", ""), order.get("arch", ""))
            if not ok:
                continue
            order["kreil"] = kreil_info
            base = float(order.get("base_score", order.get("score", 0.0)))
            order["base_score"] = round(base, 4)
            order["score"] = round(min(4.0, base + float(kreil_info.get("score_boost", 0.0) or 0.0)), 4)
            if kreil_info.get("score_boost", 0.0) > 0:
                kreil_info["boost_applied"] = True
        new_pending.append(order)
    ledger["pending_orders"] = new_pending
    # dedup per symbol keep highest score
    by_symbol = {}
    for order in ledger["pending_orders"]:
        sym = order["symbol"]
        if sym not in by_symbol or order.get("score", 0) > by_symbol[sym].get("score", 0):
            by_symbol[sym] = order
    deduped = list(by_symbol.values())
    deduped.sort(key=lambda x: x.get("score", 0), reverse=True)
    deduped = deduped[:MAX_PENDING_GLOBAL]
    ledger["pending_orders"] = deduped

    step_a_update_positions(ledger, data)

    # mark-to-market
    for pos in open_positions:
        df = data.get((pos["symbol"], pos["tf"]))
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        last = df.iloc[-1]
        C = float(last["Close"])
        risk_price = (pos["entry_price"] - pos.get("initial_stop_price", pos["initial_stop"])) if pos["direction"] == "long" else (pos.get("initial_stop_price", pos["initial_stop"]) - pos["entry_price"])
        if risk_price > 0:
            if pos["direction"] == "long":
                pos["unrealized_r"] = (C - pos["entry_price"]) / risk_price
            else:
                pos["unrealized_r"] = (pos["entry_price"] - C) / risk_price
    ledger["open_positions"] = open_positions

    # spot prices
    spot_prices = {}
    for (sym, tf), df in data.items():
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        try:
            price = float(df.iloc[-1]["Close"])
            ts = str(df.iloc[-1]["Time"]) if "Time" in df.columns else str(df.index[-1])
        except Exception:
            continue
        if sym not in spot_prices:
            spot_prices[sym] = {"price": price, "ts": ts, "tf": tf}
        else:
            cur_tf = spot_prices[sym].get("tf")
            if (cur_tf == "1d" and tf in ("15m", "1h")) or (cur_tf == "1h" and tf == "15m"):
                spot_prices[sym] = {"price": price, "ts": ts, "tf": tf}
    ledger["spot_prices"] = spot_prices

    for pos in ledger["open_positions"]:
        df = data.get((pos.get("symbol"), pos.get("tf")))
        if df is not None and not (hasattr(df, "empty") and df.empty):
            try:
                pos["last_price"] = float(df.iloc[-1]["Close"])
            except Exception:
                pass
            if "kreil" not in pos:
                try:
                    pos["kreil"] = passes_kreil_regime_gate(df, pos.get("direction", ""), pos.get("arch", ""))[1]
                except Exception:
                    pass
    # stamp pending orders
    for p in ledger["pending_orders"]:
        df = data.get((p.get("symbol"), p.get("tf")))
        if df is not None and not (hasattr(df, "empty") and df.empty):
            try:
                p["last_price"] = float(df.iloc[-1]["Close"])
            except Exception:
                pass
            if "kreil" not in p:
                try:
                    p["kreil"] = passes_kreil_regime_gate(df, p.get("direction", ""), p.get("arch", ""))[1]
                except Exception:
                    pass

    ledger["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    update_block_stats(ledger)

    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")

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
    lines.append(f"- Risk %: {acct['risk_pct']*100:.1f}% (max {MAX_RISK_PCT*100:.1f}%)")
    lines.append(f"- Open positions: {len(ledger['open_positions'])} / {MAX_OPEN_POSITIONS} (max 1 per market bucket)")
    lines.append(f"- Pending orders: {len(ledger['pending_orders'])}")
    lines.append(f"- Closed trades: {len(ledger['closed_trades'])}")
    lines.append(f"- Block stats: {ledger['block_stats']}")
    lines.append(f"- Dynamic thresholds: s_min {ledger['block_stats'].get('s_min_current')} | min_rr {ledger['block_stats'].get('min_rr_current')}")
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


if __name__ == "__main__":
    main()