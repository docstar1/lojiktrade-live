"""Standalone live forward scanner (upgraded).

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
from typing import Dict, Any, Tuple, Optional

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

MAX_OPEN_POSITIONS = 1
MAX_PENDING_GLOBAL = 8
MAX_PENDING_PER_SYMBOL = 1
COOLDOWN_BARS = 5
MAX_RISK_PCT = 0.02
ALLOWED_ARCHS = {"T10_SOROS", "F1_SOROS_MA", "MA7_TRUE", "P3_SWEEP", "F3_BREAKOUT"}
VALID_ARCHS = ALLOWED_ARCHS


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
    if delta < 0.012:
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

        # --- T10_SOROS ---
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
                        target = entry + 4.2 * R
                        direction = "long"
                    else:
                        target = entry - 4.2 * R
                        direction = "short"
                    score = 3.6 + 0.15 * min(max_zv, 5.0)
                    s = build_signal(sym, market, tf, "T10_SOROS", direction, ts, entry, stop, target, score)
                    if s:
                        t10_candidates.append(s)
            if t10_candidates:
                # keep highest score
                best = max(t10_candidates, key=lambda x: x["score"])
                out.append(best)
        except Exception:
            pass

        # --- F1_SOROS_MA ---
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
                    target = entry + 3.8 * R
                    direction = "long"
                else:
                    stop = ema30 + 1.2 * atr
                    entry = close
                    R = abs(entry - stop)
                    if R <= 0:
                        continue
                    target = entry - 3.8 * R
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
                            target = entry + 3.5 * R
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
                            target = entry - 3.5 * R
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
                                target = entry + 3.6 * R
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
                                target = entry - 3.6 * R
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
                        target = entry + 3.8 * R
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
                        target = entry - 3.8 * R
                        score = 2.75 + 0.1 * min(zv, 4.0)
                        s = build_signal(sym, market, tf, "F3_BREAKOUT", "short", ts, entry, stop, target, score)
                        if s:
                            f3_list.append(s)
            if f3_list:
                out.append(f3_list[0])
        except Exception:
            pass

    # attach kreil metadata
    for s in out:
        try:
            if "kreil" not in s:
                _, kr_info = passes_kreil_regime_gate(df, s.get("direction", ""), s.get("arch", ""))
                s["kreil"] = kr_info
                s["base_score"] = round(float(s["score"]), 4)
                if kr_info.get("score_boost", 0.0) > 0:
                    s["score"] = round(min(4.0, float(s["score"]) + float(kr_info["score_boost"])), 4)
                    kr_info["boost_applied"] = True
        except Exception:
            pass
    # keep single highest-score signal per (sym, tf)
    if not out:
        return []
    if keep_all:
        return out
    best = max(out, key=lambda x: x["score"])
    return [best]


# ---- Risk / Elder helpers ----
def get_s_min(k):
    k = int(k) if k is not None else 0
    a = 2.6 + 0.45 * k
    b = 2.0 + 0.75 * k
    return max(a, b)


def s_min_floor(k):
    return get_s_min(k)


def get_min_rr(k):
    k = int(k) if k is not None else 0
    return 3.2 + 0.4 * k


def enforce_risk_pct(ledger):
    acct = ledger.setdefault("account", {})
    rp = acct.get("risk_pct", 0.02)
    try:
        rp = float(rp)
    except (TypeError, ValueError):
        rp = 0.02
    acct["risk_pct"] = min(rp, MAX_RISK_PCT)
    if acct["risk_pct"] <= 0:
        acct["risk_pct"] = 0.02
    if "equity" not in acct:
        acct["equity"] = 100.0
    if "target" not in acct:
        acct["target"] = 1000.0
    if "month_start_equity" not in acct:
        acct["month_start_equity"] = float(acct.get("equity", 100.0))
    if "current_month" not in acct:
        acct["current_month"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
    return ledger


def update_block_stats(ledger):
    bs = ledger.setdefault("block_stats", {})
    acct = ledger.setdefault("account", {})
    bs.setdefault("consecutive_losses", 0)
    bs.setdefault("blocked", False)
    bs.setdefault("cooldowns", [])
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
    month_start = float(acct.get("month_start_equity", 100.0))
    if month_start <= 0:
        month_start = 100.0
        acct["month_start_equity"] = month_start
    dd = (month_start - equity) / month_start
    bs["monthly_dd_pct"] = float(dd)
    k = int(bs.get("consecutive_losses", 0))
    bs["s_min_current"] = float(get_s_min(k))
    bs["min_rr_current"] = round(get_min_rr(k), 4)
    blocked = False
    reason = None
    if dd >= 0.06:
        blocked = True
        reason = "ELDER_6PCT_PIRANHA_GUARD"
    if k >= 3:
        blocked = True
        reason = "ELDER_6PCT_PIRANHA_GUARD"
    bs["blocked"] = blocked
    if blocked:
        bs["block_reason"] = reason
    else:
        bs.pop("block_reason", None)
    return ledger


def load_ledger():
    if LEDGER_PATH.exists():
        try:
            L = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
            enforce_risk_pct(L)
            update_block_stats(L)
            return L
        except Exception:
            pass
    L = {
        "account": {"equity": 100.0, "risk_pct": 0.02, "target": 1000.0, "month_start_equity": 100.0, "current_month": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")},
        "open_positions": [],
        "pending_orders": [],
        "closed_trades": [],
        "block_stats": {"consecutive_losses": 0, "blocked": False, "cooldowns": [], "s_min_current": 2.6, "min_rr_current": 3.2, "monthly_dd_pct": 0.0},
        "last_run": None,
    }
    enforce_risk_pct(L)
    update_block_stats(L)
    return L


def bars_after(df, time_str):
    try:
        mask = df["Time"].astype(str) > str(time_str)
        return df[mask]
    except Exception:
        try:
            return df[df.index.astype(str) > str(time_str)]
        except Exception:
            return df.iloc[0:0]


def try_fill(order, df, tf=None):
    otf = tf if tf is not None else order["tf"]
    tf_sec = TF_SEC.get(otf, 3600)
    bars = bars_after(df, order["placed_bar_time"])
    for _, bar in bars.iterrows():
        H = float(bar["High"]); L = float(bar["Low"])
        buf = 0.15 * (H - L) * math.sqrt(2.0 / tf_sec) if (H - L) > 0 else 0.0
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


def _cooldown_hours(tf):
    if tf == "1d":
        return 48.0
    if tf == "1h":
        return 6.0
    return (TF_SEC.get(tf, 3600) * 5) / 3600.0


def record_cooldown(ledger, symbol, direction, tf, exit_time_str):
    exit_ts = _ts_naive(exit_time_str)
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


def _ts_naive(x):
    if x is None:
        return None
    try:
        ts = pd.Timestamp(x)
    except Exception:
        return None
    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
    except Exception:
        try:
            ts = pd.Timestamp(x).tz_localize(None)
        except Exception:
            pass
    return ts


def in_cooldown(ledger, symbol, direction, when_time_str, tf=None):
    when = _ts_naive(when_time_str)
    if when is None:
        return False
    # block_stats cooldowns
    for cd in ledger.get("block_stats", {}).get("cooldowns", []):
        if cd.get("symbol") == symbol and cd.get("direction") == direction:
            until = _ts_naive(cd.get("until"))
            if until is not None and when <= until:
                return True
            # fallback seconds check against stored until
            try:
                if until is not None and (when - until).total_seconds() < 0:
                    return True
            except Exception:
                pass
    # closed trades
    for ct in ledger.get("closed_trades", []):
        if ct.get("symbol") == symbol and ct.get("direction") == direction:
            if (ct.get("pnl_r", 0) or 0) < 0:
                hours = _cooldown_hours(ct.get("tf", tf or "1d"))
                exit_ts = _ts_naive(ct.get("exit_time"))
                cu = ct.get("cooldown_until")
                expiry = _ts_naive(cu) if cu is not None else None
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


def step_a_update_positions(ledger, data):
    open_positions = ledger["open_positions"]
    for pos in list(open_positions):
        df = data.get((pos["symbol"], pos["tf"]))
        if df is None:
            continue
        bars = bars_after(df, pos["entry_time"])
        exited = False
        exit_price = None
        reason = None
        cost_pct, slip_pct = get_costs(pos["market"])
        pos.setdefault("initial_stop", pos["stop_price"])
        risk_price = (pos["entry_price"] - pos["initial_stop"]) if pos["direction"] == "long" else (pos["initial_stop"] - pos["entry_price"])
        if risk_price <= 0:
            continue
        for bar_idx, (_, bar) in enumerate(bars.iterrows()):
            H = float(bar["High"]); L = float(bar["Low"]); C = float(bar["Close"])
            bar_time = str(bar["Time"])
            if pos["direction"] == "long":
                # Exit checks against this bar's range BEFORE any ratchet from Close
                if L <= pos["stop_price"]:
                    exit_price = pos["stop_price"]; reason = "stop"; exited = True
                elif H >= pos["target_price"]:
                    exit_price = pos["target_price"]; reason = "target"; exited = True
                else:
                    unreal_r = (C - pos["entry_price"]) / risk_price
                    # ratchet (applies to subsequent bars only)
                    if unreal_r >= 1.2:
                        new_stop = pos["entry_price"] + 0.30 * risk_price
                        if new_stop > pos["stop_price"]:
                            pos["stop_price"] = new_stop
                    if unreal_r >= 2.2:
                        arch = pos.get("arch", "")
                        sub = bars.iloc[:bar_idx + 1]
                        highest_high = max(float(r["High"]) for _, r in sub.iterrows())
                        lowest_low = min(float(r["Low"]) for _, r in sub.iterrows())
                        try:
                            atr_val = float(bar.get("atr", risk_price)) if hasattr(bar, "get") else float(bar["atr"])
                        except Exception:
                            try:
                                atr_val = float(bar["atr"])
                            except Exception:
                                atr_val = risk_price
                        if not np.isfinite(atr_val) or atr_val <= 0:
                            atr_val = risk_price
                        if arch in ("F3_BREAKOUT", "T10_SOROS"):
                            new_stop = max(pos["entry_price"] + 1.20 * risk_price,
                                          highest_high - 2.0 * atr_val)
                        else:
                            new_stop = pos["entry_price"] + 1.20 * risk_price
                        if new_stop > pos["stop_price"]:
                            pos["stop_price"] = new_stop
            else:
                if H >= pos["stop_price"]:
                    exit_price = pos["stop_price"]; reason = "stop"; exited = True
                elif L <= pos["target_price"]:
                    exit_price = pos["target_price"]; reason = "target"; exited = True
                else:
                    unreal_r = (pos["entry_price"] - C) / risk_price
                    if unreal_r >= 1.2:
                        new_stop = pos["entry_price"] - 0.30 * risk_price
                        if new_stop < pos["stop_price"]:
                            pos["stop_price"] = new_stop
                    if unreal_r >= 2.2:
                        arch = pos.get("arch", "")
                        sub = bars.iloc[:bar_idx + 1]
                        highest_high = max(float(r["High"]) for _, r in sub.iterrows())
                        lowest_low = min(float(r["Low"]) for _, r in sub.iterrows())
                        try:
                            atr_val = float(bar.get("atr", risk_price)) if hasattr(bar, "get") else float(bar["atr"])
                        except Exception:
                            try:
                                atr_val = float(bar["atr"])
                            except Exception:
                                atr_val = risk_price
                        if not np.isfinite(atr_val) or atr_val <= 0:
                            atr_val = risk_price
                        if arch in ("F3_BREAKOUT", "T10_SOROS"):
                            new_stop = min(pos["entry_price"] - 1.20 * risk_price,
                                          lowest_low + 2.0 * atr_val)
                        else:
                            new_stop = pos["entry_price"] - 1.20 * risk_price
                        if new_stop < pos["stop_price"]:
                            pos["stop_price"] = new_stop

            if exited:
                break

        if not exited:
            # mark to market only
            if len(bars) > 0:
                last = bars.iloc[-1]
            else:
                last = df.iloc[-1]
            C = float(last["Close"])
            if pos["direction"] == "long":
                if risk_price > 0:
                    pos["unrealized_r"] = (C - pos["entry_price"]) / risk_price
            else:
                if risk_price > 0:
                    pos["unrealized_r"] = (pos["entry_price"] - C) / risk_price
            continue

        # exited
        if pos["direction"] == "long":
            pnl_r = (exit_price - pos["entry_price"]) / risk_price
        else:
            pnl_r = (pos["entry_price"] - exit_price) / risk_price
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
        # cooldown on loss
        if pnl_r < 0:
            record_cooldown(ledger, pos["symbol"], pos["direction"], pos["tf"], str(bar["Time"]) if exited else pos["entry_time"])
        ledger["closed_trades"].append({
            "order_id": pos["order_id"],
            "symbol": pos["symbol"],
            "market": pos["market"],
            "tf": pos["tf"],
            "arch": pos["arch"],
            "direction": pos["direction"],
            "entry_time": pos["entry_time"],
            "exit_time": str(bar["Time"]) if exited else None,
            "entry_price": pos["entry_price"],
            "exit_price": float(exit_price),
            "pnl_r": round(float(pnl_r), 4),
            "reason": reason,
            "cooldown_until": None,
        })
        # set cooldown_until
        try:
            exit_ts = pd.Timestamp(bar["Time"])
            delta = pd.Timedelta(hours=_cooldown_hours(pos["tf"]))
            ledger["closed_trades"][-1]["cooldown_until"] = str(exit_ts + delta)
        except Exception:
            pass
        open_positions.remove(pos)
        update_block_stats(ledger)

    # mark-to-market remaining
    for pos in open_positions:
        df = data.get((pos["symbol"], pos["tf"]))
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        last = df.iloc[-1]
        C = float(last["Close"])
        risk_price = (pos["entry_price"] - pos["initial_stop"]) if pos["direction"] == "long" else (pos["initial_stop"] - pos["entry_price"])
        if risk_price > 0:
            if pos["direction"] == "long":
                pos["unrealized_r"] = (C - pos["entry_price"]) / risk_price
            else:
                pos["unrealized_r"] = (pos["entry_price"] - C) / risk_price


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
                if kr_info.get("score_boost", 0.0) > 0 and not order.get("kreil", {}).get("boost_applied"):
                    order["base_score"] = round(float(order["score"]), 4)
                    order["score"] = round(min(4.0, float(order["score"]) + float(kr_info["score_boost"])), 4)
                    kr_info["boost_applied"] = True
                order["kreil"] = kr_info
        if float(order.get("score", 0.0)) < s_min or float(order.get("rr", 0.0)) < min_rr:
            continue
        filtered_pending.append(order)
    ledger["pending_orders"] = filtered_pending

    # dedup pending by symbol (keep highest score) and cap
    # expire old (>3 bars)
    new_pending = []
    for order in ledger["pending_orders"]:
        df = data.get((order["symbol"], order["tf"]))
        if df is not None:
            bars = bars_after(df, order["placed_bar_time"])
            if len(bars) > 3:
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
        if len(bars) > 3:
            continue
        filled, entry_time, entry_price = try_fill(order, df)
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
        if len(open_positions) >= MAX_OPEN_POSITIONS:
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
        # once filled, do not fill more (global lock)
        # mark remaining pending kept
    # rebuild still_pending excluding filled
    filled_ids = {p["order_id"] for p in open_positions}
    # Actually open_positions may include pre-existing; track by processing. Simpler: keep pending not filled.
    # We appended to still_pending during loop only non-filled. But if global lock blocked, we appended. Good.
    # However still_pending currently contains non-filled appended. But we iterated ordered; need ensure no duplicates.
    # Reconstruct pending as still_pending filtered to deduped set.
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


# alias
step_b_pending = step_b_fill_pending


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
    step_a_update_positions(ledger, data)

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
                    if kreil_info.get("score_boost", 0.0) > 0 and not sig.get("kreil", {}).get("boost_applied"):
                        sig["base_score"] = round(float(sig["score"]), 4)
                        sig["score"] = round(min(4.0, float(sig["score"]) + float(kreil_info["score_boost"])), 4)
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

    for sig in uniq:
        if sig["order_id"] in existing_ids:
            continue
        if float(sig.get("score", 0.0)) < get_s_min(k) or float(sig.get("rr", 0.0)) < get_min_rr(k):
            continue
        df = data.get((sig["symbol"], sig["tf"]))
        if df is None:
            continue
        # global single position
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            # queue as pending instead
            pass
        market_positions = open_positions  # global
        filled, entry_time, entry_price = try_fill(sig, df, sig["tf"])
        if filled and not is_rejected(sig["order_id"], entry_time) and len(open_positions) < MAX_OPEN_POSITIONS and not ledger["block_stats"].get("blocked"):
            if in_cooldown(ledger, sig["symbol"], sig["direction"], entry_time, sig["tf"]):
                filled = False
            else:
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
                        "kreil": sig.get("kreil"),
                    }
                    open_positions.append(pos)
                    existing_ids.add(sig["order_id"])
                    continue
        # else pending
        ledger["pending_orders"].append(sig)
        existing_ids.add(sig["order_id"])

    # cleanup + dedup pending
    # expire old
    new_pending = []
    for order in ledger["pending_orders"]:
        df = data.get((order["symbol"], order["tf"]))
        if df is not None:
            bars = bars_after(df, order["placed_bar_time"])
            if len(bars) > 3:
                continue
            ok, kreil_info = passes_kreil_regime_gate(df, order.get("direction", ""), order.get("arch", ""))
            if not ok:
                continue
            order["kreil"] = kreil_info
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
        risk_price = (pos["entry_price"] - pos["initial_stop"]) if pos["direction"] == "long" else (pos["initial_stop"] - pos["entry_price"])
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

    # stamp last_price and kreil on open positions
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
