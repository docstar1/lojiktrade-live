import sys, os, json, math, datetime
from collections import Counter
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
try:
    import scanner_worker as sw
except ModuleNotFoundError:
    import live_forward_scanner as sw


def make_df(closes, highs=None, lows=None, opens=None, volumes=None, times=None):
    n = len(closes)
    if highs is None: highs = [c + 1.0 for c in closes]
    if lows is None: lows = [c - 1.0 for c in closes]
    if opens is None: opens = list(closes)
    if volumes is None: volumes = [1000.0] * n
    if times is None: times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(n)]
    df = pd.DataFrame({
        "Time": times,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })
    return sw.compute_indicators(df)


def make_base(n=80):
    closes = [100.0] * n
    highs = [101.0] * n
    lows = [99.0] * n
    opens = [100.0] * n
    volumes = [1000.0] * n
    times = [f"t{i:04d}" for i in range(n)]
    df = pd.DataFrame({
        "Time": times,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })
    return sw.compute_indicators(df)


def make_kreil_boost_df(up=True):
    n = 250
    closes, highs, lows, opens = [], [], [], []
    for i in range(n):
        if 50 <= i <= 59:
            if up:
                c, h, l = 130.0, 131.0, 99.0
            else:
                c, h, l = 70.0, 101.0, 69.0
        else:
            c, h, l = 100.01, 101.0, 99.0
        closes.append(c)
        highs.append(h)
        lows.append(l)
        opens.append(100.0)
    return make_df(closes, highs=highs, lows=lows, opens=opens,
                   times=[f"t{i:04d}" for i in range(n)])


def make_kreil_neutral_df():
    n = 250
    closes, highs, lows, opens = [], [], [], []
    for i in range(n):
        if 50 <= i <= 54:
            c, h, l = 130.0, 131.0, 99.0
        elif 55 <= i <= 59:
            c, h, l = 70.0, 101.0, 69.0
        else:
            c, h, l = 100.0, 101.0, 99.0
        closes.append(c)
        highs.append(h)
        lows.append(l)
        opens.append(100.0)
    return make_df(closes, highs=highs, lows=lows, opens=opens,
                   times=[f"t{i:04d}" for i in range(n)])


DEFAULTS = {
    "Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1000.0,
    "ema9": 100.0, "ema30": 100.0, "ema50": 100.0, "sma200": 100.0, "sma200_prev5": 99.0,
    "atr": 3.0, "z_v": 0.5, "vol_ratio": 0.9, "rsi": 50.0, "h20": 105.0, "l20": 95.0,
    "h10": 103.0, "l10": 97.0, "er10": 0.1, "atr_ratio": 0.5,
}


def set_row(df, idx, **overrides):
    vals = dict(DEFAULTS)
    vals.update(overrides)
    for col, val in vals.items():
        df.iloc[idx, df.columns.get_loc(col)] = val


def _base_ledger():
    return {
        "account": {"equity": 100.0, "risk_pct": 0.02, "target": 1000.0,
                    "month_start_equity": 100.0, "current_month": "2026-10"},
        "open_positions": [],
        "pending_orders": [],
        "closed_trades": [],
        "block_stats": {"consecutive_losses": 0, "blocked": False, "cooldowns": [],
                        "s_min_current": 2.6, "monthly_dd_pct": 0.0},
        "last_run": None,
    }


# Test 1
def test_1_no_e1():
    assert "E1" not in sw.ALLOWED_ARCHS
    assert sw.VALID_ARCHS == {"T10_SOROS", "F1_SOROS_MA", "MA7_TRUE", "P3_SWEEP", "F3_BREAKOUT"}
    cand = [os.path.join(ROOT, "scanner_worker.py"),
            os.path.join(ROOT, "scripts", "live_forward_scanner.py")]
    src = ""
    for c in cand:
        if os.path.exists(c):
            src = open(c).read()
            break
    assert src, "scanner source not found"
    assert '"E1"' not in src
    # signal-producing setup (F3 breakout) so the arch assertions are not vacuous
    n = 80
    df = make_base(n)
    for k in range(n - 6, n):
        set_row(df, k, ema9=112.0, ema30=109.0, sma200=100.0, sma200_prev5=99.0,
                Close=100.0, High=100.0, Low=100.0, rsi=50.0, atr=3.0,
                z_v=0.5, h20=105.0, l10=97.0, l20=95.0, vol_ratio=0.9, er10=0.1, atr_ratio=0.5)
    set_row(df, n - 1, Close=115.0, High=115.0, Low=108.0, ema9=112.0, ema30=109.0,
            sma200=100.0, sma200_prev5=99.0, atr=3.0, z_v=2.2, rsi=60.0, h20=112.0, l10=109.0)
    sigs = sw.run_detectors("AMD", df, "1d", "US", keep_all=True)
    assert len(sigs) > 0
    assert all(s["arch"] in sw.ALLOWED_ARCHS for s in sigs)
    assert all(s["arch"] != "E1" for s in sigs)
    print("Test 1 PASS")


# Test 2
def test_2_ma7_true():
    n = 80
    # negative: never touches ema30
    df = make_base(n)
    for k in range(n - 6, n):
        set_row(df, k, ema9=112.0, ema30=108.0, sma200=100.0, sma200_prev5=99.0,
                Low=114.0, Close=115.0, High=115.0, rsi=52.0, atr=3.0,
                z_v=0.5, h20=116.0, l20=95.0, er10=0.1, atr_ratio=0.5)
    sigs = sw.run_detectors("AAPL", df, "1d", keep_all=True)
    assert len(sigs) == 0, [s["arch"] for s in sigs]

    # positive
    df2 = make_base(n)
    for k in range(n - 6, n):
        set_row(df2, k, ema9=112.0, ema30=108.0, sma200=100.0, sma200_prev5=99.0,
                Low=114.0, Close=115.0, High=115.0, rsi=52.0, atr=3.0,
                z_v=0.5, h20=116.0, l20=95.0, er10=0.1, atr_ratio=0.5)
    set_row(df2, n - 3, High=116.0, ema30=108.0, atr=3.0, z_v=0.5)
    set_row(df2, n - 2, ema9=112.0, ema30=108.0, sma200=100.0, sma200_prev5=99.0,
            Low=114.0, Close=115.0, High=115.0, rsi=52.0, atr=3.0, z_v=0.5)
    set_row(df2, n - 1, ema9=110.0, ema30=108.0, sma200=100.0, sma200_prev5=99.0,
            Low=108.2, Close=109.0, High=109.5, rsi=52.0, atr=3.0, z_v=0.5)
    sigs2 = sw.run_detectors("AAPL", df2, "1d", keep_all=True)
    assert any(s["arch"] == "MA7_TRUE" for s in sigs2), [s["arch"] for s in sigs2]
    print("Test 2 PASS")


# Test 3
def test_3_f3_p3():
    n = 80
    # negative F3: Close>h20 but z_v=0.3
    df = make_base(n)
    for k in range(n - 6, n):
        set_row(df, k, ema9=112.0, ema30=109.0, sma200=100.0, sma200_prev5=99.0,
                Close=115.0, High=115.0, Low=108.0, rsi=60.0, atr=3.0,
                z_v=0.3, h20=112.0, l10=109.0, l20=95.0, vol_ratio=0.9, er10=0.1, atr_ratio=0.5)
    sigs = sw.run_detectors("AAPL", df, "1d", keep_all=True)
    assert not any(s["arch"] == "F3_BREAKOUT" for s in sigs)

    # overbought rsi=78
    df2 = make_base(n)
    for k in range(n - 6, n):
        set_row(df2, k, ema9=112.0, ema30=109.0, sma200=100.0, sma200_prev5=99.0,
                Close=115.0, High=115.0, Low=108.0, rsi=78.0, atr=3.0,
                z_v=2.2, h20=112.0, l10=109.0, l20=95.0, vol_ratio=0.9, er10=0.1, atr_ratio=0.5)
    sigs2 = sw.run_detectors("AAPL", df2, "1d", keep_all=True)
    assert not any(s["arch"] == "F3_BREAKOUT" for s in sigs2)

    # positive F3
    df3 = make_base(n)
    for k in range(n - 6, n):
        set_row(df3, k, ema9=112.0, ema30=109.0, sma200=100.0, sma200_prev5=99.0,
                Close=100.0, High=100.0, Low=100.0, rsi=50.0, atr=3.0,
                z_v=0.5, h20=105.0, l10=97.0, l20=95.0, vol_ratio=0.9, er10=0.1, atr_ratio=0.5)
    set_row(df3, n - 1, Close=115.0, High=115.0, Low=108.0, ema9=112.0, ema30=109.0,
            sma200=100.0, sma200_prev5=99.0, atr=3.0, z_v=2.2, rsi=60.0,
            h20=112.0, l10=109.0)
    sigs3 = sw.run_detectors("AAPL", df3, "1d", keep_all=True)
    f3 = [s for s in sigs3 if s["arch"] == "F3_BREAKOUT"]
    assert f3 and abs(f3[0]["rr"] - 2.2) < 1e-6, [s["arch"] for s in sigs3]

    # positive P3
    df4 = make_base(n)
    for k in range(n - 6, n):
        set_row(df4, k, ema9=100.0, ema30=100.0, sma200=100.0, sma200_prev5=99.0,
                Close=100.0, High=100.0, Low=100.0, rsi=45.0, atr=3.0,
                z_v=0.5, h20=108.0, l20=100.0, vol_ratio=0.9, er10=0.1, atr_ratio=0.5)
    set_row(df4, n - 1, l20=105.0, Low=103.5, Close=105.5, atr=3.0, z_v=2.0, rsi=45.0,
            sma200=100.0, ema9=100.0, ema30=100.0, h20=108.0)
    sigs4 = sw.run_detectors("AAPL", df4, "1d", keep_all=True)
    p3 = [s for s in sigs4 if s["arch"] == "P3_SWEEP"]
    assert p3 and abs(p3[0]["rr"] - 2.1) < 1e-6, [s["arch"] for s in sigs4]
    print("Test 3 PASS")


# Test 4
def test_4_soros():
    n = 80
    # T10 positive
    df = make_base(n)
    for k in range(n - 12, n):
        set_row(df, k, z_v=0.5, vol_ratio=0.5, atr=3.0, er10=0.1, atr_ratio=0.5,
                sma200=98.0, ema9=100.0, ema30=100.0, rsi=50.0, Close=100.0,
                High=100.0, Low=100.0, Open=100.0, h20=105.0, l20=95.0)
    set_row(df, n - 5, Open=100.0, Close=110.0, z_v=2.8, High=110.0, Low=100.0, atr=3.0)
    for k in range(n - 4, n):
        set_row(df, k, High=107.5, Low=105.0, Close=106.0, vol_ratio=0.5, z_v=0.5,
                atr=3.0, Open=106.0)
    set_row(df, n - 1, atr=3.0, er10=0.45, atr_ratio=1.1, sma200=98.0, ema9=106.0,
            ema30=104.0, rsi=55.0, Close=106.0, High=107.5, Low=105.0)
    sigs = sw.run_detectors("AAPL", df, "1h", keep_all=True)
    t10 = [s for s in sigs if s["arch"] == "T10_SOROS"]
    assert t10 and abs(t10[0]["rr"] - 2.5) < 1e-6, [s["arch"] for s in sigs]

    # F1 positive
    df2 = make_base(n)
    for k in range(n - 12, n):
        set_row(df2, k, z_v=0.5, vol_ratio=0.9, atr=3.0, er10=0.1, atr_ratio=0.5,
                sma200=98.0, sma200_prev5=97.0, ema9=100.0, ema30=100.0, rsi=50.0,
                Close=100.0, High=100.0, Low=100.0, Open=100.0, h20=105.0, l20=95.0)
    set_row(df2, n - 6, Open=100.0, z_v=0.5)
    set_row(df2, n - 4, z_v=2.5)
    set_row(df2, n - 2, Close=110.0, z_v=0.5)
    set_row(df2, n - 1, Close=105.0, Low=104.5, ema30=104.8, sma200=98.0,
            sma200_prev5=97.0, atr=3.0, vol_ratio=0.60, rsi=50.0, z_v=0.5,
            ema9=106.0, Open=105.0, High=105.5)
    sigs2 = sw.run_detectors("AAPL", df2, "1h", keep_all=True)
    f1 = [s for s in sigs2 if s["arch"] == "F1_SOROS_MA"]
    assert f1 and abs(f1[0]["rr"] - 2.2) < 1e-6, [s["arch"] for s in sigs2]
    print("Test 4 PASS")


# Test 5
def test_5_elder_2pct_shark():
    L = {"account": {"equity": 100.0, "risk_pct": 0.05, "target": 1000.0}}
    sw.enforce_risk_pct(L)
    assert L["account"]["risk_pct"] == 0.015
    L2 = {"account": {"equity": 100.0, "risk_pct": 0.05, "target": 1000.0}}
    sw.enforce_risk_pct(L2)
    assert L2["account"]["risk_pct"] <= sw.MAX_RISK_PCT
    assert L2["account"]["risk_pct"] == 0.015
    # a legacy 2.0% ledger is clamped down to the new 1.5% cap
    L3 = {"account": {"equity": 100.0, "risk_pct": 0.02, "target": 1000.0}}
    sw.enforce_risk_pct(L3)
    assert L3["account"]["risk_pct"] == 0.015
    assert sw.MAX_RISK_PCT == 0.015
    assert sw.MAX_OPEN_POSITIONS == 3
    # 3 slots x 1.5% = 4.5% max simultaneous risk
    assert abs(sw.MAX_RISK_PCT * sw.MAX_OPEN_POSITIONS - 0.045) < 1e-12
    D = sw.default_ledger()
    assert D["account"]["risk_pct"] == 0.015
    assert D["block_stats"]["min_rr"] == 1.8
    assert D["block_stats"]["min_rr_current"] == 1.8
    assert D["block_stats"]["s_min_current"] == 2.6
    print("Test 5 PASS")


# Test 6
def test_6_piranha_3loss():
    L = {
        "account": {"equity": 100.0, "risk_pct": 0.02, "target": 1000.0, "month_start_equity": 100.0},
        "open_positions": [],
        "pending_orders": [],
        "closed_trades": [],
        "block_stats": {"consecutive_losses": 0, "blocked": False, "cooldowns": [], "s_min_current": 2.6, "monthly_dd_pct": 0.0},
        "last_run": None,
    }
    for _ in range(3):
        L["account"]["equity"] -= 2.0
        L["block_stats"]["consecutive_losses"] += 1
        L["closed_trades"].append({"pnl_r": -1.0, "symbol": "X", "direction": "long"})
    sw.update_block_stats(L)
    assert L["block_stats"]["blocked"] is True
    assert L["block_stats"].get("block_reason") == "ELDER_6PCT_PIRANHA_GUARD"
    # 6% drawdown case
    L2 = {
        "account": {"equity": 93.0, "risk_pct": 0.02, "target": 1000.0, "month_start_equity": 100.0},
        "open_positions": [],
        "pending_orders": [],
        "closed_trades": [],
        "block_stats": {"consecutive_losses": 0, "blocked": False, "cooldowns": [], "s_min_current": 2.6, "monthly_dd_pct": 0.0},
        "last_run": None,
    }
    sw.update_block_stats(L2)
    assert L2["block_stats"]["blocked"] is True
    # blocks fills
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(5)]
    closes = [100.0] * 5
    lows = [90.0] * 5
    highs = [101.0] * 5
    df = make_df(closes, highs=highs, lows=lows, times=times)
    pending = [{"order_id": "X|1d|MA7_TRUE|long|t", "symbol": "X", "market": "US", "tf": "1d", "arch": "MA7_TRUE",
                "direction": "long", "placed_bar_time": "2026-01-01 00:00:00", "limit_price": 99.0, "stop_price": 95.0,
                "target_price": 113.0, "score": 3.0, "seal_sha256": "abc", "rr": 3.75}]
    L["pending_orders"] = pending
    data = {("X", "1d"): df}
    L["pending_orders"][0]["placed_bar_time"] = "2026-01-01 00:00:00"
    sw.step_b_fill_pending(L, data, L["open_positions"])
    assert len(L["open_positions"]) == 0, "No position should open when blocked"
    print("Test 6 PASS")


# Test 7
def test_7_s_min():
    assert sw.get_s_min(0) < sw.get_s_min(1) < sw.get_s_min(2)
    assert sw.get_s_min(0) == 2.6
    assert sw.get_s_min(1) == 2.75
    assert sw.get_s_min(2) == 2.9
    assert sw.get_s_min(3) == 2.9
    assert sw.get_s_min(10) == 2.90, "s_min must be capped at 2.90 (MA7_TRUE stays tradable)"
    assert sw.compute_s_min(0) == sw.get_s_min(0)
    assert sw.compute_min_rr(0) == 1.8
    assert sw.compute_min_rr(1) == 1.9
    assert sw.compute_min_rr(2) == 2.0
    assert sw.compute_min_rr(10) == 2.0, "min_rr must be capped at 2.0 (MA7_TRUE is a 2.0R setup)"
    assert sw.s_min_floor(0) == sw.get_s_min(0)

    # purge test: k=1 raises s_min to 2.75, so a 2.8-score order is still tradable
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(5)]
    closes = [100.0] * 5
    lows = [90.0] * 5
    highs = [101.0] * 5
    df = make_df(closes, highs=highs, lows=lows, times=times)
    data = {("X", "1d"): df}
    order = {"order_id": "X|1d|MA7_TRUE|long|t", "symbol": "X", "market": "US", "tf": "1d", "arch": "MA7_TRUE",
             "direction": "long", "placed_bar_time": times[2], "limit_price": 99.0, "stop_price": 95.0,
             "target_price": 113.0, "score": 2.8, "seal_sha256": "abc", "rr": 3.3}
    L = _base_ledger()
    L["block_stats"]["consecutive_losses"] = 1
    sw.update_block_stats(L)
    assert L["block_stats"]["s_min_current"] == 2.75
    L["pending_orders"] = [dict(order)]
    sw.step_b_fill_pending(L, data, L["open_positions"])
    assert len(L["open_positions"]) == 1, "2.8 score must clear the k=1 floor of 2.75"

    # k=2 raises s_min to 2.90 -> the same 2.8-score order is pruned
    L2 = _base_ledger()
    L2["block_stats"]["consecutive_losses"] = 2
    sw.update_block_stats(L2)
    assert L2["block_stats"]["s_min_current"] == 2.9
    L2["pending_orders"] = [dict(order)]
    sw.step_b_fill_pending(L2, data, L2["open_positions"])
    assert len(L2["pending_orders"]) == 0, L2["pending_orders"]
    assert len(L2["open_positions"]) == 0, L2["open_positions"]
    print("Test 7 PASS")


# Test 8
def test_8_multi_slot_market_buckets():
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(5)]
    closes = [100.0] * 5
    lows = [90.0] * 5          # every order fills
    highs = [101.0] * 5
    df = make_df(closes, highs=highs, lows=lows, times=times)
    entries = [("BTC-USD", "Crypto"), ("INFY.NS", "IndiaNSE"), ("COIN", "US"), ("AMD", "US")]
    data = {(sym, "1d"): df for sym, _ in entries}
    L = _base_ledger()
    L["pending_orders"] = [
        {"order_id": f"{sym}|1d|MA7_TRUE|long|{times[2]}", "symbol": sym, "market": market, "tf": "1d",
         "arch": "MA7_TRUE", "direction": "long", "placed_bar_time": times[2], "limit_price": 99.0,
         "stop_price": 95.0, "target_price": 113.0, "score": 3.0 - i * 0.01, "seal_sha256": f"s{i}", "rr": 3.5}
        for i, (sym, market) in enumerate(entries)
    ]
    sw.step_b_fill_pending(L, data, L["open_positions"])
    # three concurrent slots, one per bucket; the second US order must wait
    assert len(L["open_positions"]) == 3, [p["symbol"] for p in L["open_positions"]]
    buckets = [sw.get_market_bucket(p["market"]) for p in L["open_positions"]]
    assert sorted(buckets) == ["Crypto", "IndiaNSE", "US_FX"], buckets
    opened = [p["symbol"] for p in L["open_positions"]]
    assert "AMD" not in opened, opened
    assert [o["symbol"] for o in L["pending_orders"]] == ["AMD"], [o["symbol"] for o in L["pending_orders"]]

    # a slow IndiaNSE position never blocks a fresh Crypto fill
    L2 = _base_ledger()
    L2["open_positions"] = [{
        "order_id": "HAL.NS|1d|MA7_TRUE|long|t", "symbol": "HAL.NS", "market": "IndiaNSE", "tf": "1d",
        "arch": "MA7_TRUE", "direction": "long", "entry_time": times[0], "entry_price": 100.0,
        "stop_price": 95.0, "initial_stop": 95.0, "target_price": 110.0, "risk_capital": 1.5,
        "unrealized_r": 0.0, "seal_sha256": "h",
    }]
    L2["pending_orders"] = [{"order_id": f"BTC-USD|1d|P3_SWEEP|long|{times[2]}", "symbol": "BTC-USD",
                             "market": "Crypto", "tf": "1d", "arch": "P3_SWEEP", "direction": "long",
                             "placed_bar_time": times[2], "limit_price": 99.0, "stop_price": 95.0,
                             "target_price": 113.0, "score": 3.2, "seal_sha256": "b", "rr": 3.5}]
    sw.step_b_fill_pending(L2, {("BTC-USD", "1d"): df}, L2["open_positions"])
    assert len(L2["open_positions"]) == 2, L2["open_positions"]
    assert L2["open_positions"][1]["symbol"] == "BTC-USD"
    print("Test 8 PASS")


# Test 9
def test_9_cooldown():
    L = {
        "account": {"equity": 100.0, "risk_pct": 0.02, "target": 1000.0, "month_start_equity": 100.0},
        "open_positions": [],
        "pending_orders": [],
        "closed_trades": [
            {"order_id": "AMD|1d|MA7_TRUE|long|t", "symbol": "AMD", "market": "US", "tf": "1d", "arch": "MA7_TRUE", "direction": "long", "entry_time": "2026-01-01", "exit_time": "2026-10-25T14:00:00+00:00", "entry_price": 100.0, "exit_price": 95.0, "pnl_r": -1.0, "reason": "stop"}
        ],
        "block_stats": {"consecutive_losses": 1, "blocked": False, "cooldowns": [], "s_min_current": 3.05, "monthly_dd_pct": 0.0},
        "last_run": None,
    }
    assert sw.in_cooldown(L, "AMD", "long", "2026-10-25 15:00:00", "1d") is True
    assert sw.in_cooldown(L, "NVDA", "long", "2026-10-25 15:00:00", "1d") is False

    L2 = {
        "account": {"equity": 100.0, "risk_pct": 0.02, "target": 1000.0, "month_start_equity": 100.0},
        "open_positions": [],
        "pending_orders": [],
        "closed_trades": [],
        "block_stats": {"consecutive_losses": 1, "blocked": False, "cooldowns": [
            {"symbol": "AMD", "direction": "long", "tf": "1d", "until": "2026-10-25T16:00:00+00:00"}
        ], "s_min_current": 3.05, "monthly_dd_pct": 0.0},
        "last_run": None,
    }
    assert sw.in_cooldown(L2, "AMD", "long", "2026-10-25 15:00:00", "1d") is True
    print("Test 9 PASS")


# Test 10
def test_10_pending_cap_schema():
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(5)]
    closes = [100.0] * 5
    # lows above the 99.0 limit => no fills, so the 8-order global cap is observable
    lows = [100.0] * 5
    highs = [101.0] * 5
    df = make_df(closes, highs=highs, lows=lows, times=times)
    data = {}
    orders = []
    for i in range(15):
        sym = f"S{i % 12}"
        orders.append({
            "order_id": f"{sym}|1d|MA7_TRUE|long|t{i}",
            "symbol": sym, "market": "US", "tf": "1d", "arch": "MA7_TRUE",
            "direction": "long", "placed_bar_time": times[2], "limit_price": 99.0,
            "stop_price": 95.0, "target_price": 113.0, "score": 2.6 + i * 0.01,
            "seal_sha256": "x", "rr": 3.5,
        })
        data[(sym, "1d")] = df
    L = _base_ledger()
    L["pending_orders"] = orders
    sw.step_b_fill_pending(L, data, L["open_positions"])
    po = L["pending_orders"]
    assert len(po) == 8, len(po)
    c = Counter(o["symbol"] for o in po)
    assert all(v <= 1 for v in c.values())

    # schema
    L2 = _base_ledger()
    for k in ("account", "open_positions", "pending_orders", "closed_trades", "block_stats", "last_run"):
        assert k in L2
    assert "risk_pct" in L2["account"]
    assert "month_start_equity" in L2["account"]
    assert "consecutive_losses" in L2["block_stats"]
    assert "blocked" in L2["block_stats"]
    assert "s_min_current" in L2["block_stats"]
    assert "monthly_dd_pct" in L2["block_stats"]
    print("Test 10 PASS")


# Test 11
def test_11_no_retroactive_fill_and_min_rr():
    n = 80
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(n)]
    df = make_df([100.0] * n, times=times)
    for k in range(n - 6, n):
        set_row(df, k, ema9=112.0, ema30=108.0, sma200=100.0, sma200_prev5=99.0,
                Low=114.0, Close=115.0, High=115.0, rsi=52.0, atr=3.0,
                z_v=0.5, h20=116.0, l20=95.0, er10=0.1, atr_ratio=0.5)
    set_row(df, n - 3, High=116.0, ema30=108.0, atr=3.0, z_v=0.5)
    set_row(df, n - 2, ema9=112.0, ema30=108.0, sma200=100.0, sma200_prev5=99.0,
            Low=114.0, Close=115.0, High=115.0, rsi=52.0, atr=3.0, z_v=0.5)
    set_row(df, n - 1, ema9=110.0, ema30=108.0, sma200=100.0, sma200_prev5=99.0,
            Low=108.2, Close=109.0, High=109.5, rsi=52.0, atr=3.0, z_v=0.5)
    sigs = sw.run_detectors("HAL.NS", df, "1d", "IndiaNSE", keep_all=True)
    assert sigs, "expected at least one signal"
    # placed on the latest bar only
    for s in sigs:
        assert s["placed_bar_time"] == str(df.iloc[-1]["Time"]), (s["placed_bar_time"], str(df.iloc[-1]["Time"]))
    # try_fill on same bar should not fill
    sig = sigs[0]
    filled, et, ep = sw.try_fill(sig, df, "1d")
    assert filled is False and et is None and ep is None, (filled, et, ep)
    # update_block_stats populates min_rr_current
    L = _base_ledger()
    sw.update_block_stats(L)
    assert "min_rr_current" in L["block_stats"], L["block_stats"]
    assert L["block_stats"]["min_rr_current"] == round(sw.get_min_rr(int(L["block_stats"].get("consecutive_losses",0))),4)
    print("Test 11 PASS")


# Test 12
def test_12_chandelier_ratchet():
    import pandas as pd
    n = 5
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(n)]
    closes = [100.0] * n
    highs = [100.0] * n
    lows = [99.0] * n
    # bars after entry: last bar High 108, atr 2.0
    highs[-1] = 108.0
    lows[-1] = 107.0
    closes[-1] = 107.5
    df = make_df(closes, highs=highs, lows=lows, times=times)
    # set atr=2.0 on all bars
    for i in range(n):
        df.iloc[i, df.columns.get_loc("atr")] = 2.0
        df.iloc[i, df.columns.get_loc("ATR")] = 2.0
    entry_time = times[3]
    pos = {
        "order_id": "HAL.NS|1d|T10_SOROS|long|t",
        "symbol": "HAL.NS",
        "market": "IndiaNSE",
        "tf": "1d",
        "arch": "T10_SOROS",
        "direction": "long",
        "entry_time": entry_time,
        "entry_price": 100.0,
        "stop_price": 97.0,
        "initial_stop": 97.0,
        "target_price": 112.6,
        "risk_capital": 2.0,
        "unrealized_r": 0.0,
        "seal_sha256": "x",
    }
    L = _base_ledger()
    L["open_positions"] = [pos]
    data = {("HAL.NS", "1d"): df}
    sw.step_a_update_positions(L, data)
    # mfe 2.5R -> 1.60R chandelier floor = 104.8 (high 108 - 2*ATR 4.0 = 104.0)
    assert abs(L["open_positions"][0]["stop_price"] - 104.8) < 1e-9, L["open_positions"][0]["stop_price"]
    assert abs(L["open_positions"][0]["mfe_r"] - 2.5) < 1e-9, L["open_positions"][0]["mfe_r"]
    assert L["open_positions"][0]["bars_held"] == 1, L["open_positions"][0]["bars_held"]
    print("Test 12 PASS")


# Test 13
def test_13_kreil_metrics_and_sweet_spot():
    n = 120
    closes = [100.0 + (i % 7) * 0.5 for i in range(n)]
    highs = [c + 1.5 for c in closes]
    lows = [c - 1.5 for c in closes]
    opens = [c - 0.2 for c in closes]
    df = make_df(closes, highs=highs, lows=lows, opens=opens)
    m = sw.compute_kreil_metrics(df)
    for k in ("atr_1d", "atr_5", "atr_20", "atr_60", "atr_250", "atr_5_250",
              "atr_1_20", "mean", "std", "skew", "excess_kurt"):
        assert k in m, k
    assert np.isfinite(m["atr_5_250"])
    passed, info = sw.passes_kreil_regime_gate(make_kreil_neutral_df(), "LONG")
    assert passed is True and info["reason"] is None
    assert info["score_boost"] in (0.0, 0.15)
    print("Test 13 PASS")


# Test 14
def test_14_kreil_rejects_dead_and_exhausted_atr():
    n = 60
    closes = [100.0] * n
    df_dead = make_df(closes, highs=closes, lows=closes, opens=closes)
    passed, info = sw.passes_kreil_regime_gate(df_dead, "LONG")
    assert passed is False
    assert info["reason"] is not None

    closes2 = [100.0] * (n - 5) + [100.0 + i * 5.0 for i in range(5)]
    highs2 = [c + 0.1 for c in closes2]
    lows2 = [c - 0.1 for c in closes2]
    for i in range(n - 5, n):
        highs2[i] = closes2[i] + 20.0
        lows2[i] = closes2[i] - 20.0
    df_ex = make_df(closes2, highs=highs2, lows=lows2, opens=closes2)
    passed2, info2 = sw.passes_kreil_regime_gate(df_ex, "LONG")
    assert passed2 is False
    assert info2["reason"] == "atr_5_250_out_of_range", info2["reason"]
    assert info2["atr_5_250"] > 1.60, info2["atr_5_250"]
    print("Test 14 PASS")


# Test 15
def test_15_kreil_directional_skew_and_boost():
    n = 120
    closes = [100.0 + (i % 10) for i in range(n)]
    for i in range(0, n, 15):
        closes[i] = closes[i] + 8.0
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    opens = [c - 0.5 for c in closes]
    df = make_df(closes, highs=highs, lows=lows, opens=opens)
    passed, info = sw.passes_kreil_regime_gate(df, "LONG")
    assert passed is False
    assert info["reason"] == "low_kurtosis", info["reason"]
    assert info["skew"] < 0.0, info["skew"]
    assert info["score_boost"] == 0.0

    passed_s, info_s = sw.passes_kreil_regime_gate(df, "SHORT")
    assert passed_s is False
    assert info_s["reason"] == "low_kurtosis", info_s["reason"]
    assert info_s["score_boost"] == 0.0

    closes_n = [100.0 - (i % 10) for i in range(n)]
    for i in range(0, n, 15):
        closes_n[i] = closes_n[i] - 8.0
    df_n = make_df(closes_n, highs=[c+1.0 for c in closes_n], lows=[c-1.0 for c in closes_n],
                   opens=[c+0.5 for c in closes_n])
    passed_n, info_n = sw.passes_kreil_regime_gate(df_n, "LONG")
    assert passed_n is True
    assert info_n["reason"] is None
    assert info_n["skew"] < 0.0, info_n["skew"]
    assert info_n["score_boost"] == 0.0

    passed_ns, info_ns = sw.passes_kreil_regime_gate(df_n, "SHORT")
    assert passed_ns is True
    assert info_ns["reason"] is None
    assert info_ns["excess_kurt"] < 2.0, info_ns["excess_kurt"]
    assert info_ns["score_boost"] == 0.0

    # directional skew vetoes the opposite side on fat-tailed data
    passed_up, info_up = sw.passes_kreil_regime_gate(make_kreil_boost_df(True), "SHORT")
    assert passed_up is False
    assert info_up["reason"] == "adverse_short_skew", info_up["reason"]
    passed_dn, info_dn = sw.passes_kreil_regime_gate(make_kreil_boost_df(False), "LONG")
    assert passed_dn is False
    assert info_dn["reason"] == "adverse_long_skew", info_dn["reason"]
    print("Test 15 PASS")


# Test 16
def test_16_step_b_prunes_non_kreil_pending():
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(5)]
    closes = [100.0] * 5
    df = make_df(closes, highs=closes, lows=closes, opens=closes, times=times)
    data = {("X", "1d"): df}
    L = _base_ledger()
    pending = [{"order_id": "X|1d|MA7_TRUE|long|t", "symbol": "X", "market": "US", "tf": "1d",
                "arch": "MA7_TRUE", "direction": "long", "placed_bar_time": times[2],
                "limit_price": 99.0, "stop_price": 95.0, "target_price": 113.0,
                "score": 3.0, "seal_sha256": "abc", "rr": 3.5}]
    L["pending_orders"] = pending
    sw.step_b_fill_pending(L, data, L["open_positions"], enforce_kreil=True)
    assert len(L["pending_orders"]) == 0, L["pending_orders"]
    print("Test 16 PASS")


# Test 17
def test_17_spot_prices_and_seal_integrity():
    sig = sw.build_signal("AMD", "US", "1h", "P3_SWEEP", "long", "t", 100.0, 95.0, 113.0, 3.0)
    assert sig is not None
    import hashlib
    seal = hashlib.sha256(
        f"AMD|1h|P3_SWEEP|long|t|{sig['limit_price']:.6f}|{sig['stop_price']:.6f}|{sig['target_price']:.6f}".encode()
    ).hexdigest()
    assert sig["seal_sha256"] == seal
    sig2 = sw.build_signal("AMD", "US", "1h", "P3_SWEEP", "long", "t", 100.0, 95.0, 113.0, 3.0,
                           kreil_info={"foo": "bar"})
    assert sig2 is not None and sig2.get("kreil", {}).get("foo") == "bar"
    assert not hasattr(sw, "DASHBOARD_HTML"), "dead DASHBOARD_HTML block must be gone"
    print("Test 17 PASS")


# Test 18
def test_18_kreil_score_boost_rescues_signal_above_s_min():
    df_up = make_kreil_boost_df(True)
    passed, info = sw.passes_kreil_regime_gate(df_up, "LONG")
    assert passed is True
    assert info["reason"] is None
    assert info["skew"] > 0.5, info["skew"]
    assert info["excess_kurt"] > 2.0, info["excess_kurt"]
    assert info["score_boost"] == 0.15

    df_dn = make_kreil_boost_df(False)
    passed_dn, info_dn = sw.passes_kreil_regime_gate(df_dn, "SHORT")
    assert passed_dn is True
    assert info_dn["reason"] is None
    assert info_dn["skew"] < -0.5, info_dn["skew"]
    assert info_dn["excess_kurt"] > 2.0, info_dn["excess_kurt"]
    assert info_dn["score_boost"] == 0.15

    s_min = sw.get_s_min(0)
    assert s_min == 2.6
    base_score = 2.5
    assert base_score < s_min
    assert round(base_score + 0.15, 4) >= s_min

    last_time = str(df_up.iloc[-1]["Time"])

    def _order(sym):
        return {
            "order_id": f"{sym}|1d|MA7_TRUE|long|{last_time}",
            "symbol": sym, "market": "US", "tf": "1d", "arch": "MA7_TRUE",
            "direction": "long", "placed_bar_time": last_time,
            "limit_price": 100.0, "stop_price": 96.0, "target_price": 116.0,
            "score": base_score, "rr": 4.0, "seal_sha256": "abc",
        }

    # rescued: base score below s_min, Kreil boost lifts it over the threshold
    L = _base_ledger()
    L["pending_orders"] = [_order("BOOST")]
    sw.step_b_fill_pending(L, {("BOOST", "1d"): df_up}, L["open_positions"], enforce_kreil=True)
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    o = L["pending_orders"][0]
    assert o["base_score"] == base_score, o.get("base_score")
    assert o["score"] == 2.65, o["score"]
    assert o["score"] >= s_min
    assert o["kreil"]["boost_applied"] is True
    assert o["kreil"]["score_boost"] == 0.15
    assert len(L["open_positions"]) == 0

    # idempotent: a second pass must not stack the boost
    sw.step_b_fill_pending(L, {("BOOST", "1d"): df_up}, L["open_positions"], enforce_kreil=True)
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    assert L["pending_orders"][0]["score"] == 2.65, L["pending_orders"][0]["score"]

    # control: same sub-threshold score on gate-passing data without boost is pruned
    L2 = _base_ledger()
    L2["pending_orders"] = [_order("NEUTRAL")]
    sw.step_b_fill_pending(L2, {("NEUTRAL", "1d"): make_kreil_neutral_df()},
                           L2["open_positions"], enforce_kreil=True)
    assert len(L2["pending_orders"]) == 0, L2["pending_orders"]
    print("Test 18 PASS")


# Test 19
def test_19_nan_fail_closed_and_intrabar_order():
    # fail-closed: insufficient bars (<30) and None input
    short_df = make_df([100.0] * 29)
    passed_s, info_s = sw.passes_kreil_regime_gate(short_df, "LONG")
    assert passed_s is False
    assert info_s["reason"] == "insufficient_bars", info_s["reason"]
    assert info_s["score_boost"] == 0.0
    passed_n, info_n = sw.passes_kreil_regime_gate(None, "LONG")
    assert passed_n is False
    assert info_n["reason"] == "insufficient_bars", info_n["reason"]

    # fail-closed: NaN metrics => nan_metric
    closes = [100.0] * 60
    closes[-1] = float("nan")
    nan_df = make_df(closes)
    passed_nan, info_nan = sw.passes_kreil_regime_gate(nan_df, "LONG")
    assert passed_nan is False
    assert info_nan["reason"] == "nan_metric", info_nan["reason"]

    # intrabar order: the bar's Low hits the OLD stop before the Close-based ratchet
    times = [f"2026-02-{i+1:02d} 00:00:00" for i in range(4)]
    closes_a = [100.0, 100.0, 130.0, 130.0]
    highs_a = [100.0, 100.0, 130.0, 130.0]
    lows_a = [100.0, 100.0, 96.0, 130.0]
    df_a = make_df(closes_a, highs=highs_a, lows=lows_a, times=times)
    pos_a = {
        "order_id": "X|1d|MA7_TRUE|long|t", "symbol": "X", "market": "IndiaNSE", "tf": "1d",
        "arch": "MA7_TRUE", "direction": "long", "entry_time": times[1],
        "entry_price": 100.0, "stop_price": 97.0, "initial_stop": 97.0,
        "target_price": 112.6, "risk_capital": 2.0, "unrealized_r": 0.0, "seal_sha256": "x",
    }
    L = _base_ledger()
    L["open_positions"] = [pos_a]
    sw.step_a_update_positions(L, {("X", "1d"): df_a})
    assert len(L["open_positions"]) == 0, L["open_positions"]
    assert len(L["closed_trades"]) == 1
    tr = L["closed_trades"][0]
    assert tr["reason"] == "stop", tr["reason"]
    assert abs(tr["exit_price"] - 97.0) < 1e-9, tr["exit_price"]

    # no exit => ratchet applies, and initial_stop is preserved (setdefault on legacy pos)
    closes_b = [100.0, 100.0, 104.0, 104.0]
    highs_b = [100.0, 100.0, 105.0, 105.0]
    lows_b = [100.0, 100.0, 99.0, 101.0]
    df_b = make_df(closes_b, highs=highs_b, lows=lows_b, times=times)
    pos_b = {
        "order_id": "Y|1d|MA7_TRUE|long|t", "symbol": "Y", "market": "IndiaNSE", "tf": "1d",
        "arch": "MA7_TRUE", "direction": "long", "entry_time": times[2],
        "entry_price": 100.0, "stop_price": 97.0,
        "target_price": 112.6, "risk_capital": 2.0, "unrealized_r": 0.0, "seal_sha256": "x",
    }
    assert "initial_stop" not in pos_b
    L2 = _base_ledger()
    L2["open_positions"] = [pos_b]
    sw.step_a_update_positions(L2, {("Y", "1d"): df_b})
    assert len(L2["open_positions"]) == 1, L2["open_positions"]
    held = L2["open_positions"][0]
    assert abs(held["initial_stop"] - 97.0) < 1e-9, held["initial_stop"]
    # close-based MFE 1.333R -> +0.60R profit lock = 101.8
    assert abs(held["stop_price"] - 101.8) < 1e-9, held["stop_price"]
    print("Test 19 PASS")


def _kreil_breakout_df():
    df = make_kreil_boost_df(True)
    n = len(df)
    for k in range(n - 6, n):
        set_row(df, k, Open=100.0, High=101.0, Low=99.0, Close=100.01, Volume=1000.0,
                ema9=100.5, ema30=99.8, sma200=95.0, sma200_prev5=94.0, ema50=98.0,
                atr=3.0, z_v=2.0, rsi=60.0, h20=99.5, l10=97.0, l20=95.0,
                vol_ratio=0.9, er10=0.1, atr_ratio=0.5)
    return df


# Test 20
def test_20_kreil_base_score_no_cumulative_inflation():
    df = _kreil_breakout_df()
    passed, info = sw.passes_kreil_regime_gate(df, "LONG", "F3_BREAKOUT")
    assert passed is True and info["reason"] is None
    assert info["score_boost"] == 0.15, info["score_boost"]

    seen = []
    for _ in range(10):
        sigs = sw.run_detectors("AMD", df, "1d", "US", keep_all=True)
        f3 = [s for s in sigs if s["arch"] == "F3_BREAKOUT"]
        assert len(f3) == 1, [s["arch"] for s in sigs]
        seen.append((f3[0]["base_score"], f3[0]["score"]))
    assert len(set(seen)) == 1, seen
    base, score = seen[0]
    assert abs(base - 2.95) < 1e-9, base
    assert abs(score - 3.10) < 1e-9, score

    # the same signal re-processed 10x as a pending order must not stack the boost
    sig = dict(f3[0])
    sig["order_id"] = f"AMD|1d|F3_BREAKOUT|long|{sig['placed_bar_time']}"
    L = _base_ledger()
    L["pending_orders"] = [sig]
    for _ in range(10):
        sw.step_b_fill_pending(L, {("AMD", "1d"): df}, L["open_positions"], enforce_kreil=True)
        assert len(L["pending_orders"]) == 1, L["pending_orders"]
        o = L["pending_orders"][0]
        assert abs(o["base_score"] - 2.95) < 1e-9, o["base_score"]
        assert abs(o["score"] - 3.10) < 1e-9, o["score"]
    assert len(L["open_positions"]) == 0

    # migration repair: an order whose base_score was stored post-boost is healed
    inflated = {
        "order_id": "AMD|1d|F3_BREAKOUT|long|t", "symbol": "AMD", "market": "US", "tf": "1d",
        "arch": "F3_BREAKOUT", "direction": "long", "placed_bar_time": "t",
        "limit_price": 100.0, "stop_price": 96.0, "target_price": 108.8,
        "score": 3.10, "base_score": 3.10, "rr": 2.2, "seal_sha256": "x",
        "kreil": {"score_boost": 0.15},
    }
    Lm = _base_ledger()
    Lm["pending_orders"] = [inflated]
    sw.repair_pending_orders(Lm)
    assert abs(inflated["base_score"] - 2.95) < 1e-9, inflated["base_score"]
    assert abs(inflated["score"] - 3.10) < 1e-9, inflated["score"]

    # migration repair: a sub-1.8R legacy target is re-derived and re-sealed
    bad = {
        "order_id": "AMD|1d|F3_BREAKOUT|long|t", "symbol": "AMD", "market": "US", "tf": "1d",
        "arch": "F3_BREAKOUT", "direction": "long", "placed_bar_time": "t",
        "limit_price": 100.0, "stop_price": 96.0, "target_price": 104.0,
        "score": 3.0, "base_score": 3.0, "rr": 1.0, "seal_sha256": "old",
        "kreil": {"score_boost": 0.0},
    }
    Lb = _base_ledger()
    Lb["pending_orders"] = [bad]
    sw.repair_pending_orders(Lb)
    assert abs(bad["target_price"] - 108.8) < 1e-9, bad["target_price"]
    assert abs(bad["rr"] - 2.2) < 1e-9, bad["rr"]
    assert bad["seal_sha256"] == sw._signal_seal("AMD", "1d", "F3_BREAKOUT", "long", "t", 100.0, 96.0, 108.8)
    print("Test 20 PASS")


# Test 21
def test_21_bucket_helpers_and_ledger_migration():
    assert sw.MARKET_BUCKETS == {"Crypto": "Crypto", "IndiaNSE": "IndiaNSE",
                                 "US": "US_FX", "FXCommod": "US_FX"}
    assert sw.get_market_bucket("US") == "US_FX"
    assert sw.get_market_bucket("FXCommod") == "US_FX"
    assert sw.get_market_bucket("Crypto") == "Crypto"
    assert sw.get_market_bucket("IndiaNSE") == "IndiaNSE"

    pos = [{"market": "Crypto"}, {"market": "IndiaNSE"}, {"market": "US"}]
    assert sw.can_open_position([], "US") is True
    assert sw.can_open_position(pos[:1], "Crypto") is False
    assert sw.can_open_position(pos[:1], "US") is True
    assert sw.can_open_position(pos[:2], "US") is True
    assert sw.can_open_position(pos, "FXCommod") is False     # 3 global slots used
    assert sw.can_open_position([{"market": "FXCommod"}], "US") is False  # same bucket
    assert len(sw.bucket_positions(pos, "FXCommod")) == 1
    assert len(sw.bucket_positions(pos, "Crypto")) == 1

    # a legacy 2% ledger is migrated to 1.5% and gains the min_rr floor
    legacy = {
        "account": {"equity": 100.0, "risk_pct": 0.02, "target": 1000.0,
                    "month_start_equity": 100.0, "current_month": "2026-10"},
        "open_positions": [], "pending_orders": [], "closed_trades": [],
        "block_stats": {"consecutive_losses": 0, "blocked": False, "cooldowns": {},
                        "s_min_current": 2.6, "min_rr_current": 3.2, "monthly_dd_pct": 0.0},
        "last_run": None,
    }
    sw.enforce_risk_pct(legacy)
    sw.update_block_stats(legacy)
    assert legacy["account"]["risk_pct"] == 0.015
    assert legacy["block_stats"]["min_rr_current"] == 1.8
    assert legacy["block_stats"]["cooldowns"] == []   # {} normalised to []
    print("Test 21 PASS")


# Test 22
def test_22_profit_lock_ratchet_ladder():
    times = [f"2026-03-{i+1:02d} 00:00:00" for i in range(5)]

    def run(direction, closes, highs, lows, entry_time):
        df = make_df(closes, highs=highs, lows=lows, times=times)
        for i in range(len(df)):
            df.iloc[i, df.columns.get_loc("atr")] = 2.0
            df.iloc[i, df.columns.get_loc("ATR")] = 2.0
        if direction == "long":
            stop, target = 97.0, 130.0
        else:
            stop, target = 103.0, 70.0
        pos = {"order_id": "X|1h|MA7_TRUE|" + direction + "|t", "symbol": "X", "market": "US", "tf": "1h",
               "arch": "MA7_TRUE", "direction": direction, "entry_time": entry_time,
               "entry_price": 100.0, "stop_price": stop, "initial_stop": stop,
               "target_price": target, "risk_capital": 1.5, "unrealized_r": 0.0, "seal_sha256": "x"}
        L = _base_ledger()
        L["open_positions"] = [pos]
        sw.step_a_update_positions(L, {("X", "1h"): df})
        return L

    # LONG: 0.5R -> breakeven + 0.05R = 100.15
    L1 = run("long", [100.0, 101.5, 101.5, 101.5, 101.5],
             [100.0, 101.6, 101.6, 101.6, 101.6], [100.0, 100.1, 100.2, 100.2, 100.2], times[0])
    assert len(L1["open_positions"]) == 1, L1["closed_trades"]
    assert abs(L1["open_positions"][0]["stop_price"] - 100.15) < 1e-9, L1["open_positions"][0]["stop_price"]

    # LONG: 1.0R -> +0.60R = 101.8
    L2 = run("long", [100.0, 101.5, 103.0, 103.0, 103.0],
             [100.0, 101.6, 103.1, 103.1, 103.1], [100.0, 100.1, 101.0, 102.0, 102.0], times[0])
    assert len(L2["open_positions"]) == 1, L2["closed_trades"]
    assert abs(L2["open_positions"][0]["stop_price"] - 101.8) < 1e-9, L2["open_positions"][0]["stop_price"]

    # LONG: 1.5R -> +1.10R = 103.3
    L3 = run("long", [100.0, 101.5, 103.0, 104.5, 104.5],
             [100.0, 101.6, 103.1, 104.6, 104.6], [100.0, 100.1, 101.0, 102.5, 103.5], times[0])
    assert len(L3["open_positions"]) == 1, L3["closed_trades"]
    held3 = L3["open_positions"][0]
    assert abs(held3["stop_price"] - 103.3) < 1e-9, held3["stop_price"]
    assert abs(held3["mfe_r"] - 1.5) < 1e-9, held3["mfe_r"]

    # SHORT mirror: 0.5R -> 99.85 ; 1.0R -> 98.2 ; 1.5R -> 96.7
    S1 = run("short", [100.0, 98.5, 98.5, 98.5, 98.5],
             [100.0, 99.9, 99.8, 99.8, 99.8], [100.0, 98.4, 98.4, 98.4, 98.4], times[0])
    assert abs(S1["open_positions"][0]["stop_price"] - 99.85) < 1e-9, S1["open_positions"][0]["stop_price"]
    S2 = run("short", [100.0, 98.5, 97.0, 97.0, 97.0],
             [100.0, 99.9, 98.9, 98.1, 98.1], [100.0, 98.4, 96.9, 96.0, 96.0], times[0])
    assert abs(S2["open_positions"][0]["stop_price"] - 98.2) < 1e-9, S2["open_positions"][0]["stop_price"]
    S3 = run("short", [100.0, 98.5, 97.0, 95.5, 95.5],
             [100.0, 99.9, 98.9, 97.4, 96.6], [100.0, 98.4, 96.9, 95.4, 95.4], times[0])
    assert len(S3["open_positions"]) == 1, S3["closed_trades"]
    assert abs(S3["open_positions"][0]["stop_price"] - 96.7) < 1e-9, S3["open_positions"][0]["stop_price"]

    # the ladder never loosens a stop that is already tighter
    L4 = run("long", [100.0, 101.5, 101.5, 101.5, 101.5],
             [100.0, 101.6, 101.6, 101.6, 101.6], [100.0, 100.5, 100.5, 100.5, 100.5], times[0])
    assert abs(L4["open_positions"][0]["stop_price"] - 100.15) < 1e-9, L4["open_positions"][0]["stop_price"]
    L4["open_positions"][0]["stop_price"] = 100.3
    sw.step_a_update_positions(L4, {("X", "1h"): make_df(
        [100.0, 101.5, 101.5, 101.5, 101.5],
        highs=[100.0, 101.6, 101.6, 101.6, 101.6],
        lows=[100.0, 100.5, 100.5, 100.5, 100.5], times=times)})
    assert len(L4["open_positions"]) == 1, L4["closed_trades"]
    assert abs(L4["open_positions"][0]["stop_price"] - 100.3) < 1e-9, L4["open_positions"][0]["stop_price"]
    print("Test 22 PASS")


# Test 23
def test_23_time_profit_harvest():
    entry = 4809.487956466684
    stop = 4585.207969022488
    last = 4853.7001953125
    times = ["2026-10-27 00:00:00", "2026-10-28 00:00:00", "2026-10-29 00:00:00", "2026-10-30 00:00:00"]
    df = make_df([entry, entry, last, last],
                 highs=[entry, entry, last * 1.002, last * 1.002],
                 lows=[entry, entry, last * 0.998, last * 0.998], times=times)
    pos = {"order_id": "HAL.NS|1d|MA7_TRUE|long|2026-10-27 00:00:00", "symbol": "HAL.NS",
           "market": "IndiaNSE", "tf": "1d", "arch": "MA7_TRUE", "direction": "long",
           "entry_time": "2026-10-28 00:00:00", "entry_price": entry, "stop_price": stop,
           "initial_stop": stop, "target_price": 5594.4679125213715, "risk_capital": 2.0,
           "unrealized_r": 0.1971296652440565, "seal_sha256": "8a0ce2ca"}
    L = _base_ledger()
    L["open_positions"] = [pos]
    sw.step_a_update_positions(L, {("HAL.NS", "1d"): df})
    assert len(L["open_positions"]) == 0, L["open_positions"]
    tr = L["closed_trades"][0]
    assert tr["reason"] == "TIME_PROFIT_HARVEST", tr["reason"]
    assert abs(tr["pnl_r"] - 0.1971) < 1e-4, tr["pnl_r"]
    assert tr["bars_held"] == 2, tr["bars_held"]
    assert abs(L["account"]["equity"] - (100.0 + 0.1971 * 2.0)) < 1e-4, L["account"]["equity"]
    assert L["block_stats"]["consecutive_losses"] == 0
    assert abs(tr["exit_price"] - last) < 1e-9

    # wall-clock fallback: ONLY for a missing/empty frame, and it may advance at
    # most one bar per scan (weekends / holidays can never inflate bars_held).
    old = {"tf": "1d", "opened_at": "2020-01-02 00:00:00", "bars_held": 0}
    assert sw.bars_held_since_entry(old, None) == 1
    old2 = dict(old, bars_held=2)
    assert sw.bars_held_since_entry(old2, None) == 3
    empty_df = make_df([100.0], times=["2020-01-01 00:00:00"]).iloc[0:0]
    assert sw.bars_held_since_entry(old2, empty_df) == 3

    # a present frame with NO new bars must never count calendar/weekend days
    df_old = make_df([entry, last], highs=[entry, last], lows=[entry, last],
                     times=["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
    pos2 = dict(pos, order_id="HAL.NS|1d|MA7_TRUE|long|old",
                entry_time="2020-01-02 00:00:00", opened_at="2020-01-02 00:00:00",
                bars_held=1)
    pos2.pop("last_processed_bar_ts", None)
    L2 = _base_ledger()
    L2["open_positions"] = [pos2]
    sw.step_a_update_positions(L2, {("HAL.NS", "1d"): df_old})
    assert len(L2["open_positions"]) == 1, L2["closed_trades"]
    assert L2["open_positions"][0]["bars_held"] == 1, L2["open_positions"][0]["bars_held"]
    print("Test 23 PASS")


# Test 24
def test_24_dead_money_exit():
    times = [f"2026-04-{i+1:02d} 00:00:00" for i in range(7)]
    base = {"order_id": "X|1d|MA7_TRUE|long|t", "symbol": "X", "market": "US", "tf": "1d",
            "arch": "MA7_TRUE", "direction": "long", "entry_time": times[0], "entry_price": 100.0,
            "stop_price": 97.0, "initial_stop": 97.0, "target_price": 112.6, "risk_capital": 2.0,
            "unrealized_r": 0.0, "seal_sha256": "x"}

    # +0.10R after dead_bars(1d)=4 -> flat exit, profit banked, no loss penalty
    df = make_df([100.0] + [100.3] * 6, highs=[100.0] + [100.5] * 6,
                 lows=[100.0] + [99.9] * 6, times=times)
    L = _base_ledger()
    L["open_positions"] = [dict(base)]
    sw.step_a_update_positions(L, {("X", "1d"): df})
    assert len(L["open_positions"]) == 0, L["open_positions"]
    tr = L["closed_trades"][0]
    assert tr["reason"] == "TIME_FLAT_EXIT", tr["reason"]
    assert abs(tr["pnl_r"] - 0.1) < 1e-9, tr["pnl_r"]
    assert abs(L["account"]["equity"] - 100.2) < 1e-9, L["account"]["equity"]
    assert L["block_stats"]["consecutive_losses"] == 0

    # -0.20R scratch -> flat exit, and it does NOT count as a full loss
    df2 = make_df([100.0] + [99.4] * 6, highs=[100.0] + [99.6] * 6,
                  lows=[100.0] + [99.0] * 6, times=times)
    L2 = _base_ledger()
    L2["open_positions"] = [dict(base, order_id="Y")]
    sw.step_a_update_positions(L2, {("X", "1d"): df2})
    tr2 = L2["closed_trades"][0]
    assert tr2["reason"] == "TIME_FLAT_EXIT", tr2["reason"]
    assert abs(tr2["pnl_r"] + 0.2) < 1e-9, tr2["pnl_r"]
    assert L2["block_stats"]["consecutive_losses"] == 0, L2["block_stats"]

    # -0.30R is outside the [-0.25R, +0.15R) dead-money band -> keeps running
    df3 = make_df([100.0] + [99.1] * 6, highs=[100.0] + [99.3] * 6,
                  lows=[100.0] + [98.9] * 6, times=times)
    L3 = _base_ledger()
    L3["open_positions"] = [dict(base, order_id="Z")]
    sw.step_a_update_positions(L3, {("X", "1d"): df3})
    assert len(L3["open_positions"]) == 1, L3["closed_trades"]
    assert abs(L3["open_positions"][0]["unrealized_r"] + 0.3) < 1e-9, L3["open_positions"][0]["unrealized_r"]

    # before dead_bars the flat trade is NOT closed (time gate is real)
    L4 = _base_ledger()
    L4["open_positions"] = [dict(base, entry_time=times[3])]
    sw.step_a_update_positions(L4, {("X", "1d"): df})
    assert len(L4["open_positions"]) == 1, L4["closed_trades"]
    print("Test 24 PASS")


# Test 25
def test_25_loss_block_12h_expiry():
    now = datetime.datetime.now(datetime.timezone.utc)

    def _L(hours_ago, equity=100.0):
        exit_ts = (now - datetime.timedelta(hours=hours_ago)).isoformat()
        return {
            "account": {"equity": equity, "risk_pct": 0.015, "target": 1000.0,
                        "month_start_equity": 100.0, "current_month": now.strftime("%Y-%m")},
            "open_positions": [], "pending_orders": [],
            "closed_trades": [{"pnl_r": -1.0, "symbol": "X", "direction": "long", "tf": "1d",
                               "exit_time": exit_ts} for _ in range(3)],
            "block_stats": {"consecutive_losses": 3, "blocked": True, "cooldowns": [],
                            "s_min_current": 3.1, "min_rr": 1.8, "min_rr_current": 2.1,
                            "monthly_dd_pct": 0.0},
            "last_run": None,
        }

    L = _L(1)
    sw.update_block_stats(L)
    assert L["block_stats"]["blocked"] is True
    assert L["block_stats"]["block_reason"] == "ELDER_12H_LOSS_COOLDOWN", L["block_stats"]
    assert "block_until" in L["block_stats"]

    L2 = _L(13)
    sw.update_block_stats(L2)
    assert L2["block_stats"]["blocked"] is False, L2["block_stats"]
    assert L2["block_stats"]["consecutive_losses"] == 2
    assert L2["block_stats"]["s_min_current"] == 2.9
    assert L2["block_stats"]["min_rr_current"] == 2.0
    assert "block_until" not in L2["block_stats"]
    assert "loss_block_expired_at" in L2["block_stats"]
    assert "block_reason" not in L2["block_stats"]

    # and the un-blocked ledger can actually trade again (no permanent deadlock)
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(5)]
    df = make_df([100.0] * 5, highs=[101.0] * 5, lows=[90.0] * 5, times=times)
    L2["pending_orders"] = [{"order_id": "AMD|1d|F3_BREAKOUT|long|t", "symbol": "AMD",
                             "market": "US", "tf": "1d", "arch": "F3_BREAKOUT", "direction": "long",
                             "placed_bar_time": times[2], "limit_price": 99.0, "stop_price": 95.0,
                             "target_price": 113.0, "score": 3.05, "seal_sha256": "a", "rr": 3.5}]
    sw.step_b_fill_pending(L2, {("AMD", "1d"): df}, L2["open_positions"])
    assert len(L2["open_positions"]) == 1, L2["open_positions"]
    print("Test 25 PASS")


# Test 26
def test_26_stop_floor_and_target_multiples():
    # intraday floor 0.35%: 0.30% rejected, 0.40% accepted
    assert sw.build_signal("AMD", "US", "15m", "MA7_TRUE", "long", "t", 100.0, 99.7, 104.0, 3.0) is None
    s15 = sw.build_signal("AMD", "US", "15m", "MA7_TRUE", "long", "t", 100.0, 99.6, 104.0, 3.0)
    assert s15 is not None and abs(s15["delta"] - 0.004) < 1e-9
    # daily equity floor 0.60%: 0.50% rejected, 0.70% accepted
    assert sw.build_signal("AMD", "US", "1d", "MA7_TRUE", "long", "t", 100.0, 99.5, 104.0, 3.0) is None
    s1d = sw.build_signal("AMD", "US", "1d", "MA7_TRUE", "long", "t", 100.0, 99.3, 104.0, 3.0)
    assert s1d is not None and abs(s1d["delta"] - 0.007) < 1e-9
    # FX/Commod daily uses the intraday floor
    assert sw.build_signal("EURUSD=X", "FXCommod", "1d", "MA7_TRUE", "long", "t", 100.0, 99.6, 104.0, 3.0) is not None
    # crypto daily uses the 0.60% floor
    assert sw.build_signal("BTC-USD", "Crypto", "1d", "MA7_TRUE", "long", "t", 100.0, 99.5, 104.0, 3.0) is None
    # targets are realistic 1.8R-2.5R
    assert sw.ARCH_TARGET_R == {"T10_SOROS": 2.5, "F1_SOROS_MA": 2.2, "MA7_TRUE": 2.0,
                                "P3_SWEEP": 2.1, "F3_BREAKOUT": 2.2}
    for arch, mult in sw.ARCH_TARGET_R.items():
        assert 1.8 <= mult <= 2.5, (arch, mult)
    print("Test 26 PASS")


# Test 27
def test_27_near_spot_fill_and_pending_lifetime():
    assert [sw.max_pending_bars(t) for t in ("15m", "1h", "4h", "1d", "1w")] == [12, 8, 6, 5, 6]
    times = ["2026-05-01 00:00:00", "2026-05-02 00:00:00"]

    def order(direction, limit, stop):
        return {"order_id": "X|1d|MA7_TRUE|" + direction + "|t", "symbol": "X", "market": "US", "tf": "1d",
                "arch": "MA7_TRUE", "direction": direction, "placed_bar_time": times[0],
                "limit_price": limit, "stop_price": stop, "target_price": 110.0}

    # 0.20R (0.8) near-spot tolerance when no ATR is supplied
    df_near = make_df([100.0, 100.0], highs=[100.6, 100.6], lows=[100.5, 100.5], times=times)
    filled, et, ep = sw.try_fill(order("long", 100.0, 96.0), df_near, "1d")
    assert filled is True and abs(ep - 100.0) < 1e-12 and et == times[1], (filled, et, ep)
    df_far = make_df([100.0, 100.0], highs=[101.0, 101.0], lows=[101.0, 101.0], times=times)
    assert sw.try_fill(order("long", 100.0, 96.0), df_far, "1d")[0] is False
    # 0.25*ATR tolerance when an ATR is supplied (0.25*2.0 = 0.5)
    df_atr = make_df([100.0, 100.0], highs=[100.6, 100.6], lows=[100.4, 100.4], times=times)
    assert sw.try_fill(order("long", 100.0, 96.0), df_atr, "1d", atr=2.0)[0] is True
    df_atr2 = make_df([100.0, 100.0], highs=[100.7, 100.7], lows=[100.6, 100.6], times=times)
    assert sw.try_fill(order("long", 100.0, 96.0), df_atr2, "1d", atr=2.0)[0] is False
    # short mirror
    df_s = make_df([100.0, 100.0], highs=[99.5, 99.5], lows=[99.4, 99.4], times=times)
    assert sw.try_fill(order("short", 100.0, 104.0), df_s, "1d")[0] is True
    df_s2 = make_df([100.0, 100.0], highs=[99.0, 99.0], lows=[98.9, 98.9], times=times)
    assert sw.try_fill(order("short", 100.0, 104.0), df_s2, "1d")[0] is False
    # strict touch still fills
    df_touch = make_df([100.0, 100.0], highs=[100.0, 100.0], lows=[99.0, 99.0], times=times)
    assert sw.try_fill(order("long", 100.0, 96.0), df_touch, "1d")[0] is True
    # placed on the latest bar -> never fills retroactively
    sig = dict(order("long", 100.0, 96.0), placed_bar_time=times[1])
    assert sw.try_fill(sig, df_near, "1d")[0] is False

    # pending lifetime: 15m survives 10 bars, expires after 12
    idx = pd.date_range("2026-05-01", periods=14, freq="15min")
    t15 = [str(x) for x in idx]
    df15 = make_df([100.0] * 14, highs=[101.0] * 14, lows=[100.0] * 14, times=t15)

    def pending15(placed):
        return {"order_id": f"X|15m|MA7_TRUE|long|{placed}", "symbol": "X", "market": "US", "tf": "15m",
                "arch": "MA7_TRUE", "direction": "long", "placed_bar_time": placed,
                "limit_price": 99.0, "stop_price": 95.0, "target_price": 107.0,
                "score": 3.0, "seal_sha256": "x", "rr": 2.0}

    L = _base_ledger()
    L["pending_orders"] = [pending15(t15[3])]     # 10 bars elapsed <= 12 -> kept
    sw.step_b_fill_pending(L, {("X", "15m"): df15}, L["open_positions"])
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    L2 = _base_ledger()
    L2["pending_orders"] = [pending15(t15[0])]    # 13 bars elapsed > 12 -> expired
    sw.step_b_fill_pending(L2, {("X", "15m"): df15}, L2["open_positions"])
    assert len(L2["pending_orders"]) == 0, L2["pending_orders"]

    # daily orders expire after 5 bars (not the old hard-coded 3)
    idxd = pd.date_range("2026-05-01", periods=8, freq="D")
    td = [str(x) for x in idxd]
    dfd = make_df([100.0] * 8, highs=[101.0] * 8, lows=[100.0] * 8, times=td)

    def pending1d(placed):
        return {"order_id": f"Y|1d|MA7_TRUE|long|{placed}", "symbol": "Y", "market": "US", "tf": "1d",
                "arch": "MA7_TRUE", "direction": "long", "placed_bar_time": placed,
                "limit_price": 99.0, "stop_price": 95.0, "target_price": 107.0,
                "score": 3.0, "seal_sha256": "x", "rr": 2.0}

    L3 = _base_ledger()
    L3["pending_orders"] = [pending1d(td[2])]     # 5 bars elapsed -> kept
    sw.step_b_fill_pending(L3, {("Y", "1d"): dfd}, L3["open_positions"])
    assert len(L3["pending_orders"]) == 1, L3["pending_orders"]
    L4 = _base_ledger()
    L4["pending_orders"] = [pending1d(td[1])]     # 6 bars elapsed -> expired
    sw.step_b_fill_pending(L4, {("Y", "1d"): dfd}, L4["open_positions"])
    assert len(L4["pending_orders"]) == 0, L4["pending_orders"]
    print("Test 27 PASS")


# Test 28 — historical base_score inflation repair + target RR migration
def test_28_historical_inflation_and_target_rr_migration():
    assert sw.ARCH_MAX_BASE_SCORE == {"T10_SOROS": 4.0, "F1_SOROS_MA": 3.6,
                                      "MA7_TRUE": 2.95, "P3_SWEEP": 3.5, "F3_BREAKOUT": 3.15}
    assert sw.ARCH_TARGET_RR == {"T10_SOROS": 2.5, "F1_SOROS_MA": 2.2, "MA7_TRUE": 2.0,
                                 "P3_SWEEP": 2.1, "F3_BREAKOUT": 2.2}
    assert sw.ARCH_TARGET_R is sw.ARCH_TARGET_RR

    # exact pre-inflation state of the 8 live orders (faaa097 + one live scan)
    legacy = [
        # order_id, arch, direction, limit, stop, inflated base, boost, original base
        ("XRP-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00", "P3_SWEEP", "long",
         1.256762981414795, 1.191654235124588, 3.7345, 0.15, 3.4345),
        ("LTC-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00", "P3_SWEEP", "long",
         55.197975158691406, 52.637742042541504, 3.6689, 0.15, 3.3689),
        ("ETH-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00", "P3_SWEEP", "long",
         2233.372802734375, 2160.716077532087, 3.5493, 0.15, 3.3993),
        ("NEAR-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00", "P3_SWEEP", "long",
         4.045522212982178, 3.831010341644287, 3.4028, 0.15, 3.2528),
        ("INFY.NS|1h|F3_BREAKOUT|long|2026-10-30 15:15:00+05:30", "F3_BREAKOUT", "long",
         1176.4925838655354, 1132.2103332519532, 3.3, 0.15, 3.15),
        ("SOL-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00", "P3_SWEEP", "long",
         95.49413299560547, 91.92192309243339, 3.4072, 0.0, 3.4072),
        ("COIN|1d|F3_BREAKOUT|short|2026-10-30 00:00:00", "F3_BREAKOUT", "short",
         157.84661739400207, 184.43256530761718, 2.9638, 0.0, 2.9638),
        ("USDJPY=X|1d|MA7_TRUE|short|2026-10-30 00:00:00", "MA7_TRUE", "short",
         157.71430236689594, 159.19819082950752, 2.8, 0.15, 2.8),
    ]
    orders = []
    for oid, arch, direction, limit, stop, inflated, boost, _orig in legacy:
        r_dist = abs(limit - stop)
        target = limit + 3.6 * r_dist if direction == "long" else limit - 3.6 * r_dist
        orders.append({
            "order_id": oid, "symbol": oid.split("|")[0], "market": "US", "tf": "1d",
            "arch": arch, "direction": direction, "placed_bar_time": oid.split("|")[-1],
            "limit_price": limit, "stop_price": stop, "target_price": target,
            "score": round(inflated + boost, 4), "base_score": inflated,
            "rr": 3.6, "seal_sha256": "legacy",
            "kreil": {"score_boost": boost, "boost_applied": boost > 0},
        })
    L = _base_ledger()
    L["pending_orders"] = orders
    sw.repair_pending_orders(L)

    for order, (_oid, arch, direction, limit, stop, _inflated, boost, orig) in zip(orders, legacy):
        r_dist = abs(limit - stop)
        design = sw.ARCH_TARGET_RR[arch]
        # 1. base_score restored to the exact original creation value
        assert abs(order["base_score"] - orig) < 1e-9, (order["order_id"], order["base_score"], orig)
        # 2. score rebuilt exactly once, never above 4.0
        assert abs(order["score"] - min(4.0, orig + boost)) < 1e-9, (order["order_id"], order["score"])
        assert order["base_score"] <= sw.ARCH_MAX_BASE_SCORE[arch] + 1e-9
        # 3. legacy 3.6R target migrated to the architecture multiple + re-sealed
        assert abs(order["rr"] - design) < 1e-9, (order["order_id"], order["rr"], design)
        exp_target = limit + design * r_dist if direction == "long" else limit - design * r_dist
        assert abs(order["target_price"] - exp_target) < 1e-9, order["order_id"]
        assert order["seal_sha256"] == sw._signal_seal(
            order["symbol"], order["tf"], arch, direction, order["placed_bar_time"], limit, stop, exp_target)
        assert order["integrity_seal"] == order["seal_sha256"]

    # idempotent: a second repair pass changes nothing (no re-inflation, no drift)
    snapshot = [(o["base_score"], o["score"], o["rr"], o["target_price"], o["seal_sha256"]) for o in orders]
    sw.repair_pending_orders(L)
    assert snapshot == [(o["base_score"], o["score"], o["rr"], o["target_price"], o["seal_sha256"]) for o in orders]

    # FALSE-POSITIVE GUARD: a legitimate unboosted 4.0 T10_SOROS keeps its base
    legit = {
        "order_id": "BTC-USD|1d|T10_SOROS|long|2026-10-30 00:00:00", "symbol": "BTC-USD",
        "market": "Crypto", "tf": "1d", "arch": "T10_SOROS", "direction": "long",
        "placed_bar_time": "2026-10-30 00:00:00", "limit_price": 100.0, "stop_price": 95.0,
        "target_price": 112.5, "score": 4.0, "base_score": 4.0, "rr": 2.5, "seal_sha256": "s",
        "kreil": {"score_boost": 0.15, "boost_applied": True},
    }
    Ll = _base_ledger()
    Ll["pending_orders"] = [legit]
    sw.repair_pending_orders(Ll)
    assert abs(legit["base_score"] - 4.0) < 1e-12, legit["base_score"]
    assert abs(legit["score"] - 4.0) < 1e-12, legit["score"]
    assert abs(legit["rr"] - 2.5) < 1e-12, legit["rr"]

    # generic heuristic (no known order id) still strips whole +0.15 steps
    generic = dict(legit, order_id="NEW|1d|P3_SWEEP|long|t", arch="P3_SWEEP",
                   base_score=3.65, score=3.8, target_price=107.5, rr=1.5, seal_sha256="x")
    Lg = _base_ledger()
    Lg["pending_orders"] = [generic]
    sw.repair_pending_orders(Lg)
    # 3.65 is 0.15 above the 3.50 cap -> exactly one +0.15 step is removed
    assert abs(generic["base_score"] - 3.5) < 1e-9, generic["base_score"]
    assert abs(generic["rr"] - 2.1) < 1e-9, generic["rr"]
    # 3.80 is two whole steps above the cap -> lands on the cap, never below it
    generic2 = dict(generic, order_id="NEW2|1d|P3_SWEEP|long|t",
                    base_score=3.80, score=3.95, seal_sha256="y")
    Lg2 = _base_ledger()
    Lg2["pending_orders"] = [generic2]
    sw.repair_pending_orders(Lg2)
    assert abs(generic2["base_score"] - 3.5) < 1e-9, generic2["base_score"]

    # LIVE LEDGER INVARIANTS (unconditional, read from disk)
    ledger_path = os.path.join(ROOT, "ledger.json")
    with open(ledger_path, encoding="utf-8") as fh:
        live = json.load(fh)
    assert len(live["pending_orders"]) == 8, len(live["pending_orders"])
    for o in live["pending_orders"]:
        assert o["base_score"] <= sw.ARCH_MAX_BASE_SCORE[o["arch"]] + 1e-9, o["order_id"]
        assert abs(o["score"] - min(4.0, o["base_score"] + float(o["kreil"].get("score_boost", 0.0)))) < 1e-9, o["order_id"]
        design = sw.ARCH_TARGET_RR[o["arch"]]
        assert 1.8 - 1e-9 <= o["rr"] <= design + 0.05 + 1e-9, (o["order_id"], o["rr"])
        r_dist = abs(o["limit_price"] - o["stop_price"])
        exp = o["limit_price"] + design * r_dist if o["direction"] == "long" else o["limit_price"] - design * r_dist
        assert abs(o["target_price"] - exp) < 1e-6, o["order_id"]
        assert o["seal_sha256"] == sw._signal_seal(
            o["symbol"], o["tf"], o["arch"], o["direction"], o["placed_bar_time"],
            o["limit_price"], o["stop_price"], o["target_price"]), o["order_id"]
    by_id = {o["order_id"]: o for o in live["pending_orders"]}
    assert abs(by_id["XRP-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00"]["base_score"] - 3.4345) < 1e-9
    assert abs(by_id["XRP-USD|1d|P3_SWEEP|long|2026-10-30 00:00:00"]["score"] - 3.5845) < 1e-9
    print("Test 28 PASS")


# Test 29 — no hindsight across repeated step_a calls, intrabar and gap-through
def test_29_step_a_no_hindsight_and_gap_exit():
    times = [f"2026-06-{i+1:02d} 00:00:00" for i in range(4)]

    def pos(stop=95.0):
        return {"order_id": "X|1h|MA7_TRUE|long|t", "symbol": "X", "market": "US", "tf": "1h",
                "arch": "MA7_TRUE", "direction": "long", "entry_time": times[0],
                "entry_price": 100.0, "stop_price": stop, "initial_stop": 95.0,
                "initial_stop_price": 95.0, "target_price": 120.0, "risk_capital": 1.5,
                "unrealized_r": 0.0, "mfe_r": 0.0, "bars_held": 0, "seal_sha256": "x"}

    closes = [100.0, 102.0, 109.0, 110.0]
    highs = [100.0, 103.0, 110.0, 111.0]
    lows = [100.0, 99.0, 104.0, 106.0]
    df = make_df(closes, highs=highs, lows=lows, times=times)
    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc("atr")] = 2.0
        df.iloc[i, df.columns.get_loc("ATR")] = 2.0

    L = _base_ledger()
    L["open_positions"] = [pos()]
    sw.step_a_update_positions(L, {("X", "1h"): df})
    held = L["open_positions"][0]
    # bar1 low 99 > 95 (no exit) -> bar2 mfe 1.8R ratchets to 100 + 1.10*5 = 105.5
    assert len(L["open_positions"]) == 1, L["closed_trades"]
    assert abs(held["stop_price"] - 105.5) < 1e-9, held["stop_price"]
    assert abs(held["mfe_r"] - 2.0) < 1e-9, held["mfe_r"]
    assert held["bars_held"] == 3, held["bars_held"]
    assert held["last_processed_bar_ts"] == times[3], held["last_processed_bar_ts"]
    assert abs(held["initial_stop_price"] - 95.0) < 1e-9
    assert abs(held["initial_stop"] - 95.0) < 1e-9

    # REPLAY the same scan 5x: bar1 (low 99) must NEVER be re-tested against 105.5
    for _ in range(5):
        sw.step_a_update_positions(L, {("X", "1h"): df})
        assert len(L["open_positions"]) == 1, L["closed_trades"]
        again = L["open_positions"][0]
        assert abs(again["stop_price"] - 105.5) < 1e-9, again["stop_price"]
        assert abs(again["mfe_r"] - 2.0) < 1e-9, again["mfe_r"]
        assert again["bars_held"] == 3, again["bars_held"]

    # INTRABAR: the ratchet of bar b is only valid from bar b+1 onwards
    t2 = ["2026-07-01 00:00:00", "2026-07-02 00:00:00", "2026-07-03 00:00:00"]
    df_i1 = make_df([100.0, 110.0, 110.0], highs=[100.0, 110.0, 110.0],
                    lows=[100.0, 99.0, 110.0], times=t2)
    for i in range(len(df_i1)):
        df_i1.iloc[i, df_i1.columns.get_loc("atr")] = 2.0
        df_i1.iloc[i, df_i1.columns.get_loc("ATR")] = 2.0
    p2 = pos()
    p2["entry_time"] = t2[0]
    Li = _base_ledger()
    Li["open_positions"] = [p2]
    sw.step_a_update_positions(Li, {("X", "1h"): df_i1.iloc[:2]})
    assert len(Li["open_positions"]) == 1, Li["closed_trades"]   # low 99 > 95, no exit
    assert abs(Li["open_positions"][0]["stop_price"] - 105.5) < 1e-9, Li["open_positions"][0]["stop_price"]
    # next bar opens ABOVE the ratchet (106) and dips to 103 -> intrabar stop at 105.5
    df_i2 = make_df([100.0, 110.0, 104.0], highs=[100.0, 110.0, 106.0],
                    lows=[100.0, 99.0, 103.0], opens=[100.0, 110.0, 106.0], times=t2)
    for i in range(len(df_i2)):
        df_i2.iloc[i, df_i2.columns.get_loc("atr")] = 2.0
        df_i2.iloc[i, df_i2.columns.get_loc("ATR")] = 2.0
    sw.step_a_update_positions(Li, {("X", "1h"): df_i2})
    assert len(Li["open_positions"]) == 0, Li["open_positions"]
    assert abs(Li["closed_trades"][0]["exit_price"] - 105.5) < 1e-9, Li["closed_trades"][0]

    # GAP-THROUGH: open 90 < stop 95 -> fill at the open (90), not at 95
    t3 = ["2026-08-01 00:00:00", "2026-08-02 00:00:00"]
    df_gap = make_df([100.0, 91.0], highs=[100.0, 92.0], lows=[100.0, 88.0],
                     opens=[100.0, 90.0], times=t3)
    pg = pos()
    pg["entry_time"] = t3[0]
    Lg = _base_ledger()
    Lg["open_positions"] = [pg]
    sw.step_a_update_positions(Lg, {("X", "1h"): df_gap})
    assert len(Lg["open_positions"]) == 0, Lg["open_positions"]
    tr = Lg["closed_trades"][0]
    assert tr["reason"] == "stop", tr["reason"]
    assert abs(tr["exit_price"] - 90.0) < 1e-9, tr["exit_price"]
    print("Test 29 PASS")


# Test 30 — real UTC normalisation in bars_after / cooldowns / block_until
def test_30_utc_normalisation():
    ist = sw._parse_utc("2026-10-30 15:15:00+05:30")
    assert ist == sw._parse_utc("2026-10-30 09:45:00+00:00"), ist
    assert ist.hour == 9 and ist.minute == 45
    ny = sw._parse_utc("2026-10-30 14:00:00-04:00")
    assert ny == sw._parse_utc("2026-10-30 18:00:00+00:00"), ny
    # naive timestamps are treated as UTC (never silently stripped/shifted)
    assert sw._parse_utc("2026-10-30 10:00:00") == sw._parse_utc("2026-10-30 10:00:00+00:00")
    assert sw._parse_utc("t0249") is None
    assert sw._parse_utc(None) is None

    t0 = "2026-10-30 15:15:00+05:30"      # 09:45 UTC
    t1 = "2026-10-30 14:00:00-04:00"      # 18:00 UTC
    t2 = "2026-10-30 20:00:00+00:00"      # 20:00 UTC
    df = make_df([100.0, 101.0, 102.0], times=[t0, t1, t2])
    got = [str(x) for x in sw.bars_after(df, "2026-10-30 10:00:00+00:00")["Time"]]
    assert got == [t1, t2], got
    # the old naive string comparison wrongly included t0 (09:45 UTC < 10:00 UTC)
    assert t0 > "2026-10-30 10:00:00"

    # 12h block_until is computed on real UTC instants
    now = datetime.datetime.now(datetime.timezone.utc)
    exit_utc = now - datetime.timedelta(hours=1)
    exit_ist = pd.Timestamp(exit_utc).tz_convert("Asia/Kolkata").isoformat()
    L = _base_ledger()
    L["block_stats"]["consecutive_losses"] = 3
    L["closed_trades"] = [{"pnl_r": -1.0, "symbol": f"S{i}", "direction": "long", "tf": "1d",
                           "exit_time": exit_ist} for i in range(3)]
    sw.update_block_stats(L)
    assert L["block_stats"]["blocked"] is True, L["block_stats"]
    assert L["block_stats"]["block_reason"] == "ELDER_12H_LOSS_COOLDOWN"
    assert sw._parse_utc(L["block_stats"]["block_until"]) == sw._parse_utc(exit_ist) + pd.Timedelta(hours=12)

    # cooldown comparisons across +05:30 / -04:00 offsets
    Lc = _base_ledger()
    Lc["block_stats"]["cooldowns"] = [{"symbol": "X", "direction": "long", "tf": "1d",
                                       "until": "2026-10-30 15:15:00+05:30"}]   # 09:45 UTC
    assert sw.in_cooldown(Lc, "X", "long", "2026-10-30 05:00:00-04:00") is True    # 09:00 UTC
    assert sw.in_cooldown(Lc, "X", "long", "2026-10-30 06:00:00-04:00") is False   # 10:00 UTC
    print("Test 30 PASS")


# Test 31 — monotonic bars_held, no shrink on window roll-off, no weekend inflation
def test_31_bars_held_monotonic():
    times6 = [f"2026-09-{i+1:02d} 00:00:00" for i in range(7)]
    # close 97.0 with stop 90 -> unreal -0.30R: outside the dead-money/harvest bands,
    # so the trade stays open while bars_held is exercised.
    df6 = make_df([100.0] + [97.0] * 6, highs=[100.5] + [97.5] * 6,
                  lows=[99.5] + [96.5] * 6, times=times6)
    p = {"order_id": "X|1d|MA7_TRUE|long|t", "symbol": "X", "market": "US", "tf": "1d",
         "arch": "MA7_TRUE", "direction": "long", "entry_time": times6[0], "entry_price": 100.0,
         "stop_price": 90.0, "initial_stop": 90.0, "initial_stop_price": 90.0,
         "target_price": 130.0, "risk_capital": 1.5, "unrealized_r": 0.0, "mfe_r": 0.0,
         "bars_held": 0, "seal_sha256": "x"}
    L = _base_ledger()
    L["open_positions"] = [p]
    sw.step_a_update_positions(L, {("X", "1d"): df6})
    assert L["open_positions"][0]["bars_held"] == 6, L["open_positions"][0]["bars_held"]

    # window roll-off: the frame now only carries 2 bars after entry -> must not shrink
    df2 = make_df([97.0] * 3, highs=[97.5] * 3, lows=[96.5] * 3, times=times6[-3:])
    sw.step_a_update_positions(L, {("X", "1d"): df2})
    assert L["open_positions"][0]["bars_held"] == 6, L["open_positions"][0]["bars_held"]

    # weekend: Friday entry, Monday bar -> ONE trading bar, never 3 calendar days
    fri = "2026-10-30 00:00:00"    # Friday
    mon = "2026-11-02 00:00:00"    # Monday
    dfw = make_df([100.0, 100.5], highs=[100.5, 101.0], lows=[99.5, 100.0], times=[fri, mon])
    pw = dict(p, entry_time=fri, bars_held=0)
    pw.pop("last_processed_bar_ts", None)
    Lw = _base_ledger()
    Lw["open_positions"] = [pw]
    sw.step_a_update_positions(Lw, {("X", "1d"): dfw})
    assert Lw["open_positions"][0]["bars_held"] == 1, Lw["open_positions"][0]["bars_held"]
    print("Test 31 PASS")


# Test 32 — MA7_TRUE (2.0R target, 2.95 max base) tradable at k = 1, 2, 3
def test_32_ma7_tradable_under_step_up():
    assert sw.compute_s_min(1) == 2.75 and sw.compute_min_rr(1) == 1.9
    assert sw.compute_s_min(2) == 2.9 and sw.compute_min_rr(2) == 2.0
    assert sw.compute_s_min(3) == 2.9 and sw.compute_min_rr(3) == 2.0
    assert sw.compute_s_min(9) == 2.9 and sw.compute_min_rr(9) == 2.0
    # the setup's own limits stay inside the step-up floors
    assert sw.ARCH_MAX_BASE_SCORE["MA7_TRUE"] >= sw.compute_s_min(2)
    assert sw.ARCH_TARGET_RR["MA7_TRUE"] >= sw.compute_min_rr(2)

    times = [f"2026-10-{i+1:02d} 00:00:00" for i in range(5)]
    df = make_df([100.0] * 5, highs=[101.0] * 5, lows=[90.0] * 5, times=times)
    for k in (1, 2, 3):
        L = _base_ledger()
        L["block_stats"]["consecutive_losses"] = k
        L["block_stats"]["blocked"] = False
        sw.update_block_stats(L)
        assert L["block_stats"]["s_min_current"] == sw.compute_s_min(k)
        assert L["block_stats"]["min_rr_current"] == sw.compute_min_rr(k)
        L["block_stats"]["blocked"] = False      # isolate the score/rr gates
        L["pending_orders"] = [{
            "order_id": f"M{k}|1d|MA7_TRUE|long|{times[2]}", "symbol": f"M{k}", "market": "US",
            "tf": "1d", "arch": "MA7_TRUE", "direction": "long", "placed_bar_time": times[2],
            "limit_price": 99.0, "stop_price": 95.0, "target_price": 107.0,
            "score": 2.95, "base_score": 2.95, "rr": 2.0, "seal_sha256": "x",
            "kreil": {"score_boost": 0.0},
        }]
        sw.step_b_fill_pending(L, {(f"M{k}", "1d"): df}, L["open_positions"])
        assert len(L["open_positions"]) == 1, (k, L["pending_orders"])
        assert L["open_positions"][0]["symbol"] == f"M{k}"
        assert L["open_positions"][0]["initial_stop_price"] == 95.0
        assert L["open_positions"][0]["last_processed_bar_ts"] == str(L["open_positions"][0]["entry_time"])
    print("Test 32 PASS")


# Test 33 — try_fill near-spot tolerance capped at 0.20R and latest bar only
def test_33_try_fill_cap_and_latest_bar_only():
    times = ["2026-11-10 00:00:00", "2026-11-11 00:00:00", "2026-11-12 00:00:00"]

    def order(direction, limit, stop):
        return {"order_id": "X|1d|MA7_TRUE|" + direction + "|t", "symbol": "X", "market": "US",
                "tf": "1d", "arch": "MA7_TRUE", "direction": direction,
                "placed_bar_time": times[0], "limit_price": limit, "stop_price": stop,
                "target_price": 110.0 if direction == "long" else 90.0}

    # r_dist 10, ATR 100 -> 0.25*ATR = 25 would be absurd; capped at 0.20R = 2.0
    assert min(0.25 * 100.0, 0.20 * 10.0) == 2.0
    o = order("long", 100.0, 90.0)
    df_out = make_df([100.0] * 3, highs=[101.0] * 3, lows=[102.5, 102.5, 102.5], times=times)
    assert sw.try_fill(o, df_out, "1d", atr=100.0)[0] is False      # 102.5 > 100 + 2.0
    df_in = make_df([100.0] * 3, highs=[101.0] * 3, lows=[102.5, 102.5, 101.9], times=times)
    filled, et, ep = sw.try_fill(o, df_in, "1d", atr=100.0)
    assert filled is True and et == times[2] and abs(ep - 100.0) < 1e-12, (filled, et, ep)

    # the near-spot band is NOT applied to historical bars (no retroactive fills)
    df_hist = make_df([100.0] * 3, highs=[101.0] * 3, lows=[101.9, 105.0, 105.0], times=times)
    assert sw.try_fill(o, df_hist, "1d", atr=100.0)[0] is False
    # strict touch still fills historical bars
    df_touch = make_df([100.0] * 3, highs=[101.0] * 3, lows=[105.0, 99.0, 105.0], times=times)
    filled_t, et_t, _ = sw.try_fill(o, df_touch, "1d", atr=100.0)
    assert filled_t is True and et_t == times[1], (filled_t, et_t)

    # short mirror: cap 2.0 on the latest bar only
    so = order("short", 100.0, 110.0)
    df_s_out = make_df([100.0] * 3, highs=[97.5, 97.5, 97.5], lows=[96.0] * 3, times=times)
    assert sw.try_fill(so, df_s_out, "1d", atr=100.0)[0] is False   # 97.5 < 100 - 2.0
    df_s_in = make_df([100.0] * 3, highs=[97.5, 97.5, 98.1], lows=[96.0] * 3, times=times)
    assert sw.try_fill(so, df_s_in, "1d", atr=100.0)[0] is True
    df_s_hist = make_df([100.0] * 3, highs=[98.1, 90.0, 90.0], lows=[96.0] * 3, times=times)
    assert sw.try_fill(so, df_s_hist, "1d", atr=100.0)[0] is False

    # no ATR: 0.20R band, never wider
    df_near = make_df([100.0, 100.0], highs=[100.6, 100.6], lows=[100.5, 100.5],
                      times=times[:2])
    assert sw.try_fill(order("long", 100.0, 96.0), df_near, "1d")[0] is True   # 0.8 = 0.20R
    df_far = make_df([100.0, 100.0], highs=[100.9, 100.9], lows=[100.81, 100.81],
                     times=times[:2])
    assert sw.try_fill(order("long", 100.0, 96.0), df_far, "1d")[0] is False   # > 0.8
    print("Test 33 PASS")


# Test 34 — sticky rejection: a rejected fill is never parked on the pending book
def test_34_sticky_rejection():
    times = ["2026-12-01 00:00:00", "2026-12-02 00:00:00"]
    entry_time = times[1]
    df = make_df([100.0, 100.0], highs=[101.0, 101.0], lows=[90.0, 90.0], times=times)

    def sig(oid):
        return {"order_id": oid, "symbol": "AMD", "market": "US", "tf": "1d",
                "arch": "MA7_TRUE", "direction": "long", "placed_bar_time": times[0],
                "limit_price": 99.0, "stop_price": 95.0, "target_price": 107.0,
                "score": 3.0, "base_score": 3.0, "rr": 2.0, "seal_sha256": "x"}

    rejected = None
    accepted = None
    for i in range(4000):
        oid = f"AMD|1d|MA7_TRUE|long|rej{i}"
        if sw.is_rejected(oid, entry_time):
            rejected = oid
            break
    for i in range(4000):
        oid = f"AMD|1d|MA7_TRUE|long|ok{i}"
        if not sw.is_rejected(oid, entry_time):
            accepted = oid
            break
    assert rejected is not None and accepted is not None
    assert sw.is_rejected(rejected, entry_time) is True
    assert sw.is_rejected(accepted, entry_time) is False

    L = _base_ledger()
    sw.intake_signals(L, {("AMD", "1d"): df}, [sig(rejected)], L["open_positions"], set())
    assert L["pending_orders"] == [], L["pending_orders"]      # NOT queued for next scan
    assert L["open_positions"] == [], L["open_positions"]

    L2 = _base_ledger()
    sw.intake_signals(L2, {("AMD", "1d"): df}, [sig(accepted)], L2["open_positions"], set())
    assert len(L2["open_positions"]) == 1, L2
    assert L2["pending_orders"] == []
    assert L2["open_positions"][0]["entry_time"] == entry_time
    assert L2["open_positions"][0]["initial_stop_price"] == 95.0
    print("Test 34 PASS")


# Test 35 — Medium #8: a newer bar supersedes the stale resting order on the
# same symbol (order_id already carries the signal time), and per-symbol
# dedup means the pending book never stacks two orders for one symbol.
def test_35_pending_supersede_same_symbol():
    times = ["2026-12-01 00:00:00", "2026-12-02 00:00:00", "2026-12-03 00:00:00"]
    df = make_df([100.0] * 3, highs=[101.0] * 3, lows=[99.0] * 3, times=times)
    data = {("AMD", "1d"): df, ("NVDA", "1d"): df}

    def sig(bar, score, limit, stop, oid=None, symbol="AMD", arch="MA7_TRUE", direction="long"):
        oid = oid or f"{symbol}|1d|{arch}|{direction}|{bar}"
        return {"order_id": oid, "symbol": symbol, "market": "US", "tf": "1d",
                "arch": arch, "direction": direction, "placed_bar_time": bar,
                "limit_price": limit, "stop_price": stop,
                "target_price": limit + 2.0 * (limit - stop),
                "score": score, "base_score": score, "rr": 2.0, "seal_sha256": "x"}

    # order_id must encode the signal time: same setup on two bars = two ids
    s_a = sig(times[0], 3.0, 90.0, 86.0)
    s_b = sig(times[1], 3.0, 90.0, 86.0)
    assert s_a["order_id"] != s_b["order_id"]
    assert s_a["order_id"].endswith(times[0]) and s_b["order_id"].endswith(times[1])

    # (a) newer bar + higher score replaces the resting order
    L = _base_ledger()
    sw.intake_signals(L, data, [s_a], L["open_positions"], set())
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    assert L["pending_orders"][0]["placed_bar_time"] == times[0]
    sw.intake_signals(L, data, [sig(times[1], 3.5, 90.0, 86.0)], L["open_positions"], set())
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    assert L["pending_orders"][0]["placed_bar_time"] == times[1]
    assert L["pending_orders"][0]["score"] == 3.5

    # (b) newer bar + lower score but clean ratchet (entry up, stop up) replaces
    L = _base_ledger()
    sw.intake_signals(L, data, [s_a], L["open_positions"], set())
    sw.intake_signals(L, data, [sig(times[1], 2.7, 92.0, 88.0)], L["open_positions"], set())
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    assert L["pending_orders"][0]["placed_bar_time"] == times[1]
    assert L["pending_orders"][0]["limit_price"] == 92.0
    assert L["pending_orders"][0]["stop_price"] == 88.0

    # (c) newer bar + lower score + dirty shift (entry down, stop down) keeps old
    L = _base_ledger()
    sw.intake_signals(L, data, [s_a], L["open_positions"], set())
    sw.intake_signals(L, data, [sig(times[1], 2.7, 88.0, 84.0)], L["open_positions"], set())
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    assert L["pending_orders"][0]["placed_bar_time"] == times[0]
    assert L["pending_orders"][0]["limit_price"] == 90.0

    # (d) same bar, different order_id, lower score -> never stacks a second order
    L = _base_ledger()
    sw.intake_signals(L, data, [sig(times[0], 3.0, 90.0, 86.0, oid="A1")], L["open_positions"], set())
    sw.intake_signals(L, data, [sig(times[0], 2.7, 90.0, 86.0, oid="A2")], L["open_positions"], set())
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    assert L["pending_orders"][0]["order_id"] == "A1"

    # (e) an older bar never evicts a fresher resting order, even with a
    #     higher score and a clean shift
    L = _base_ledger()
    sw.intake_signals(L, data, [sig(times[2], 3.0, 90.0, 86.0)], L["open_positions"], set())
    sw.intake_signals(L, data, [sig(times[0], 3.9, 92.0, 88.0)], L["open_positions"], set())
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    assert L["pending_orders"][0]["placed_bar_time"] == times[2]
    assert L["pending_orders"][0]["score"] == 3.0

    # (f) different symbols still both rest
    L = _base_ledger()
    sw.intake_signals(L, data, [s_a, sig(times[0], 2.8, 90.0, 86.0, symbol="NVDA")],
                      L["open_positions"], set())
    assert len(L["pending_orders"]) == 2, L["pending_orders"]
    assert {o["symbol"] for o in L["pending_orders"]} == {"AMD", "NVDA"}

    # (g) short side: newer bar + clean ratchet (entry down, stop down) replaces
    L = _base_ledger()
    sw.intake_signals(L, data, [sig(times[0], 3.0, 110.0, 114.0, direction="short")],
                      L["open_positions"], set())
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    sw.intake_signals(L, data, [sig(times[1], 2.7, 108.0, 112.0, direction="short")],
                      L["open_positions"], set())
    assert len(L["pending_orders"]) == 1, L["pending_orders"]
    assert L["pending_orders"][0]["limit_price"] == 108.0
    assert L["pending_orders"][0]["stop_price"] == 112.0

    # (h) superseded ids leave the dedup set so the fresh order is not blocked
    L = _base_ledger()
    ids = set()
    sw.intake_signals(L, data, [s_a], L["open_positions"], ids)
    assert s_a["order_id"] in ids
    sw.intake_signals(L, data, [sig(times[1], 3.5, 90.0, 86.0)], L["open_positions"], ids)
    assert s_a["order_id"] not in ids
    assert len(ids) == 1
    print("Test 35 PASS")


if __name__ == "__main__":
    test_1_no_e1()
    test_2_ma7_true()
    test_3_f3_p3()
    test_4_soros()
    test_5_elder_2pct_shark()
    test_6_piranha_3loss()
    test_7_s_min()
    test_8_multi_slot_market_buckets()
    test_9_cooldown()
    test_10_pending_cap_schema()
    test_11_no_retroactive_fill_and_min_rr()
    test_12_chandelier_ratchet()
    test_13_kreil_metrics_and_sweet_spot()
    test_14_kreil_rejects_dead_and_exhausted_atr()
    test_15_kreil_directional_skew_and_boost()
    test_16_step_b_prunes_non_kreil_pending()
    test_17_spot_prices_and_seal_integrity()
    test_18_kreil_score_boost_rescues_signal_above_s_min()
    test_19_nan_fail_closed_and_intrabar_order()
    test_20_kreil_base_score_no_cumulative_inflation()
    test_21_bucket_helpers_and_ledger_migration()
    test_22_profit_lock_ratchet_ladder()
    test_23_time_profit_harvest()
    test_24_dead_money_exit()
    test_25_loss_block_12h_expiry()
    test_26_stop_floor_and_target_multiples()
    test_27_near_spot_fill_and_pending_lifetime()
    test_28_historical_inflation_and_target_rr_migration()
    test_29_step_a_no_hindsight_and_gap_exit()
    test_30_utc_normalisation()
    test_31_bars_held_monotonic()
    test_32_ma7_tradable_under_step_up()
    test_33_try_fill_cap_and_latest_bar_only()
    test_34_sticky_rejection()
    test_35_pending_supersede_same_symbol()
    print("ALL 35 TESTS PASSED")
