import sys, os, json, math
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
    """250 bars with 10 fat-tail bars (5 up / 5 down) that trip the Kreil sweet-spot boost."""
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
    """250 bars with symmetric fat tails: Kreil gate passes but no directional boost."""
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
    assert f3 and f3[0]["rr"] >= 3.8, [s["arch"] for s in sigs3]

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
    assert p3 and p3[0]["rr"] >= 3.6, [s["arch"] for s in sigs4]
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
    assert t10 and t10[0]["rr"] >= 4.2, [s["arch"] for s in sigs]

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
    assert f1 and f1[0]["rr"] >= 3.8, [s["arch"] for s in sigs2]
    print("Test 4 PASS")


# Test 5
def test_5_elder_2pct_shark():
    L = {"account": {"equity": 100.0, "risk_pct": 0.05, "target": 1000.0}}
    sw.enforce_risk_pct(L)
    assert L["account"]["risk_pct"] == 0.02
    L2 = {"account": {"equity": 100.0, "risk_pct": 0.05, "target": 1000.0}}
    sw.enforce_risk_pct(L2)
    assert L2["account"]["risk_pct"] <= sw.MAX_RISK_PCT
    assert L2["account"]["risk_pct"] == 0.02
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
    score = 2.8
    assert score >= sw.get_s_min(0)
    assert score < sw.get_s_min(1)
    assert score < sw.get_s_min(2)
    assert sw.s_min_floor(0) == sw.get_s_min(0)

    # purge test
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(5)]
    closes = [100.0] * 5
    lows = [90.0] * 5
    highs = [101.0] * 5
    df = make_df(closes, highs=highs, lows=lows, times=times)
    data = {("X", "1d"): df}
    L = _base_ledger()
    L["block_stats"]["consecutive_losses"] = 1
    sw.update_block_stats(L)
    pending = [{"order_id": "X|1d|MA7_TRUE|long|t", "symbol": "X", "market": "US", "tf": "1d", "arch": "MA7_TRUE",
                "direction": "long", "placed_bar_time": times[2], "limit_price": 99.0, "stop_price": 95.0,
                "target_price": 113.0, "score": 2.8, "seal_sha256": "abc", "rr": 3.3}]
    L["pending_orders"] = pending
    sw.step_b_fill_pending(L, data, L["open_positions"])
    assert len(L["pending_orders"]) == 0, L["pending_orders"]
    assert len(L["open_positions"]) == 0, L["open_positions"]
    print("Test 7 PASS")


# Test 8
def test_8_global_single_position():
    times = [f"2026-01-{i+1:02d} 00:00:00" for i in range(5)]
    closes = [100.0] * 5
    lows = [90.0] * 5
    highs = [101.0] * 5
    df = make_df(closes, highs=highs, lows=lows, times=times)
    data = {("CL=F", "1h"): df, ("AMD", "1h"): df}
    L = _base_ledger()
    L["account"]["month_start_equity"] = 100.0
    pending = [
        {"order_id": "CL=F|1h|MA7_TRUE|long|2026-01-01 00:00:00", "symbol": "CL=F", "market": "FXCommod", "tf": "1h", "arch": "MA7_TRUE", "direction": "long", "placed_bar_time": "2026-01-01 00:00:00", "limit_price": 99.0, "stop_price": 95.0, "target_price": 113.0, "score": 3.0, "seal_sha256": "a", "rr": 3.5},
        {"order_id": "CL=F|1h|MA7_TRUE|long|2026-01-02 00:00:00", "symbol": "CL=F", "market": "FXCommod", "tf": "1h", "arch": "MA7_TRUE", "direction": "long", "placed_bar_time": "2026-01-02 00:00:00", "limit_price": 99.0, "stop_price": 95.0, "target_price": 113.0, "score": 2.9, "seal_sha256": "b", "rr": 3.5},
        {"order_id": "AMD|1h|MA7_TRUE|long|2026-01-01 00:00:00", "symbol": "AMD", "market": "US", "tf": "1h", "arch": "MA7_TRUE", "direction": "long", "placed_bar_time": "2026-01-01 00:00:00", "limit_price": 99.0, "stop_price": 95.0, "target_price": 113.0, "score": 2.8, "seal_sha256": "c", "rr": 3.5},
    ]
    L["pending_orders"] = pending
    sw.step_b_fill_pending(L, data, L["open_positions"])
    assert len(L["open_positions"]) == 1, L["open_positions"]
    cl_pending = [o for o in L["pending_orders"] if o["symbol"] == "CL=F"]
    assert len(cl_pending) <= 1
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
    entry_time = times[1]
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
    assert abs(L["open_positions"][0]["stop_price"] - 104.0) < 1e-9, L["open_positions"][0]["stop_price"]
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
    assert sw.DASHBOARD_HTML is None
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
        "arch": "MA7_TRUE", "direction": "long", "entry_time": times[1],
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
    assert abs(held["stop_price"] - 100.9) < 1e-9, held["stop_price"]
    print("Test 19 PASS")


if __name__ == "__main__":
    test_1_no_e1()
    test_2_ma7_true()
    test_3_f3_p3()
    test_4_soros()
    test_5_elder_2pct_shark()
    test_6_piranha_3loss()
    test_7_s_min()
    test_8_global_single_position()
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
    print("ALL 19 TESTS PASSED")
