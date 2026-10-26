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
    times = [f"t{i}" for i in range(n)]
    df = pd.DataFrame({
        "Time": times,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })
    return sw.compute_indicators(df)


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
    closes = [100 + i * 0.5 for i in range(120)]
    df = make_df(closes)
    sigs = sw.run_detectors("AMD", df, "1h", "US")
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
    lows = [90.0] * 5
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
    assert len(po) <= 8
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
    print("ALL 10 TESTS PASSED")
