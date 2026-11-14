# Live Forward Scanner Status

Last run: 2026-11-14T09:41:53.314883+00:00

## Account
- Equity: $100.00 (target $1000.00)
- Risk %: 3.0% (max 3.0%)
- Open positions: 1 / 3 (max 1 per market bucket)
- Pending orders: 1
- Closed trades: 0
- Block stats: {'consecutive_losses': 0, 's_min_current': 2.6, 'min_rr_current': 1.8, 'blocked': False, 'cooldowns': [], 'min_rr': 1.8, 'monthly_dd_pct': 0.0}
- Dynamic thresholds: s_min 2.6 | min_rr 1.8

## Open Positions
| Symbol | Market | TF | Arch | Dir | Entry | Stop | Target | Unreal R |
|---|---|---|---|---|---|---|---|---|
| HAL.NS | IndiaNSE | 1d | MA7_TRUE | long | 4874.7896 | 4625.8882 | 5372.5924 | -0.26 |

## Pending Orders
| Symbol | Market | TF | Arch | Dir | Limit | Stop | Target | Score |
|---|---|---|---|---|---|---|---|---|
| HAL.NS | IndiaNSE | 1d | MA7_TRUE | long | 4880.4175 | 4631.2461 | 5378.7604 | 2.80 |

## Recent Closed Trades
_None_


## Two-pot plan (2026-11-09)
- Pot 1 (income): buyable 3-leg sleeve - 50% Nifty ETF + 25% gold ETF + 25% S&P 500 in rupees (Indian index fund-of-fund; no USD, no oil), rebalanced monthly. Tracked by pot1.py in ledger.json under 'pot1'. Tested +10.9%/yr, maxDD -13% (5y).
- Pot 2 (growth): this engine, risk per trade raised 1.5% -> 3.0% per the 5-year walk-forward ($100 -> $235 at 3.0%, maxDD ~$14; 6% collapses). Raise to 4.5% only after live results confirm the backtest.
- Harvest floor raised 0.15R -> 0.50R (5y: banking at 0.15R cut winners). Soros gate dropped (5y: it blocked 917 of ~1,040 entries and cut $177 to $110).
- Pool restricted to IndiaNSE symbols (2026-11-09, cost model): US/crypto/FX legs stay out; existing non-NSE positions keep being managed until they close.