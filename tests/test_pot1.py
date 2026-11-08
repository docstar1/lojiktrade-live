import unittest
from datetime import date
from pot1 import sleeve_update, next_rebalance_date

BASE = {"^NSEI": 100.0, "GOLDBEES.NS": 60.0, "INR=X": 88.0, "CL=F": 70.0}


class TestPot1(unittest.TestCase):
    def test_init(self):
        st = sleeve_update(None, dict(BASE), date(2026, 11, 9))
        self.assertAlmostEqual(st["nav"], 1.0, places=9)
        self.assertEqual(st["next_rebalance"], "2026-12-01")
        self.assertAlmostEqual(sum(st["target_weights"].values()), 1.0, places=9)

    def test_drift_and_return(self):
        st = sleeve_update(None, dict(BASE), date(2026, 11, 9))
        p2 = dict(BASE); p2["^NSEI"] = 110.0
        st2 = sleeve_update(st, p2, date(2026, 11, 10))
        self.assertAlmostEqual(st2["nav"], 1.05, places=9)  # 0.5*1.1 + 0.5*1.0
        self.assertAlmostEqual(st2["current_weights"]["^NSEI"], round(0.55 / 1.05, 4), places=9)
        self.assertFalse(st2["rebalanced_today"])

    def test_monthly_rebalance(self):
        st = sleeve_update(None, dict(BASE), date(2026, 11, 9))
        p2 = dict(BASE); p2["^NSEI"] = 110.0
        st2 = sleeve_update(st, p2, date(2026, 11, 10))
        st3 = sleeve_update(st2, p2, date(2026, 12, 1))
        self.assertTrue(st3["rebalanced_today"])
        self.assertAlmostEqual(st3["current_weights"]["^NSEI"], 0.5, places=9)
        self.assertEqual(st3["next_rebalance"], "2027-01-01")
        self.assertAlmostEqual(st3["nav"], 1.05, places=9)  # rebalance preserves NAV

    def test_next_rebalance_december(self):
        self.assertEqual(next_rebalance_date(date(2026, 12, 15)), date(2027, 1, 1))

    def test_missing_price_raises(self):
        with self.assertRaises(ValueError):
            sleeve_update(None, {"^NSEI": 100.0}, date(2026, 11, 9))


if __name__ == "__main__":
    unittest.main()
