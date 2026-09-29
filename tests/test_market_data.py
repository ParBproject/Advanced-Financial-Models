import math
import unittest

import numpy as np
import pandas as pd

from financial_models.market_data import (
    align_prices,
    annualize_sample,
    backtest_rebalanced_portfolio,
    cumulative_wealth,
    download_adjusted_close,
    historical_risk_summary,
    historical_var_expected_shortfall,
    max_drawdown,
    performance_summary,
    portfolio_return_series,
    simple_returns,
    worked_example_prices,
)


class MarketDataTests(unittest.TestCase):
    def test_simple_returns_match_the_first_price_change(self):
        returns = simple_returns(worked_example_prices())
        self.assertEqual(returns.shape, (4, 2))
        self.assertAlmostEqual(returns.iloc[0]["AAA"], 0.02)

    def test_missing_prices_are_rejected_instead_of_filled(self):
        prices = worked_example_prices()
        prices.iloc[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            simple_returns(prices)

        aligned = align_prices(prices)
        self.assertEqual(len(aligned), 4)
        self.assertNotIn(prices.index[2], aligned.index)
        self.assertFalse(aligned.isna().any(axis=None))

    def test_price_drawdown_is_peak_to_trough(self):
        prices = pd.DataFrame({"AAA": [100.0, 120.0, 90.0]})
        summary = historical_risk_summary(prices, periods_per_year=1)
        self.assertAlmostEqual(float(summary.max_drawdown["AAA"]), 90.0 / 120.0 - 1.0)
        self.assertAlmostEqual(max_drawdown([0.20, -0.25]), -0.25)

    def test_historical_var_and_expected_shortfall_use_the_worst_tail(self):
        value_at_risk, expected_shortfall = historical_var_expected_shortfall(
            [-0.10, -0.06, -0.02, 0.01, 0.03, 0.04],
            confidence=0.80,
        )
        self.assertAlmostEqual(value_at_risk, 0.06)
        self.assertAlmostEqual(expected_shortfall, 0.08)
        # (1 - 0.80) * 6 = 1.2, so the tail is the worst 2 returns.
        # Both have full weight: (0.10 + 0.06) / 2 = 0.08, not a fractional blend.

        gain_var, gain_shortfall = historical_var_expected_shortfall(
            [0.01, 0.02, 0.03, 0.04],
            confidence=0.75,
        )
        self.assertAlmostEqual(gain_var, -0.01)
        self.assertAlmostEqual(gain_shortfall, -0.01)

    def test_short_samples_can_stay_in_period_units(self):
        self.assertFalse(annualize_sample(4))
        self.assertTrue(annualize_sample(60))
        returns = portfolio_return_series(simple_returns(worked_example_prices()), [0.5, 0.5])
        values = returns.to_numpy(dtype=float)
        summary = performance_summary(
            returns,
            periods_per_year=252,
            risk_free_rate=0.03,
            annualize=False,
        )
        threshold = 0.03 / 252
        excess = values - threshold
        sample_volatility = float(np.std(values, ddof=1))
        self.assertAlmostEqual(summary.annualized_volatility, sample_volatility)
        self.assertAlmostEqual(summary.sharpe_ratio, float(excess.mean()) / sample_volatility)
        self.assertAlmostEqual(summary.cagr, summary.cumulative_return)
        annualized = performance_summary(returns, periods_per_year=252, risk_free_rate=0.03)
        self.assertGreater(annualized.cagr, 1.0)
        table = historical_risk_summary(worked_example_prices(), periods_per_year=252)
        self.assertGreater(float(table.annualized_returns["AAA"]), 1.0)

    def test_sortino_uses_the_risk_free_threshold(self):
        summary = performance_summary(
            [0.10, -0.05, 0.02],
            periods_per_year=1,
            risk_free_rate=0.01,
            confidence=0.67,
        )
        downside = math.sqrt((0.06**2) / 3.0)
        self.assertAlmostEqual(summary.sortino_ratio, (0.04 / 3.0) / downside)
        self.assertLess(summary.max_drawdown, 0.0)
        self.assertGreaterEqual(summary.expected_shortfall, summary.value_at_risk)

    def test_zero_excess_volatility_sharpe_is_negative_infinity(self):
        summary = performance_summary([0.01, 0.01], periods_per_year=1, risk_free_rate=0.02)
        self.assertEqual(summary.sharpe_ratio, float("-inf"))
        self.assertAlmostEqual(summary.sortino_ratio, -1.0)

    def test_non_finite_returns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            performance_summary([0.01, float("nan")])
        with self.assertRaisesRegex(ValueError, "-100%"):
            performance_summary([0.01, -1.01])

    def test_cumulative_wealth_matches_compounding(self):
        wealth = cumulative_wealth([0.10, -0.05, 0.02], initial_value=100.0)
        self.assertAlmostEqual(wealth[-1], 100.0 * 1.10 * 0.95 * 1.02)

    def test_portfolio_returns_require_weights_that_sum_to_one(self):
        returns = simple_returns(worked_example_prices())
        portfolio = portfolio_return_series(returns, [0.6, 0.4])
        expected = returns.iloc[0]["AAA"] * 0.6 + returns.iloc[0]["BBB"] * 0.4
        self.assertAlmostEqual(portfolio.iloc[0], expected)
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            portfolio_return_series(returns, [0.5, 0.4])

    def test_rebalance_cost_matches_one_way_turnover(self):
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 110.0, 110.0],
                "BBB": [100.0, 100.0, 100.0],
            }
        )
        result = backtest_rebalanced_portfolio(
            prices,
            [0.5, 0.5],
            rebalance_every=1,
            transaction_cost_bps=100.0,
        )
        self.assertAlmostEqual(result.transaction_costs.iloc[0], 0.00025)
        self.assertAlmostEqual(result.wealth.iloc[-1], 1.04975)
        self.assertAlmostEqual(result.transaction_costs.iloc[-1], 0.0)

    def test_daily_rebalance_without_costs_matches_target_weights(self):
        prices = worked_example_prices()
        returns = simple_returns(prices)
        expected = portfolio_return_series(returns, [0.6, 0.4])
        result = backtest_rebalanced_portfolio(
            prices,
            [0.6, 0.4],
            rebalance_every=1,
            transaction_cost_bps=0.0,
        )
        np.testing.assert_allclose(result.returns, expected)
        self.assertAlmostEqual(result.total_transaction_cost, 0.0)

    def test_download_requires_a_ticker_and_yfinance(self):
        with self.assertRaisesRegex(ValueError, "ticker"):
            download_adjusted_close([" "], start="2020-01-01")
        import sys

        sys.modules["yfinance"] = None
        try:
            with self.assertRaises(ImportError):
                download_adjusted_close(["SPY"], start="2020-01-01")
        finally:
            sys.modules.pop("yfinance", None)


if __name__ == "__main__":
    unittest.main()
