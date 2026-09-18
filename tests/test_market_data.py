import unittest

import numpy as np
import pandas as pd

from financial_models.market_data import (
    backtest_rebalanced_portfolio,
    cumulative_wealth,
    historical_risk_summary,
    historical_var_expected_shortfall,
    performance_summary,
    portfolio_return_series,
    simple_returns,
)


class MarketDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = pd.DataFrame(
            {
                "AAA": [100.0, 102.0, 101.0, 104.0, 106.0],
                "BBB": [50.0, 49.0, 51.0, 52.0, 51.5],
            },
            index=pd.date_range("2025-01-01", periods=5, freq="D"),
        )

    def test_simple_returns_have_expected_shape(self) -> None:
        returns = simple_returns(self.prices)
        self.assertEqual(returns.shape, (4, 2))
        self.assertAlmostEqual(returns.iloc[0]["AAA"], 0.02)

    def test_historical_summary_is_covariance_consistent(self) -> None:
        summary = historical_risk_summary(self.prices, periods_per_year=252)
        self.assertEqual(summary.annualized_covariance.shape, (2, 2))
        self.assertTrue(np.allclose(summary.annualized_covariance, summary.annualized_covariance.T))
        self.assertTrue((summary.annualized_volatility >= 0).all())
        self.assertTrue((summary.max_drawdown <= 0).all())

    def test_portfolio_return_series_requires_fully_invested_weights(self) -> None:
        returns = simple_returns(self.prices)
        portfolio = portfolio_return_series(returns, [0.6, 0.4])
        expected_first = returns.iloc[0]["AAA"] * 0.6 + returns.iloc[0]["BBB"] * 0.4
        self.assertAlmostEqual(portfolio.iloc[0], expected_first)
        with self.assertRaises(ValueError):
            portfolio_return_series(returns, [0.5, 0.4])

    def test_historical_var_and_expected_shortfall(self) -> None:
        returns = pd.Series([-0.10, -0.06, -0.02, 0.01, 0.03, 0.04])
        var, expected_shortfall = historical_var_expected_shortfall(returns, confidence=0.80)
        self.assertGreaterEqual(var, 0.0)
        self.assertGreaterEqual(expected_shortfall, var)

    def test_performance_summary_reports_growth_and_downside_metrics(self) -> None:
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.015])
        summary = performance_summary(
            returns,
            periods_per_year=12,
            risk_free_rate=0.0,
            confidence=0.80,
        )
        self.assertGreater(summary.cumulative_return, 0.0)
        self.assertGreater(summary.cagr, 0.0)
        self.assertGreater(summary.annualized_volatility, 0.0)
        self.assertLessEqual(summary.max_drawdown, 0.0)
        self.assertGreaterEqual(summary.expected_shortfall, summary.value_at_risk)

    def test_cumulative_wealth_matches_compounding(self) -> None:
        wealth = cumulative_wealth([0.10, -0.05, 0.02], initial_value=100.0)
        expected = 100.0 * 1.10 * 0.95 * 1.02
        self.assertAlmostEqual(wealth[-1], expected)

    def test_returns_below_negative_one_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            performance_summary([0.01, -1.01])


    def test_daily_rebalancing_without_costs_matches_weighted_returns(self) -> None:
        returns = simple_returns(self.prices)
        expected = portfolio_return_series(returns, [0.6, 0.4])
        result = backtest_rebalanced_portfolio(
            self.prices,
            [0.6, 0.4],
            rebalance_every=1,
            transaction_cost_bps=0.0,
        )
        self.assertTrue(np.allclose(result.returns, expected))
        self.assertAlmostEqual(result.total_transaction_cost, 0.0)

    def test_transaction_costs_reduce_backtest_wealth(self) -> None:
        free = backtest_rebalanced_portfolio(
            self.prices,
            [0.5, 0.5],
            rebalance_every=1,
            transaction_cost_bps=0.0,
        )
        costly = backtest_rebalanced_portfolio(
            self.prices,
            [0.5, 0.5],
            rebalance_every=1,
            transaction_cost_bps=25.0,
        )
        self.assertGreater(costly.total_turnover, 0.0)
        self.assertGreater(costly.total_transaction_cost, 0.0)
        self.assertLess(costly.wealth.iloc[-1], free.wealth.iloc[-1])


if __name__ == "__main__":
    unittest.main()
