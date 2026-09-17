import unittest

import numpy as np
import pandas as pd

from financial_models.market_data import (
    historical_risk_summary,
    historical_var_expected_shortfall,
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


if __name__ == "__main__":
    unittest.main()
