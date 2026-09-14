import unittest

import numpy as np

from financial_models.advanced_portfolio import (
    approximate_efficient_frontier,
    correlated_portfolio_metrics,
    covariance_from_correlation,
    illustrative_correlation_matrix,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    simulate_long_only_portfolios,
    validate_covariance_matrix,
)
from financial_models.portfolio import AssetAllocation, workbook_balanced_portfolio


class CovarianceTests(unittest.TestCase):
    def test_identity_correlation_produces_diagonal_covariance(self):
        volatilities = np.array([0.10, 0.20])
        covariance = covariance_from_correlation(volatilities, np.eye(2))

        np.testing.assert_allclose(covariance, np.diag([0.01, 0.04]))

    def test_known_two_asset_portfolio_matches_manual_variance(self):
        assets = (
            AssetAllocation("A", 0.08, 0.20, 0.50),
            AssetAllocation("B", 0.12, 0.30, 0.50),
        )
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])

        metrics = correlated_portfolio_metrics(
            assets,
            covariance,
            risk_free_rate=0.02,
            investment_amount=100_000.0,
            horizon_years=1,
        )

        self.assertAlmostEqual(metrics.expected_return, 0.10)
        self.assertAlmostEqual(metrics.volatility**2, 0.0375)
        self.assertAlmostEqual(metrics.sharpe_ratio, 0.08 / np.sqrt(0.0375))
        self.assertAlmostEqual(metrics.future_value, 110_000.0)

    def test_covariance_validation_rejects_asymmetric_matrix(self):
        covariance = np.array([[0.04, 0.01], [0.02, 0.09]])

        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_covariance_matrix(covariance, 2)

    def test_correlation_validation_rejects_non_psd_matrix(self):
        correlation = np.array(
            [
                [1.0, 0.9, 0.9],
                [0.9, 1.0, -0.9],
                [0.9, -0.9, 1.0],
            ]
        )

        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            covariance_from_correlation([0.1, 0.1, 0.1], correlation)

    def test_illustrative_correlation_is_valid_and_psd(self):
        correlation = illustrative_correlation_matrix()
        covariance = covariance_from_correlation(
            [asset.volatility for asset in workbook_balanced_portfolio()],
            correlation,
        )

        self.assertEqual(correlation.shape, (6, 6))
        np.testing.assert_allclose(np.diag(correlation), 1.0)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-12)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.assets = workbook_balanced_portfolio()
        self.covariance = covariance_from_correlation(
            [asset.volatility for asset in self.assets],
            illustrative_correlation_matrix(),
        )

    def test_simulation_is_reproducible_and_fully_invested(self):
        first = simulate_long_only_portfolios(
            self.assets,
            self.covariance,
            n_portfolios=500,
            random_state=123,
        )
        second = simulate_long_only_portfolios(
            self.assets,
            self.covariance,
            n_portfolios=500,
            random_state=123,
        )

        np.testing.assert_allclose(first.weights, second.weights)
        np.testing.assert_allclose(first.expected_returns, second.expected_returns)
        np.testing.assert_allclose(first.volatilities, second.volatilities)
        np.testing.assert_allclose(first.weights.sum(axis=1), 1.0)
        self.assertTrue(np.all(first.weights >= 0.0))
        self.assertTrue(np.all(first.weights <= 1.0))

    def test_selected_portfolios_match_simulation_extrema(self):
        simulation = simulate_long_only_portfolios(
            self.assets,
            self.covariance,
            n_portfolios=1_000,
            random_state=9,
        )

        max_sharpe = max_sharpe_portfolio(simulation)
        min_volatility = min_volatility_portfolio(simulation)

        self.assertAlmostEqual(max_sharpe.sharpe_ratio, float(simulation.sharpe_ratios.max()))
        self.assertAlmostEqual(
            min_volatility.volatility,
            float(simulation.volatilities.min()),
        )
        self.assertAlmostEqual(sum(max_sharpe.weights), 1.0)
        self.assertAlmostEqual(sum(min_volatility.weights), 1.0)

    def test_approximate_frontier_is_monotonic_and_non_dominated(self):
        simulation = simulate_long_only_portfolios(
            self.assets,
            self.covariance,
            n_portfolios=2_000,
            random_state=7,
        )
        frontier = approximate_efficient_frontier(simulation, max_points=15)

        self.assertGreaterEqual(len(frontier), 2)
        self.assertLessEqual(len(frontier), 15)
        for previous, current in zip(frontier, frontier[1:]):
            self.assertLessEqual(previous.volatility, current.volatility + 1e-12)
            self.assertLess(previous.expected_return, current.expected_return)

    def test_simulation_rejects_invalid_portfolio_count(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            simulate_long_only_portfolios(
                self.assets,
                self.covariance,
                n_portfolios=0,
            )


if __name__ == "__main__":
    unittest.main()
