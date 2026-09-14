import unittest

import numpy as np

from financial_models.advanced_portfolio import (
    covariance_from_correlation,
    illustrative_correlation_matrix,
)
from financial_models.cash_flow import CashFlowAssumptions
from financial_models.credit_risk import Loan
from financial_models.portfolio import AssetAllocation, workbook_balanced_portfolio
from financial_models.stress_testing import (
    CashFlowStressScenario,
    CreditStressScenario,
    simulate_portfolio_terminal_values,
    stress_cash_flow,
    stress_credit_portfolio,
)


class CashFlowStressTests(unittest.TestCase):
    def test_neutral_scenario_reproduces_baseline(self):
        result = stress_cash_flow(
            CashFlowAssumptions(),
            CashFlowStressScenario("Neutral"),
        )

        self.assertAlmostEqual(
            result.stressed.periods[-1].ending_cash,
            result.baseline.periods[-1].ending_cash,
        )
        self.assertAlmostEqual(
            result.stressed.npv_of_net_cash_flows,
            result.baseline.npv_of_net_cash_flows,
        )
        self.assertAlmostEqual(result.ending_cash_change, 0.0)
        self.assertAlmostEqual(result.npv_change, 0.0)

    def test_severe_scenario_reduces_liquidity_and_npv(self):
        scenario = CashFlowStressScenario(
            "Severe downturn",
            revenue_multiplier=0.80,
            monthly_growth_delta=-0.03,
            operating_expense_ratio_delta=0.10,
            fixed_cost_multiplier=1.10,
            capex_multiplier=1.20,
        )
        result = stress_cash_flow(CashFlowAssumptions(), scenario)

        self.assertLess(result.ending_cash_change, 0.0)
        self.assertLess(result.npv_change, 0.0)
        self.assertLess(
            result.stressed.periods[-1].ending_cash,
            result.baseline.periods[-1].ending_cash,
        )

    def test_invalid_stress_can_not_create_expense_ratio_above_one(self):
        scenario = CashFlowStressScenario(
            "Impossible OpEx",
            operating_expense_ratio_delta=0.50,
        )

        with self.assertRaisesRegex(ValueError, "operating expense ratio"):
            stress_cash_flow(CashFlowAssumptions(), scenario)


class CreditStressTests(unittest.TestCase):
    def setUp(self):
        self.loans = (
            Loan("L-1", "Strong", 100_000.0, 780),
            Loan("L-2", "Mid", 150_000.0, 620),
            Loan("L-3", "Weak", 200_000.0, 520),
        )

    def test_neutral_credit_scenario_matches_baseline_loss(self):
        result = stress_credit_portfolio(
            self.loans,
            CreditStressScenario("Neutral"),
        )

        self.assertAlmostEqual(
            result.stressed_expected_loss,
            result.baseline_expected_loss,
        )
        self.assertAlmostEqual(result.expected_loss_change, 0.0)

    def test_credit_stress_increases_expected_loss(self):
        result = stress_credit_portfolio(
            self.loans,
            CreditStressScenario(
                "Recession",
                pd_multiplier=1.75,
                lgd_multiplier=1.25,
            ),
        )

        self.assertGreater(result.stressed_expected_loss, result.baseline_expected_loss)
        self.assertGreater(result.stressed_expected_loss_ratio, 0.0)
        self.assertLessEqual(result.stressed_expected_loss, result.total_exposure)

    def test_pd_and_lgd_stress_are_capped_at_one(self):
        result = stress_credit_portfolio(
            [Loan("L-1", "Distressed", 100_000.0, 450)],
            CreditStressScenario("Extreme", pd_multiplier=10.0, lgd_multiplier=10.0),
        )
        loan = result.loans[0]

        self.assertEqual(loan.stressed_probability_of_default, 1.0)
        self.assertEqual(loan.stressed_loss_given_default, 1.0)
        self.assertEqual(loan.stressed_expected_loss, 100_000.0)


class PortfolioMonteCarloTests(unittest.TestCase):
    def test_zero_volatility_case_matches_exact_compounding(self):
        assets = (AssetAllocation("Cash-like", 0.05, 0.0, 1.0),)
        result = simulate_portfolio_terminal_values(
            assets,
            np.array([[0.0]]),
            initial_investment=100_000.0,
            horizon_years=3,
            n_simulations=100,
            random_state=7,
        )
        expected = 100_000.0 * (1.05**3)

        np.testing.assert_allclose(result.terminal_values, expected)
        self.assertAlmostEqual(result.mean_terminal_value, expected)
        self.assertAlmostEqual(result.probability_of_loss, 0.0)

    def test_seeded_monte_carlo_is_reproducible(self):
        assets = workbook_balanced_portfolio()
        covariance = covariance_from_correlation(
            [asset.volatility for asset in assets],
            illustrative_correlation_matrix(),
        )
        first = simulate_portfolio_terminal_values(
            assets,
            covariance,
            n_simulations=2_000,
            random_state=123,
        )
        second = simulate_portfolio_terminal_values(
            assets,
            covariance,
            n_simulations=2_000,
            random_state=123,
        )

        np.testing.assert_allclose(first.terminal_values, second.terminal_values)
        self.assertAlmostEqual(first.mean_terminal_value, second.mean_terminal_value)
        self.assertAlmostEqual(first.percentile_05, second.percentile_05)

    def test_percentiles_are_ordered_and_loss_probability_is_bounded(self):
        assets = workbook_balanced_portfolio()
        covariance = covariance_from_correlation(
            [asset.volatility for asset in assets],
            illustrative_correlation_matrix(),
        )
        result = simulate_portfolio_terminal_values(
            assets,
            covariance,
            n_simulations=3_000,
            random_state=99,
        )

        self.assertLessEqual(result.percentile_05, result.percentile_25)
        self.assertLessEqual(result.percentile_25, result.median_terminal_value)
        self.assertLessEqual(result.median_terminal_value, result.percentile_75)
        self.assertLessEqual(result.percentile_75, result.percentile_95)
        self.assertGreaterEqual(result.probability_of_loss, 0.0)
        self.assertLessEqual(result.probability_of_loss, 1.0)


if __name__ == "__main__":
    unittest.main()
