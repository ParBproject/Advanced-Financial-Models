import math
import unittest

from financial_models.cash_flow import CashFlowAssumptions, forecast_cash_flow
from financial_models.credit_risk import (
    Loan,
    assess_loan,
    probability_of_default,
    risk_rating,
    summarize_portfolio,
)
from financial_models.portfolio import (
    AssetAllocation,
    portfolio_metrics,
    workbook_balanced_portfolio,
)


class CashFlowTests(unittest.TestCase):
    def test_base_forecast_matches_documented_month_one_math(self):
        result = forecast_cash_flow(CashFlowAssumptions())
        first = result.periods[0]

        self.assertAlmostEqual(first.revenue, 100_000.0)
        self.assertAlmostEqual(first.operating_expense, 60_000.0)
        self.assertAlmostEqual(first.net_cash_flow, 27_000.0)
        self.assertAlmostEqual(first.ending_cash, 77_000.0)
        self.assertEqual(len(result.periods), 12)

    def test_default_capex_is_applied_to_documented_months(self):
        result = forecast_cash_flow()
        capex = {period.month: period.capex for period in result.periods}

        self.assertEqual(capex[2], 15_000.0)
        self.assertEqual(capex[5], 15_000.0)
        self.assertEqual(capex[9], 15_000.0)
        self.assertEqual(capex[1], 0.0)

    def test_invalid_operating_expense_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "operating expense ratio"):
            forecast_cash_flow(CashFlowAssumptions(operating_expense_ratio=1.1))

    def test_monthly_discount_uses_the_effective_annual_rate(self):
        result = forecast_cash_flow(
            CashFlowAssumptions(
                starting_cash=0.0,
                revenue_month1=100.0,
                monthly_revenue_growth=0.0,
                operating_expense_ratio=0.0,
                monthly_loan_payment=0.0,
                monthly_rent=0.0,
                annual_discount_rate=0.10,
            ),
            months=1,
            capex_by_month={},
        )

        self.assertAlmostEqual(result.periods[0].net_cash_flow, 100.0)
        self.assertAlmostEqual(result.npv_of_net_cash_flows, 100.0 / (1.10 ** (1.0 / 12.0)))

    def test_non_finite_discount_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            forecast_cash_flow(CashFlowAssumptions(annual_discount_rate=float("nan")))


class CreditRiskTests(unittest.TestCase):
    def test_documented_credit_score_bands(self):
        self.assertEqual(probability_of_default(820), 0.01)
        self.assertEqual(probability_of_default(620), 0.15)
        self.assertEqual(probability_of_default(520), 0.35)
        self.assertEqual(probability_of_default(480), 0.50)

    def test_expected_loss_matches_guide_example(self):
        result = assess_loan(Loan("L-001", "Example", 100_000.0, 620))

        self.assertAlmostEqual(result.expected_loss, 6_750.0)
        self.assertEqual(result.probability_of_default, 0.15)
        self.assertEqual(result.loss_given_default, 0.45)
        self.assertEqual(result.risk_rating, "Medium")

    def test_risk_rating_uses_workbook_thresholds(self):
        self.assertEqual(risk_rating(0.05), "Low")
        self.assertEqual(risk_rating(0.07), "Medium")
        self.assertEqual(risk_rating(0.16), "Medium")
        self.assertEqual(risk_rating(0.20), "High")

    def test_non_finite_exposure_and_fractional_score_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            assess_loan(Loan("L-1", "Borrower", float("nan"), 700))
        with self.assertRaisesRegex(TypeError, "integer"):
            probability_of_default(620.4)

    def test_portfolio_summary_aggregates_exposure_and_risk_counts(self):
        summary = summarize_portfolio(
            [
                Loan("L-1", "Low", 50_000.0, 780),
                Loan("L-2", "Medium", 100_000.0, 620),
                Loan("L-3", "High", 150_000.0, 520),
            ]
        )

        self.assertAlmostEqual(summary.total_exposure, 300_000.0)
        self.assertEqual(summary.low_risk_count, 1)
        self.assertEqual(summary.medium_risk_count, 1)
        self.assertEqual(summary.high_risk_count, 1)


class PortfolioTests(unittest.TestCase):
    def test_balanced_portfolio_weights_sum_to_one(self):
        assets = workbook_balanced_portfolio()
        self.assertAlmostEqual(sum(asset.weight for asset in assets), 1.0)

    def test_balanced_portfolio_return_uses_weighted_average(self):
        metrics = portfolio_metrics(workbook_balanced_portfolio())

        self.assertAlmostEqual(metrics.expected_return, 0.0864)
        self.assertGreater(metrics.volatility, 0.0)
        self.assertTrue(math.isfinite(metrics.sharpe_ratio))
        self.assertGreater(metrics.future_value, 1_000_000.0)

    def test_portfolio_rejects_weights_that_do_not_sum_to_one(self):
        assets = list(workbook_balanced_portfolio())
        assets[0] = type(assets[0])(
            assets[0].name,
            assets[0].expected_return,
            assets[0].volatility,
            0.25,
        )
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            portfolio_metrics(assets)

    def test_zero_volatility_below_cash_has_negative_sharpe(self):
        lagging = portfolio_metrics(
            [AssetAllocation("Cash", 0.01, 0.0, 1.0)],
            risk_free_rate=0.03,
        )
        matching = portfolio_metrics(
            [AssetAllocation("Cash", 0.03, 0.0, 1.0)],
            risk_free_rate=0.03,
        )

        self.assertEqual(lagging.sharpe_ratio, float("-inf"))
        self.assertEqual(matching.sharpe_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
