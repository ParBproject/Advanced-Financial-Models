import math
import unittest

from financial_models.cash_flow import CashFlowAssumptions, forecast_cash_flow
from financial_models.credit_risk import (
    Loan,
    assess_loan,
    probability_of_default,
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

    def test_non_finite_cash_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "starting cash must be finite"):
            forecast_cash_flow(CashFlowAssumptions(starting_cash=float("nan")))

    def test_boolean_capex_month_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "capex month"):
            forecast_cash_flow(capex_by_month={True: 1_000.0})

    def test_month_twelve_revenue_compounds_eleven_times(self):
        result = forecast_cash_flow()

        self.assertAlmostEqual(result.periods[-1].revenue, 100_000.0 * (1.05**11))

    def test_npv_uses_equivalent_monthly_discount_rate(self):
        result = forecast_cash_flow(
            CashFlowAssumptions(),
            months=1,
            capex_by_month={},
        )
        monthly_rate = (1.10 ** (1.0 / 12.0)) - 1.0

        self.assertAlmostEqual(result.periods[0].net_cash_flow, 27_000.0)
        self.assertAlmostEqual(result.npv_of_net_cash_flows, 27_000.0 / (1.0 + monthly_rate))


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

    def test_non_finite_exposure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exposure must be finite"):
            assess_loan(Loan("L-001", "Example", float("nan"), 620))

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

        weights = [asset.weight for asset in workbook_balanced_portfolio()]
        returns = [asset.expected_return for asset in workbook_balanced_portfolio()]
        volatilities = [asset.volatility for asset in workbook_balanced_portfolio()]
        expected_return = sum(weight * ret for weight, ret in zip(weights, returns, strict=True))
        variance = sum((weight * vol) ** 2 for weight, vol in zip(weights, volatilities, strict=True))
        volatility = math.sqrt(variance)

        self.assertAlmostEqual(metrics.expected_return, expected_return)
        self.assertAlmostEqual(metrics.expected_return, 0.0864)
        self.assertAlmostEqual(metrics.volatility, volatility)
        self.assertAlmostEqual(round(metrics.volatility * 100, 2), 6.82)
        self.assertAlmostEqual(metrics.sharpe_ratio, (expected_return - 0.03) / volatility)
        self.assertAlmostEqual(round(metrics.sharpe_ratio, 2), 0.83)
        self.assertAlmostEqual(metrics.future_value, 1_000_000.0 * ((1.0 + expected_return) ** 10))
        self.assertEqual(round(metrics.future_value), 2_290_327)

    def test_zero_volatility_sharpe_sign_follows_excess_return(self):
        above = portfolio_metrics(
            (AssetAllocation("Cash", 0.05, 0.0, 1.0),),
            risk_free_rate=0.03,
            horizon_years=1,
        )
        below = portfolio_metrics(
            (AssetAllocation("Cash", 0.01, 0.0, 1.0),),
            risk_free_rate=0.03,
            horizon_years=1,
        )
        tied = portfolio_metrics(
            (AssetAllocation("Cash", 0.03, 0.0, 1.0),),
            risk_free_rate=0.03,
            horizon_years=1,
        )

        self.assertEqual(above.sharpe_ratio, math.inf)
        self.assertEqual(below.sharpe_ratio, -math.inf)
        self.assertEqual(tied.sharpe_ratio, 0.0)

    def test_compounding_rejects_total_loss_or_worse(self):
        with self.assertRaisesRegex(ValueError, "greater than -100%"):
            portfolio_metrics(
                (AssetAllocation("Wipeout", -1.0, 0.10, 1.0),),
                horizon_years=1,
            )

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


if __name__ == "__main__":
    unittest.main()
