import csv
import io
import json
import unittest

import numpy as np

from financial_models.advanced_portfolio import (
    correlated_portfolio_metrics,
    covariance_from_correlation,
    illustrative_correlation_matrix,
)
from financial_models.cash_flow import CashFlowAssumptions, forecast_cash_flow
from financial_models.credit_concentration import credit_concentration
from financial_models.credit_risk import Loan, summarize_portfolio
from financial_models.portfolio import PortfolioMetrics, workbook_balanced_portfolio
from financial_models.reporting import (
    build_executive_report,
    report_to_csv,
    report_to_json,
)
from financial_models.stress_testing import (
    CreditStressScenario,
    simulate_portfolio_terminal_values,
    stress_credit_portfolio,
)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.cash = forecast_cash_flow(CashFlowAssumptions())
        self.loans = (
            Loan("L-1", "Borrower A", 100_000.0, 780),
            Loan("L-2", "Borrower B", 150_000.0, 620),
            Loan("L-3", "Borrower C", 200_000.0, 520),
        )
        self.credit = summarize_portfolio(self.loans)
        self.concentration = credit_concentration(self.loans, top_n=2)
        self.assets = workbook_balanced_portfolio()
        self.covariance = covariance_from_correlation(
            [asset.volatility for asset in self.assets],
            illustrative_correlation_matrix(),
        )
        self.portfolio = correlated_portfolio_metrics(self.assets, self.covariance)
        self.credit_stress = stress_credit_portfolio(
            self.loans,
            CreditStressScenario("Recession", pd_multiplier=1.75, lgd_multiplier=1.25),
        )
        self.monte_carlo = simulate_portfolio_terminal_values(
            self.assets,
            self.covariance,
            n_simulations=500,
            random_state=123,
        )

    def build_report(self):
        return build_executive_report(
            self.cash,
            self.credit,
            self.portfolio,
            concentration=self.concentration,
            credit_stress=self.credit_stress,
            monte_carlo=self.monte_carlo,
            metadata={"scenario": "Base + recession", "model_version": np.int64(6)},
        )

    def test_report_contains_core_and_optional_sections(self):
        report = self.build_report()

        self.assertEqual(report["report_version"], "1.0")
        self.assertEqual(report["metadata"]["model_version"], 6)
        self.assertAlmostEqual(
            report["cash_flow"]["ending_cash"],
            self.cash.periods[-1].ending_cash,
        )
        self.assertAlmostEqual(
            report["credit"]["total_expected_loss"],
            self.credit.total_expected_loss,
        )
        self.assertAlmostEqual(
            report["credit_concentration"]["borrower_hhi"],
            self.concentration.herfindahl_hirschman_index,
        )
        self.assertEqual(report["credit_stress"]["scenario"], "Recession")
        self.assertAlmostEqual(
            report["portfolio_monte_carlo"]["median_terminal_value"],
            self.monte_carlo.median_terminal_value,
        )

    def test_json_export_is_deterministic_and_standards_compliant(self):
        report = self.build_report()

        first = report_to_json(report)
        second = report_to_json(report)
        parsed = json.loads(first)

        self.assertEqual(first, second)
        self.assertEqual(parsed["metadata"]["scenario"], "Base + recession")
        self.assertNotIn("NaN", first)
        self.assertNotIn("Infinity", first)

    def test_csv_export_flattens_nested_metrics(self):
        report = self.build_report()
        csv_text = report_to_csv(report)
        rows = list(csv.reader(io.StringIO(csv_text)))
        metrics = {row[0]: row[1] for row in rows[1:]}

        self.assertEqual(rows[0], ["metric", "value"])
        self.assertIn("cash_flow.ending_cash", metrics)
        self.assertIn("credit.risk_counts.high", metrics)
        self.assertIn("credit_concentration.borrower_hhi", metrics)
        self.assertIn("portfolio_monte_carlo.percentile_05", metrics)

    def test_non_finite_values_are_rejected(self):
        broken_portfolio = PortfolioMetrics(
            expected_return=0.08,
            volatility=0.0,
            sharpe_ratio=float("inf"),
            future_value=2_000_000.0,
            expected_gain=1_000_000.0,
        )

        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_executive_report(self.cash, self.credit, broken_portfolio)

    def test_json_export_rejects_non_finite_caller_payload(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            report_to_json({"metric": np.nan})

    def test_invalid_json_indent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            report_to_json({"ok": 1}, indent=-1)
        with self.assertRaisesRegex(TypeError, "non-negative integer"):
            report_to_json({"ok": 1}, indent=True)


if __name__ == "__main__":
    unittest.main()
