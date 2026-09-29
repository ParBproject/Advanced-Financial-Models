import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from financial_models.credit_concentration import credit_concentration
from financial_models.credit_risk import assess_loan, summarize_portfolio
from financial_models.loan_book import (
    credit_book_memo_audit,
    format_credit_book_decision,
    loans_from_credit_book,
)


class CreditBookAdapterTests(unittest.TestCase):
    def test_book_feeds_expected_loss_and_concentration(self):
        loans = loans_from_credit_book()
        summary = summarize_portfolio(loans)
        concentration = credit_concentration(loans)
        first = assess_loan(loans[0])

        self.assertEqual(len(loans), 1000)
        self.assertEqual(len(summary.loans), 1000)
        self.assertEqual(loans[0].loan_id, "CUST_0000")
        self.assertEqual(loans[0].borrower, "CUST_0000")
        self.assertEqual(loans[0].credit_score, 720)
        self.assertAlmostEqual(loans[0].exposure, 105_858.9, places=2)
        self.assertAlmostEqual(first.probability_of_default, 0.04)
        self.assertAlmostEqual(first.expected_loss, 105_858.9 * 0.04 * 0.45, places=2)

        self.assertAlmostEqual(summary.total_exposure, 68_121_079.07, places=2)
        self.assertAlmostEqual(summary.total_expected_loss, 1_946_680.74, places=2)
        self.assertAlmostEqual(concentration.total_exposure, summary.total_exposure)
        self.assertAlmostEqual(
            concentration.herfindahl_hirschman_index,
            0.0013076635,
            places=8,
        )
        self.assertGreater(concentration.effective_borrower_count, 700)

    def test_memo_var_and_operational_risk_claims_are_file_facts(self):
        audit = credit_book_memo_audit()

        self.assertEqual(audit.loan_count, 1000)
        self.assertAlmostEqual(audit.total_exposure, 68_121_079.07, places=2)
        self.assertAlmostEqual(audit.average_reported_pd, 0.2980, places=4)
        self.assertEqual(audit.reported_pd_above_20_count, 504)
        self.assertAlmostEqual(audit.net_income_p05, 43_146.55, places=2)
        self.assertAlmostEqual(audit.net_income_p01, 25_890.70, places=2)
        self.assertEqual(audit.count_operational_risk_above_60, 91)
        self.assertEqual(audit.default_count_operational_risk_above_60, 27)
        self.assertEqual(audit.count_operational_risk_at_or_below_60, 909)
        self.assertEqual(audit.default_count_operational_risk_at_or_below_60, 271)
        self.assertAlmostEqual(
            audit.default_rate_operational_risk_above_60,
            27 / 91,
        )
        self.assertAlmostEqual(
            audit.default_rate_operational_risk_at_or_below_60,
            271 / 909,
        )
        self.assertLess(
            audit.default_rate_operational_risk_above_60
            / audit.default_rate_operational_risk_at_or_below_60,
            1.05,
        )
        self.assertAlmostEqual(audit.combined_stress_net_income, -12_559_240.00, places=2)

    def test_decision_line_states_library_totals_and_memo_corrections(self):
        loans = loans_from_credit_book()
        summary = summarize_portfolio(loans)
        concentration = credit_concentration(loans)
        audit = credit_book_memo_audit()
        line = format_credit_book_decision(summary, concentration, audit)

        self.assertIn("1,000-loan book", line)
        self.assertIn("$1,946,680.74", line)
        self.assertIn("0.001308", line)
        self.assertIn("$43,146.55", line)
        self.assertIn("$25,890.70", line)
        self.assertIn("not a loss VaR", line)
        self.assertIn("29.67%", line)
        self.assertIn("29.81%", line)
        self.assertNotIn("2.5", line)

    def test_missing_column_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "portfolio_data.csv"
            path.write_text("Customer_ID,Credit_Score\nCUST_1,700\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing column Loan_Amount"):
                loans_from_credit_book(path)


if __name__ == "__main__":
    unittest.main()
