import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from financial_models.credit_concentration import credit_concentration
from financial_models.credit_risk import assess_loan, summarize_portfolio
from financial_models.loan_book import (
    credit_book_benchmark,
    credit_book_memo_audit,
    format_credit_book_benchmark,
    format_credit_book_decision,
    loans_from_credit_book,
)
from financial_models.stress_testing import CreditStressScenario, stress_credit_portfolio


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
        self.assertEqual(concentration.borrower_exposures[0].name, "CUST_0209")
        self.assertAlmostEqual(concentration.largest_borrower_share, 0.00258721, places=7)
        self.assertAlmostEqual(concentration.effective_borrower_count, 764.7227, places=4)

        stressed = stress_credit_portfolio(
            loans,
            CreditStressScenario("Severe", pd_multiplier=1.75, lgd_multiplier=1.25),
        )
        self.assertAlmostEqual(stressed.stressed_expected_loss, 4_258_364.1177, places=3)

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
        self.assertNotIn("retired memo", line.lower())
        self.assertIn("29.67%", line)
        self.assertIn("29.81%", line)
        self.assertNotIn("2.5", line)

    def test_score_bands_are_below_the_realized_default_rates(self):
        loans = loans_from_credit_book()
        summary = summarize_portfolio(loans)
        benchmark = credit_book_benchmark()

        self.assertAlmostEqual(benchmark.score_band_expected_loss, summary.total_expected_loss)
        self.assertEqual(benchmark.loan_count, 1000)
        self.assertEqual(benchmark.default_count, 298)
        self.assertAlmostEqual(benchmark.count_default_rate, 0.298)
        self.assertAlmostEqual(benchmark.exposure_weighted_default_rate, 0.17225862, places=7)
        self.assertAlmostEqual(benchmark.exposure_weighted_score_band_pd, 0.06350394, places=7)
        self.assertAlmostEqual(benchmark.realized_loss_at_lgd, 5_280_499.2645, places=3)
        # The stored PD matches the default flag in aggregate, to the cent.
        # It is not an out-of-sample forecast.
        self.assertAlmostEqual(benchmark.file_pd_expected_loss, 5_280_499.2708, places=3)
        self.assertLess(
            abs(benchmark.file_pd_expected_loss - benchmark.realized_loss_at_lgd),
            0.02,
        )
        self.assertAlmostEqual(benchmark.exposure_weighted_file_pd, 0.17225862, places=7)

        expected = {
            "800-850": (76, 0, 0.01),
            "750-799": (106, 2, 0.02),
            "700-749": (221, 23, 0.04),
            "650-699": (240, 54, 0.08),
            "600-649": (214, 106, 0.15),
            "550-599": (103, 75, 0.25),
            "500-549": (33, 31, 0.35),
            "300-499": (7, 7, 0.50),
        }
        self.assertEqual([band.label for band in benchmark.bands], list(expected))
        for band in benchmark.bands:
            count, defaults, assumed = expected[band.label]
            self.assertEqual(band.loan_count, count)
            self.assertEqual(band.default_count, defaults)
            self.assertEqual(band.assumed_pd, assumed)
            self.assertAlmostEqual(band.realized_default_rate, defaults / count)
        from_700_down = [band for band in benchmark.bands if band.assumed_pd >= 0.04]
        self.assertEqual(len(from_700_down), 6)
        for band in from_700_down:
            self.assertGreater(band.realized_default_rate, band.assumed_pd)

        line = format_credit_book_benchmark(benchmark)
        self.assertIn("$1,946,680.74", line)
        self.assertIn("$5,280,499.26", line)
        self.assertIn("29.80%", line)
        self.assertIn("17.23%", line)
        self.assertIn("6.35%", line)
        self.assertIn("not a walk-forward test", line)
        self.assertIn("not a second forecast", line)

    def test_band_calibration_matches_a_two_loan_hand_calculation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "book.csv"
            path.write_text(
                "Customer_ID,Credit_Score,Loan_Amount,Default,PD_Score\n"
                "A,820,100,0,0.10\n"
                "B,620,100,1,0.90\n",
                encoding="utf-8",
            )
            benchmark = credit_book_benchmark(path, loss_given_default=0.45)

        self.assertEqual([band.label for band in benchmark.bands], ["800-850", "600-649"])
        self.assertAlmostEqual(benchmark.score_band_expected_loss, 7.20)
        self.assertAlmostEqual(benchmark.realized_loss_at_lgd, 45.0)
        self.assertAlmostEqual(benchmark.file_pd_expected_loss, 45.0)
        self.assertAlmostEqual(benchmark.count_default_rate, 0.5)
        self.assertEqual(benchmark.bands[0].default_count, 0)
        self.assertEqual(benchmark.bands[1].default_count, 1)
        self.assertEqual(benchmark.bands[1].assumed_pd, 0.15)

    def test_default_flag_must_be_zero_or_one(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "book.csv"
            path.write_text(
                "Customer_ID,Credit_Score,Loan_Amount,Default,PD_Score\n"
                "A,700,100,2,0.10\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Default must be 0 or 1"):
                credit_book_benchmark(path)

    def test_missing_column_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "portfolio_data.csv"
            path.write_text("Customer_ID,Credit_Score\nCUST_1,700\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing column Loan_Amount"):
                loans_from_credit_book(path)


if __name__ == "__main__":
    unittest.main()
