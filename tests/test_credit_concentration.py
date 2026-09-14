import unittest

from financial_models.credit_concentration import (
    credit_concentration,
    segment_stress_attribution,
)
from financial_models.credit_risk import Loan
from financial_models.stress_testing import CreditStressScenario, stress_credit_portfolio


class CreditConcentrationTests(unittest.TestCase):
    def setUp(self):
        self.loans = (
            Loan("L-1", "Borrower A", 60.0, 780),
            Loan("L-2", "Borrower A", 40.0, 720),
            Loan("L-3", "Borrower B", 100.0, 620),
            Loan("L-4", "Borrower C", 200.0, 520),
        )

    def test_borrower_concentration_matches_hand_calculation(self):
        summary = credit_concentration(self.loans, top_n=2)

        self.assertAlmostEqual(summary.total_exposure, 400.0)
        self.assertAlmostEqual(summary.largest_borrower_share, 0.50)
        self.assertAlmostEqual(summary.top_n_share, 0.75)
        self.assertAlmostEqual(summary.herfindahl_hirschman_index, 0.375)
        self.assertAlmostEqual(summary.effective_borrower_count, 1.0 / 0.375)
        self.assertEqual(summary.borrower_exposures[0].name, "Borrower C")
        self.assertAlmostEqual(summary.borrower_exposures[0].share, 0.50)

    def test_risk_rating_exposure_sums_to_total(self):
        summary = credit_concentration(self.loans)

        self.assertAlmostEqual(
            sum(group.exposure for group in summary.risk_rating_exposures),
            summary.total_exposure,
        )
        self.assertAlmostEqual(
            sum(group.share for group in summary.risk_rating_exposures),
            1.0,
        )

    def test_zero_exposure_portfolio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "total exposure must be positive"):
            credit_concentration([Loan("L-0", "Zero", 0.0, 700)])

    def test_invalid_top_n_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            credit_concentration(self.loans, top_n=0)


class SegmentStressAttributionTests(unittest.TestCase):
    def setUp(self):
        self.loans = (
            Loan("L-1", "A", 100_000.0, 780),
            Loan("L-2", "B", 150_000.0, 620),
            Loan("L-3", "C", 200_000.0, 520),
        )
        self.segments = {
            "L-1": "Corporate",
            "L-2": "Corporate",
            "L-3": "Growth",
        }
        self.scenario = CreditStressScenario(
            "Recession",
            pd_multiplier=1.75,
            lgd_multiplier=1.25,
        )

    def test_segment_changes_reconcile_to_portfolio_stress(self):
        attributions = segment_stress_attribution(
            self.loans,
            self.segments,
            self.scenario,
        )
        total = stress_credit_portfolio(self.loans, self.scenario)

        self.assertAlmostEqual(
            sum(result.exposure for result in attributions),
            total.total_exposure,
        )
        self.assertAlmostEqual(
            sum(result.baseline_expected_loss for result in attributions),
            total.baseline_expected_loss,
        )
        self.assertAlmostEqual(
            sum(result.stressed_expected_loss for result in attributions),
            total.stressed_expected_loss,
        )
        self.assertAlmostEqual(
            sum(result.expected_loss_change for result in attributions),
            total.expected_loss_change,
        )
        self.assertAlmostEqual(
            sum(result.share_of_total_loss_change for result in attributions),
            1.0,
        )

    def test_attribution_is_sorted_by_loss_change(self):
        attributions = segment_stress_attribution(
            self.loans,
            self.segments,
            self.scenario,
        )

        self.assertGreaterEqual(
            attributions[0].expected_loss_change,
            attributions[-1].expected_loss_change,
        )

    def test_missing_segment_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing segment"):
            segment_stress_attribution(
                self.loans,
                {"L-1": "Corporate"},
                self.scenario,
            )


if __name__ == "__main__":
    unittest.main()
