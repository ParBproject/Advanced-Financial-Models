from financial_models.credit_concentration import (
    credit_concentration,
    segment_stress_attribution,
)
from financial_models.credit_risk import Loan
from financial_models.stress_testing import CreditStressScenario


loans = (
    Loan("L-001", "Northstar Inc.", 250_000.0, 760),
    Loan("L-002", "Northstar Inc.", 100_000.0, 720),
    Loan("L-003", "Growth Labs", 300_000.0, 620),
    Loan("L-004", "Main Street Co.", 175_000.0, 540),
)

summary = credit_concentration(loans, top_n=2)
print("Credit concentration")
print(f"  largest borrower: {summary.largest_borrower_share:.2%}")
print(f"  top-{summary.top_n} share:     {summary.top_n_share:.2%}")
print(f"  borrower HHI:      {summary.herfindahl_hirschman_index:.3f}")
print(f"  effective borrowers: {summary.effective_borrower_count:.2f}")

segments = {
    "L-001": "Corporate",
    "L-002": "Corporate",
    "L-003": "Growth",
    "L-004": "SME",
}
attribution = segment_stress_attribution(
    loans,
    segments,
    CreditStressScenario("Recession", pd_multiplier=1.75, lgd_multiplier=1.25),
)

print("\nStress attribution")
for segment in attribution:
    print(
        f"  {segment.segment:<12} "
        f"loss change ${segment.expected_loss_change:>10,.0f} "
        f"({segment.share_of_total_loss_change:>6.1%})"
    )
