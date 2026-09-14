from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .credit_risk import Loan, assess_loan
from .stress_testing import CreditStressScenario, stress_credit_portfolio


@dataclass(frozen=True)
class ExposureGroup:
    name: str
    exposure: float
    share: float


@dataclass(frozen=True)
class CreditConcentrationSummary:
    total_exposure: float
    borrower_exposures: tuple[ExposureGroup, ...]
    risk_rating_exposures: tuple[ExposureGroup, ...]
    largest_borrower_share: float
    top_n_share: float
    top_n: int
    herfindahl_hirschman_index: float
    effective_borrower_count: float


@dataclass(frozen=True)
class SegmentStressAttribution:
    segment: str
    exposure: float
    baseline_expected_loss: float
    stressed_expected_loss: float
    expected_loss_change: float
    share_of_total_loss_change: float


def _validated_loans(loans: Iterable[Loan]) -> tuple[Loan, ...]:
    loan_tuple = tuple(loans)
    if not loan_tuple:
        raise ValueError("portfolio must contain at least one loan")
    for loan in loan_tuple:
        loan.validate()
    return loan_tuple


def _exposure_groups(exposure_by_name: Mapping[str, float], total: float) -> tuple[ExposureGroup, ...]:
    return tuple(
        ExposureGroup(name=name, exposure=exposure, share=exposure / total)
        for name, exposure in sorted(
            exposure_by_name.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def credit_concentration(
    loans: Iterable[Loan],
    *,
    top_n: int = 3,
) -> CreditConcentrationSummary:
    """Measure borrower and risk-rating concentration for a credit portfolio."""
    loan_tuple = _validated_loans(loans)
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise TypeError("top_n must be a positive integer")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    total_exposure = sum(loan.exposure for loan in loan_tuple)
    if total_exposure <= 0:
        raise ValueError("portfolio total exposure must be positive")

    borrower_totals: defaultdict[str, float] = defaultdict(float)
    rating_totals: defaultdict[str, float] = defaultdict(float)
    for loan in loan_tuple:
        borrower_totals[loan.borrower] += loan.exposure
        rating_totals[assess_loan(loan).risk_rating] += loan.exposure

    borrower_groups = _exposure_groups(borrower_totals, total_exposure)
    rating_groups = _exposure_groups(rating_totals, total_exposure)
    shares = [group.share for group in borrower_groups]
    hhi = sum(share**2 for share in shares)
    effective_count = 1.0 / hhi if hhi > 0 else 0.0
    selected_count = min(top_n, len(borrower_groups))
    top_share = sum(group.share for group in borrower_groups[:selected_count])

    return CreditConcentrationSummary(
        total_exposure=total_exposure,
        borrower_exposures=borrower_groups,
        risk_rating_exposures=rating_groups,
        largest_borrower_share=borrower_groups[0].share,
        top_n_share=top_share,
        top_n=selected_count,
        herfindahl_hirschman_index=hhi,
        effective_borrower_count=effective_count,
    )


def segment_stress_attribution(
    loans: Iterable[Loan],
    segment_by_loan_id: Mapping[str, str],
    scenario: CreditStressScenario,
    *,
    base_loss_given_default: float = 0.45,
) -> tuple[SegmentStressAttribution, ...]:
    """Attribute stressed expected-loss changes to caller-defined credit segments."""
    loan_tuple = _validated_loans(loans)
    grouped: defaultdict[str, list[Loan]] = defaultdict(list)

    for loan in loan_tuple:
        if loan.loan_id not in segment_by_loan_id:
            raise ValueError(f"missing segment for loan {loan.loan_id!r}")
        segment = str(segment_by_loan_id[loan.loan_id]).strip()
        if not segment:
            raise ValueError(f"segment for loan {loan.loan_id!r} must not be empty")
        grouped[segment].append(loan)

    raw_results: list[tuple[str, float, float, float, float]] = []
    for segment, segment_loans in grouped.items():
        result = stress_credit_portfolio(
            segment_loans,
            scenario,
            base_loss_given_default=base_loss_given_default,
        )
        raw_results.append(
            (
                segment,
                result.total_exposure,
                result.baseline_expected_loss,
                result.stressed_expected_loss,
                result.expected_loss_change,
            )
        )

    total_change = sum(item[4] for item in raw_results)
    attributions = [
        SegmentStressAttribution(
            segment=segment,
            exposure=exposure,
            baseline_expected_loss=baseline_loss,
            stressed_expected_loss=stressed_loss,
            expected_loss_change=loss_change,
            share_of_total_loss_change=(loss_change / total_change if total_change else 0.0),
        )
        for segment, exposure, baseline_loss, stressed_loss, loss_change in raw_results
    ]
    return tuple(
        sorted(
            attributions,
            key=lambda result: (-result.expected_loss_change, result.segment),
        )
    )
