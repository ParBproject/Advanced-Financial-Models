from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

from .validation import require_credit_score, require_finite, require_probability


@dataclass(frozen=True)
class Loan:
    loan_id: str
    borrower: str
    exposure: float
    credit_score: int

    def validate(self) -> None:
        if not str(self.loan_id).strip() or not str(self.borrower).strip():
            raise ValueError("loan id and borrower must not be empty")
        exposure = require_finite("loan exposure", self.exposure)
        if exposure < 0:
            raise ValueError("loan exposure must be non-negative")
        require_credit_score(self.credit_score)


@dataclass(frozen=True)
class LoanRiskResult:
    loan: Loan
    probability_of_default: float
    loss_given_default: float
    expected_loss: float
    risk_rating: str


@dataclass(frozen=True)
class PortfolioCreditSummary:
    loans: tuple[LoanRiskResult, ...]
    total_exposure: float
    total_expected_loss: float
    expected_loss_ratio: float
    average_credit_score: float
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int


def probability_of_default(credit_score: int) -> float:
    """Map a FICO-style credit score to the workbook's documented PD bands."""
    credit_score = require_credit_score(credit_score)
    if credit_score >= 800:
        return 0.01
    if credit_score >= 750:
        return 0.02
    if credit_score >= 700:
        return 0.04
    if credit_score >= 650:
        return 0.08
    if credit_score >= 600:
        return 0.15
    if credit_score >= 550:
        return 0.25
    if credit_score >= 500:
        return 0.35
    return 0.50


def risk_rating(probability: float) -> str:
    """Rate a default probability using the workbook thresholds.

    The loan sheet labels PD >= 20% as High and PD >= 7% as Medium. The
    score-band table lands on the same labels, because its medium bands are
    8% and 15% and its high bands start at 25%.
    """
    probability = require_probability("probability", probability)
    if probability >= 0.20:
        return "High"
    if probability >= 0.07:
        return "Medium"
    return "Low"


def assess_loan(loan: Loan, *, loss_given_default: float = 0.45) -> LoanRiskResult:
    loan.validate()
    require_probability("loss given default", loss_given_default)

    pd = probability_of_default(loan.credit_score)
    expected_loss = loan.exposure * pd * loss_given_default
    return LoanRiskResult(
        loan=loan,
        probability_of_default=pd,
        loss_given_default=loss_given_default,
        expected_loss=expected_loss,
        risk_rating=risk_rating(pd),
    )


def summarize_portfolio(
    loans: Iterable[Loan],
    *,
    loss_given_default: float = 0.45,
) -> PortfolioCreditSummary:
    assessed = tuple(
        assess_loan(loan, loss_given_default=loss_given_default) for loan in loans
    )
    if not assessed:
        raise ValueError("portfolio must contain at least one loan")

    total_exposure = sum(result.loan.exposure for result in assessed)
    total_expected_loss = sum(result.expected_loss for result in assessed)
    expected_loss_ratio = total_expected_loss / total_exposure if total_exposure else 0.0

    return PortfolioCreditSummary(
        loans=assessed,
        total_exposure=total_exposure,
        total_expected_loss=total_expected_loss,
        expected_loss_ratio=expected_loss_ratio,
        average_credit_score=fmean(result.loan.credit_score for result in assessed),
        low_risk_count=sum(result.risk_rating == "Low" for result in assessed),
        medium_risk_count=sum(result.risk_rating == "Medium" for result in assessed),
        high_risk_count=sum(result.risk_rating == "High" for result in assessed),
    )
