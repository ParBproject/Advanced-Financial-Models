from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable


@dataclass(frozen=True)
class Loan:
    loan_id: str
    borrower: str
    exposure: float
    credit_score: int

    def validate(self) -> None:
        if self.exposure < 0:
            raise ValueError("loan exposure must be non-negative")
        if not 300 <= self.credit_score <= 850:
            raise ValueError("credit score must be between 300 and 850")


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
    if not 300 <= credit_score <= 850:
        raise ValueError("credit score must be between 300 and 850")
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
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if probability <= 0.04:
        return "Low"
    if probability <= 0.15:
        return "Medium"
    return "High"


def assess_loan(loan: Loan, *, loss_given_default: float = 0.45) -> LoanRiskResult:
    loan.validate()
    if not 0 <= loss_given_default <= 1:
        raise ValueError("loss given default must be between 0 and 1")

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
    expected_loss_ratio = (
        total_expected_loss / total_exposure if total_exposure else 0.0
    )

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
