"""Map the committed credit-portfolio CSV onto the tested Loan schema.

``data/portfolio_data.csv`` is a byte copy of ``portfolio_data.csv`` in
ParBproject/portfolio-risk-analysis-credit-risk-modeling. That repository
remains the original source. This module does not recompute expected loss
or concentration; callers pass the returned loans into ``summarize_portfolio``
and ``credit_concentration``.

Column mapping:

- ``Customer_ID`` → ``loan_id`` and ``borrower`` (the file has no obligor name)
- ``Loan_Amount`` → ``exposure``
- ``Credit_Score`` → ``credit_score``

``PD_Score`` is the file's reported probability. Expected loss ignores it and
uses the score-band PD in ``credit_risk.probability_of_default``.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np

from .credit_concentration import CreditConcentrationSummary
from .credit_risk import Loan, PortfolioCreditSummary

LOAN_COLUMNS = ("Customer_ID", "Credit_Score", "Loan_Amount")
AUDIT_COLUMNS = (
    "Customer_ID",
    "Credit_Score",
    "Loan_Amount",
    "Operational_Risk_Score",
    "Revenue",
    "Expenses",
    "Net_Income",
    "Default",
    "PD_Score",
)


@dataclass(frozen=True)
class CreditBookMemoAudit:
    """File facts that correct the retired credit-repo memo.

    ``net_income_p05`` and ``net_income_p01`` are NumPy's higher sample
    quantiles of per-customer net income. On this file they equal the memo's
    printed "$43,146.55" and "$25,890.70". They are income levels, not a
    portfolio loss VaR.
    """

    loan_count: int
    total_exposure: float
    average_reported_pd: float
    reported_pd_above_20_count: int
    net_income_p05: float
    net_income_p01: float
    default_rate_operational_risk_above_60: float
    default_count_operational_risk_above_60: int
    count_operational_risk_above_60: int
    default_rate_operational_risk_at_or_below_60: float
    default_count_operational_risk_at_or_below_60: int
    count_operational_risk_at_or_below_60: int
    combined_stress_net_income: float


def credit_book_csv_path() -> Path:
    """Return the committed copy at ``<repo>/data/portfolio_data.csv``."""
    path = Path(__file__).resolve().parents[2] / "data" / "portfolio_data.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "data/portfolio_data.csv is missing. It is the committed copy of "
            "portfolio_data.csv from portfolio-risk-analysis-credit-risk-modeling."
        )
    return path


def loans_from_credit_book(path: Path | None = None) -> tuple[Loan, ...]:
    """Build Loan records from the credit-book CSV."""
    rows = _read_credit_book(str(_resolve_path(path)))
    _require_columns(rows, LOAN_COLUMNS)
    loans: list[Loan] = []
    for row in rows:
        loan_id = row["Customer_ID"].strip()
        if not loan_id:
            raise ValueError("Customer_ID must not be empty")
        loans.append(
            Loan(
                loan_id,
                loan_id,
                float(row["Loan_Amount"]),
                _credit_score(row["Credit_Score"]),
            )
        )
    if not loans:
        raise ValueError("credit book must contain at least one loan")
    return tuple(loans)


def credit_book_memo_audit(path: Path | None = None) -> CreditBookMemoAudit:
    """Recompute the retired memo's headline figures from the CSV columns."""
    rows = _read_credit_book(str(_resolve_path(path)))
    _require_columns(rows, AUDIT_COLUMNS)
    if not rows:
        raise ValueError("credit book must contain at least one loan")

    exposure = Decimal("0")
    reported_pd = Decimal("0")
    pd_above_20 = 0
    net_income: list[float] = []
    high_defaults = 0
    high_count = 0
    low_defaults = 0
    low_count = 0
    combined_stress = Decimal("0")

    for row in rows:
        exposure += Decimal(row["Loan_Amount"])
        pd_score = Decimal(row["PD_Score"])
        reported_pd += pd_score
        if pd_score > Decimal("0.20"):
            pd_above_20 += 1
        net_income.append(float(row["Net_Income"]))
        defaulted = int(row["Default"])
        if Decimal(row["Operational_Risk_Score"]) > 60:
            high_count += 1
            high_defaults += defaulted
        else:
            low_count += 1
            low_defaults += defaulted
        combined_stress += Decimal(row["Revenue"]) * Decimal("0.8") - Decimal(
            row["Expenses"]
        ) * Decimal("1.1")

    income = np.asarray(net_income, dtype=float)
    count = len(rows)
    return CreditBookMemoAudit(
        loan_count=count,
        total_exposure=float(exposure),
        average_reported_pd=float(reported_pd / count),
        reported_pd_above_20_count=pd_above_20,
        net_income_p05=float(np.quantile(income, 0.05, method="higher")),
        net_income_p01=float(np.quantile(income, 0.01, method="higher")),
        default_rate_operational_risk_above_60=_rate(high_defaults, high_count),
        default_count_operational_risk_above_60=high_defaults,
        count_operational_risk_above_60=high_count,
        default_rate_operational_risk_at_or_below_60=_rate(low_defaults, low_count),
        default_count_operational_risk_at_or_below_60=low_defaults,
        count_operational_risk_at_or_below_60=low_count,
        combined_stress_net_income=float(combined_stress),
    )


def format_credit_book_decision(
    summary: PortfolioCreditSummary,
    concentration: CreditConcentrationSummary,
    audit: CreditBookMemoAudit,
) -> str:
    """One decision line: library EL and HHI, plus the memo corrections."""
    above = audit.default_rate_operational_risk_above_60
    below = audit.default_rate_operational_risk_at_or_below_60
    return (
        f"On this {len(summary.loans):,}-loan book, expected loss is "
        f"${summary.total_expected_loss:,.2f} and borrower HHI is "
        f"{concentration.herfindahl_hirschman_index:.6f} "
        f"({concentration.effective_borrower_count:.1f} effective borrowers). "
        f"The retired memo's ${audit.net_income_p05:,.2f} and "
        f"${audit.net_income_p01:,.2f} are the 5th and 1st percentiles of "
        f"per-customer net income, not a loss VaR. Operational-risk scores "
        f"above 60 default at {above:.2%} "
        f"({audit.default_count_operational_risk_above_60} of "
        f"{audit.count_operational_risk_above_60}), against {below:.2%} "
        f"({audit.default_count_operational_risk_at_or_below_60} of "
        f"{audit.count_operational_risk_at_or_below_60}) at or below 60."
    )


def _resolve_path(path: Path | None) -> Path:
    return credit_book_csv_path() if path is None else Path(path)


def _require_columns(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> None:
    if not rows:
        return
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError("credit book is missing column " + ", ".join(missing))


def _credit_score(value: str) -> int:
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"credit score must be a whole number, got {value}")
    score = int(number)
    if not 300 <= score <= 850:
        raise ValueError("credit score must be between 300 and 850")
    return score


def _rate(defaults: int, count: int) -> float:
    if count == 0:
        raise ValueError("operational-risk slice must contain at least one loan")
    return defaults / count


@lru_cache(maxsize=4)
def _read_credit_book(path: str) -> tuple[dict[str, str], ...]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("credit book CSV has no header")
        return tuple(dict(row) for row in reader)
