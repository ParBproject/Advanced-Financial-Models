from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .cash_flow import CashFlowForecast
from .credit_concentration import CreditConcentrationSummary
from .credit_risk import PortfolioCreditSummary
from .portfolio import PortfolioMetrics
from .stress_testing import CreditStressResult, PortfolioMonteCarloResult


REPORT_VERSION = "1.0"


def _to_builtin(value: Any) -> Any:
    """Convert supported NumPy/scalar containers to JSON-safe built-in values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _validate_finite(value: Any, path: str = "report") -> None:
    """Reject non-finite numeric output before it reaches JSON/CSV exports."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, (int, bool)) or value is None or isinstance(value, str):
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"{path} contains a non-finite numeric value")


def build_executive_report(
    cash_flow: CashFlowForecast,
    credit: PortfolioCreditSummary,
    portfolio: PortfolioMetrics,
    *,
    concentration: CreditConcentrationSummary | None = None,
    credit_stress: CreditStressResult | None = None,
    monte_carlo: PortfolioMonteCarloResult | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact executive report from already-calculated model results.

    The report contains aggregate decision metrics only. Large simulation arrays
    and loan-level records remain in their source model objects rather than being
    copied into the export payload.
    """
    if not cash_flow.periods:
        raise ValueError("cash-flow forecast must contain at least one period")

    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "metadata": _to_builtin(dict(metadata or {})),
        "cash_flow": {
            "ending_cash": cash_flow.periods[-1].ending_cash,
            "npv_of_net_cash_flows": cash_flow.npv_of_net_cash_flows,
            "average_monthly_net_cash_flow": cash_flow.average_monthly_net_cash_flow,
            "minimum_cash_balance": cash_flow.minimum_cash_balance,
            "maximum_cash_balance": cash_flow.maximum_cash_balance,
        },
        "credit": {
            "total_exposure": credit.total_exposure,
            "total_expected_loss": credit.total_expected_loss,
            "expected_loss_ratio": credit.expected_loss_ratio,
            "average_credit_score": credit.average_credit_score,
            "risk_counts": {
                "low": credit.low_risk_count,
                "medium": credit.medium_risk_count,
                "high": credit.high_risk_count,
            },
        },
        "portfolio": {
            "expected_return": portfolio.expected_return,
            "volatility": portfolio.volatility,
            "sharpe_ratio": portfolio.sharpe_ratio,
            "future_value": portfolio.future_value,
            "expected_gain": portfolio.expected_gain,
        },
    }

    if concentration is not None:
        report["credit_concentration"] = {
            "largest_borrower_share": concentration.largest_borrower_share,
            "top_n": concentration.top_n,
            "top_n_share": concentration.top_n_share,
            "borrower_hhi": concentration.herfindahl_hirschman_index,
            "effective_borrower_count": concentration.effective_borrower_count,
        }

    if credit_stress is not None:
        report["credit_stress"] = {
            "scenario": credit_stress.scenario.name,
            "baseline_expected_loss": credit_stress.baseline_expected_loss,
            "stressed_expected_loss": credit_stress.stressed_expected_loss,
            "stressed_expected_loss_ratio": credit_stress.stressed_expected_loss_ratio,
            "expected_loss_change": credit_stress.expected_loss_change,
        }

    if monte_carlo is not None:
        report["portfolio_monte_carlo"] = {
            "initial_investment": monte_carlo.initial_investment,
            "horizon_years": monte_carlo.horizon_years,
            "expected_annual_return": monte_carlo.expected_annual_return,
            "annual_volatility": monte_carlo.annual_volatility,
            "mean_terminal_value": monte_carlo.mean_terminal_value,
            "median_terminal_value": monte_carlo.median_terminal_value,
            "percentile_05": monte_carlo.percentile_05,
            "percentile_25": monte_carlo.percentile_25,
            "percentile_75": monte_carlo.percentile_75,
            "percentile_95": monte_carlo.percentile_95,
            "probability_of_loss": monte_carlo.probability_of_loss,
        }

    report = _to_builtin(report)
    _validate_finite(report)
    return report


def report_to_json(report: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize a report deterministically as standards-compliant JSON."""
    if isinstance(indent, bool) or not isinstance(indent, int):
        raise TypeError("indent must be a non-negative integer")
    if indent < 0:
        raise ValueError("indent must be non-negative")

    normalized = _to_builtin(dict(report))
    _validate_finite(normalized)
    return json.dumps(
        normalized,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
    )


def _flatten_mapping(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key in sorted(value):
        item = value[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            rows.extend(_flatten_mapping(item, prefix=path))
        elif isinstance(item, (list, tuple)):
            rows.append((path, json.dumps(_to_builtin(item), sort_keys=True, allow_nan=False)))
        else:
            rows.append((path, item))
    return rows


def report_to_csv(report: Mapping[str, Any]) -> str:
    """Serialize aggregate report metrics to a two-column metric/value CSV."""
    normalized = _to_builtin(dict(report))
    _validate_finite(normalized)

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["metric", "value"])
    for metric, value in _flatten_mapping(normalized):
        writer.writerow([metric, value])
    return buffer.getvalue()
