from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt

from .validation import require_finite


@dataclass(frozen=True)
class AssetAllocation:
    name: str
    expected_return: float
    volatility: float
    weight: float

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ValueError("asset name must not be empty")
        require_finite("asset expected return", self.expected_return)
        volatility = require_finite("asset volatility", self.volatility)
        weight = require_finite("asset weight", self.weight)
        if volatility < 0:
            raise ValueError("asset volatility must be non-negative")
        if not 0 <= weight <= 1:
            raise ValueError("asset weights must be between 0 and 1")


def compound_future_value(
    investment_amount: float,
    expected_return: float,
    horizon_years: int,
) -> float:
    """Compound a portfolio return for ``horizon_years``.

    A gross return of zero or less is not a meaningful base for multi-year
    compounding: ``(1 - 150%) ** 3`` is negative. One-year and longer horizons
    therefore require an expected return above -100%. A zero-year horizon does
    not compound, so the investment amount is returned unchanged.
    """
    if horizon_years >= 1 and expected_return <= -1.0:
        raise ValueError(
            "expected return must be greater than -100% when compounding over a year or more"
        )
    return investment_amount * (1.0 + expected_return) ** horizon_years


def sharpe_ratio(expected_return: float, volatility: float, risk_free_rate: float) -> float:
    """Return excess return per unit of volatility.

    A zero-volatility portfolio is a sure outcome. Its Sharpe ratio is
    positive infinity when that outcome beats the risk-free rate, negative
    infinity when it lags cash, and zero when the two returns are equal.
    """
    excess = expected_return - risk_free_rate
    if volatility == 0.0:
        if excess > 0.0:
            return float("inf")
        if excess < 0.0:
            return float("-inf")
        return 0.0
    return excess / volatility


@dataclass(frozen=True)
class PortfolioMetrics:
    expected_return: float
    volatility: float
    sharpe_ratio: float
    future_value: float
    expected_gain: float


def portfolio_metrics(
    assets: Iterable[AssetAllocation],
    *,
    risk_free_rate: float = 0.03,
    investment_amount: float = 1_000_000.0,
    horizon_years: int = 10,
) -> PortfolioMetrics:
    """Calculate the workbook's independent-asset portfolio metrics.

    The volatility formula intentionally mirrors the Excel model's documented
    zero-correlation simplification: sqrt(sum((weight * volatility) ** 2)).
    """
    allocations = tuple(assets)
    if not allocations:
        raise ValueError("portfolio must contain at least one asset")
    for asset in allocations:
        asset.validate()

    total_weight = sum(asset.weight for asset in allocations)
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError("portfolio weights must sum to 1.0")
    investment_amount = require_finite("investment amount", investment_amount)
    risk_free_rate = require_finite("risk-free rate", risk_free_rate)
    if investment_amount < 0:
        raise ValueError("investment amount must be non-negative")
    if isinstance(horizon_years, bool) or not isinstance(horizon_years, int):
        raise TypeError("horizon years must be a non-negative integer")
    if horizon_years < 0:
        raise ValueError("horizon years must be a non-negative integer")

    expected_return = sum(
        asset.weight * asset.expected_return for asset in allocations
    )
    variance = sum(
        (asset.weight * asset.volatility) ** 2 for asset in allocations
    )
    volatility = sqrt(variance)
    sharpe = sharpe_ratio(expected_return, volatility, risk_free_rate)

    future_value = compound_future_value(investment_amount, expected_return, horizon_years)
    return PortfolioMetrics(
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        future_value=future_value,
        expected_gain=future_value - investment_amount,
    )


def workbook_balanced_portfolio() -> tuple[AssetAllocation, ...]:
    """Return the six-asset baseline documented in the workbook guide."""
    return (
        AssetAllocation("Large Cap Stocks", 0.10, 0.15, 0.35),
        AssetAllocation("Small Cap Stocks", 0.12, 0.20, 0.15),
        AssetAllocation("International Stocks", 0.09, 0.17, 0.15),
        AssetAllocation("Bonds", 0.05, 0.06, 0.25),
        AssetAllocation("Real Estate", 0.08, 0.14, 0.07),
        AssetAllocation("Commodities", 0.06, 0.18, 0.03),
    )
