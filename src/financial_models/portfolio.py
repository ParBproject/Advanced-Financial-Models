from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class AssetAllocation:
    name: str
    expected_return: float
    volatility: float
    weight: float

    def validate(self) -> None:
        if self.volatility < 0:
            raise ValueError("asset volatility must be non-negative")
        if not 0 <= self.weight <= 1:
            raise ValueError("asset weights must be between 0 and 1")


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
    if investment_amount < 0:
        raise ValueError("investment amount must be non-negative")
    if isinstance(horizon_years, bool) or not isinstance(horizon_years, int):
        raise ValueError("horizon years must be a non-negative integer")
    if horizon_years < 0:
        raise ValueError("horizon years must be a non-negative integer")

    expected_return = sum(
        asset.weight * asset.expected_return for asset in allocations
    )
    variance = sum(
        (asset.weight * asset.volatility) ** 2 for asset in allocations
    )
    volatility = sqrt(variance)
    if volatility == 0:
        sharpe = float("inf") if expected_return > risk_free_rate else 0.0
    else:
        sharpe = (expected_return - risk_free_rate) / volatility

    future_value = investment_amount * (1.0 + expected_return) ** horizon_years
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
