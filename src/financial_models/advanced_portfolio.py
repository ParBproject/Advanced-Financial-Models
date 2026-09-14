from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .portfolio import AssetAllocation, PortfolioMetrics


@dataclass(frozen=True)
class PortfolioPoint:
    """One portfolio on a simulated risk/return surface."""

    weights: tuple[float, ...]
    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass(frozen=True)
class PortfolioSimulation:
    """Vectorized long-only portfolio simulation results."""

    weights: np.ndarray
    expected_returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray

    @property
    def n_portfolios(self) -> int:
        return int(self.weights.shape[0])


def _assets_tuple(assets: Iterable[AssetAllocation]) -> tuple[AssetAllocation, ...]:
    allocations = tuple(assets)
    if not allocations:
        raise ValueError("portfolio must contain at least one asset")
    for asset in allocations:
        asset.validate()
    return allocations


def validate_covariance_matrix(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    n_assets: int,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Return a validated symmetric positive-semidefinite covariance matrix."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"covariance matrix must have shape ({n_assets}, {n_assets})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("covariance matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0):
        raise ValueError("covariance matrix must be symmetric")

    scale = max(1.0, float(np.max(np.abs(matrix))))
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(eigenvalues.min()) < -(tolerance * scale):
        raise ValueError("covariance matrix must be positive semidefinite")
    if np.any(np.diag(matrix) < -(tolerance * scale)):
        raise ValueError("covariance variances must be non-negative")
    return matrix


def covariance_from_correlation(
    volatilities: Sequence[float] | np.ndarray,
    correlation: Sequence[Sequence[float]] | np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Convert asset volatilities and a valid correlation matrix to covariance."""
    vols = np.asarray(volatilities, dtype=float)
    if vols.ndim != 1 or len(vols) == 0:
        raise ValueError("volatilities must be a non-empty one-dimensional vector")
    if not np.isfinite(vols).all() or np.any(vols < 0):
        raise ValueError("volatilities must be finite and non-negative")

    corr = np.asarray(correlation, dtype=float)
    n_assets = len(vols)
    if corr.shape != (n_assets, n_assets):
        raise ValueError(
            f"correlation matrix must have shape ({n_assets}, {n_assets})"
        )
    if not np.isfinite(corr).all():
        raise ValueError("correlation matrix must contain only finite values")
    if not np.allclose(corr, corr.T, atol=tolerance, rtol=0.0):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("correlation matrix diagonal must equal 1")
    if np.any(corr < -1.0 - tolerance) or np.any(corr > 1.0 + tolerance):
        raise ValueError("correlations must be between -1 and 1")

    eigenvalues = np.linalg.eigvalsh(corr)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("correlation matrix must be positive semidefinite")

    return np.outer(vols, vols) * corr


def correlated_portfolio_metrics(
    assets: Iterable[AssetAllocation],
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    risk_free_rate: float = 0.03,
    investment_amount: float = 1_000_000.0,
    horizon_years: int = 10,
) -> PortfolioMetrics:
    """Calculate portfolio metrics using wᵀΣw covariance-aware risk."""
    allocations = _assets_tuple(assets)
    total_weight = sum(asset.weight for asset in allocations)
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError("portfolio weights must sum to 1.0")
    if investment_amount < 0:
        raise ValueError("investment amount must be non-negative")
    if isinstance(horizon_years, bool) or not isinstance(horizon_years, int):
        raise TypeError("horizon years must be a non-negative integer")
    if horizon_years < 0:
        raise ValueError("horizon years must be a non-negative integer")

    cov = validate_covariance_matrix(covariance, len(allocations))
    weights = np.asarray([asset.weight for asset in allocations], dtype=float)
    returns = np.asarray([asset.expected_return for asset in allocations], dtype=float)

    expected_return = float(weights @ returns)
    variance = float(weights @ cov @ weights)
    variance = max(variance, 0.0)
    volatility = float(np.sqrt(variance))
    if volatility == 0.0:
        excess_return = expected_return - risk_free_rate
        sharpe = float("inf") if excess_return > 0 else 0.0
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


def simulate_long_only_portfolios(
    assets: Iterable[AssetAllocation],
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_portfolios: int = 10_000,
    risk_free_rate: float = 0.03,
    random_state: int | None = 42,
) -> PortfolioSimulation:
    """Sample reproducible long-only, fully invested portfolios from a simplex."""
    allocations = _assets_tuple(assets)
    if isinstance(n_portfolios, bool) or not isinstance(n_portfolios, int):
        raise TypeError("n_portfolios must be a positive integer")
    if n_portfolios < 1:
        raise ValueError("n_portfolios must be at least 1")

    cov = validate_covariance_matrix(covariance, len(allocations))
    expected_asset_returns = np.asarray(
        [asset.expected_return for asset in allocations], dtype=float
    )

    rng = np.random.default_rng(random_state)
    weights = rng.dirichlet(np.ones(len(allocations)), size=n_portfolios)
    expected_returns = weights @ expected_asset_returns
    variances = np.einsum("ij,jk,ik->i", weights, cov, weights)
    volatilities = np.sqrt(np.maximum(variances, 0.0))

    excess_returns = expected_returns - risk_free_rate
    sharpe_ratios = np.zeros_like(excess_returns)
    nonzero_risk = volatilities > 1e-15
    sharpe_ratios[nonzero_risk] = (
        excess_returns[nonzero_risk] / volatilities[nonzero_risk]
    )
    zero_risk = ~nonzero_risk
    sharpe_ratios[zero_risk & (excess_returns > 0)] = np.inf
    sharpe_ratios[zero_risk & (excess_returns < 0)] = -np.inf

    return PortfolioSimulation(
        weights=weights,
        expected_returns=expected_returns,
        volatilities=volatilities,
        sharpe_ratios=sharpe_ratios,
    )


def _point_at(simulation: PortfolioSimulation, index: int) -> PortfolioPoint:
    return PortfolioPoint(
        weights=tuple(float(value) for value in simulation.weights[index]),
        expected_return=float(simulation.expected_returns[index]),
        volatility=float(simulation.volatilities[index]),
        sharpe_ratio=float(simulation.sharpe_ratios[index]),
    )


def max_sharpe_portfolio(simulation: PortfolioSimulation) -> PortfolioPoint:
    """Return the sampled portfolio with the highest Sharpe ratio."""
    if simulation.n_portfolios < 1:
        raise ValueError("simulation must contain at least one portfolio")
    return _point_at(simulation, int(np.argmax(simulation.sharpe_ratios)))


def min_volatility_portfolio(simulation: PortfolioSimulation) -> PortfolioPoint:
    """Return the sampled portfolio with the lowest volatility."""
    if simulation.n_portfolios < 1:
        raise ValueError("simulation must contain at least one portfolio")
    return _point_at(simulation, int(np.argmin(simulation.volatilities)))


def approximate_efficient_frontier(
    simulation: PortfolioSimulation,
    *,
    max_points: int = 25,
    return_tolerance: float = 1e-12,
) -> tuple[PortfolioPoint, ...]:
    """Extract non-dominated risk/return points from a portfolio simulation.

    The returned frontier is approximate because it is selected from sampled
    portfolios rather than solved analytically.
    """
    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise TypeError("max_points must be a positive integer")
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    if simulation.n_portfolios < 1:
        raise ValueError("simulation must contain at least one portfolio")

    order = np.lexsort((-simulation.expected_returns, simulation.volatilities))
    efficient_indices: list[int] = []
    best_return = -np.inf
    for index in order:
        candidate_return = float(simulation.expected_returns[index])
        if candidate_return > best_return + return_tolerance:
            efficient_indices.append(int(index))
            best_return = candidate_return

    if len(efficient_indices) > max_points:
        positions = np.linspace(0, len(efficient_indices) - 1, max_points, dtype=int)
        efficient_indices = [efficient_indices[position] for position in np.unique(positions)]

    return tuple(_point_at(simulation, index) for index in efficient_indices)


def illustrative_correlation_matrix() -> np.ndarray:
    """Return a PSD illustrative correlation matrix for the six workbook assets.

    This is a transparent one-factor example for demonstrations; it is not
    presented as an estimate from historical market data.
    """
    factor_loadings = np.asarray([0.80, 0.75, 0.70, -0.15, 0.45, 0.20])
    correlation = np.outer(factor_loadings, factor_loadings)
    np.fill_diagonal(correlation, 1.0)
    return correlation
