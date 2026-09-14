"""Reproducible financial-modeling core for the portfolio workbook."""

from .advanced_portfolio import (
    PortfolioPoint,
    PortfolioSimulation,
    approximate_efficient_frontier,
    correlated_portfolio_metrics,
    covariance_from_correlation,
    illustrative_correlation_matrix,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    simulate_long_only_portfolios,
    validate_covariance_matrix,
)
from .cash_flow import CashFlowAssumptions, CashFlowForecast, CashFlowPeriod, forecast_cash_flow
from .credit_risk import (
    Loan,
    LoanRiskResult,
    PortfolioCreditSummary,
    assess_loan,
    summarize_portfolio,
)
from .portfolio import (
    AssetAllocation,
    PortfolioMetrics,
    portfolio_metrics,
    workbook_balanced_portfolio,
)

__all__ = [
    "AssetAllocation",
    "CashFlowAssumptions",
    "CashFlowForecast",
    "CashFlowPeriod",
    "Loan",
    "LoanRiskResult",
    "PortfolioCreditSummary",
    "PortfolioMetrics",
    "PortfolioPoint",
    "PortfolioSimulation",
    "approximate_efficient_frontier",
    "assess_loan",
    "correlated_portfolio_metrics",
    "covariance_from_correlation",
    "forecast_cash_flow",
    "illustrative_correlation_matrix",
    "max_sharpe_portfolio",
    "min_volatility_portfolio",
    "portfolio_metrics",
    "simulate_long_only_portfolios",
    "summarize_portfolio",
    "validate_covariance_matrix",
    "workbook_balanced_portfolio",
]
