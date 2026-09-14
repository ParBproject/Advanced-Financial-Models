"""Reproducible financial-modeling core for the portfolio workbook."""

from .cash_flow import CashFlowAssumptions, CashFlowForecast, CashFlowPeriod, forecast_cash_flow
from .credit_risk import Loan, LoanRiskResult, PortfolioCreditSummary, assess_loan, summarize_portfolio
from .portfolio import AssetAllocation, PortfolioMetrics, portfolio_metrics

__all__ = [
    "AssetAllocation",
    "CashFlowAssumptions",
    "CashFlowForecast",
    "CashFlowPeriod",
    "Loan",
    "LoanRiskResult",
    "PortfolioCreditSummary",
    "PortfolioMetrics",
    "assess_loan",
    "forecast_cash_flow",
    "portfolio_metrics",
    "summarize_portfolio",
]
