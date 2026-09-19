"""Real-market example: estimate portfolio risk from observed ETF prices.

Requires the optional market-data dependency:
    pip install -e ".[market-data]"
"""

from financial_models import (
    download_adjusted_close,
    historical_risk_summary,
    historical_var_expected_shortfall,
    portfolio_return_series,
    simple_returns,
)

TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
WEIGHTS = [0.40, 0.25, 0.20, 0.15]

prices = download_adjusted_close(TICKERS, start="2020-01-01")
summary = historical_risk_summary(prices)
returns = simple_returns(prices)
portfolio_returns = portfolio_return_series(returns, WEIGHTS)
var_95, es_95 = historical_var_expected_shortfall(portfolio_returns, confidence=0.95)

print("Observed assets:", ", ".join(prices.columns))
print("\nAnnualized return estimates")
print(summary.annualized_returns.round(4))
print("\nAnnualized volatility")
print(summary.annualized_volatility.round(4))
print("\nMaximum drawdown")
print(summary.max_drawdown.round(4))
print(f"\nPortfolio historical 95% VaR: {var_95:.2%}")
print(f"Portfolio historical 95% Expected Shortfall: {es_95:.2%}")
