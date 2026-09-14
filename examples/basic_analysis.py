from financial_models import (
    CashFlowAssumptions,
    Loan,
    forecast_cash_flow,
    portfolio_metrics,
    summarize_portfolio,
    workbook_balanced_portfolio,
)


cash = forecast_cash_flow(CashFlowAssumptions())
print(f"12-month ending cash: ${cash.periods[-1].ending_cash:,.0f}")
print(f"NPV of monthly net cash flows: ${cash.npv_of_net_cash_flows:,.0f}")

credit = summarize_portfolio(
    [
        Loan("L-001", "Borrower A", 100_000.0, 620),
        Loan("L-002", "Borrower B", 250_000.0, 760),
        Loan("L-003", "Borrower C", 75_000.0, 540),
    ]
)
print(f"Credit expected loss: ${credit.total_expected_loss:,.0f}")
print(f"Expected-loss ratio: {credit.expected_loss_ratio:.2%}")

portfolio = portfolio_metrics(workbook_balanced_portfolio())
print(f"Portfolio expected return: {portfolio.expected_return:.2%}")
print(f"Portfolio volatility: {portfolio.volatility:.2%}")
print(f"Portfolio Sharpe ratio: {portfolio.sharpe_ratio:.2f}")
