from financial_models.advanced_portfolio import (
    covariance_from_correlation,
    illustrative_correlation_matrix,
)
from financial_models.cash_flow import CashFlowAssumptions
from financial_models.credit_risk import Loan
from financial_models.portfolio import workbook_balanced_portfolio
from financial_models.stress_testing import (
    CashFlowStressScenario,
    CreditStressScenario,
    simulate_portfolio_terminal_values,
    stress_cash_flow,
    stress_credit_portfolio,
)


cash_stress = stress_cash_flow(
    CashFlowAssumptions(),
    CashFlowStressScenario(
        "Severe downturn",
        revenue_multiplier=0.80,
        monthly_growth_delta=-0.03,
        operating_expense_ratio_delta=0.10,
        fixed_cost_multiplier=1.10,
        capex_multiplier=1.20,
    ),
)
print("Cash-flow stress")
print(f"  ending cash change: ${cash_stress.ending_cash_change:,.0f}")
print(f"  NPV change:         ${cash_stress.npv_change:,.0f}")

credit_stress = stress_credit_portfolio(
    [
        Loan("L-001", "Borrower A", 100_000.0, 780),
        Loan("L-002", "Borrower B", 150_000.0, 620),
        Loan("L-003", "Borrower C", 200_000.0, 520),
    ],
    CreditStressScenario("Recession", pd_multiplier=1.75, lgd_multiplier=1.25),
)
print("\nCredit stress")
print(f"  baseline expected loss: ${credit_stress.baseline_expected_loss:,.0f}")
print(f"  stressed expected loss: ${credit_stress.stressed_expected_loss:,.0f}")

assets = workbook_balanced_portfolio()
covariance = covariance_from_correlation(
    [asset.volatility for asset in assets],
    illustrative_correlation_matrix(),
)
monte_carlo = simulate_portfolio_terminal_values(
    assets,
    covariance,
    n_simulations=10_000,
    random_state=42,
)
print("\n10-year portfolio Monte Carlo")
print(f"  median terminal value: ${monte_carlo.median_terminal_value:,.0f}")
print(f"  5th percentile:        ${monte_carlo.percentile_05:,.0f}")
print(f"  95th percentile:       ${monte_carlo.percentile_95:,.0f}")
print(f"  probability of loss:   {monte_carlo.probability_of_loss:.2%}")
