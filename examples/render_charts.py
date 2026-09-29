"""Write the repository charts from the tested models."""

from financial_models.advanced_portfolio import (
    approximate_efficient_frontier,
    correlated_portfolio_metrics,
    covariance_from_correlation,
    illustrative_correlation_matrix,
    simulate_long_only_portfolios,
)
from financial_models.cash_flow import CashFlowAssumptions, forecast_cash_flow
from financial_models.charts import (
    allocation_matplotlib,
    cash_flow_matplotlib,
    credit_ratings_matplotlib,
    frontier_matplotlib,
    monte_carlo_matplotlib,
    save_figure,
)
from financial_models.credit_concentration import credit_concentration
from financial_models.credit_risk import summarize_portfolio
from financial_models.loan_book import loans_from_credit_book
from financial_models.portfolio import portfolio_metrics, workbook_balanced_portfolio
from financial_models.stress_testing import simulate_portfolio_terminal_values

OUTPUTS = {
    "docs/images/cash-flow.png": "cash",
    "docs/images/credit-book.png": "credit",
    "docs/images/portfolio-allocation.png": "allocation",
    "docs/images/efficient-frontier.png": "frontier",
    "docs/images/monte-carlo.png": "monte_carlo",
}


def main() -> None:
    cash = forecast_cash_flow(CashFlowAssumptions())
    loans = loans_from_credit_book()
    credit = summarize_portfolio(loans)
    concentration = credit_concentration(loans)
    assets = workbook_balanced_portfolio()
    covariance = covariance_from_correlation(
        [asset.volatility for asset in assets],
        illustrative_correlation_matrix(),
    )
    zero_correlation = portfolio_metrics(assets)
    aware = correlated_portfolio_metrics(assets, covariance)
    simulation = simulate_long_only_portfolios(
        assets,
        covariance,
        n_portfolios=10_000,
        random_state=42,
    )
    frontier = approximate_efficient_frontier(simulation, max_points=18)
    monte_carlo = simulate_portfolio_terminal_values(
        assets,
        covariance,
        n_simulations=10_000,
        random_state=42,
    )

    figures = {
        "cash": cash_flow_matplotlib(cash),
        "credit": credit_ratings_matplotlib(credit, concentration),
        "allocation": allocation_matplotlib(assets, zero_correlation, aware),
        "frontier": frontier_matplotlib(
            simulation.volatilities,
            simulation.expected_returns,
            simulation.sharpe_ratios,
            [point.volatility for point in frontier],
            [point.expected_return for point in frontier],
            aware.volatility,
            aware.expected_return,
        ),
        "monte_carlo": monte_carlo_matplotlib(monte_carlo),
    }
    for path, key in OUTPUTS.items():
        save_figure(figures[key], path)
        print(path)

    print(f"ending_cash {cash.periods[-1].ending_cash:.2f}")
    print(f"npv {cash.npv_of_net_cash_flows:.2f}")
    print(f"minimum_cash {cash.minimum_cash_balance:.2f}")
    print(f"expected_loss {credit.total_expected_loss:.2f}")
    print(f"exposure {credit.total_exposure:.2f}")
    print(f"el_ratio {credit.expected_loss_ratio:.6f}")
    print(f"hhi {concentration.herfindahl_hirschman_index:.8f}")
    print(f"effective_borrowers {concentration.effective_borrower_count:.4f}")
    print(f"largest_borrower {concentration.largest_borrower_share:.6f}")
    print(f"zero_corr_return {zero_correlation.expected_return:.6f}")
    print(f"zero_corr_vol {zero_correlation.volatility:.6f}")
    print(f"zero_corr_sharpe {zero_correlation.sharpe_ratio:.6f}")
    print(f"zero_corr_future {zero_correlation.future_value:.2f}")
    print(f"aware_vol {aware.volatility:.6f}")
    print(f"aware_sharpe {aware.sharpe_ratio:.6f}")
    print(f"mc_median {monte_carlo.median_terminal_value:.2f}")
    print(f"mc_p05 {monte_carlo.percentile_05:.2f}")
    print(f"mc_p95 {monte_carlo.percentile_95:.2f}")
    print(f"mc_loss {monte_carlo.probability_of_loss:.6f}")


if __name__ == "__main__":
    main()
