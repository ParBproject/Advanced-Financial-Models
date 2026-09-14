from financial_models.advanced_portfolio import (
    approximate_efficient_frontier,
    correlated_portfolio_metrics,
    covariance_from_correlation,
    illustrative_correlation_matrix,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    simulate_long_only_portfolios,
)
from financial_models.portfolio import workbook_balanced_portfolio


assets = workbook_balanced_portfolio()
correlation = illustrative_correlation_matrix()
covariance = covariance_from_correlation(
    [asset.volatility for asset in assets],
    correlation,
)

baseline = correlated_portfolio_metrics(assets, covariance)
simulation = simulate_long_only_portfolios(
    assets,
    covariance,
    n_portfolios=10_000,
    random_state=42,
)
max_sharpe = max_sharpe_portfolio(simulation)
min_volatility = min_volatility_portfolio(simulation)
frontier = approximate_efficient_frontier(simulation, max_points=12)

print("Covariance-aware baseline")
print(f"  expected return: {baseline.expected_return:.2%}")
print(f"  volatility:      {baseline.volatility:.2%}")
print(f"  Sharpe ratio:    {baseline.sharpe_ratio:.2f}")
print()
print("Best sampled portfolios")
print(f"  max Sharpe:      {max_sharpe.sharpe_ratio:.2f}")
print(f"  min volatility:  {min_volatility.volatility:.2%}")
print(f"  frontier points: {len(frontier)}")
print()
print("Max-Sharpe weights")
for asset, weight in zip(assets, max_sharpe.weights, strict=True):
    print(f"  {asset.name:<24} {weight:>7.2%}")
