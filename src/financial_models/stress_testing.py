from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .advanced_portfolio import correlated_portfolio_metrics
from .cash_flow import (
    DEFAULT_CAPEX,
    CashFlowAssumptions,
    CashFlowForecast,
    forecast_cash_flow,
)
from .credit_risk import Loan, probability_of_default, summarize_portfolio
from .portfolio import AssetAllocation


@dataclass(frozen=True)
class CashFlowStressScenario:
    """Multipliers/deltas applied to the baseline liquidity forecast."""

    name: str
    revenue_multiplier: float = 1.0
    monthly_growth_delta: float = 0.0
    operating_expense_ratio_delta: float = 0.0
    fixed_cost_multiplier: float = 1.0
    capex_multiplier: float = 1.0

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if self.revenue_multiplier < 0:
            raise ValueError("revenue multiplier must be non-negative")
        if self.fixed_cost_multiplier < 0:
            raise ValueError("fixed cost multiplier must be non-negative")
        if self.capex_multiplier < 0:
            raise ValueError("capex multiplier must be non-negative")


@dataclass(frozen=True)
class CashFlowStressResult:
    scenario: CashFlowStressScenario
    baseline: CashFlowForecast
    stressed: CashFlowForecast
    ending_cash_change: float
    npv_change: float


@dataclass(frozen=True)
class CreditStressScenario:
    """Stress multipliers for probability of default and loss given default."""

    name: str
    pd_multiplier: float = 1.0
    lgd_multiplier: float = 1.0

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if self.pd_multiplier < 0 or self.lgd_multiplier < 0:
            raise ValueError("credit stress multipliers must be non-negative")


@dataclass(frozen=True)
class StressedLoanRisk:
    loan: Loan
    base_probability_of_default: float
    stressed_probability_of_default: float
    stressed_loss_given_default: float
    stressed_expected_loss: float


@dataclass(frozen=True)
class CreditStressResult:
    scenario: CreditStressScenario
    loans: tuple[StressedLoanRisk, ...]
    total_exposure: float
    baseline_expected_loss: float
    stressed_expected_loss: float
    stressed_expected_loss_ratio: float
    expected_loss_change: float


@dataclass(frozen=True)
class PortfolioMonteCarloResult:
    terminal_values: np.ndarray
    initial_investment: float
    horizon_years: int
    expected_annual_return: float
    annual_volatility: float
    mean_terminal_value: float
    median_terminal_value: float
    percentile_05: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    probability_of_loss: float


def stress_cash_flow(
    assumptions: CashFlowAssumptions,
    scenario: CashFlowStressScenario,
    *,
    months: int = 12,
    capex_by_month: Mapping[int, float] | None = None,
) -> CashFlowStressResult:
    """Compare a stressed cash-flow forecast with its unchanged baseline."""
    scenario.validate()
    base_capex = dict(DEFAULT_CAPEX if capex_by_month is None else capex_by_month)
    baseline = forecast_cash_flow(
        assumptions,
        months=months,
        capex_by_month=base_capex,
    )

    stressed_assumptions = CashFlowAssumptions(
        starting_cash=assumptions.starting_cash,
        revenue_month1=assumptions.revenue_month1 * scenario.revenue_multiplier,
        monthly_revenue_growth=(
            assumptions.monthly_revenue_growth + scenario.monthly_growth_delta
        ),
        operating_expense_ratio=(
            assumptions.operating_expense_ratio + scenario.operating_expense_ratio_delta
        ),
        monthly_loan_payment=(
            assumptions.monthly_loan_payment * scenario.fixed_cost_multiplier
        ),
        monthly_rent=assumptions.monthly_rent * scenario.fixed_cost_multiplier,
        annual_discount_rate=assumptions.annual_discount_rate,
    )
    stressed_capex = {
        month: amount * scenario.capex_multiplier
        for month, amount in base_capex.items()
    }
    stressed = forecast_cash_flow(
        stressed_assumptions,
        months=months,
        capex_by_month=stressed_capex,
    )

    return CashFlowStressResult(
        scenario=scenario,
        baseline=baseline,
        stressed=stressed,
        ending_cash_change=(
            stressed.periods[-1].ending_cash - baseline.periods[-1].ending_cash
        ),
        npv_change=stressed.npv_of_net_cash_flows - baseline.npv_of_net_cash_flows,
    )


def stress_credit_portfolio(
    loans: Iterable[Loan],
    scenario: CreditStressScenario,
    *,
    base_loss_given_default: float = 0.45,
) -> CreditStressResult:
    """Apply capped PD/LGD stress multipliers to a loan portfolio."""
    scenario.validate()
    loan_tuple = tuple(loans)
    if not loan_tuple:
        raise ValueError("portfolio must contain at least one loan")
    if not 0 <= base_loss_given_default <= 1:
        raise ValueError("base loss given default must be between 0 and 1")

    baseline = summarize_portfolio(
        loan_tuple,
        loss_given_default=base_loss_given_default,
    )
    stressed_lgd = min(base_loss_given_default * scenario.lgd_multiplier, 1.0)
    stressed_loans: list[StressedLoanRisk] = []

    for loan in loan_tuple:
        loan.validate()
        base_pd = probability_of_default(loan.credit_score)
        stressed_pd = min(base_pd * scenario.pd_multiplier, 1.0)
        stressed_loans.append(
            StressedLoanRisk(
                loan=loan,
                base_probability_of_default=base_pd,
                stressed_probability_of_default=stressed_pd,
                stressed_loss_given_default=stressed_lgd,
                stressed_expected_loss=loan.exposure * stressed_pd * stressed_lgd,
            )
        )

    stressed_total = sum(result.stressed_expected_loss for result in stressed_loans)
    total_exposure = baseline.total_exposure
    stressed_ratio = stressed_total / total_exposure if total_exposure else 0.0

    return CreditStressResult(
        scenario=scenario,
        loans=tuple(stressed_loans),
        total_exposure=total_exposure,
        baseline_expected_loss=baseline.total_expected_loss,
        stressed_expected_loss=stressed_total,
        stressed_expected_loss_ratio=stressed_ratio,
        expected_loss_change=stressed_total - baseline.total_expected_loss,
    )


def simulate_portfolio_terminal_values(
    assets: Iterable[AssetAllocation],
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    initial_investment: float = 1_000_000.0,
    horizon_years: int = 10,
    n_simulations: int = 10_000,
    random_state: int | None = 42,
) -> PortfolioMonteCarloResult:
    """Simulate terminal wealth with a lognormal portfolio-return approximation.

    The annual arithmetic return and volatility are first calculated from the
    supplied asset weights and covariance matrix. They are then converted to
    lognormal parameters so simulated gross returns remain positive while
    matching the requested first two moments.
    """
    if initial_investment < 0 or not np.isfinite(initial_investment):
        raise ValueError("initial investment must be finite and non-negative")
    if isinstance(horizon_years, bool) or not isinstance(horizon_years, int):
        raise TypeError("horizon years must be a positive integer")
    if horizon_years < 1:
        raise ValueError("horizon years must be at least 1")
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, int):
        raise TypeError("n_simulations must be a positive integer")
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")

    asset_tuple = tuple(assets)
    metrics = correlated_portfolio_metrics(
        asset_tuple,
        covariance,
        investment_amount=initial_investment,
        horizon_years=horizon_years,
    )

    gross_mean = 1.0 + metrics.expected_return
    if gross_mean <= 0:
        raise ValueError("expected annual return must be greater than -100%")

    log_variance = np.log1p((metrics.volatility**2) / (gross_mean**2))
    log_sigma = float(np.sqrt(log_variance))
    log_mean = float(np.log(gross_mean) - 0.5 * log_variance)

    rng = np.random.default_rng(random_state)
    annual_log_returns = rng.normal(
        loc=log_mean,
        scale=log_sigma,
        size=(n_simulations, horizon_years),
    )
    terminal_values = initial_investment * np.exp(annual_log_returns.sum(axis=1))
    percentiles = np.quantile(terminal_values, [0.05, 0.25, 0.50, 0.75, 0.95])

    return PortfolioMonteCarloResult(
        terminal_values=terminal_values,
        initial_investment=initial_investment,
        horizon_years=horizon_years,
        expected_annual_return=metrics.expected_return,
        annual_volatility=metrics.volatility,
        mean_terminal_value=float(terminal_values.mean()),
        median_terminal_value=float(percentiles[2]),
        percentile_05=float(percentiles[0]),
        percentile_25=float(percentiles[1]),
        percentile_75=float(percentiles[3]),
        percentile_95=float(percentiles[4]),
        probability_of_loss=float(np.mean(terminal_values < initial_investment)),
    )
