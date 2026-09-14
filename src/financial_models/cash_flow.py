from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean


DEFAULT_CAPEX = {2: 15_000.0, 5: 15_000.0, 9: 15_000.0}


@dataclass(frozen=True)
class CashFlowAssumptions:
    starting_cash: float = 50_000.0
    revenue_month1: float = 100_000.0
    monthly_revenue_growth: float = 0.05
    operating_expense_ratio: float = 0.60
    monthly_loan_payment: float = 5_000.0
    monthly_rent: float = 8_000.0
    annual_discount_rate: float = 0.10

    def validate(self) -> None:
        if self.starting_cash < 0 or self.revenue_month1 < 0:
            raise ValueError("starting cash and revenue must be non-negative")
        if self.monthly_revenue_growth <= -1:
            raise ValueError("monthly revenue growth must be greater than -100%")
        if not 0 <= self.operating_expense_ratio <= 1:
            raise ValueError("operating expense ratio must be between 0 and 1")
        if self.monthly_loan_payment < 0 or self.monthly_rent < 0:
            raise ValueError("fixed monthly outflows must be non-negative")
        if self.annual_discount_rate < 0:
            raise ValueError("annual discount rate must be non-negative")


@dataclass(frozen=True)
class CashFlowPeriod:
    month: int
    beginning_cash: float
    revenue: float
    operating_expense: float
    rent: float
    loan_payment: float
    capex: float
    net_cash_flow: float
    ending_cash: float


@dataclass(frozen=True)
class CashFlowForecast:
    periods: tuple[CashFlowPeriod, ...]
    npv_of_net_cash_flows: float
    average_monthly_net_cash_flow: float
    minimum_cash_balance: float
    maximum_cash_balance: float


def _monthly_discount_rate(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def forecast_cash_flow(
    assumptions: CashFlowAssumptions | None = None,
    *,
    months: int = 12,
    capex_by_month: Mapping[int, float] | None = None,
) -> CashFlowForecast:
    """Build a monthly cash-flow forecast from the workbook's documented assumptions."""
    assumptions = assumptions or CashFlowAssumptions()
    assumptions.validate()
    if isinstance(months, bool) or not isinstance(months, int) or months < 1:
        raise ValueError("months must be a positive integer")

    capex_schedule = dict(DEFAULT_CAPEX if capex_by_month is None else capex_by_month)
    for month, amount in capex_schedule.items():
        if not isinstance(month, int) or month < 1 or month > months:
            raise ValueError("capex month keys must fall inside the forecast horizon")
        if amount < 0:
            raise ValueError("capex amounts must be non-negative")

    periods: list[CashFlowPeriod] = []
    beginning_cash = float(assumptions.starting_cash)

    for month in range(1, months + 1):
        revenue = assumptions.revenue_month1 * (
            1.0 + assumptions.monthly_revenue_growth
        ) ** (month - 1)
        operating_expense = revenue * assumptions.operating_expense_ratio
        capex = float(capex_schedule.get(month, 0.0))
        net_cash_flow = (
            revenue
            - operating_expense
            - assumptions.monthly_rent
            - assumptions.monthly_loan_payment
            - capex
        )
        ending_cash = beginning_cash + net_cash_flow

        periods.append(
            CashFlowPeriod(
                month=month,
                beginning_cash=beginning_cash,
                revenue=revenue,
                operating_expense=operating_expense,
                rent=assumptions.monthly_rent,
                loan_payment=assumptions.monthly_loan_payment,
                capex=capex,
                net_cash_flow=net_cash_flow,
                ending_cash=ending_cash,
            )
        )
        beginning_cash = ending_cash

    monthly_discount = _monthly_discount_rate(assumptions.annual_discount_rate)
    npv = sum(
        period.net_cash_flow / (1.0 + monthly_discount) ** period.month
        for period in periods
    )
    balances = [period.ending_cash for period in periods]
    net_cash_flows = [period.net_cash_flow for period in periods]

    return CashFlowForecast(
        periods=tuple(periods),
        npv_of_net_cash_flows=npv,
        average_monthly_net_cash_flow=fmean(net_cash_flows),
        minimum_cash_balance=min(balances),
        maximum_cash_balance=max(balances),
    )
