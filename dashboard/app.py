from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from financial_models import (
    AssetAllocation,
    CashFlowAssumptions,
    CashFlowStressScenario,
    CreditStressScenario,
    Loan,
    approximate_efficient_frontier,
    correlated_portfolio_metrics,
    covariance_from_correlation,
    forecast_cash_flow,
    illustrative_correlation_matrix,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    simulate_long_only_portfolios,
    simulate_portfolio_terminal_values,
    stress_cash_flow,
    stress_credit_portfolio,
    summarize_portfolio,
    workbook_balanced_portfolio,
)


st.set_page_config(
    page_title="Advanced Financial Models",
    page_icon="📊",
    layout="wide",
)


def money(value: float) -> str:
    return f"${value:,.0f}"


def percent(value: float) -> str:
    return f"{value:.2%}"


def default_loans_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Loan ID": "L-001", "Borrower": "Strong Co.", "Exposure": 100_000.0, "Credit Score": 780},
            {"Loan ID": "L-002", "Borrower": "Mid Market", "Exposure": 150_000.0, "Credit Score": 620},
            {"Loan ID": "L-003", "Borrower": "Growth Co.", "Exposure": 200_000.0, "Credit Score": 540},
        ]
    )


def loans_from_frame(frame: pd.DataFrame) -> tuple[Loan, ...]:
    loans: list[Loan] = []
    for row in frame.itertuples(index=False, name=None):
        loan_id, borrower, exposure, credit_score = row
        loans.append(
            Loan(
                str(loan_id),
                str(borrower),
                float(exposure),
                int(credit_score),
            )
        )
    return tuple(loans)


def baseline_portfolio_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": asset.name,
                "Expected Return %": asset.expected_return * 100,
                "Volatility %": asset.volatility * 100,
                "Weight %": asset.weight * 100,
            }
            for asset in workbook_balanced_portfolio()
        ]
    )


def assets_from_frame(frame: pd.DataFrame) -> tuple[AssetAllocation, ...]:
    raw_weights = frame["Weight %"].astype(float).to_numpy()
    if np.any(raw_weights < 0):
        raise ValueError("portfolio weights must be non-negative")
    total_weight = float(raw_weights.sum())
    if total_weight <= 0:
        raise ValueError("portfolio weights must have a positive total")

    normalized = raw_weights / total_weight
    assets: list[AssetAllocation] = []
    for position, row in enumerate(frame.itertuples(index=False, name=None)):
        name, expected_return, volatility, _ = row
        assets.append(
            AssetAllocation(
                str(name),
                float(expected_return) / 100.0,
                float(volatility) / 100.0,
                float(normalized[position]),
            )
        )
    return tuple(assets)


def cash_flow_table(assumptions: CashFlowAssumptions) -> tuple[pd.DataFrame, object]:
    forecast = forecast_cash_flow(assumptions)
    frame = pd.DataFrame(
        [
            {
                "Month": period.month,
                "Revenue": period.revenue,
                "Net Cash Flow": period.net_cash_flow,
                "Ending Cash": period.ending_cash,
            }
            for period in forecast.periods
        ]
    ).set_index("Month")
    return frame, forecast


def render_overview() -> None:
    cash = forecast_cash_flow(CashFlowAssumptions())
    loans = loans_from_frame(default_loans_frame())
    credit = summarize_portfolio(loans)
    assets = workbook_balanced_portfolio()
    covariance = covariance_from_correlation(
        [asset.volatility for asset in assets],
        illustrative_correlation_matrix(),
    )
    portfolio = correlated_portfolio_metrics(assets, covariance)

    st.subheader("Executive snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("12-month ending cash", money(cash.periods[-1].ending_cash))
    col2.metric("Credit expected loss", money(credit.total_expected_loss))
    col3.metric("Portfolio expected return", percent(portfolio.expected_return))
    col4.metric("Covariance-aware volatility", percent(portfolio.volatility))

    st.markdown(
        "This dashboard uses the same tested Python models as the repository. "
        "Change assumptions in the tabs below to compare liquidity, credit, and portfolio risk."
    )

    overview = pd.DataFrame(
        {
            "Metric": [
                "Cash-flow NPV",
                "Expected-loss ratio",
                "Portfolio Sharpe ratio",
                "10-year projected value",
            ],
            "Value": [
                money(cash.npv_of_net_cash_flows),
                percent(credit.expected_loss_ratio),
                f"{portfolio.sharpe_ratio:.2f}",
                money(portfolio.future_value),
            ],
        }
    )
    st.dataframe(overview, hide_index=True, use_container_width=True)


def render_cash_flow() -> None:
    st.subheader("Cash-flow forecast")
    c1, c2, c3, c4 = st.columns(4)
    starting_cash = c1.number_input("Starting cash", min_value=0.0, value=50_000.0, step=5_000.0)
    revenue = c2.number_input("Month 1 revenue", min_value=0.0, value=100_000.0, step=5_000.0)
    growth = c3.slider("Monthly revenue growth", -10.0, 15.0, 5.0, 0.5) / 100.0
    opex = c4.slider("Operating expense ratio", 0.0, 100.0, 60.0, 1.0) / 100.0

    assumptions = CashFlowAssumptions(
        starting_cash=starting_cash,
        revenue_month1=revenue,
        monthly_revenue_growth=growth,
        operating_expense_ratio=opex,
    )
    frame, forecast = cash_flow_table(assumptions)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ending cash", money(forecast.periods[-1].ending_cash))
    m2.metric("Cash-flow NPV", money(forecast.npv_of_net_cash_flows))
    m3.metric("Minimum cash", money(forecast.minimum_cash_balance))
    m4.metric("Average monthly FCF", money(forecast.average_monthly_net_cash_flow))

    st.line_chart(frame[["Revenue", "Ending Cash"]])
    st.dataframe(
        frame.style.format("${:,.0f}"),
        use_container_width=True,
    )


def render_credit() -> None:
    st.subheader("Credit expected loss")
    lgd = st.slider("Loss given default", 0.0, 100.0, 45.0, 1.0) / 100.0
    edited = st.data_editor(
        default_loans_frame(),
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Exposure": st.column_config.NumberColumn(min_value=0.0, format="$%.0f"),
            "Credit Score": st.column_config.NumberColumn(min_value=300, max_value=850, step=1),
        },
        key="credit_editor",
    )

    try:
        loans = loans_from_frame(edited)
        summary = summarize_portfolio(loans, loss_given_default=lgd)
    except (TypeError, ValueError) as exc:
        st.error(str(exc))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total exposure", money(summary.total_exposure))
    c2.metric("Expected loss", money(summary.total_expected_loss))
    c3.metric("Expected-loss ratio", percent(summary.expected_loss_ratio))
    c4.metric("Average credit score", f"{summary.average_credit_score:.0f}")

    rows = pd.DataFrame(
        [
            {
                "Loan ID": result.loan.loan_id,
                "Borrower": result.loan.borrower,
                "Exposure": result.loan.exposure,
                "PD": result.probability_of_default,
                "LGD": result.loss_given_default,
                "Expected Loss": result.expected_loss,
                "Risk Rating": result.risk_rating,
            }
            for result in summary.loans
        ]
    )
    st.dataframe(
        rows.style.format(
            {
                "Exposure": "${:,.0f}",
                "PD": "{:.1%}",
                "LGD": "{:.1%}",
                "Expected Loss": "${:,.0f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_portfolio() -> None:
    st.subheader("Covariance-aware portfolio analytics")
    st.caption("Edit expected returns, volatility, or weights. Weights are normalized to 100% for analysis.")
    edited = st.data_editor(
        baseline_portfolio_frame(),
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key="portfolio_editor",
    )
    n_portfolios = st.slider("Simulated portfolios", 1_000, 20_000, 6_000, 1_000)

    try:
        assets = assets_from_frame(edited)
        covariance = covariance_from_correlation(
            [asset.volatility for asset in assets],
            illustrative_correlation_matrix(),
        )
        metrics = correlated_portfolio_metrics(assets, covariance)
        simulation = simulate_long_only_portfolios(
            assets,
            covariance,
            n_portfolios=n_portfolios,
            random_state=42,
        )
    except (TypeError, ValueError) as exc:
        st.error(str(exc))
        return

    max_sharpe = max_sharpe_portfolio(simulation)
    min_vol = min_volatility_portfolio(simulation)
    frontier = approximate_efficient_frontier(simulation, max_points=20)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected return", percent(metrics.expected_return))
    c2.metric("Volatility", percent(metrics.volatility))
    c3.metric("Sharpe ratio", f"{metrics.sharpe_ratio:.2f}")
    c4.metric("10-year value", money(metrics.future_value))

    sample = pd.DataFrame(
        {
            "Volatility": simulation.volatilities,
            "Expected Return": simulation.expected_returns,
            "Sharpe": simulation.sharpe_ratios,
        }
    )
    st.scatter_chart(sample, x="Volatility", y="Expected Return", size=12)

    frontier_frame = pd.DataFrame(
        {
            "Volatility": [point.volatility for point in frontier],
            "Expected Return": [point.expected_return for point in frontier],
        }
    ).set_index("Volatility")
    st.markdown("**Approximate efficient frontier**")
    st.line_chart(frontier_frame)

    best = pd.DataFrame(
        [
            {
                "Portfolio": "Max Sharpe",
                "Expected Return": max_sharpe.expected_return,
                "Volatility": max_sharpe.volatility,
                "Sharpe": max_sharpe.sharpe_ratio,
            },
            {
                "Portfolio": "Minimum Volatility",
                "Expected Return": min_vol.expected_return,
                "Volatility": min_vol.volatility,
                "Sharpe": min_vol.sharpe_ratio,
            },
        ]
    )
    st.dataframe(
        best.style.format(
            {
                "Expected Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_stress() -> None:
    st.subheader("Stress testing and Monte Carlo")
    left, right = st.columns(2)

    with left:
        st.markdown("### Liquidity stress")
        revenue_multiplier = st.slider("Revenue level", 50.0, 120.0, 80.0, 5.0) / 100.0
        growth_delta = st.slider("Monthly growth shock", -10.0, 5.0, -3.0, 0.5) / 100.0
        opex_delta = st.slider("OpEx ratio shock", -10.0, 30.0, 10.0, 1.0) / 100.0
        fixed_multiplier = st.slider("Fixed-cost multiplier", 80.0, 150.0, 110.0, 5.0) / 100.0

        try:
            cash_stress = stress_cash_flow(
                CashFlowAssumptions(),
                CashFlowStressScenario(
                    "Dashboard stress",
                    revenue_multiplier=revenue_multiplier,
                    monthly_growth_delta=growth_delta,
                    operating_expense_ratio_delta=opex_delta,
                    fixed_cost_multiplier=fixed_multiplier,
                ),
            )
            st.metric("Ending cash impact", money(cash_stress.ending_cash_change))
            st.metric("NPV impact", money(cash_stress.npv_change))
        except ValueError as exc:
            st.error(str(exc))

        st.markdown("### Credit stress")
        pd_multiplier = st.slider("PD multiplier", 0.5, 4.0, 1.75, 0.25)
        lgd_multiplier = st.slider("LGD multiplier", 0.5, 2.5, 1.25, 0.25)
        credit_stress = stress_credit_portfolio(
            loans_from_frame(default_loans_frame()),
            CreditStressScenario(
                "Dashboard credit stress",
                pd_multiplier=pd_multiplier,
                lgd_multiplier=lgd_multiplier,
            ),
        )
        st.metric(
            "Stressed expected loss",
            money(credit_stress.stressed_expected_loss),
            delta=money(credit_stress.expected_loss_change),
            delta_color="inverse",
        )

    with right:
        st.markdown("### Portfolio Monte Carlo")
        horizon = st.slider("Horizon (years)", 1, 30, 10)
        simulations = st.slider("Monte Carlo paths", 1_000, 25_000, 10_000, 1_000)
        assets = workbook_balanced_portfolio()
        covariance = covariance_from_correlation(
            [asset.volatility for asset in assets],
            illustrative_correlation_matrix(),
        )
        result = simulate_portfolio_terminal_values(
            assets,
            covariance,
            horizon_years=horizon,
            n_simulations=simulations,
            random_state=42,
        )

        c1, c2 = st.columns(2)
        c1.metric("Median terminal value", money(result.median_terminal_value))
        c2.metric("Probability of loss", percent(result.probability_of_loss))
        c3, c4 = st.columns(2)
        c3.metric("5th percentile", money(result.percentile_05))
        c4.metric("95th percentile", money(result.percentile_95))

        counts, edges = np.histogram(result.terminal_values, bins=30)
        histogram = pd.DataFrame(
            {
                "Terminal Value": (edges[:-1] + edges[1:]) / 2.0,
                "Paths": counts,
            }
        ).set_index("Terminal Value")
        st.bar_chart(histogram)


st.title("Advanced Financial Models")
st.caption("Excel-backed financial modeling with reproducible Python analytics")

overview_tab, cash_tab, credit_tab, portfolio_tab, stress_tab = st.tabs(
    ["Overview", "Cash Flow", "Credit Risk", "Portfolio", "Stress & Monte Carlo"]
)

with overview_tab:
    render_overview()
with cash_tab:
    render_cash_flow()
with credit_tab:
    render_credit()
with portfolio_tab:
    render_portfolio()
with stress_tab:
    render_stress()

st.divider()
st.caption(
    "Educational portfolio project only — not investment, lending, accounting, or financial advice."
)
