from __future__ import annotations

from datetime import date

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
    backtest_rebalanced_portfolio,
    correlated_portfolio_metrics,
    covariance_from_correlation,
    cumulative_wealth,
    download_adjusted_close,
    forecast_cash_flow,
    historical_risk_summary,
    illustrative_correlation_matrix,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    performance_summary,
    simple_returns,
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_prices(
    tickers: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    return download_adjusted_close(tickers, start=start, end=end)


def default_loans_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Loan ID": "L-001",
                "Borrower": "Strong Co.",
                "Exposure": 100_000.0,
                "Credit Score": 780,
            },
            {
                "Loan ID": "L-002",
                "Borrower": "Mid Market",
                "Exposure": 150_000.0,
                "Credit Score": 620,
            },
            {
                "Loan ID": "L-003",
                "Borrower": "Growth Co.",
                "Exposure": 200_000.0,
                "Credit Score": 540,
            },
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
        "This dashboard combines tested financial models with real-market analytics. "
        "Use the tabs below to inspect liquidity, credit, market, portfolio, and stress risk."
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
    starting_cash = c1.number_input(
        "Starting cash",
        min_value=0.0,
        value=50_000.0,
        step=5_000.0,
    )
    revenue = c2.number_input(
        "Month 1 revenue",
        min_value=0.0,
        value=100_000.0,
        step=5_000.0,
    )
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
            "Credit Score": st.column_config.NumberColumn(
                min_value=300,
                max_value=850,
                step=1,
            ),
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


def render_market_risk() -> None:
    st.subheader("Real-market risk & performance")
    st.caption(
        "Download historical adjusted prices, estimate return/covariance risk, "
        "and compare a portfolio with a selected benchmark."
    )

    with st.form("market_risk_form"):
        c1, c2 = st.columns(2)
        tickers_text = c1.text_input("Tickers", value="SPY, QQQ, TLT, GLD")
        weights_text = c2.text_input("Weights (%)", value="25, 25, 25, 25")
        c3, c4, c5 = st.columns(3)
        start_date = c3.date_input("Start date", value=date(2021, 1, 1))
        end_date = c4.date_input("End date", value=date.today())
        benchmark_text = c5.text_input("Benchmark ticker", value="SPY")
        c6, c7 = st.columns(2)
        rebalance_every = c6.number_input(
            "Rebalance every (trading days)",
            min_value=1,
            max_value=252,
            value=21,
            step=1,
        )
        transaction_cost_bps = c7.number_input(
            "Transaction cost (basis points)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
        )
        submitted = st.form_submit_button("Run historical analysis")

    if not submitted:
        st.info("Run the analysis to load historical market data and portfolio risk metrics.")
        return

    symbols = tuple(
        symbol.strip().upper()
        for symbol in tickers_text.split(",")
        if symbol.strip()
    )
    benchmark = benchmark_text.strip().upper()
    try:
        weights = np.asarray(
            [float(value.strip()) for value in weights_text.split(",")],
            dtype=float,
        )
    except ValueError:
        st.error("Weights must be comma-separated numbers.")
        return

    if not symbols:
        st.error("Enter at least one ticker.")
        return
    if len(weights) != len(symbols):
        st.error("Provide one weight for each ticker.")
        return
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        st.error("Weights must be non-negative with a positive total.")
        return
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return

    weights = weights / weights.sum()
    request_symbols = tuple(dict.fromkeys((*symbols, benchmark)))

    try:
        with st.spinner("Loading historical market data..."):
            prices = load_market_prices(
                request_symbols,
                start_date.isoformat(),
                end_date.isoformat(),
            )
        missing = [symbol for symbol in request_symbols if symbol not in prices.columns]
        if missing:
            raise ValueError(f"missing downloaded prices for: {', '.join(missing)}")

        portfolio_prices = prices.loc[:, list(symbols)]
        market_summary = historical_risk_summary(portfolio_prices)
        backtest = backtest_rebalanced_portfolio(
            portfolio_prices,
            weights,
            rebalance_every=int(rebalance_every),
            transaction_cost_bps=float(transaction_cost_bps),
        )
        portfolio_returns = backtest.returns

        benchmark_prices = prices[[benchmark]]
        benchmark_returns = simple_returns(benchmark_prices)[benchmark]
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns.rename("benchmark_return")],
            axis=1,
            join="inner",
        ).dropna()
        portfolio_summary = performance_summary(aligned["portfolio_return"])
        benchmark_summary = performance_summary(aligned["benchmark_return"])
    except (ImportError, TypeError, ValueError) as exc:
        st.error(str(exc))
        st.caption('Install market-data support with: pip install -e ".[dashboard,market-data]"')
        return

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Portfolio CAGR", percent(portfolio_summary.cagr))
    m2.metric("Volatility", percent(portfolio_summary.annualized_volatility))
    m3.metric("Sharpe", f"{portfolio_summary.sharpe_ratio:.2f}")
    m4.metric("Sortino", f"{portfolio_summary.sortino_ratio:.2f}")
    m5.metric("Max drawdown", percent(portfolio_summary.max_drawdown))
    m6.metric("95% Expected Shortfall", percent(portfolio_summary.expected_shortfall))
    st.caption(
        f"Rebalanced every {int(rebalance_every)} trading days | "
        f"transaction cost: {float(transaction_cost_bps):.1f} bps | "
        f"cumulative turnover: {backtest.total_turnover:.2f}x | "
        f"modeled costs: {backtest.total_transaction_cost:.2%} of initial capital"
    )

    asset_statistics = pd.DataFrame(
        {
            "Annualized Return": market_summary.annualized_returns,
            "Annualized Volatility": market_summary.annualized_volatility,
            "Maximum Drawdown": market_summary.max_drawdown,
            "Weight": pd.Series(weights, index=symbols),
        }
    )
    st.markdown("### Asset-level statistics")
    st.dataframe(
        asset_statistics.style.format("{:.2%}"),
        use_container_width=True,
    )

    wealth = pd.DataFrame(
        {
            "Portfolio": cumulative_wealth(aligned["portfolio_return"]),
            benchmark: cumulative_wealth(aligned["benchmark_return"]),
        },
        index=aligned.index,
    )
    st.markdown("### Portfolio vs benchmark")
    st.line_chart(wealth)

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Cumulative return",
                "CAGR",
                "Annualized volatility",
                "Sharpe ratio",
                "Sortino ratio",
                "Maximum drawdown",
                "95% historical VaR",
                "95% Expected Shortfall",
            ],
            "Portfolio": [
                percent(portfolio_summary.cumulative_return),
                percent(portfolio_summary.cagr),
                percent(portfolio_summary.annualized_volatility),
                f"{portfolio_summary.sharpe_ratio:.2f}",
                f"{portfolio_summary.sortino_ratio:.2f}",
                percent(portfolio_summary.max_drawdown),
                percent(portfolio_summary.value_at_risk),
                percent(portfolio_summary.expected_shortfall),
            ],
            benchmark: [
                percent(benchmark_summary.cumulative_return),
                percent(benchmark_summary.cagr),
                percent(benchmark_summary.annualized_volatility),
                f"{benchmark_summary.sharpe_ratio:.2f}",
                f"{benchmark_summary.sortino_ratio:.2f}",
                percent(benchmark_summary.max_drawdown),
                percent(benchmark_summary.value_at_risk),
                percent(benchmark_summary.expected_shortfall),
            ],
        }
    )
    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Correlation matrix")
    st.dataframe(
        market_summary.correlation.style.format("{:.2f}"),
        use_container_width=True,
    )


def render_portfolio() -> None:
    st.subheader("Covariance-aware portfolio analytics")
    st.caption(
        "Edit expected returns, volatility, or weights. "
        "Weights are normalized to 100% for analysis."
    )
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
        revenue_multiplier = st.slider(
            "Revenue level",
            50.0,
            120.0,
            80.0,
            5.0,
        ) / 100.0
        growth_delta = st.slider(
            "Monthly growth shock",
            -10.0,
            5.0,
            -3.0,
            0.5,
        ) / 100.0
        opex_delta = st.slider(
            "OpEx ratio shock",
            -10.0,
            30.0,
            10.0,
            1.0,
        ) / 100.0
        fixed_multiplier = st.slider(
            "Fixed-cost multiplier",
            80.0,
            150.0,
            110.0,
            5.0,
        ) / 100.0

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
        simulations = st.slider(
            "Monte Carlo paths",
            1_000,
            25_000,
            10_000,
            1_000,
        )
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
st.caption(
    "Real-market risk analytics, portfolio modeling, credit risk, "
    "liquidity forecasting, and Monte Carlo simulation"
)

(
    overview_tab,
    cash_tab,
    credit_tab,
    market_tab,
    portfolio_tab,
    stress_tab,
) = st.tabs(
    [
        "Overview",
        "Cash Flow",
        "Credit Risk",
        "Market Risk",
        "Portfolio",
        "Stress & Monte Carlo",
    ]
)

with overview_tab:
    render_overview()
with cash_tab:
    render_cash_flow()
with credit_tab:
    render_credit()
with market_tab:
    render_market_risk()
with portfolio_tab:
    render_portfolio()
with stress_tab:
    render_stress()

st.divider()
st.caption(
    "Educational portfolio project only — not investment, lending, accounting, "
    "or financial advice."
)
