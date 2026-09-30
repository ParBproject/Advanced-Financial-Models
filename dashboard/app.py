"""Streamlit decision board for the tested financial-model package."""

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
    assess_loan,
    correlated_portfolio_metrics,
    covariance_from_correlation,
    credit_book_memo_audit,
    credit_concentration,
    forecast_cash_flow,
    format_credit_book_decision,
    illustrative_correlation_matrix,
    loans_from_credit_book,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    portfolio_metrics,
    simulate_long_only_portfolios,
    simulate_portfolio_terminal_values,
    stress_cash_flow,
    stress_credit_portfolio,
    summarize_portfolio,
    workbook_balanced_portfolio,
)
from financial_models.charts import (
    allocation_plotly,
    cash_flow_plotly,
    credit_ratings_plotly,
    frontier_plotly,
    monte_carlo_plotly,
    stressed_cash_plotly,
    wealth_plotly,
)
from financial_models.market_data import (
    align_prices,
    annualize_sample,
    backtest_rebalanced_portfolio,
    cumulative_wealth,
    download_adjusted_close,
    historical_risk_summary,
    performance_summary,
    portfolio_return_series,
    simple_returns,
    worked_example_prices,
)
from financial_models.theme import streamlit_css

st.set_page_config(
    page_title="Advanced Financial Models",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(streamlit_css(), unsafe_allow_html=True)


def money(value: float, *, decimals: int = 0) -> str:
    return f"${value:,.{decimals}f}"


def prose(text: str) -> str:
    """Keep currency signs as text. Streamlit markdown treats $...$ as LaTeX."""
    return text.replace("$", r"\$")


def percent(value: float) -> str:
    return f"{value:.2%}"


def show(figure, key: str) -> None:
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False},
        key=key,
    )


def portfolio_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": asset.name,
                "Expected return %": asset.expected_return * 100,
                "Volatility %": asset.volatility * 100,
                "Weight %": asset.weight * 100,
            }
            for asset in workbook_balanced_portfolio()
        ]
    )


def assets_from_frame(frame: pd.DataFrame) -> tuple[AssetAllocation, ...]:
    raw_weights = frame["Weight %"].astype(float).to_numpy()
    if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0):
        raise ValueError("portfolio weights must be finite and non-negative")
    total_weight = float(raw_weights.sum())
    if total_weight <= 0:
        raise ValueError("portfolio weights must have a positive total")
    normalized = raw_weights / total_weight
    assets: list[AssetAllocation] = []
    for position, row in enumerate(frame.itertuples(index=False, name=None)):
        name, expected_return, volatility, _weight = row
        assets.append(
            AssetAllocation(
                str(name),
                float(expected_return) / 100.0,
                float(volatility) / 100.0,
                float(normalized[position]),
            )
        )
    return tuple(assets)


def prices_from_upload(upload) -> pd.DataFrame:
    frame = pd.read_csv(upload)
    date_column = next(
        (column for column in frame.columns if str(column).lower() in {"date", "datetime"}),
        None,
    )
    if date_column is not None:
        frame = frame.set_index(date_column)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame.apply(pd.to_numeric, errors="coerce")


def render_header() -> None:
    st.markdown('<p class="afm-kicker">Liquidity · Credit · Portfolio risk</p>', unsafe_allow_html=True)
    st.title("Advanced Financial Models")
    st.markdown('<div class="afm-rule"></div>', unsafe_allow_html=True)
    st.caption(
        prose(
            "Cash-flow forecasting, credit expected loss and portfolio risk "
            "on one tested Python engine."
        )
    )


def render_overview() -> None:
    cash = forecast_cash_flow(CashFlowAssumptions())
    loans = loans_from_credit_book()
    credit = summarize_portfolio(loans)
    concentration = credit_concentration(loans)
    audit = credit_book_memo_audit()
    assets = workbook_balanced_portfolio()
    covariance = covariance_from_correlation(
        [asset.volatility for asset in assets],
        illustrative_correlation_matrix(),
    )
    portfolio = correlated_portfolio_metrics(assets, covariance)

    columns = st.columns(4)
    columns[0].metric("12-month ending cash", money(cash.periods[-1].ending_cash))
    columns[1].metric("Credit expected loss", money(credit.total_expected_loss, decimals=2))
    columns[2].metric("Borrower HHI", f"{concentration.herfindahl_hirschman_index:.6f}")
    columns[3].metric("Covariance-aware volatility", percent(portfolio.volatility))

    st.markdown(prose(format_credit_book_decision(credit, concentration, audit)))
    show(cash_flow_plotly(cash), key="overview-cash")
    st.caption(
        prose(
            "The cash forecast uses the workbook assumptions. Expected loss uses a 45% loss given "
            "default and the score-band default rate. The file's reported probability of default "
            f"averages {audit.average_reported_pd:.2%} and is not the rate in that calculation."
        )
    )


def render_cash_flow() -> None:
    st.subheader("Cash-flow forecast")
    controls = st.columns(4)
    starting_cash = controls[0].number_input("Starting cash", min_value=0.0, value=50_000.0, step=5_000.0)
    revenue = controls[1].number_input("Month 1 revenue", min_value=0.0, value=100_000.0, step=5_000.0)
    growth = controls[2].slider("Monthly revenue growth", -10.0, 15.0, 5.0, 0.5) / 100.0
    opex = controls[3].slider("Operating expense ratio", 0.0, 100.0, 60.0, 1.0) / 100.0
    assumptions = CashFlowAssumptions(
        starting_cash=starting_cash,
        revenue_month1=revenue,
        monthly_revenue_growth=growth,
        operating_expense_ratio=opex,
    )
    forecast = forecast_cash_flow(assumptions)
    metrics = st.columns(4)
    metrics[0].metric("Ending cash", money(forecast.periods[-1].ending_cash))
    metrics[1].metric("NPV of net cash flows", money(forecast.npv_of_net_cash_flows))
    metrics[2].metric("Minimum cash", money(forecast.minimum_cash_balance))
    metrics[3].metric("Average monthly net cash flow", money(forecast.average_monthly_net_cash_flow))
    show(cash_flow_plotly(forecast), key="cash-flow-chart")
    st.caption(
        prose("NPV discounts each month at the effective monthly rate from the 10% annual assumption.")
    )
    frame = pd.DataFrame(
        {
            "Month": [period.month for period in forecast.periods],
            "Revenue": [period.revenue for period in forecast.periods],
            "Net cash flow": [period.net_cash_flow for period in forecast.periods],
            "Ending cash": [period.ending_cash for period in forecast.periods],
        }
    )
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Revenue": st.column_config.NumberColumn(format="$%.0f"),
            "Net cash flow": st.column_config.NumberColumn(format="$%.0f"),
            "Ending cash": st.column_config.NumberColumn(format="$%.0f"),
        },
    )


def render_credit() -> None:
    st.subheader("Credit book")
    st.caption(
        prose(
            "1,000 loans. Expected loss uses the score-band probability of default, "
            "not the probability stored on each row."
        )
    )
    loss_given_default = st.slider("Loss given default", 0.0, 100.0, 45.0, 1.0) / 100.0
    loans = loans_from_credit_book()
    summary = summarize_portfolio(loans, loss_given_default=loss_given_default)
    concentration = credit_concentration(loans)
    audit = credit_book_memo_audit()
    checked = assess_loan(
        Loan("L-001", "Guide example", 100_000.0, 620),
        loss_given_default=loss_given_default,
    )

    metrics = st.columns(4)
    metrics[0].metric("Exposure", money(summary.total_exposure, decimals=2))
    metrics[1].metric("Expected loss", money(summary.total_expected_loss, decimals=2))
    metrics[2].metric("Expected-loss ratio", percent(summary.expected_loss_ratio))
    metrics[3].metric("Effective borrowers", f"{concentration.effective_borrower_count:.1f}")
    second = st.columns(4)
    second[0].metric("Borrower HHI", f"{concentration.herfindahl_hirschman_index:.6f}")
    second[1].metric("Largest borrower", percent(concentration.largest_borrower_share))
    second[2].metric("High-risk loans", f"{summary.high_risk_count}")
    second[3].metric(
        "Guide check, score 620",
        money(checked.expected_loss, decimals=2),
        help="100,000 exposure times the 15% band PD times the selected LGD.",
    )
    st.markdown(prose(format_credit_book_decision(summary, concentration, audit)))
    show(credit_ratings_plotly(summary, concentration), key="credit-ratings")
    rows = pd.DataFrame(
        [
            {
                "Loan ID": result.loan.loan_id,
                "Exposure": result.loan.exposure,
                "Credit score": result.loan.credit_score,
                "PD": result.probability_of_default,
                "LGD": result.loss_given_default,
                "Expected loss": result.expected_loss,
                "Rating": result.risk_rating,
            }
            for result in summary.loans
        ]
    )
    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
        height=360,
        column_config={
            "Exposure": st.column_config.NumberColumn(format="$%.2f"),
            "PD": st.column_config.NumberColumn(format="%.1%"),
            "LGD": st.column_config.NumberColumn(format="%.1%"),
            "Expected loss": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


def render_portfolio() -> None:
    st.subheader("Portfolio")
    st.caption(
        prose(
            "The correlation matrix is a one-factor illustration for these six sleeves. "
            "It is not an estimate from market history. Weights that do not sum to 100% are rescaled, "
            "and the rescale is shown below."
        )
    )
    # st.data_editor needs pyarrow.arrow_table, which the browser runtime does not provide.
    baseline = portfolio_frame()
    headers = st.columns([1.6, 1, 1, 1])
    headers[0].caption("Asset")
    headers[1].caption("Expected return %")
    headers[2].caption("Volatility %")
    headers[3].caption("Weight %")
    edited_rows = []
    for position, row in enumerate(baseline.itertuples(index=False)):
        name, expected_return, volatility, weight = row
        fields = st.columns([1.6, 1, 1, 1])
        fields[0].markdown(f"**{name}**")
        expected_value = fields[1].number_input(
            f"Expected return % for {name}",
            value=float(expected_return),
            step=0.5,
            format="%.2f",
            key=f"return-{position}",
            label_visibility="collapsed",
        )
        volatility_value = fields[2].number_input(
            f"Volatility % for {name}",
            min_value=0.0,
            value=float(volatility),
            step=0.5,
            format="%.2f",
            key=f"volatility-{position}",
            label_visibility="collapsed",
        )
        weight_value = fields[3].number_input(
            f"Weight % for {name}",
            min_value=0.0,
            value=float(weight),
            step=0.5,
            format="%.2f",
            key=f"weight-{position}",
            label_visibility="collapsed",
        )
        edited_rows.append(
            {
                "Asset": str(name),
                "Expected return %": expected_value,
                "Volatility %": volatility_value,
                "Weight %": weight_value,
            }
        )
    edited = pd.DataFrame(edited_rows)
    n_portfolios = st.slider("Simulated portfolios", 1_000, 20_000, 8_000, 1_000)
    weight_sum = float(edited["Weight %"].sum())
    if abs(weight_sum - 100.0) > 0.05:
        st.warning(
            prose(f"Entered weights sum to {weight_sum:.2f}%. Analysis uses weights rescaled to 100%.")
        )
    try:
        assets = assets_from_frame(edited)
        covariance = covariance_from_correlation(
            [asset.volatility for asset in assets],
            illustrative_correlation_matrix(),
        )
        zero_correlation = portfolio_metrics(assets)
        aware = correlated_portfolio_metrics(assets, covariance)
        simulation = simulate_long_only_portfolios(
            assets,
            covariance,
            n_portfolios=int(n_portfolios),
            random_state=42,
        )
    except (TypeError, ValueError) as exc:
        st.error(str(exc))
        return

    max_sharpe = max_sharpe_portfolio(simulation)
    min_vol = min_volatility_portfolio(simulation)
    frontier = approximate_efficient_frontier(simulation, max_points=20)
    metrics = st.columns(4)
    metrics[0].metric("Expected return", percent(aware.expected_return))
    metrics[1].metric("Zero-correlation volatility", percent(zero_correlation.volatility))
    metrics[2].metric("Covariance-aware volatility", percent(aware.volatility))
    metrics[3].metric("Sharpe ratio", f"{aware.sharpe_ratio:.2f}")
    left, right = st.columns(2)
    with left:
        show(allocation_plotly(assets), key="allocation")
    with right:
        show(
            frontier_plotly(
                simulation.volatilities,
                simulation.expected_returns,
                simulation.sharpe_ratios,
                [point.volatility for point in frontier],
                [point.expected_return for point in frontier],
                aware.volatility,
                aware.expected_return,
            ),
            key="frontier",
        )
    best = pd.DataFrame(
        [
            {
                "Portfolio": "Workbook mix",
                "Expected return": aware.expected_return,
                "Volatility": aware.volatility,
                "Sharpe": aware.sharpe_ratio,
            },
            {
                "Portfolio": "Highest sampled Sharpe",
                "Expected return": max_sharpe.expected_return,
                "Volatility": max_sharpe.volatility,
                "Sharpe": max_sharpe.sharpe_ratio,
            },
            {
                "Portfolio": "Lowest sampled volatility",
                "Expected return": min_vol.expected_return,
                "Volatility": min_vol.volatility,
                "Sharpe": min_vol.sharpe_ratio,
            },
        ]
    )
    st.dataframe(
        best,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Expected return": st.column_config.NumberColumn(format="%.2%"),
            "Volatility": st.column_config.NumberColumn(format="%.2%"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_stress() -> None:
    st.subheader("Stress and Monte Carlo")
    left, right = st.columns(2)
    with left:
        st.markdown(prose("**Liquidity**"))
        revenue_multiplier = st.slider("Revenue level", 50.0, 120.0, 80.0, 5.0) / 100.0
        growth_delta = st.slider("Monthly growth shock", -10.0, 5.0, -3.0, 0.5) / 100.0
        opex_delta = st.slider("Operating-expense shock", -10.0, 30.0, 10.0, 1.0) / 100.0
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
        except ValueError as exc:
            st.error(str(exc))
        else:
            impact = st.columns(2)
            impact[0].metric("Ending cash impact", money(cash_stress.ending_cash_change))
            impact[1].metric("NPV impact", money(cash_stress.npv_change))
            show(
                stressed_cash_plotly(cash_stress.baseline, cash_stress.stressed, "Stressed"),
                key="stress-cash",
            )

        st.markdown(prose("**Credit**"))
        st.caption(
            prose("Probability of default and loss given default are scaled on the 1,000-loan book and capped at 100%.")
        )
        pd_multiplier = st.slider("PD multiplier", 0.5, 4.0, 1.75, 0.25)
        lgd_multiplier = st.slider("LGD multiplier", 0.5, 2.5, 1.25, 0.25)
        credit_stress = stress_credit_portfolio(
            loans_from_credit_book(),
            CreditStressScenario(
                "Dashboard credit stress",
                pd_multiplier=pd_multiplier,
                lgd_multiplier=lgd_multiplier,
            ),
        )
        st.metric(
            "Stressed expected loss",
            money(credit_stress.stressed_expected_loss, decimals=0),
            delta=money(credit_stress.expected_loss_change, decimals=0),
            delta_color="inverse",
        )

    with right:
        st.markdown(prose("**Terminal wealth**"))
        horizon = st.slider("Horizon (years)", 1, 30, 10)
        simulations = st.slider("Monte Carlo paths", 1_000, 20_000, 8_000, 1_000)
        assets = workbook_balanced_portfolio()
        covariance = covariance_from_correlation(
            [asset.volatility for asset in assets],
            illustrative_correlation_matrix(),
        )
        result = simulate_portfolio_terminal_values(
            assets,
            covariance,
            horizon_years=int(horizon),
            n_simulations=int(simulations),
            random_state=42,
        )
        terminal = st.columns(2)
        terminal[0].metric("Median terminal value", money(result.median_terminal_value))
        terminal[1].metric("Probability of loss", percent(result.probability_of_loss))
        tails = st.columns(2)
        tails[0].metric("5th percentile", money(result.percentile_05))
        tails[1].metric("95th percentile", money(result.percentile_95))
        show(monte_carlo_plotly(result), key="monte-carlo")
        st.caption(prose("Loss means terminal value below the $1,000,000 starting investment."))


def render_market() -> None:
    st.subheader("Market risk")
    st.caption(
        prose(
            "Upload adjusted closes, run the worked example, or download prices. "
            "The worked example is a constructed price path, not a market history. "
            "Rows with a missing price are dropped."
        )
    )
    source = st.radio(
        "Price source",
        ["Worked example", "Upload CSV", "Download adjusted closes"],
        horizontal=True,
    )
    prices = None
    if source == "Worked example":
        prices = worked_example_prices()
    elif source == "Upload CSV":
        upload = st.file_uploader("CSV with a date column and one price column per asset", type="csv")
        if upload is not None:
            try:
                prices = align_prices(prices_from_upload(upload))
            except (TypeError, ValueError) as exc:
                st.error(str(exc))
                return
    else:
        st.caption(
            prose(
                "This download needs the optional yfinance package. "
                "The GitHub Pages demo runs in the browser and does not include it."
            )
        )
        tickers = st.text_input("Tickers", "SPY QQQ TLT GLD")
        start = st.text_input("Start date", "2020-01-01")
        if st.button("Download"):
            try:
                prices = download_adjusted_close(tickers.split(), start=start)
                st.session_state["downloaded_prices"] = prices
            except (ImportError, TypeError, ValueError) as exc:
                st.error(str(exc))
                return
        prices = st.session_state.get("downloaded_prices")

    if prices is None:
        st.info(
            prose("Choose a price source to calculate returns, drawdown, VaR, and a rebalanced backtest.")
        )
        return

    try:
        returns = simple_returns(prices)
    except (TypeError, ValueError) as exc:
        st.error(str(exc))
        return

    columns = list(prices.columns)
    share = round(1.0 / len(columns), 2)
    default_weights = [share] * (len(columns) - 1)
    default_weights.append(round(1.0 - share * (len(columns) - 1), 2))
    weight_inputs = st.columns(len(columns))
    weights = []
    for column, weight_input, default in zip(columns, weight_inputs, default_weights, strict=True):
        weights.append(
            weight_input.number_input(
                str(column),
                min_value=0.0,
                max_value=1.0,
                value=float(default),
                step=0.05,
                format="%.2f",
            )
        )
    weight_total = float(sum(weights))
    if abs(weight_total - 1.0) > 1e-9:
        st.warning(prose(f"Weights sum to {weight_total:.2f}. Rescale them to 1.00 to run the backtest."))
        return

    sample_periods = len(returns)
    periods_per_year = int(
        st.number_input(
            "Periods per year",
            min_value=1,
            value=252,
            step=1,
            key=f"periods-{source}-{sample_periods}",
            help=(
                "Divides the 3% risk-free rate into one period. "
                "Samples of 60 or more returns are also annualized with this factor."
            ),
        )
    )
    annualize = annualize_sample(sample_periods)
    if not annualize:
        st.caption(
            prose(
                f"This sample has {sample_periods} returns, so return, volatility, and Sharpe "
                "are not annualized. "
                f"The 3% risk-free rate is charged as 3% × {sample_periods} / {periods_per_year} "
                "over the window."
            )
        )
    portfolio_returns = portfolio_return_series(returns, weights)
    try:
        summary = historical_risk_summary(
            prices,
            periods_per_year=periods_per_year if annualize else 1,
        )
        performance = performance_summary(
            portfolio_returns,
            periods_per_year=periods_per_year,
            risk_free_rate=0.03,
            annualize=annualize,
        )
    except (TypeError, ValueError) as exc:
        st.error(str(exc))
        return
    backtest = backtest_rebalanced_portfolio(
        prices,
        weights,
        rebalance_every=max(1, len(returns) // 2),
        transaction_cost_bps=5.0,
    )
    volatility_label = "Annualized volatility" if annualize else "Period volatility"
    sharpe_label = "Sharpe" if annualize else "Sharpe (not annualized)"
    sortino_label = "Sortino" if annualize else "Sortino (not annualized)"
    show_sortino = performance.sortino_ratio is not None
    show_tail_risk = performance.value_at_risk is not None
    metrics = st.columns(4)
    metrics[0].metric("Cumulative return", percent(performance.cumulative_return))
    metrics[1].metric("Max drawdown", percent(performance.max_drawdown))
    if show_tail_risk:
        metrics[2].metric("95% historical VaR", percent(performance.value_at_risk))
    else:
        metrics[2].metric("95% historical VaR", "n/a", help=performance.tail_risk_reason)
    metrics[3].metric(volatility_label, percent(performance.annualized_volatility))
    more = st.columns(4)
    more[0].metric(sharpe_label, f"{performance.sharpe_ratio:.2f}")
    if show_sortino:
        more[1].metric(sortino_label, f"{performance.sortino_ratio:.2f}")
    else:
        more[1].metric("Sortino", "n/a", help=performance.sortino_reason)
    if show_tail_risk:
        more[2].metric("95% expected shortfall", percent(performance.expected_shortfall))
    else:
        more[2].metric("95% expected shortfall", "n/a", help=performance.tail_risk_reason)
    more[3].metric("Backtest ending wealth", f"{backtest.wealth.iloc[-1]:.4f}")

    wealth = {"Portfolio": cumulative_wealth(portfolio_returns)}
    for column in columns:
        wealth[str(column)] = cumulative_wealth(returns[column])
    show(
        wealth_plotly(list(portfolio_returns.index), wealth, title="Growth of $1"),
        key="market-wealth",
    )
    return_column = "Annualized return" if annualize else "Mean period return"
    volatility_column = "Annualized volatility" if annualize else "Period volatility"
    risk = pd.DataFrame(
        {
            "Asset": summary.annualized_volatility.index.astype(str),
            return_column: summary.annualized_returns.to_numpy(),
            volatility_column: summary.annualized_volatility.to_numpy(),
            "Max drawdown": summary.max_drawdown.to_numpy(),
        }
    )
    st.dataframe(
        risk,
        hide_index=True,
        use_container_width=True,
        column_config={
            return_column: st.column_config.NumberColumn(format="%.2%"),
            volatility_column: st.column_config.NumberColumn(format="%.2%"),
            "Max drawdown": st.column_config.NumberColumn(format="%.2%"),
        },
    )
    if show_tail_risk:
        tail_note = (
            "VaR and expected shortfall are per-period loss fractions from the worst "
            "ceil((1 − 95%) × n) returns, each with full weight. "
            "A negative loss means that point was a gain. "
        )
    else:
        tail_note = (
            "95% VaR and expected shortfall are not shown: "
            f"{performance.tail_risk_reason}. "
        )
    if show_sortino:
        ratio_note = (
            "Sharpe and Sortino are annualized with a 3% risk-free rate divided across periods per year. "
            if annualize
            else "Sharpe and Sortino use that scaled risk-free rate and are not annualized. "
        )
    else:
        ratio_note = (
            "Sharpe is annualized with a 3% risk-free rate divided across periods per year. "
            if annualize
            else "Sharpe uses that scaled risk-free rate and is not annualized. "
        )
        ratio_note += f"Sortino is not shown: {performance.sortino_reason}. "
    st.caption(prose(tail_note + ratio_note + "The backtest charges 5 bps of one-way turnover."))


render_header()
overview_tab, cash_tab, credit_tab, portfolio_tab, stress_tab, market_tab = st.tabs(
    ["Overview", "Cash flow", "Credit book", "Portfolio", "Stress", "Market risk"]
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
with market_tab:
    render_market()

st.divider()
st.caption(
    prose("Educational portfolio project. Not investment, lending, accounting, or financial advice.")
)
