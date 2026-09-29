"""Chart builders for the workbook models.

Every figure uses :mod:`financial_models.theme`. Callers pass results that
were already calculated by the model functions; these builders do not invent
summary statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .cash_flow import CashFlowForecast
from .credit_concentration import CreditConcentrationSummary
from .credit_risk import PortfolioCreditSummary
from .portfolio import AssetAllocation, PortfolioMetrics
from .stress_testing import PortfolioMonteCarloResult
from .theme import (
    AMBER,
    BG,
    EMERALD,
    GRID,
    INK,
    MUTED,
    RATING_COLORS,
    ROSE,
    SERIES,
    SKY,
    WHITE,
    apply_matplotlib_theme,
    plotly_layout,
)

RATING_ORDER = ("Low", "Medium", "High")


def save_figure(figure: Figure, path: str | Path) -> Path:
    """Write a themed matplotlib figure and close it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.clf()
    import matplotlib.pyplot as plt

    plt.close(figure)
    return destination


def _figure(width: float = 11.0, height: float = 6.2) -> tuple[Figure, Axes]:
    apply_matplotlib_theme()
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(width, height))
    axis.set_facecolor(axis.get_facecolor())
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    return figure, axis


def _title(axis: Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", pad=14)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def _note(figure: Figure, text: str) -> None:
    figure.text(0.01, 0.01, text, color=MUTED, fontsize=8, fontfamily="monospace")
    figure.tight_layout(rect=(0, 0.03, 1, 1))


def _currency_axis(axis: Axes) -> None:
    from matplotlib.ticker import FuncFormatter

    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.0f}"))


def _percent_axis(axis: Axes, *, horizontal: bool = False) -> None:
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(lambda value, _pos: f"{value * 100:.1f}%")
    target = axis.xaxis if horizontal else axis.yaxis
    target.set_major_formatter(formatter)


def cash_flow_matplotlib(forecast: CashFlowForecast) -> Figure:
    """Plot monthly net cash flow and the ending cash balance."""
    figure, axis = _figure()
    months = [period.month for period in forecast.periods]
    net = [period.net_cash_flow for period in forecast.periods]
    ending = [period.ending_cash for period in forecast.periods]
    bar_colors = [EMERALD if value >= 0 else ROSE for value in net]
    axis.bar(months, net, color=bar_colors, width=0.62, label="Net cash flow", zorder=2)
    axis.plot(
        months,
        ending,
        color=WHITE,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Ending cash",
        zorder=3,
    )
    axis.set_xticks(months)
    _title(axis, "Cash balance over the forecast", "Month", "US dollars")
    _currency_axis(axis)
    axis.legend(loc="upper left")
    _note(figure, "Workbook assumptions: 5% monthly growth, 60% operating expenses, 10% annual discount rate")
    return figure


def _rating_amounts(
    summary: PortfolioCreditSummary,
    concentration: CreditConcentrationSummary,
) -> tuple[list[str], list[float], list[float]]:
    exposure = {group.name: group.exposure for group in concentration.risk_rating_exposures}
    expected_loss = {name: 0.0 for name in RATING_ORDER}
    for result in summary.loans:
        expected_loss[result.risk_rating] = (
            expected_loss.get(result.risk_rating, 0.0) + result.expected_loss
        )
    labels = [name for name in RATING_ORDER if exposure.get(name, 0.0) or expected_loss[name]]
    return (
        labels,
        [exposure.get(name, 0.0) for name in labels],
        [expected_loss[name] for name in labels],
    )


def credit_ratings_matplotlib(
    summary: PortfolioCreditSummary,
    concentration: CreditConcentrationSummary,
) -> Figure:
    """Plot exposure and expected loss by rating on separate scales."""
    apply_matplotlib_theme()
    import matplotlib.pyplot as plt

    labels, exposure, expected_loss = _rating_amounts(summary, concentration)
    colors = [RATING_COLORS[name] for name in labels]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.2))
    axes[0].bar(labels, exposure, color=colors)
    axes[1].bar(labels, expected_loss, color=colors)
    _title(axes[0], "Exposure by rating", "Risk rating", "US dollars")
    _title(axes[1], "Expected loss by rating", "Risk rating", "US dollars")
    for axis in axes:
        _currency_axis(axis)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.spines["left"].set_color(GRID)
        axis.spines["bottom"].set_color(GRID)
    hhi = concentration.herfindahl_hirschman_index
    _note(
        figure,
        f"Expected loss = exposure x score-band PD x 45% LGD. Borrower HHI {hhi:.6f}. "
        f"Effective borrowers {concentration.effective_borrower_count:.1f}.",
    )
    return figure


def allocation_matplotlib(
    assets: Sequence[AssetAllocation],
    zero_correlation: PortfolioMetrics,
    covariance_aware: PortfolioMetrics,
) -> Figure:
    """Plot the workbook weights beside the two volatility calculations."""
    apply_matplotlib_theme()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    figure.set_facecolor(figure.get_facecolor())
    names = [asset.name.replace(" Stocks", "") for asset in assets]
    weights = [asset.weight for asset in assets]
    axes[0].barh(names[::-1], weights[::-1], color=SERIES[: len(assets)][::-1])
    axes[0].set_xlim(0, max(weights) * 1.25)
    _title(axes[0], "Baseline allocation", "Portfolio weight", "")
    _percent_axis(axes[0], horizontal=True)
    axes[0].xaxis.grid(True, color=GRID, linewidth=0.6)
    axes[0].yaxis.grid(False)

    labels = ["Expected return", "Zero-correlation vol", "Covariance-aware vol"]
    values = [
        zero_correlation.expected_return,
        zero_correlation.volatility,
        covariance_aware.volatility,
    ]
    colors = [EMERALD, SKY, AMBER]
    axes[1].barh(labels[::-1], values[::-1], color=colors[::-1])
    _title(axes[1], "Risk and return", "Annual rate", "")
    _percent_axis(axes[1], horizontal=True)
    axes[1].xaxis.grid(True, color=GRID, linewidth=0.6)
    axes[1].yaxis.grid(False)
    for axis in axes:
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.spines["left"].set_color(GRID)
        axis.spines["bottom"].set_color(GRID)
    sharpe = covariance_aware.sharpe_ratio
    _note(
        figure,
        "Zero-correlation vol is sqrt of the sum of squared weighted volatilities. "
        f"Covariance-aware Sharpe {sharpe:.2f} at a 3% risk-free rate.",
    )
    return figure


def frontier_matplotlib(
    volatilities: np.ndarray,
    expected_returns: np.ndarray,
    sharpe_ratios: np.ndarray,
    frontier_volatility: Sequence[float],
    frontier_return: Sequence[float],
    baseline_volatility: float,
    baseline_return: float,
) -> Figure:
    """Plot simulated long-only portfolios and the sampled efficient frontier."""
    figure, axis = _figure()
    finite = np.isfinite(sharpe_ratios)
    scatter = axis.scatter(
        volatilities[finite],
        expected_returns[finite],
        c=sharpe_ratios[finite],
        s=8,
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
        zorder=2,
    )
    axis.plot(
        frontier_volatility,
        frontier_return,
        color=WHITE,
        linewidth=2.0,
        label="Sampled frontier",
        zorder=3,
    )
    axis.scatter(
        [baseline_volatility],
        [baseline_return],
        s=70,
        color=EMERALD,
        edgecolor=BG,
        linewidth=0.8,
        label="Workbook mix",
        zorder=4,
    )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Sharpe ratio", color=MUTED)
    colorbar.ax.yaxis.set_tick_params(color=MUTED)
    plt_set_colorbar_tick_color(colorbar)
    _title(axis, "Long-only portfolios and sampled frontier", "Annual volatility", "Expected annual return")
    _percent_axis(axis)
    _percent_axis(axis, horizontal=True)
    axis.legend(loc="upper left")
    _note(figure, "Illustrative one-factor correlation. Frontier is selected from the sample, not solved analytically.")
    return figure


def plt_set_colorbar_tick_color(colorbar: object) -> None:
    colorbar.ax.tick_params(labelcolor=MUTED)  # type: ignore[attr-defined]


def monte_carlo_matplotlib(result: PortfolioMonteCarloResult) -> Figure:
    """Plot the terminal-wealth distribution and its reported percentiles."""
    figure, axis = _figure()
    axis.hist(
        result.terminal_values,
        bins=40,
        color=EMERALD,
        alpha=0.88,
        edgecolor=BG,
        linewidth=0.4,
    )
    markers = (
        (result.percentile_05, "5th", ROSE),
        (result.median_terminal_value, "Median", WHITE),
        (result.percentile_95, "95th", SKY),
    )
    for value, label, color in markers:
        axis.axvline(value, color=color, linewidth=1.4, linestyle="--", label=f"{label} ${value:,.0f}")
    _title(
        axis,
        f"{result.horizon_years}-year terminal value",
        "Terminal portfolio value (US dollars)",
        "Simulated paths",
    )
    _currency_axis_x(axis)
    axis.legend(loc="upper right")
    _note(
        figure,
        "Lognormal yearly returns matched to the covariance-aware mean and volatility. "
        f"Probability of ending below the initial investment: {result.probability_of_loss:.2%}.",
    )
    return figure


def _currency_axis_x(axis: Axes) -> None:
    from matplotlib.ticker import FuncFormatter

    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.0f}"))


def drawdown_matplotlib(index: Sequence[object], drawdown: Sequence[float], name: str) -> Figure:
    """Plot a drawdown path. ``drawdown`` must already be computed."""
    figure, axis = _figure(height=5.6)
    axis.fill_between(list(index), list(drawdown), 0.0, color=ROSE, alpha=0.35)
    axis.plot(list(index), list(drawdown), color=ROSE, linewidth=1.6)
    _title(axis, f"Drawdown: {name}", "Date", "Decline from peak")
    _percent_axis(axis)
    _note(figure, "Peak-to-trough decline of the price or wealth index. Zero is a new high.")
    return figure


def _plotly():
    import plotly.graph_objects as go

    return go


def cash_flow_plotly(forecast: CashFlowForecast):
    """Interactive cash-flow chart for the dashboard."""
    go = _plotly()
    months = [period.month for period in forecast.periods]
    net = [period.net_cash_flow for period in forecast.periods]
    ending = [period.ending_cash for period in forecast.periods]
    figure = go.Figure()
    figure.add_bar(
        x=months,
        y=net,
        name="Net cash flow",
        marker_color=[EMERALD if value >= 0 else ROSE for value in net],
        hovertemplate="Month %{x}<br>Net cash flow $%{y:,.0f}<extra></extra>",
    )
    figure.add_scatter(
        x=months,
        y=ending,
        name="Ending cash",
        mode="lines+markers",
        line={"color": WHITE, "width": 2.4},
        marker={"size": 7, "color": WHITE},
        hovertemplate="Month %{x}<br>Ending cash $%{y:,.0f}<extra></extra>",
    )
    figure.update_layout(
        **plotly_layout(
            "Cash balance over the forecast",
            x_title="Month",
            y_title="US dollars",
        )
    )
    return figure


def stressed_cash_plotly(baseline: CashFlowForecast, stressed: CashFlowForecast, scenario_name: str):
    """Compare baseline and stressed ending cash."""
    go = _plotly()
    months = [period.month for period in baseline.periods]
    figure = go.Figure()
    figure.add_scatter(
        x=months,
        y=[period.ending_cash for period in baseline.periods],
        name="Baseline",
        mode="lines",
        line={"color": EMERALD, "width": 2.4},
        hovertemplate="Month %{x}<br>Baseline $%{y:,.0f}<extra></extra>",
    )
    figure.add_scatter(
        x=months,
        y=[period.ending_cash for period in stressed.periods],
        name=scenario_name,
        mode="lines",
        line={"color": ROSE, "width": 2.4},
        hovertemplate="Month %{x}<br>Stressed $%{y:,.0f}<extra></extra>",
    )
    figure.update_layout(
        **plotly_layout("Ending cash under stress", x_title="Month", y_title="US dollars")
    )
    return figure


def credit_ratings_plotly(
    summary: PortfolioCreditSummary,
    concentration: CreditConcentrationSummary,
):
    """Interactive exposure and expected-loss charts on separate scales."""
    from plotly.subplots import make_subplots

    labels, exposure, expected_loss = _rating_amounts(summary, concentration)
    colors = [RATING_COLORS[name] for name in labels]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Exposure", "Expected loss"),
    )
    figure.add_bar(
        x=labels,
        y=exposure,
        marker_color=colors,
        name="Exposure",
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        row=1,
        col=1,
    )
    figure.add_bar(
        x=labels,
        y=expected_loss,
        marker_color=colors,
        name="Expected loss",
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        row=1,
        col=2,
    )
    layout = plotly_layout(
        "Credit exposure and expected loss by rating",
        x_title="Risk rating",
        y_title="US dollars",
    )
    figure.update_layout(**layout, showlegend=False)
    figure.update_yaxes(title_text="US dollars")
    figure.update_annotations(font={"color": INK, "family": "Inter, sans-serif"})
    return figure


def frontier_plotly(
    volatilities: np.ndarray,
    expected_returns: np.ndarray,
    sharpe_ratios: np.ndarray,
    frontier_volatility: Sequence[float],
    frontier_return: Sequence[float],
    baseline_volatility: float,
    baseline_return: float,
):
    """Interactive efficient-frontier scatter."""
    go = _plotly()
    finite = np.isfinite(sharpe_ratios)
    figure = go.Figure()
    figure.add_scatter(
        x=volatilities[finite],
        y=expected_returns[finite],
        mode="markers",
        name="Simulated mix",
        marker={
            "size": 6,
            "color": sharpe_ratios[finite],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": "Sharpe", "tickfont": {"color": MUTED}},
        },
        hovertemplate="Vol %{x:.2%}<br>Return %{y:.2%}<extra></extra>",
    )
    figure.add_scatter(
        x=list(frontier_volatility),
        y=list(frontier_return),
        mode="lines",
        name="Sampled frontier",
        line={"color": WHITE, "width": 2.4},
    )
    figure.add_scatter(
        x=[baseline_volatility],
        y=[baseline_return],
        mode="markers",
        name="Workbook mix",
        marker={"size": 14, "color": EMERALD, "line": {"color": INK, "width": 1}},
    )
    figure.update_layout(
        **plotly_layout(
            "Long-only portfolios and sampled frontier",
            x_title="Annual volatility",
            y_title="Expected annual return",
        )
    )
    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")
    return figure


def monte_carlo_plotly(result: PortfolioMonteCarloResult):
    """Interactive terminal-value histogram."""
    go = _plotly()
    figure = go.Figure()
    figure.add_histogram(
        x=result.terminal_values,
        name="Paths",
        marker_color=EMERALD,
        nbinsx=40,
        hovertemplate="Value $%{x:,.0f}<br>Paths %{y}<extra></extra>",
    )
    for value, label, color in (
        (result.percentile_05, "5th percentile", ROSE),
        (result.median_terminal_value, "Median", WHITE),
        (result.percentile_95, "95th percentile", SKY),
    ):
        figure.add_vline(
            x=value,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_font_color=color,
        )
    figure.update_layout(
        **plotly_layout(
            f"{result.horizon_years}-year terminal value",
            x_title="Terminal portfolio value (US dollars)",
            y_title="Simulated paths",
        )
    )
    return figure


def allocation_plotly(assets: Sequence[AssetAllocation]):
    """Interactive weight chart."""
    go = _plotly()
    figure = go.Figure()
    figure.add_bar(
        y=[asset.name for asset in assets][::-1],
        x=[asset.weight for asset in assets][::-1],
        orientation="h",
        marker_color=list(SERIES[: len(assets)])[::-1],
        hovertemplate="%{y}<br>Weight %{x:.1%}<extra></extra>",
    )
    figure.update_layout(
        **plotly_layout("Baseline allocation", x_title="Portfolio weight", y_title="")
    )
    figure.update_xaxes(tickformat=".0%")
    return figure


def wealth_plotly(
    dates: Sequence[object],
    series: dict[str, Sequence[float]],
    *,
    title: str,
):
    """Plot one or more wealth indexes that the caller has already compounded."""
    go = _plotly()
    figure = go.Figure()
    for color, (name, values) in zip(SERIES, series.items(), strict=False):
        figure.add_scatter(
            x=list(dates),
            y=list(values),
            name=name,
            mode="lines",
            line={"color": color, "width": 2.2},
            hovertemplate="%{x}<br>" + name + " %{y:.3f}<extra></extra>",
        )
    figure.update_layout(**plotly_layout(title, x_title="Date", y_title="Growth of $1"))
    return figure
