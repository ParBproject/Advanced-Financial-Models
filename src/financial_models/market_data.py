"""Historical return, drawdown, VaR, and rebalanced-backtest analytics.

Price history is never forward-filled. A missing print would otherwise become
a zero return and understate both volatility and drawdown. Incomplete rows are
dropped only when several assets are aligned, and that choice is explicit.
Rows must already be in time order. A reversed file is rejected rather than
silently turned into returns between the wrong neighbors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .validation import require_finite


@dataclass(frozen=True)
class MarketRiskSummary:
    """Historical market-risk statistics estimated from observed prices."""

    annualized_returns: pd.Series
    annualized_covariance: pd.DataFrame
    annualized_volatility: pd.Series
    correlation: pd.DataFrame
    max_drawdown: pd.Series


@dataclass(frozen=True)
class PerformanceSummary:
    """Risk-adjusted performance statistics for one return series.

    ``value_at_risk`` and ``expected_shortfall`` are per-period loss fractions,
    not annualized figures. Positive means a loss. A negative value means that
    point in the distribution was a gain. Both are ``None`` when the sample
    has fewer than 20 returns; ``tail_risk_reason`` then says why.
    """

    cumulative_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float | None
    max_drawdown: float
    value_at_risk: float | None
    expected_shortfall: float | None
    sortino_reason: str | None = None
    tail_risk_reason: str | None = None


@dataclass(frozen=True)
class RebalancedBacktestResult:
    """Historical fixed-allocation backtest with explicit rebalancing costs."""

    returns: pd.Series
    wealth: pd.Series
    turnover: pd.Series
    transaction_costs: pd.Series

    @property
    def total_turnover(self) -> float:
        return float(self.turnover.sum())

    @property
    def total_transaction_cost(self) -> float:
        return float(self.transaction_costs.sum())


def worked_example_prices() -> pd.DataFrame:
    """Return a short constructed price path for tests and the dashboard.

    The path is not a market history and must not be labeled as one.
    """
    return pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 101.0, 104.0, 106.0],
            "BBB": [50.0, 49.0, 51.0, 52.0, 51.5],
        },
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
    )


def _validate_periods_per_year(periods_per_year: int) -> int:
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int):
        raise TypeError("periods_per_year must be a positive integer")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be a positive integer")
    return periods_per_year


def _price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a DataFrame")
    if prices.empty or prices.shape[1] < 1:
        raise ValueError("prices must contain at least one asset")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any(axis=None):
        raise ValueError("prices must be numeric and must not contain missing values")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("prices must contain only finite values")
    if (numeric <= 0).any(axis=None):
        raise ValueError("prices must be strictly positive")
    if len(numeric) < 2:
        raise ValueError("prices must contain at least two observations")
    if not numeric.index.is_monotonic_increasing:
        raise ValueError("prices must be ordered by time")
    if numeric.index.has_duplicates:
        raise ValueError("prices must not contain duplicate timestamps")
    return numeric


def _return_array(returns: pd.Series | Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if values.size == 0:
        raise ValueError("returns must contain at least one observation")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite observations")
    if np.any(values < -1.0):
        raise ValueError("returns cannot be less than -100%")
    return values


def _sortino_ratio(
    excess: float,
    downside_deviation: float,
    *,
    n_observations: int,
    downside_count: int,
) -> tuple[float | None, str | None]:
    """Return Sortino, or ``None`` and a short reason when it is not estimated.

    The sample must have at least ``TAIL_RISK_MIN_OBSERVATIONS`` returns and
    at least ``SORTINO_MIN_DOWNSIDE`` observations strictly below the
    risk-free threshold. A deviation built from one or two shortfalls is not
    reported.
    """
    if n_observations < TAIL_RISK_MIN_OBSERVATIONS:
        return None, (
            f"needs at least {TAIL_RISK_MIN_OBSERVATIONS} returns "
            f"(this sample has {n_observations})"
        )
    if downside_count < SORTINO_MIN_DOWNSIDE:
        return None, (
            f"needs at least {SORTINO_MIN_DOWNSIDE} returns below the risk-free rate "
            f"(this sample has {downside_count})"
        )
    return _ratio(excess, downside_deviation), None


def _ratio(excess: float, risk: float) -> float:
    if risk > 1e-15:
        return excess / risk
    if excess > 0.0:
        return float("inf")
    if excess < 0.0:
        return float("-inf")
    return 0.0


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple periodic returns from an aligned price matrix."""
    clean = _price_frame(prices)
    returns = clean.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("prices do not produce a valid return series")
    return returns


def align_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop dates that are missing a price for any asset.

    Rows are removed rather than filled. Forward-filling a missing close
    would invent a zero-return day.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a DataFrame")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(how="any")
    return _price_frame(numeric)


def historical_risk_summary(
    prices: pd.DataFrame,
    *,
    periods_per_year: int = 252,
) -> MarketRiskSummary:
    """Estimate annualized return, covariance, volatility, correlation, and drawdown.

    Annualized covariance is the sample covariance (divisor ``n - 1``) times
    ``periods_per_year``. That matches ``std(ddof=1) * sqrt(periods_per_year)``
    on the diagonal. Drawdown is the peak-to-trough decline of each price.
    """
    periods_per_year = _validate_periods_per_year(periods_per_year)
    clean = _price_frame(prices)
    returns = simple_returns(clean)
    annualized_returns = returns.mean() * periods_per_year
    annualized_covariance = returns.cov() * periods_per_year
    annualized_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    correlation = returns.corr()
    if not np.isfinite(correlation.to_numpy(dtype=float)).all():
        raise ValueError("correlation is undefined when an asset has zero variance")
    drawdowns = clean / clean.cummax() - 1.0
    return MarketRiskSummary(
        annualized_returns=annualized_returns,
        annualized_covariance=annualized_covariance,
        annualized_volatility=annualized_volatility,
        correlation=correlation,
        max_drawdown=drawdowns.min(),
    )


def portfolio_return_series(
    returns: pd.DataFrame,
    weights: Sequence[float],
) -> pd.Series:
    """Calculate a fully invested portfolio return series from asset returns."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1 or len(weight_array) != returns.shape[1]:
        raise ValueError("weights must match the number of return columns")
    if not np.isfinite(weight_array).all():
        raise ValueError("weights must contain only finite values")
    if abs(float(weight_array.sum()) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.0")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite observations")
    return pd.Series(
        values @ weight_array,
        index=returns.index,
        name="portfolio_return",
    )


ANNUALIZE_MIN_OBSERVATIONS = 60
TAIL_RISK_MIN_OBSERVATIONS = 20
SORTINO_MIN_DOWNSIDE = 3


def annualize_sample(n_observations: int) -> bool:
    """Return whether a return sample is long enough to annualize.

    Fewer than 60 observations stay in period units. Annualizing a four-day
    path at 252 periods per year turns a small drift into a triple-digit rate.
    """
    if isinstance(n_observations, bool) or not isinstance(n_observations, int):
        raise TypeError("observation count must be a positive integer")
    if n_observations < 1:
        raise ValueError("observation count must be a positive integer")
    return n_observations >= ANNUALIZE_MIN_OBSERVATIONS


def historical_var_expected_shortfall(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return historical VaR and Expected Shortfall as loss fractions.

    The tail holds the worst ``k`` observations, where
    ``k = ceil((1 - confidence) * n)`` and ``k`` is at least 1. VaR is the
    loss on the least severe observation in that tail. Expected Shortfall is
    the equal-weighted average of those ``k`` losses. The boundary observation
    (the least severe loss in the tail) gets the same full weight as the worse
    ones. When ``(1 - confidence) * n`` is not an integer, the count is rounded
    up with ``ceil`` rather than giving that boundary observation a fractional
    weight. Neither figure is clipped at zero, so a gain at the quantile stays
    a negative loss.
    """
    confidence = require_finite("confidence", confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    ordered = np.sort(_return_array(returns))
    tail_count = max(1, int(np.ceil((1.0 - confidence) * ordered.size)))
    tail = ordered[:tail_count]
    value_at_risk = float(-tail[-1])
    expected_shortfall = float(-tail.mean())
    return value_at_risk, expected_shortfall


def max_drawdown(returns: pd.Series | Sequence[float] | np.ndarray) -> float:
    """Return the most severe peak-to-trough decline of compounded wealth.

    Wealth starts at 1. The result is zero or negative.
    """
    wealth = np.cumprod(1.0 + _return_array(returns))
    wealth = np.concatenate(([1.0], wealth))
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def performance_summary(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.03,
    confidence: float = 0.95,
    annualize: bool = True,
) -> PerformanceSummary:
    """Summarize growth, downside risk, and risk-adjusted performance.

    Sharpe and Sortino both use the same minimum acceptable return: the
    risk-free rate spread evenly across periods (``risk_free_rate /
    periods_per_year``). Downside deviation is the square root of the mean
    squared shortfall below that threshold, including zeros for observations
    that clear it.

    When ``annualize`` is true, excess return is multiplied by
    ``periods_per_year`` and volatility by its square root. When it is false,
    those scales stay at 1, so volatility and the ratios are per period.
    ``cagr`` is then the cumulative return, not an annual rate. The risk-free
    rate is still divided by ``periods_per_year``, so a four-day sample with
    252 periods per year is charged about four days of interest, not a full
    year.

    ``sortino_ratio`` is ``None`` when the sample has fewer than 20 returns,
    or fewer than 3 returns fall below the per-period risk-free rate.
    ``sortino_reason`` then says which gate failed. One or two shortfalls are
    not enough to estimate downside deviation.

    ``value_at_risk`` and ``expected_shortfall`` are ``None`` below 20
    returns, with the reason in ``tail_risk_reason``. The lower-level
    ``historical_var_expected_shortfall`` still evaluates a short sample when
    called directly.
    """
    periods_per_year = _validate_periods_per_year(periods_per_year)
    if not isinstance(annualize, bool):
        raise TypeError("annualize must be true or false")
    risk_free_rate = require_finite("risk-free rate", risk_free_rate)
    values = _return_array(returns)
    return_scale = float(periods_per_year if annualize else 1)
    volatility_scale = float(np.sqrt(periods_per_year) if annualize else 1.0)

    wealth = np.cumprod(1.0 + values)
    final_wealth = float(wealth[-1])
    cumulative_return = final_wealth - 1.0
    if not annualize:
        cagr = cumulative_return
    elif final_wealth <= 0.0:
        cagr = -1.0
    else:
        cagr = float(final_wealth ** (periods_per_year / values.size) - 1.0)

    threshold = risk_free_rate / periods_per_year
    excess = values - threshold
    annualized_excess = float(excess.mean() * return_scale)
    if values.size > 1:
        annualized_volatility = float(np.std(values, ddof=1) * volatility_scale)
    else:
        annualized_volatility = 0.0
    downside = np.minimum(excess, 0.0)
    downside_count = int(np.count_nonzero(excess < 0.0))
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * volatility_scale)
    sortino_ratio, sortino_reason = _sortino_ratio(
        annualized_excess,
        downside_deviation,
        n_observations=int(values.size),
        downside_count=downside_count,
    )

    if values.size < TAIL_RISK_MIN_OBSERVATIONS:
        value_at_risk = None
        expected_shortfall = None
        tail_risk_reason = (
            f"needs at least {TAIL_RISK_MIN_OBSERVATIONS} returns "
            f"(this sample has {int(values.size)})"
        )
    else:
        value_at_risk, expected_shortfall = historical_var_expected_shortfall(
            values,
            confidence=confidence,
        )
        tail_risk_reason = None
    return PerformanceSummary(
        cumulative_return=cumulative_return,
        cagr=cagr,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=_ratio(annualized_excess, annualized_volatility),
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown(values),
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
        sortino_reason=sortino_reason,
        tail_risk_reason=tail_risk_reason,
    )


def cumulative_wealth(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    initial_value: float = 1.0,
) -> np.ndarray:
    """Return a cumulative wealth index for a periodic return series."""
    initial_value = require_finite("initial value", initial_value)
    if initial_value <= 0:
        raise ValueError("initial value must be positive")
    values = _return_array(returns)
    return initial_value * np.cumprod(1.0 + values)


def download_adjusted_close(
    tickers: Sequence[str],
    *,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted closes with the optional yfinance dependency.

    Dates missing a price for any requested symbol are dropped. They are not
    forward-filled. The symbol list is whatever the caller passes, so a
    backtest of names that are listed today does not add names that delisted
    earlier.
    """
    symbols = [ticker.strip().upper() for ticker in tickers if str(ticker).strip()]
    if not symbols:
        raise ValueError("tickers must contain at least one symbol")
    if not str(start).strip():
        raise ValueError("start date must not be empty")

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'yfinance is required for downloads; install with pip install -e ".[market-data]"'
        ) from exc

    data = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if data is None or len(data) == 0:
        raise ValueError("market-data download returned no observations")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("download did not include adjusted close prices")
        prices = data["Close"]
    else:
        if "Close" not in data.columns:
            raise ValueError("download did not include adjusted close prices")
        prices = data[["Close"]].rename(columns={"Close": symbols[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])
    return align_prices(prices)


def _long_only_weights(weights: Sequence[float], n_assets: int) -> np.ndarray:
    target = np.asarray(weights, dtype=float)
    if target.ndim != 1 or len(target) != n_assets:
        raise ValueError("weights must match the number of price columns")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("weights must be finite and non-negative")
    if abs(float(target.sum()) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.0")
    return target


def buy_and_hold_returns(
    prices: pd.DataFrame,
    weights: Sequence[float],
) -> pd.Series:
    """Return a drifted long-only book with no rebalance and no trading cost.

    Weights are applied to the first price and then held. The return from one
    row to the next uses only those two prices, so a later close cannot change
    an earlier weight. This is the no-trade benchmark for
    ``backtest_rebalanced_portfolio``.
    """
    clean = _price_frame(prices)
    target = _long_only_weights(weights, clean.shape[1])
    values = clean.to_numpy(dtype=float)
    wealth = (values / values[0]) @ target
    returns = wealth[1:] / wealth[:-1] - 1.0
    if not np.isfinite(returns).all() or np.any(returns < -1.0):
        raise ValueError("buy-and-hold returns must be finite and at least -100%")
    return pd.Series(returns, index=clean.index[1:], name="buy_and_hold_return")


def backtest_rebalanced_portfolio(
    prices: pd.DataFrame,
    weights: Sequence[float],
    *,
    rebalance_every: int = 21,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    initial_value: float = 1.0,
) -> RebalancedBacktestResult:
    """Backtest a long-only target allocation with periodic rebalancing.

    The weight that earns period t is the weight held at the previous close.
    After that return, if the period is a rebalance date and a later period
    still remains, the book trades back to the target at that close. The
    close that sets the new weight is not used to earn the return just
    realized.

    Turnover is one-way turnover: half the sum of absolute weight changes.
    Commission and slippage are both charged as that turnover times their
    basis-point rates, on post-return wealth. Slippage here is a proportional
    execution haircut, not a delay of the fill. Rebalancing at the final
    observation is skipped because it cannot affect a later return. The book
    is assumed to start on the target weights, so there is no opening trade.
    """
    if isinstance(rebalance_every, bool) or not isinstance(rebalance_every, int):
        raise TypeError("rebalance_every must be a positive integer")
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be a positive integer")
    transaction_cost_bps = require_finite("transaction cost", transaction_cost_bps)
    slippage_bps = require_finite("slippage", slippage_bps)
    initial_value = require_finite("initial value", initial_value)
    if transaction_cost_bps < 0:
        raise ValueError("transaction cost must be non-negative")
    if slippage_bps < 0:
        raise ValueError("slippage must be non-negative")
    if initial_value <= 0:
        raise ValueError("initial value must be positive")

    returns = simple_returns(prices)
    target = _long_only_weights(weights, returns.shape[1])

    current_weights = target.copy()
    wealth_value = float(initial_value)
    net_returns: list[float] = []
    wealth_values: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    n_periods = len(returns)

    for position, (_, row) in enumerate(returns.iterrows(), start=1):
        starting_wealth = wealth_value
        asset_returns = row.to_numpy(dtype=float)
        gross_portfolio_return = float(current_weights @ asset_returns)
        if gross_portfolio_return <= -1.0:
            raise ValueError("portfolio return cannot be -100% or worse")
        wealth_value *= 1.0 + gross_portfolio_return
        drifted_weights = current_weights * (1.0 + asset_returns) / (1.0 + gross_portfolio_return)

        turnover = 0.0
        transaction_cost = 0.0
        if position % rebalance_every == 0 and position < n_periods:
            turnover = 0.5 * float(np.abs(target - drifted_weights).sum())
            cost_rate = turnover * (transaction_cost_bps + slippage_bps) / 10_000.0
            if cost_rate >= 1.0:
                raise ValueError("transaction cost cannot consume the whole portfolio")
            transaction_cost = wealth_value * cost_rate
            wealth_value -= transaction_cost
            current_weights = target.copy()
        else:
            current_weights = drifted_weights

        net_returns.append(wealth_value / starting_wealth - 1.0)
        wealth_values.append(wealth_value)
        turnover_values.append(turnover)
        cost_values.append(transaction_cost)

    index = returns.index
    return RebalancedBacktestResult(
        returns=pd.Series(net_returns, index=index, name="portfolio_return"),
        wealth=pd.Series(wealth_values, index=index, name="portfolio_wealth"),
        turnover=pd.Series(turnover_values, index=index, name="turnover"),
        transaction_costs=pd.Series(cost_values, index=index, name="transaction_cost"),
    )
