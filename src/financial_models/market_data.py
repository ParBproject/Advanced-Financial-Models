from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


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
    """Risk-adjusted performance statistics for one return series."""

    cumulative_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    value_at_risk: float
    expected_shortfall: float


def _validate_periods_per_year(periods_per_year: int) -> None:
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int):
        raise TypeError("periods_per_year must be a positive integer")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be a positive integer")


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("prices must contain at least one observation")
    if prices.shape[1] < 1:
        raise ValueError("prices must contain at least one asset")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all(axis=None):
        raise ValueError("prices must contain numeric observations")
    numeric = numeric.dropna(how="all").ffill().dropna()
    if len(numeric) < 2:
        raise ValueError("prices must contain at least two complete observations")
    if (numeric <= 0).any(axis=None):
        raise ValueError("prices must be strictly positive")
    return numeric


def _finite_returns(
    returns: pd.Series | Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("returns must contain at least one finite observation")
    if np.any(values < -1.0):
        raise ValueError("returns cannot be less than -100%")
    return values


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple periodic returns from an aligned price matrix."""
    clean = _validate_prices(prices)
    returns = clean.pct_change(fill_method=None).dropna(how="any")
    if returns.empty:
        raise ValueError("prices do not produce a valid return series")
    return returns


def historical_risk_summary(
    prices: pd.DataFrame,
    *,
    periods_per_year: int = 252,
) -> MarketRiskSummary:
    """Estimate annualized return, covariance, volatility, correlation and drawdown."""
    _validate_periods_per_year(periods_per_year)

    clean = _validate_prices(prices)
    returns = simple_returns(clean)
    annualized_returns = returns.mean() * periods_per_year
    annualized_covariance = returns.cov() * periods_per_year
    annualized_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    correlation = returns.corr()

    wealth = clean / clean.iloc[0]
    running_peak = wealth.cummax()
    drawdowns = wealth / running_peak - 1.0
    max_drawdown = drawdowns.min()

    return MarketRiskSummary(
        annualized_returns=annualized_returns,
        annualized_covariance=annualized_covariance,
        annualized_volatility=annualized_volatility,
        correlation=correlation,
        max_drawdown=max_drawdown,
    )


def portfolio_return_series(
    returns: pd.DataFrame,
    weights: Sequence[float],
) -> pd.Series:
    """Calculate a fully invested portfolio return series from asset returns."""
    if returns.empty:
        raise ValueError("returns must not be empty")
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1 or len(weight_array) != returns.shape[1]:
        raise ValueError("weights must match the number of return columns")
    if not np.isfinite(weight_array).all():
        raise ValueError("weights must contain only finite values")
    if abs(float(weight_array.sum()) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.0")
    return pd.Series(
        returns.to_numpy(dtype=float) @ weight_array,
        index=returns.index,
        name="portfolio_return",
    )


def historical_var_expected_shortfall(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return positive loss magnitudes for historical VaR and Expected Shortfall."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    values = _finite_returns(returns)

    cutoff = float(np.quantile(values, 1.0 - confidence))
    tail = values[values <= cutoff]
    var = max(0.0, -cutoff)
    expected_shortfall = max(0.0, -float(tail.mean()))
    return var, expected_shortfall


def performance_summary(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.03,
    confidence: float = 0.95,
) -> PerformanceSummary:
    """Summarize growth, downside risk, and risk-adjusted performance."""
    _validate_periods_per_year(periods_per_year)
    values = _finite_returns(returns)

    wealth = np.cumprod(1.0 + values)
    final_wealth = float(wealth[-1])
    cumulative_return = final_wealth - 1.0
    if final_wealth <= 0.0:
        cagr = -1.0
    else:
        cagr = final_wealth ** (periods_per_year / values.size) - 1.0

    annualized_volatility = (
        float(np.std(values, ddof=1) * np.sqrt(periods_per_year))
        if values.size > 1
        else 0.0
    )
    annualized_return = float(np.mean(values) * periods_per_year)
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = (
        excess_return / annualized_volatility
        if annualized_volatility > 1e-15
        else (float("inf") if excess_return > 0 else 0.0)
    )

    downside = np.minimum(values, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(periods_per_year))
    sortino_ratio = (
        excess_return / downside_deviation
        if downside_deviation > 1e-15
        else (float("inf") if excess_return > 0 else 0.0)
    )

    wealth_with_origin = np.concatenate(([1.0], wealth))
    running_peak = np.maximum.accumulate(wealth_with_origin)
    max_drawdown = float(np.min(wealth_with_origin / running_peak - 1.0))
    var, expected_shortfall = historical_var_expected_shortfall(
        values,
        confidence=confidence,
    )

    return PerformanceSummary(
        cumulative_return=cumulative_return,
        cagr=float(cagr),
        annualized_volatility=annualized_volatility,
        sharpe_ratio=float(sharpe_ratio),
        sortino_ratio=float(sortino_ratio),
        max_drawdown=max_drawdown,
        value_at_risk=var,
        expected_shortfall=expected_shortfall,
    )


def cumulative_wealth(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    initial_value: float = 1.0,
) -> np.ndarray:
    """Return a cumulative wealth index for a periodic return series."""
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")
    values = _finite_returns(returns)
    return initial_value * np.cumprod(1.0 + values)


def download_adjusted_close(
    tickers: Sequence[str],
    *,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted-close market data using the optional yfinance dependency."""
    symbols = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not symbols:
        raise ValueError("tickers must contain at least one symbol")

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
    if data.empty:
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
    return _validate_prices(prices)
