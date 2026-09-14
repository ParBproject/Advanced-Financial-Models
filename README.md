# Advanced Financial Modeling Suite

[![Financial model quality](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](src/financial_models)
[![NumPy](https://img.shields.io/badge/NumPy-Risk_Analytics-013243?logo=numpy&logoColor=white)](src/financial_models)
[![Excel](https://img.shields.io/badge/Microsoft_Excel-Financial_Modeling-217346?logo=microsoftexcel&logoColor=white)](Advanced_Financial_Models.xlsx)

A portfolio-grade financial analytics project combining an executive-ready Excel workbook with a reproducible Python modeling core. It covers **liquidity forecasting, credit expected loss, covariance-aware portfolio analytics, stress testing, and Monte Carlo risk analysis** while keeping assumptions explicit and calculations testable.

## What this project demonstrates

- Financial forecasting and liquidity scenario design
- Credit-risk modeling with **PD × LGD × EAD**
- Cash-flow and credit stress testing
- Portfolio return, volatility, Sharpe ratio, and long-horizon projection
- Covariance/correlation validation and **wᵀΣw** portfolio risk
- Reproducible long-only portfolio simulation and approximate efficient frontiers
- Seeded Monte Carlo terminal-value distributions and downside percentiles
- Advanced Excel modeling plus testable Python implementations
- Input validation, numerical regression tests, and multi-version CI

## Analytics layers

| Layer | Decision supported | Excel | Python |
|---|---|---:|---:|
| Cash-flow forecast | Liquidity planning and funding needs | ✅ | ✅ |
| Credit expected loss | Loan monitoring and portfolio loss estimation | ✅ | ✅ |
| Baseline portfolio allocation | Workbook-comparable risk/return analysis | ✅ | ✅ |
| Covariance-aware portfolio analytics | Diversification and efficient-frontier analysis | — | ✅ |
| Cash-flow / credit stress tests | Downturn and loss-severity scenarios | — | ✅ |
| Portfolio Monte Carlo | Terminal wealth distribution and downside risk | — | ✅ |

## 1. Cash-Flow Forecast

![Cash-flow forecast model](cash-flow-model.png)

The 12-month model compounds revenue growth, forecasts operating expenses, rent, debt service and CapEx, then tracks liquidity month by month. The Python implementation converts the annual discount assumption to an equivalent monthly rate before discounting monthly net cash flows.

**Core outputs:** ending cash, average monthly net cash flow, minimum/maximum liquidity, and NPV.

### Cash-flow stress testing

`CashFlowStressScenario` can independently shock:

- starting revenue through a revenue multiplier;
- monthly revenue growth;
- operating-expense ratio;
- rent and loan service through a fixed-cost multiplier;
- planned CapEx.

The result reports stressed ending cash and NPV alongside the unchanged baseline, making the financial impact of a scenario explicit.

## 2. Credit Expected-Loss Model

![Credit expected-loss model](loan-risk-model.png)

The credit model maps FICO-style score bands to probability of default, applies an adjustable LGD assumption, and calculates expected loss:

> **Expected Loss = EAD × PD × LGD**

The reproducible core aggregates portfolio exposure, expected loss, expected-loss ratio, average score, and Low/Medium/High risk counts.

### Credit stress testing

`CreditStressScenario` applies PD and LGD multipliers to every exposure. Stressed PD and LGD are capped at 100%, so extreme scenarios remain economically bounded. Outputs include baseline expected loss, stressed expected loss, expected-loss ratio, and loan-level stressed parameters.

## 3. Portfolio Allocation Model

![Portfolio allocation model](portfolio-allocation.png)

The Excel workbook compares six asset classes using expected return, volatility, allocation weights, Sharpe ratio, and a 10-year value projection. The baseline Python model deliberately mirrors the workbook's documented **zero-correlation simplification**:

> **Baseline volatility = √Σ(weight × asset volatility)²**

This baseline remains available for workbook-to-code reconciliation.

### Covariance-aware extension

The advanced Python layer adds professional portfolio-risk mechanics without rewriting the baseline model:

> **Portfolio variance = wᵀΣw**

It provides:

- correlation-to-covariance conversion;
- symmetry, finiteness, bounds, unit-diagonal, and positive-semidefinite validation;
- covariance-aware expected return, volatility, Sharpe ratio, and projected value;
- seeded long-only portfolio simulation using simplex/Dirichlet weights;
- sampled maximum-Sharpe and minimum-volatility portfolios;
- approximate non-dominated efficient-frontier extraction.

The bundled six-asset correlation matrix is explicitly **illustrative**, generated from transparent one-factor loadings. It is not presented as a historical estimate.

### Monte Carlo terminal-value simulation

The Monte Carlo layer first calculates the selected portfolio's covariance-aware annual return and volatility, then converts those moments into a lognormal gross-return approximation. This keeps simulated wealth non-negative and supports reproducible terminal-value distributions.

Outputs include:

- mean and median terminal value;
- 5th, 25th, 75th and 95th percentiles;
- probability of finishing below the initial investment;
- the full terminal-value sample for downstream charts or analysis.

All simulations support an explicit `random_state`.

## Repository architecture

```text
Advanced-Financial-Models/
├── Advanced_Financial_Models.xlsx       # Original Excel modeling suite
├── Financial_Models_User_Guide.txt      # Workbook assumptions and usage
├── src/financial_models/
│   ├── cash_flow.py                      # 12-month liquidity forecast
│   ├── credit_risk.py                    # PD/LGD/EAD expected-loss model
│   ├── portfolio.py                      # Excel-comparable baseline allocation
│   ├── advanced_portfolio.py             # Covariance, simulation, efficient frontier
│   └── stress_testing.py                 # Stress scenarios + Monte Carlo
├── examples/
│   ├── basic_analysis.py                 # Baseline end-to-end example
│   ├── efficient_frontier.py             # Correlated portfolio example
│   └── stress_analysis.py                # Liquidity/credit/Monte Carlo example
├── tests/
│   ├── test_models.py                    # Workbook-aligned regression tests
│   ├── test_advanced_portfolio.py        # Matrix/frontier tests
│   └── test_stress_testing.py            # Stress/Monte Carlo tests
├── .github/workflows/ci.yml              # Python 3.10 + 3.12 checks
└── CONTRIBUTING.md                       # Modeling and test conventions
```

## Quick start

```bash
git clone https://github.com/ParBproject/Advanced-Financial-Models.git
cd Advanced-Financial-Models
python -m pip install -e .

python examples/basic_analysis.py
python examples/efficient_frontier.py
python examples/stress_analysis.py
```

Run the full regression suite:

```bash
python -m unittest discover -s tests -v
```

For development quality checks:

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
```

## Reproducibility and numerical checks

The automated suite pins important behavior, including:

- Month 1 cash flow: **$100,000 revenue − 60% OpEx − $8,000 rent − $5,000 loan = $27,000 net cash flow**
- Credit example: **$100,000 exposure × 15% PD × 45% LGD = $6,750 expected loss**
- Baseline allocation: weights sum to 100% and expected return is a weighted average
- A hand-calculated two-asset covariance example for **wᵀΣw**
- rejection of asymmetric and non-positive-semidefinite risk matrices
- seeded portfolio-simulation reproducibility and long-only/full-investment constraints
- neutral stress scenarios reproducing baseline results exactly
- severe cash-flow scenarios reducing liquidity and NPV
- PD/LGD stress caps at 100%
- zero-volatility Monte Carlo collapsing to exact deterministic compound growth
- ordered Monte Carlo percentiles and probability-of-loss bounds

CI runs correctness linting, compilation, and the full numerical regression suite on Python 3.10 and 3.12.

## Excel workbook

- **[Download the Excel workbook](Advanced_Financial_Models.xlsx)**
- **[Read the full user guide](Financial_Models_User_Guide.txt)**

The workbook remains the visual scenario-analysis deliverable. The Python package complements it with reproducible calculations, automated validation, advanced risk analytics, and stress/simulation tooling.

## Model assumptions

The project keeps baseline and advanced assumptions separate:

- The Excel-comparable portfolio calculation assumes zero correlation between asset classes.
- The covariance-aware layer only uses correlations supplied explicitly by the caller; its bundled matrix is illustrative.
- Portfolio Monte Carlo uses a lognormal approximation calibrated to the selected portfolio's annual expected return and covariance-aware volatility.
- Credit loss is **expected loss**, not regulatory capital or unexpected loss.
- Credit PD mappings and stress multipliers are illustrative portfolio assumptions, not underwriting advice.
- Cash-flow scenarios are deterministic stresses around user-supplied assumptions and do not estimate macroeconomic probabilities.
- Historical returns and volatility assumptions do not guarantee future performance.

## Roadmap

- Credit concentration analytics and segment-level stress attribution
- Interactive analytics dashboard
- Exportable scenario/comparison reports
- Optional historical-data estimation for returns/covariance

## Disclaimer

This repository is an educational portfolio project and is not investment, lending, accounting, or financial advice.

## License

MIT — see [LICENSE](LICENSE).
