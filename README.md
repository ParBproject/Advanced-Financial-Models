# Advanced Financial Modeling Suite

[![Financial model quality](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](src/financial_models)
[![NumPy](https://img.shields.io/badge/NumPy-Matrix_Analytics-013243?logo=numpy&logoColor=white)](src/financial_models/advanced_portfolio.py)
[![Excel](https://img.shields.io/badge/Microsoft_Excel-Financial_Modeling-217346?logo=microsoftexcel&logoColor=white)](Advanced_Financial_Models.xlsx)

A portfolio-grade financial analytics project combining an executive-ready Excel workbook with a reproducible Python modeling core. The suite covers **liquidity forecasting**, **credit expected loss**, and both **baseline and covariance-aware portfolio analytics** while keeping assumptions explicit and calculations testable.

## What this project demonstrates

- Financial forecasting and scenario design
- Credit-risk modeling with **PD × LGD × EAD**
- Portfolio return, volatility, Sharpe ratio, and long-horizon projection
- Covariance/correlation matrix validation and **wᵀΣw** portfolio risk
- Reproducible long-only portfolio simulation and approximate efficient frontiers
- Advanced Excel modeling and visualization
- Input validation, regression tests, CI, and auditable model boundaries

## Models

| Model | Decision supported | Excel | Python core |
|---|---|---:|---:|
| Cash-flow forecast | Liquidity planning and funding needs | ✅ | ✅ |
| Credit expected loss | Loan monitoring and portfolio loss estimation | ✅ | ✅ |
| Portfolio allocation | Risk/return trade-offs and scenario comparison | ✅ | ✅ |
| Covariance-aware portfolio analytics | Diversification, max-Sharpe and minimum-risk analysis | — | ✅ |

## 1. Cash-Flow Forecast

![Cash-flow forecast model](cash-flow-model.png)

The 12-month model compounds revenue growth, forecasts operating expenses, rent, loan service and CapEx, then tracks liquidity month by month. The Python implementation converts the annual discount assumption to an equivalent monthly rate before discounting monthly net cash flows.

**Core outputs:** ending cash, average monthly net cash flow, minimum/maximum liquidity, and NPV.

## 2. Credit Expected-Loss Model

![Credit expected-loss model](loan-risk-model.png)

The credit model maps FICO-style score bands to probability of default, applies an adjustable LGD assumption, and calculates expected loss for each exposure:

> **Expected Loss = EAD × PD × LGD**

The reproducible core aggregates portfolio exposure, expected loss, expected-loss ratio, average score, and Low/Medium/High risk counts.

## 3. Portfolio Allocation Model

![Portfolio allocation model](portfolio-allocation.png)

The Excel workbook compares six asset classes using expected return, volatility, allocation weights, Sharpe ratio, and a 10-year value projection. The baseline Python model deliberately mirrors the workbook's documented **zero-correlation simplification**:

> **Baseline volatility = √Σ(weight × asset volatility)²**

That baseline remains available for workbook-to-code reconciliation.

### Covariance-aware extension

The advanced Python layer adds professional portfolio-risk mechanics without rewriting the baseline model:

> **Portfolio variance = wᵀΣw**

It provides:

- correlation-to-covariance conversion;
- symmetry, finiteness, bounds, and positive-semidefinite matrix validation;
- covariance-aware expected return, volatility, Sharpe ratio, and projected value;
- seeded long-only portfolio simulation using simplex/Dirichlet weights;
- sampled max-Sharpe and minimum-volatility portfolio selection;
- approximate non-dominated efficient-frontier extraction.

The built-in six-asset correlation matrix is explicitly **illustrative**, generated from transparent one-factor loadings. It is not presented as historical market estimation.

## Repository architecture

```text
Advanced-Financial-Models/
├── Advanced_Financial_Models.xlsx       # Original Excel modeling suite
├── Financial_Models_User_Guide.txt      # Detailed workbook assumptions and usage
├── src/financial_models/
│   ├── cash_flow.py                      # 12-month liquidity forecast
│   ├── credit_risk.py                    # PD/LGD/EAD expected-loss model
│   ├── portfolio.py                      # Excel-comparable baseline allocation metrics
│   └── advanced_portfolio.py             # Covariance, simulation, efficient frontier
├── examples/
│   ├── basic_analysis.py                 # Baseline end-to-end example
│   └── efficient_frontier.py             # Correlated portfolio simulation example
├── tests/
│   ├── test_models.py                    # Workbook-aligned regression tests
│   └── test_advanced_portfolio.py        # Matrix/simulation/frontier tests
├── .github/workflows/ci.yml              # Python 3.10 + 3.12 quality checks
└── CONTRIBUTING.md                       # Modeling and test conventions
```

## Quick start

```bash
git clone https://github.com/ParBproject/Advanced-Financial-Models.git
cd Advanced-Financial-Models
python -m pip install -e .
python examples/basic_analysis.py
python examples/efficient_frontier.py
```

Run the regression suite:

```bash
python -m unittest discover -s tests -v
```

For development quality checks:

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
```

## Reproducibility checks

The automated tests pin important numerical behavior, including:

- Month 1 cash flow: **$100,000 revenue − 60% OpEx − $8,000 rent − $5,000 loan = $27,000 net cash flow**
- Credit example: **$100,000 exposure × 15% PD × 45% LGD = $6,750 expected loss**
- Baseline allocation: weights sum to 100% and expected return is a weighted average
- A hand-calculated two-asset covariance example for **wᵀΣw**
- rejection of asymmetric and non-positive-semidefinite risk matrices
- seeded simulation reproducibility and long-only/full-investment constraints
- max-Sharpe/minimum-volatility selection and monotonic sampled-frontier behavior

CI runs correctness linting, compilation, and the full numerical regression suite on Python 3.10 and 3.12.

## Excel workbook

- **[Download the Excel workbook](Advanced_Financial_Models.xlsx)**
- **[Read the full user guide](Financial_Models_User_Guide.txt)**

The workbook retains the original blue-input / black-formula / green-link convention and remains the visual scenario-analysis deliverable. The Python package complements it with reproducible calculations and automated validation.

## Model assumptions

The project keeps baseline and advanced assumptions separate:

- The Excel-comparable portfolio calculation assumes zero correlation between asset classes.
- The covariance-aware Python layer only uses correlations supplied explicitly by the caller; its bundled matrix is illustrative.
- Credit loss is **expected loss**, not regulatory capital or unexpected loss.
- Default-rate mappings are illustrative portfolio assumptions, not underwriting advice.
- Forecasts exclude tax, transaction-cost, and macroeconomic regime modeling unless explicitly added by a scenario.
- Historical returns and volatility assumptions do not guarantee future performance.

## Roadmap

Planned extensions are separated so each can be validated independently:

- Monte Carlo portfolio and cash-flow simulation
- Credit stress scenarios and concentration analysis
- Interactive analytics dashboard
- Exportable scenario/comparison reports

## Disclaimer

This repository is an educational portfolio project and is not investment, lending, accounting, or financial advice.

## License

MIT — see [LICENSE](LICENSE).
