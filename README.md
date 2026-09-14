# Advanced Financial Modeling Suite

[![Financial model quality](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](src/financial_models)
[![Excel](https://img.shields.io/badge/Microsoft_Excel-Financial_Modeling-217346?logo=microsoftexcel&logoColor=white)](Advanced_Financial_Models.xlsx)
[![Models](https://img.shields.io/badge/Decision_Models-3-1f6feb)](#models)

A portfolio-grade financial analytics project combining an executive-ready Excel workbook with a reproducible Python modeling core. The suite covers **liquidity forecasting**, **credit expected loss**, and **portfolio risk/return analysis** while keeping assumptions explicit and calculations testable.

## What this project demonstrates

- Financial forecasting and scenario design
- Credit-risk modeling with **PD × LGD × EAD**
- Portfolio return, volatility, Sharpe ratio, and long-horizon projection
- Advanced Excel modeling and visualization
- Reproducible Python implementations of workbook logic
- Input validation, regression tests, CI, and auditable model boundaries

## Models

| Model | Decision supported | Excel | Python core |
|---|---|---:|---:|
| Cash-flow forecast | Liquidity planning and funding needs | ✅ | ✅ |
| Credit expected loss | Loan monitoring and portfolio loss estimation | ✅ | ✅ |
| Portfolio allocation | Risk/return trade-offs and scenario comparison | ✅ | ✅ |

## 1. Cash-Flow Forecast

![Cash-flow forecast model](cash-flow-model.png)

The 12-month model compounds revenue growth, forecasts operating expenses, rent, loan service and CapEx, then tracks liquidity month by month. The Python implementation also converts the annual discount assumption to an equivalent monthly rate before discounting monthly net cash flows.

**Core outputs:** ending cash, average monthly net cash flow, minimum/maximum liquidity, and NPV.

## 2. Credit Expected-Loss Model

![Credit expected-loss model](loan-risk-model.png)

The credit model maps FICO-style score bands to probability of default, applies an adjustable LGD assumption, and calculates expected loss for each exposure:

> **Expected Loss = EAD × PD × LGD**

The reproducible core aggregates portfolio exposure, expected loss, expected-loss ratio, average score, and Low/Medium/High risk counts.

## 3. Portfolio Allocation Model

![Portfolio allocation model](portfolio-allocation.png)

The workbook compares six asset classes using expected return, volatility, allocation weights, Sharpe ratio, and a 10-year value projection. The initial Python implementation deliberately mirrors the workbook's documented **zero-correlation simplification**:

> **Portfolio volatility = √Σ(weight × asset volatility)²**

This keeps the code and workbook directly comparable while leaving covariance-aware analytics as a clearly separated advanced extension.

## Repository architecture

```text
Advanced-Financial-Models/
├── Advanced_Financial_Models.xlsx      # Original Excel modeling suite
├── Financial_Models_User_Guide.txt     # Detailed workbook assumptions and usage
├── src/financial_models/
│   ├── cash_flow.py                     # 12-month liquidity forecast
│   ├── credit_risk.py                   # PD/LGD/EAD expected-loss model
│   └── portfolio.py                     # Baseline allocation metrics
├── examples/basic_analysis.py           # Executable end-to-end example
├── tests/test_models.py                 # Hand-checkable regression tests
├── .github/workflows/ci.yml             # Multi-version quality checks
└── CONTRIBUTING.md                      # Modeling and test conventions
```

## Quick start

```bash
git clone https://github.com/ParBproject/Advanced-Financial-Models.git
cd Advanced-Financial-Models
python -m pip install -e .
python examples/basic_analysis.py
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

The automated tests pin important documented examples, including:

- Month 1 cash flow: **$100,000 revenue − 60% OpEx − $8,000 rent − $5,000 loan = $27,000 net cash flow**
- Credit example: **$100,000 exposure × 15% PD × 45% LGD = $6,750 expected loss**
- Baseline allocation: portfolio weights sum to 100% and expected return is calculated as a weighted average
- Invalid ratios, credit scores, and allocation totals fail explicitly instead of producing silent model errors

CI runs linting, compilation, and the full numerical regression suite on Python 3.10 and 3.12.

## Excel workbook

- **[Download the Excel workbook](Advanced_Financial_Models.xlsx)**
- **[Read the full user guide](Financial_Models_User_Guide.txt)**

The workbook retains the original blue-input / black-formula / green-link convention and remains the visual scenario-analysis deliverable. The Python package complements it with reproducible calculations and automated validation.

## Baseline assumptions

The project intentionally documents its simplifications:

- The Excel portfolio model assumes zero correlation between asset classes.
- Credit loss is **expected loss**, not regulatory capital or unexpected loss.
- Default-rate mappings are illustrative portfolio assumptions, not underwriting advice.
- Forecasts exclude tax, transaction-cost, and macroeconomic regime modeling unless explicitly added by a later scenario.
- Historical returns and volatility assumptions do not guarantee future performance.

## Roadmap

Planned extensions are separated from the baseline models so each can be validated independently:

- Covariance-aware portfolio risk and efficient-frontier optimization
- Monte Carlo portfolio and cash-flow simulation
- Credit stress scenarios and concentration analysis
- Interactive analytics dashboard and exportable scenario reports

## Disclaimer

This repository is an educational portfolio project and is not investment, lending, accounting, or financial advice.

## License

MIT — see [LICENSE](LICENSE).
