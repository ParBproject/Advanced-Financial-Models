# Advanced Financial Modeling Suite

[![Financial model quality](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](src/financial_models)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Dashboard-FF4B4B?logo=streamlit&logoColor=white)](dashboard/app.py)
[![Excel](https://img.shields.io/badge/Microsoft_Excel-Financial_Modeling-217346?logo=microsoftexcel&logoColor=white)](Advanced_Financial_Models.xlsx)

A portfolio-grade financial analytics project combining an executive-ready Excel workbook, a reproducible Python modeling core, and an interactive decision dashboard. It covers **liquidity forecasting, credit expected loss, covariance-aware portfolio analytics, stress testing, and Monte Carlo risk analysis** while keeping assumptions explicit and calculations testable.

## What this project demonstrates

- Financial forecasting and liquidity scenario design
- Credit-risk modeling with **PD × LGD × EAD**
- Cash-flow and credit stress testing
- Portfolio return, volatility, Sharpe ratio, and long-horizon projection
- Covariance/correlation validation and **wᵀΣw** portfolio risk
- Reproducible long-only portfolio simulation and approximate efficient frontiers
- Seeded Monte Carlo terminal-value distributions and downside percentiles
- Streamlit decision-support UI backed by the tested model package
- Advanced Excel modeling plus reproducible Python implementations
- Input validation, numerical regression tests, and multi-version CI

## Analytics layers

| Layer | Decision supported | Excel | Python | Dashboard |
|---|---|---:|---:|---:|
| Cash-flow forecast | Liquidity planning and funding needs | ✅ | ✅ | ✅ |
| Credit expected loss | Loan monitoring and portfolio loss estimation | ✅ | ✅ | ✅ |
| Baseline portfolio allocation | Workbook-comparable risk/return analysis | ✅ | ✅ | ✅ |
| Covariance-aware portfolio analytics | Diversification and frontier analysis | — | ✅ | ✅ |
| Cash-flow / credit stress tests | Downturn and loss-severity scenarios | — | ✅ | ✅ |
| Portfolio Monte Carlo | Terminal wealth distribution and downside risk | — | ✅ | ✅ |

## Interactive decision dashboard

The Streamlit app turns the model library into a usable scenario-analysis tool rather than duplicating calculations inside the UI.

```bash
python -m pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

The dashboard includes five tabs:

- **Overview** — executive snapshot across liquidity, credit, and portfolio risk;
- **Cash Flow** — editable starting cash, revenue, growth, and OpEx assumptions with monthly charts;
- **Credit Risk** — editable loan exposures/scores, LGD control, PD bands, expected-loss table and risk ratings;
- **Portfolio** — editable return/volatility/weight assumptions, covariance-aware metrics, simulated portfolios, efficient frontier, max-Sharpe and minimum-volatility views;
- **Stress & Monte Carlo** — liquidity shocks, PD/LGD stress, terminal-value percentiles, loss probability, and distribution chart.

The UI uses the same functions covered by the repository's numerical regression suite. Model logic stays in `src/financial_models/`; `dashboard/app.py` is an orchestration and visualization layer.

## 1. Cash-Flow Forecast

![Cash-flow forecast model](cash-flow-model.png)

The 12-month model compounds revenue growth, forecasts operating expenses, rent, debt service and CapEx, then tracks liquidity month by month. The Python implementation converts the annual discount assumption to an equivalent monthly rate before discounting monthly net cash flows.

**Core outputs:** ending cash, average monthly net cash flow, minimum/maximum liquidity, and NPV.

### Cash-flow stress testing

`CashFlowStressScenario` can independently shock starting revenue, monthly growth, operating-expense ratio, fixed costs, and CapEx. The result reports stressed ending cash and NPV alongside the unchanged baseline.

## 2. Credit Expected-Loss Model

![Credit expected-loss model](loan-risk-model.png)

The credit model maps FICO-style score bands to probability of default, applies an adjustable LGD assumption, and calculates expected loss:

> **Expected Loss = EAD × PD × LGD**

The reproducible core aggregates portfolio exposure, expected loss, expected-loss ratio, average score, and Low/Medium/High risk counts.

### Credit stress testing

`CreditStressScenario` applies PD and LGD multipliers to every exposure. Stressed PD and LGD are capped at 100%, so extreme scenarios remain economically bounded.

## 3. Portfolio Allocation Model

![Portfolio allocation model](portfolio-allocation.png)

The Excel workbook compares six asset classes using expected return, volatility, allocation weights, Sharpe ratio, and a 10-year value projection. The baseline Python model deliberately mirrors the workbook's documented **zero-correlation simplification**:

> **Baseline volatility = √Σ(weight × asset volatility)²**

This baseline remains available for workbook-to-code reconciliation.

### Covariance-aware extension

The advanced Python layer adds professional portfolio-risk mechanics without rewriting the baseline model:

> **Portfolio variance = wᵀΣw**

It provides correlation-to-covariance conversion, matrix validation, covariance-aware metrics, seeded long-only simulation, sampled maximum-Sharpe/minimum-volatility portfolios, and approximate efficient-frontier extraction.

The bundled six-asset correlation matrix is explicitly **illustrative**, generated from transparent one-factor loadings rather than presented as a historical estimate.

### Monte Carlo terminal-value simulation

The Monte Carlo layer calculates covariance-aware portfolio return/volatility and converts those moments into a lognormal gross-return approximation. Outputs include terminal-value percentiles, probability of loss, mean/median wealth, and the full reproducible terminal-value sample.

## Repository architecture

```text
Advanced-Financial-Models/
├── Advanced_Financial_Models.xlsx       # Original Excel modeling suite
├── Financial_Models_User_Guide.txt      # Workbook assumptions and usage
├── dashboard/
│   └── app.py                            # Interactive Streamlit decision dashboard
├── .streamlit/
│   └── config.toml                       # Dashboard theme/server settings
├── src/financial_models/
│   ├── cash_flow.py                      # 12-month liquidity forecast
│   ├── credit_risk.py                    # PD/LGD/EAD expected-loss model
│   ├── portfolio.py                      # Excel-comparable baseline allocation
│   ├── advanced_portfolio.py             # Covariance, simulation, efficient frontier
│   └── stress_testing.py                 # Stress scenarios + Monte Carlo
├── examples/
│   ├── basic_analysis.py
│   ├── efficient_frontier.py
│   └── stress_analysis.py
├── tests/                                # Numerical regression suite
├── .github/workflows/ci.yml              # Python 3.10 + 3.12 checks
└── CONTRIBUTING.md
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

For the dashboard:

```bash
python -m pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

Run the full regression suite:

```bash
python -m unittest discover -s tests -v
```

For development quality checks:

```bash
python -m pip install -e ".[dev,dashboard]"
ruff check src tests examples dashboard
```

## Reproducibility and numerical checks

The automated suite pins important behavior, including:

- Month 1 cash flow: **$100,000 revenue − 60% OpEx − $8,000 rent − $5,000 loan = $27,000 net cash flow**
- Credit example: **$100,000 exposure × 15% PD × 45% LGD = $6,750 expected loss**
- Baseline allocation weights/return calculation
- Hand-calculated two-asset covariance risk using **wᵀΣw**
- rejection of asymmetric and non-positive-semidefinite matrices
- seeded portfolio-simulation reproducibility and allocation constraints
- neutral stress scenarios reproducing baseline results
- severe cash-flow stress lowering liquidity and NPV
- PD/LGD caps at 100%
- zero-volatility Monte Carlo matching deterministic compounding
- ordered terminal percentiles and bounded loss probabilities

CI installs the package plus dashboard dependencies, runs correctness linting, executes the full numerical suite, compiles the source/UI, and verifies imports on Python 3.10 and 3.12.

## Excel workbook

- **[Download the Excel workbook](Advanced_Financial_Models.xlsx)**
- **[Read the full user guide](Financial_Models_User_Guide.txt)**

The workbook remains the spreadsheet scenario-analysis deliverable. The Python package adds reproducibility and advanced analytics; the dashboard makes those analytics interactive.

## Model assumptions

- The Excel-comparable portfolio calculation assumes zero correlation between asset classes.
- The covariance-aware layer only uses correlations supplied explicitly by the caller; its bundled matrix is illustrative.
- Portfolio Monte Carlo uses a lognormal approximation calibrated to portfolio expected return and covariance-aware volatility.
- Credit loss is **expected loss**, not regulatory capital or unexpected loss.
- Credit PD mappings and stress multipliers are illustrative portfolio assumptions, not underwriting advice.
- Cash-flow stresses are deterministic scenarios and do not estimate macroeconomic probabilities.
- Historical returns and volatility assumptions do not guarantee future performance.

## Roadmap

- Credit concentration analytics and segment-level stress attribution
- Exportable scenario/comparison reports
- Optional historical-data estimation for returns/covariance
- Deployable hosted dashboard configuration

## Disclaimer

This repository is an educational portfolio project and is not investment, lending, accounting, or financial advice.

## License

MIT — see [LICENSE](LICENSE).
