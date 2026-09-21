# Advanced Financial Modeling Suite

## For a data analyst application

**Use this when the role is finance, FP&A, or credit.** It is the one finance dashboard to show: Excel assumptions, a tested Python core (expected loss, concentration, stress), and the Streamlit board that calls that core. The board defaults to the 1,000-loan credit book in `data/portfolio_data.csv`. Do not also lead with the separate trading dashboards.

<p align="center"><img src="cash-flow-model.png" alt="Cash-flow forecast" width="100%"></p>
<p align="center"><img src="loan-risk-model.png" alt="Credit expected-loss model" width="100%"></p>
<p align="center"><img src="portfolio-allocation.png" alt="Portfolio allocation" width="100%"></p>

[![Financial model quality](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/ParBproject/Advanced-Financial-Models/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](src/financial_models)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Dashboard-FF4B4B?logo=streamlit&logoColor=white)](dashboard/app.py)
[![Excel](https://img.shields.io/badge/Microsoft_Excel-Financial_Modeling-217346?logo=microsoftexcel&logoColor=white)](Advanced_Financial_Models.xlsx)

A portfolio-grade financial analytics project combining an executive-ready Excel workbook, a reproducible Python modeling core, and an interactive decision dashboard. It covers **liquidity forecasting, credit expected loss and concentration, covariance-aware portfolio analytics, stress testing, and Monte Carlo risk analysis** while keeping assumptions explicit and calculations testable.

## What this project demonstrates

- Financial forecasting and liquidity scenario design
- Credit-risk modeling with **PD × LGD × EAD**
- Borrower concentration, HHI, top-N exposure, and effective borrower count
- Segment-level stressed-loss attribution
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
| Credit concentration | Borrower concentration and diversification risk | — | ✅ | ✅ |
| Segment stress attribution | Sources of stressed expected-loss change | — | ✅ | — |
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

The dashboard includes executive KPIs, editable cash-flow assumptions, loan expected-loss analysis, covariance-aware portfolio simulation/frontier views, liquidity/credit stress controls, and Monte Carlo terminal-value distributions.

The credit tab and the overview open on the committed 1,000-loan book. Cash-flow and market-portfolio tabs still use the workbook assumptions. Expected loss and borrower HHI on the credit book come from `summarize_portfolio()` and `credit_concentration()`.

## 1. Cash-Flow Forecast

![Cash-flow forecast model](cash-flow-model.png)

The 12-month model compounds revenue growth, forecasts operating expenses, rent, debt service and CapEx, then tracks liquidity month by month. The Python implementation converts the annual discount assumption to an equivalent monthly rate before discounting monthly net cash flows.

**Core outputs:** ending cash, average monthly net cash flow, minimum/maximum liquidity, and NPV.

`CashFlowStressScenario` can independently shock starting revenue, monthly growth, operating-expense ratio, fixed costs, and CapEx. The result reports stressed ending cash and NPV alongside the unchanged baseline.

## 2. Credit Risk Analytics

![Credit expected-loss model](loan-risk-model.png)

The credit model maps FICO-style score bands to probability of default, applies an adjustable LGD assumption, and calculates expected loss:

> **Expected Loss = EAD × PD × LGD**

The reproducible core aggregates portfolio exposure, expected loss, expected-loss ratio, average score, and Low/Medium/High risk counts. The single-loan formula check in the tests ($100,000 × 15% PD × 45% LGD = $6,750) is not the book on the dashboard.

### Default book

`data/portfolio_data.csv` is a copy of the loan file in [ParBproject/portfolio-risk-analysis-credit-risk-modeling](https://github.com/ParBproject/portfolio-risk-analysis-credit-risk-modeling). That repository is the original source. The adapter in `loan_book.py` maps columns as follows:

| CSV column | Loan field |
|---|---|
| `Customer_ID` | `loan_id` and `borrower` (the file has no obligor name) |
| `Loan_Amount` | `exposure` |
| `Credit_Score` | `credit_score` |

`PD_Score` stays on the file. Expected loss uses the score-band PD above, at the library's 45% LGD. Headline results pinned by the tests:

| Metric | Value |
|---|---:|
| Loans | 1,000 |
| Total exposure | $68,121,079.07 |
| Expected loss | $1,946,680.74 |
| Borrower HHI | 0.00130766 |
| Effective borrowers | 764.7 |

The file's own average `PD_Score` is 29.80%, and 504 loans have `PD_Score` above 20%. A 20% revenue cut and a 10% expense increase together produce −$12,559,240.00 of net income. Those file facts are not expected-loss outputs.

The credit repository's Word memo is retired. Two of its lines do not survive the file:

- $43,146.55 and $25,890.70 are NumPy's higher 5th and 1st percentiles of per-customer `Net_Income`. They are positive income levels, not a portfolio loss VaR.
- Operational-risk score above 60 does not show a 2.5× default rate. The default rate is 29.67% (27 of 91) above 60 and 29.81% (271 of 909) at or below 60.

The overview and credit tab print that decision in one line.

### Stress testing

`CreditStressScenario` applies PD and LGD multipliers to every exposure. Stressed PD and LGD are capped at 100%, so extreme scenarios remain economically bounded.

### Concentration risk

`credit_concentration()` aggregates multiple facilities to the borrower level and reports:

- largest-borrower exposure share;
- configurable top-N exposure share;
- **Herfindahl-Hirschman Index (HHI)** from borrower exposure shares;
- effective borrower count (`1 / HHI`);
- exposure distribution by Low/Medium/High risk rating.

### Segment stress attribution

`segment_stress_attribution()` accepts caller-defined loan-to-segment mappings and reconciles each segment's baseline expected loss, stressed expected loss, loss change, and share of total portfolio loss change. Tests require segment totals to reconcile exactly to the portfolio-level stress result.

Run the example:

```bash
python examples/credit_concentration.py
```

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
├── dashboard/app.py                     # Streamlit decision dashboard
├── src/financial_models/
│   ├── cash_flow.py                      # 12-month liquidity forecast
│   ├── credit_risk.py                    # PD/LGD/EAD expected-loss model
│   ├── credit_concentration.py           # HHI, top-N and stress attribution
│   ├── loan_book.py                      # CSV adapter for the 1,000-loan book
│   ├── portfolio.py                      # Excel-comparable baseline allocation
│   ├── advanced_portfolio.py             # Covariance, simulation, frontier
│   └── stress_testing.py                 # Stress scenarios + Monte Carlo
├── data/portfolio_data.csv               # Committed copy of the credit-repo loan file
├── examples/
│   ├── basic_analysis.py
│   ├── efficient_frontier.py
│   ├── stress_analysis.py
│   └── credit_concentration.py
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
python examples/credit_concentration.py
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
- Credit formula check: **$100,000 exposure × 15% PD × 45% LGD = $6,750 expected loss**
- 1,000-loan book: **1,000 loans, $68,121,079.07 exposure, $1,946,680.74 expected loss** at 45% LGD, borrower HHI **0.00130766**
- hand-calculated borrower concentration where HHI = **0.375** and top-2 share = **75%**
- segment stress contributions reconciling exactly to total stressed portfolio loss
- baseline allocation weights/return calculation
- hand-calculated two-asset covariance risk using **wᵀΣw**
- rejection of asymmetric and non-positive-semidefinite matrices
- seeded portfolio-simulation reproducibility and allocation constraints
- neutral/severe stress behavior and PD/LGD caps
- zero-volatility Monte Carlo matching deterministic compounding
- ordered terminal percentiles and bounded loss probabilities

CI installs the package plus dashboard dependencies, runs correctness linting, executes the full numerical suite, compiles the source/UI, and verifies imports on Python 3.10 and 3.12.

## Excel workbook

- **[Download the Excel workbook](Advanced_Financial_Models.xlsx)**
- **[Read the full user guide](Financial_Models_User_Guide.txt)**

The workbook remains the spreadsheet scenario-analysis deliverable. The Python package adds reproducibility and advanced analytics; the dashboard makes core analytics interactive.

## Model assumptions

- The Excel-comparable portfolio calculation assumes zero correlation between asset classes.
- The covariance-aware layer only uses correlations supplied explicitly by the caller; its bundled matrix is illustrative.
- Portfolio Monte Carlo uses a lognormal approximation calibrated to portfolio expected return and covariance-aware volatility.
- Credit loss is **expected loss**, not regulatory capital or unexpected loss.
- Credit PD mappings and stress multipliers are illustrative portfolio assumptions, not underwriting advice.
- HHI and top-N metrics describe exposure concentration; they do not estimate default correlation.
- Cash-flow stresses are deterministic scenarios and do not estimate macroeconomic probabilities.
- Historical returns and volatility assumptions do not guarantee future performance.

## Roadmap

- Exportable scenario/comparison reports
- Optional historical-data estimation for returns/covariance
- Deployable hosted dashboard configuration
- Additional sector/geography concentration dimensions

## Disclaimer

This repository is an educational portfolio project and is not investment, lending, accounting, or financial advice.

## License

MIT — see [LICENSE](LICENSE).
