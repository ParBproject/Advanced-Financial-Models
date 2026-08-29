# Advanced Financial Modeling Suite

[![Excel](https://img.shields.io/badge/Microsoft_Excel-Financial_Modeling-217346?logo=microsoftexcel&logoColor=white)](Advanced_Financial_Models.xlsx)
[![Models](https://img.shields.io/badge/Models-3-1f6feb)](Advanced_Financial_Models.xlsx)

A portfolio-ready Excel workbook containing three decision-support models: a 12-month cash-flow forecast, a credit expected-loss model, and a portfolio allocation model.

## Models at a Glance

| Model | Decision supported | Core techniques |
|---|---|---|
| Cash-flow forecast | Liquidity and planning | Revenue growth, expense forecasting, capex, debt service, NPV |
| Credit-risk model | Loan portfolio monitoring | PD × LGD × EAD, exposure summaries, risk bands |
| Portfolio allocation | Risk/return trade-offs | Weighted return, volatility, Sharpe ratio, scenario projection |

## 1. Cash-Flow Forecast

![Cash-flow forecast model](cash-flow-model.png)

- Monthly compounding revenue assumptions
- Operating-expense and capital-expenditure schedules
- Loan-repayment structure
- NPV calculation
- Conditional liquidity alerts

## 2. Credit Expected-Loss Model

![Credit expected-loss model](loan-risk-model.png)

- 50-loan illustrative portfolio
- Probability of default, loss given default, and exposure at default
- Expected-loss calculation and risk categorization
- Portfolio-level exposure summary

## 3. Portfolio Allocation Model

![Portfolio allocation model](portfolio-allocation.png)

- Six illustrative asset classes
- Weighted return and volatility calculations
- Sharpe-ratio comparison
- Ten-year projection and Solver-ready allocation inputs

## Files

- **[Download the Excel workbook](Advanced_Financial_Models.xlsx)**
- **[Read the user guide](Financial_Models_User_Guide.txt)**

## Skills Demonstrated

Advanced Excel formulas, XLOOKUP, SUMPRODUCT, NPV, scenario analysis, financial forecasting, credit-risk concepts, portfolio theory, conditional formatting, and executive-ready model presentation.

## Modelling Assumptions

This is an educational portfolio project. The portfolio model assumes zero correlation, the credit model estimates expected rather than unexpected loss, and the workbook does not include Monte Carlo simulation. These simplifications are documented so the outputs can be interpreted appropriately.
