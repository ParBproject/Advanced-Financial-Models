# Contributing

This repository combines an Excel decision-support workbook with a reproducible Python analytics layer.

## Development workflow

1. Create a focused feature or fix branch from `main`.
2. Keep model assumptions explicit and document any change to them.
3. Add or update tests for every calculation change.
4. Run:
   ```bash
   python -m pip install -e ".[dev]"
   ruff check src tests examples
   python -m unittest discover -s tests -v
   ```
5. Open a pull request that explains the financial behavior being changed and why.

## Modeling rules

- Do not silently change the Excel workbook's documented baseline assumptions.
- Validate percentages, weights, credit scores, horizons, and cash inputs at model boundaries.
- Keep forward-looking assumptions distinct from realized/historical data.
- Prefer small pure functions and dataclasses so calculations are easy to audit.
- Tests should include at least one numerical example with a hand-checkable expected result.

## Scope

The Excel workbook remains a first-class artifact. The Python package exists to make the same model logic reproducible, testable, and easier to extend with professional analytics such as covariance-aware risk, stress testing, and simulation.
