"""Smoke test for the static GitHub Pages bundle."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from demo.build_site import (
    BROWSER_PATH_BOOTSTRAP,
    ENTRYPOINT,
    PLOTLY_REQUIREMENT,
    REQUIREMENTS,
    STLITE_VERSION,
    build_site,
    published_dashboard,
)
from financial_models.advanced_portfolio import (
    correlated_portfolio_metrics,
    covariance_from_correlation,
    illustrative_correlation_matrix,
)
from financial_models.cash_flow import CashFlowAssumptions, forecast_cash_flow
from financial_models.portfolio import workbook_balanced_portfolio

REPO = Path(__file__).resolve().parents[1]


class DemoBuildTests(unittest.TestCase):
    def test_published_dashboard_only_adds_the_import_path(self) -> None:
        source = (REPO / "dashboard/app.py").read_text(encoding="utf-8")
        published = published_dashboard(source)
        self.assertEqual(
            published.replace(BROWSER_PATH_BOOTSTRAP, "", 1),
            source,
        )
        ast.parse(published)

    def test_bundle_matches_package_numbers(self) -> None:
        source_cash = forecast_cash_flow(CashFlowAssumptions())
        assets = workbook_balanced_portfolio()
        source_portfolio = correlated_portfolio_metrics(
            assets,
            covariance_from_correlation(
                [asset.volatility for asset in assets],
                illustrative_correlation_matrix(),
            ),
        )

        with TemporaryDirectory() as directory:
            destination = Path(directory)
            mounted = build_site(destination)
            index = (destination / "index.html").read_text(encoding="utf-8")

            self.assertEqual(ENTRYPOINT, "dashboard/app.py")
            self.assertIn("matplotlib", REQUIREMENTS)
            self.assertIn(PLOTLY_REQUIREMENT, REQUIREMENTS)
            self.assertNotIn("yfinance", REQUIREMENTS)
            self.assertNotIn("numpy", REQUIREMENTS)
            self.assertNotIn("pandas", REQUIREMENTS)
            self.assertIn(f"@stlite/browser@{STLITE_VERSION}", index)
            self.assertIn('"#10B981"', index)
            self.assertIn('"#0B0F14"', index)
            self.assertIn('"dashboard/app.py"', index)
            self.assertTrue((destination / ".nojekyll").is_file())

            for relative in mounted:
                self.assertIn(relative, index)
                self.assertTrue((destination / relative).is_file(), relative)

            dashboard_source = (REPO / "dashboard/app.py").read_bytes()
            published = (destination / "dashboard/app.py").read_bytes()
            self.assertNotEqual(published, dashboard_source)
            self.assertIn(b"yfinance", published)
            for path in (REPO / "src/financial_models").rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                relative = path.relative_to(REPO / "src/financial_models").as_posix()
                bundled = destination / "src" / "financial_models" / relative
                self.assertEqual(bundled.read_bytes(), path.read_bytes(), relative)
            self.assertEqual(
                (destination / "data/portfolio_data.csv").read_bytes(),
                (REPO / "data/portfolio_data.csv").read_bytes(),
            )

            ending_cash, expected_loss, hhi, volatility = _bundled_headline_numbers(destination)

        self.assertAlmostEqual(ending_cash, source_cash.periods[-1].ending_cash)
        self.assertAlmostEqual(ending_cash, 485_685, places=0)
        self.assertAlmostEqual(expected_loss, 1_946_680.74, places=2)
        self.assertAlmostEqual(hhi, 0.0013076635, places=8)
        self.assertAlmostEqual(volatility, source_portfolio.volatility)
        self.assertAlmostEqual(volatility, 0.0968, places=4)


def _bundled_headline_numbers(destination: Path) -> tuple[float, float, float, float]:
    """Import the built package, not the checkout on ``sys.path``."""
    saved_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "financial_models" or name.startswith("financial_models.")
    }
    for name in saved_modules:
        del sys.modules[name]
    sys.path.insert(0, str(destination / "src"))
    try:
        from financial_models.advanced_portfolio import (
            correlated_portfolio_metrics,
            covariance_from_correlation,
            illustrative_correlation_matrix,
        )
        from financial_models.cash_flow import CashFlowAssumptions, forecast_cash_flow
        from financial_models.credit_concentration import credit_concentration
        from financial_models.credit_risk import summarize_portfolio
        from financial_models.loan_book import loans_from_credit_book
        from financial_models.portfolio import workbook_balanced_portfolio

        loaded_from = Path(loans_from_credit_book.__code__.co_filename).resolve()
        if not loaded_from.is_relative_to(destination.resolve()):
            raise AssertionError(f"loan book loaded from {loaded_from}")

        cash = forecast_cash_flow(CashFlowAssumptions())
        loans = loans_from_credit_book()
        summary = summarize_portfolio(loans)
        concentration = credit_concentration(loans)
        assets = workbook_balanced_portfolio()
        portfolio = correlated_portfolio_metrics(
            assets,
            covariance_from_correlation(
                [asset.volatility for asset in assets],
                illustrative_correlation_matrix(),
            ),
        )
        return (
            cash.periods[-1].ending_cash,
            summary.total_expected_loss,
            concentration.herfindahl_hirschman_index,
            portfolio.volatility,
        )
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name == "financial_models" or name.startswith("financial_models."):
                del sys.modules[name]
        sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
