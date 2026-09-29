import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - dashboard extra is optional for local dev
    AppTest = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(AppTest is not None, "dashboard extra is not installed")
class DashboardSmokeTests(unittest.TestCase):
    def test_app_renders_baseline_metrics(self):
        app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=90)
        app.run()
        if app.exception:
            self.fail(app.exception[0].message)

        labels = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(labels["12-month ending cash"], "$485,685")
        self.assertEqual(labels["Credit expected loss"], "$42,525")
        self.assertEqual(labels["Ending cash"], "$485,685")
        self.assertEqual(labels["Cash-flow NPV"], "$410,751")
        self.assertEqual(labels["Minimum cash"], "$77,000")

    def test_month_one_revenue_input_changes_liquidity(self):
        app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=90)
        app.run()
        revenue = next(item for item in app.number_input if item.label == "Month 1 revenue")
        revenue.set_value(0.0)
        app.run()
        if app.exception:
            self.fail(app.exception[0].message)

        labels = {metric.label: metric.value for metric in app.metric}
        self.assertNotEqual(labels["Ending cash"], "$485,685")
        self.assertEqual(labels["12-month ending cash"], "$485,685")


if __name__ == "__main__":
    unittest.main()
