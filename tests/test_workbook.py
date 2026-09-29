import unittest
import zipfile
from xml.etree import ElementTree as ET

from financial_models.cash_flow import forecast_cash_flow
from financial_models.portfolio import portfolio_metrics, workbook_balanced_portfolio


class WorkbookFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with zipfile.ZipFile("Advanced_Financial_Models.xlsx") as workbook:
            cls.cash_flow = workbook.read("xl/worksheets/sheet2.xml").decode("utf-8")
            cls.portfolio = workbook.read("xl/worksheets/sheet4.xml").decode("utf-8")

    def _value(self, xml: str, ref: str) -> float:
        root = ET.fromstring(xml)
        namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for cell in root.findall(".//m:c", namespace):
            if cell.attrib.get("r") != ref:
                continue
            value = cell.find("m:v", namespace)
            self.assertIsNotNone(value)
            return float(value.text)
        self.fail(f"missing cell {ref}")

    def test_cash_flow_formulas_include_revenue_and_a_monthly_discount(self):
        self.assertIn("<f aca=\"false\">B18-B26</f>", self.cash_flow)
        self.assertIn("<f aca=\"false\">B15+B27</f>", self.cash_flow)
        self.assertIn("<f aca=\"false\">B20+B22+B24+B25</f>", self.cash_flow)
        self.assertIn("NPV((1+$B$10)^(1/12)-1,B27:M27)", self.cash_flow)
        self.assertNotIn("B19-B26", self.cash_flow)
        self.assertNotIn("NPV($B$10,C27:M27)+B27", self.cash_flow)

        forecast = forecast_cash_flow()
        self.assertAlmostEqual(self._value(self.cash_flow, "B27"), 27_000.0)
        self.assertAlmostEqual(self._value(self.cash_flow, "C25"), 15_000.0)
        self.assertAlmostEqual(self._value(self.cash_flow, "F25"), 15_000.0)
        self.assertAlmostEqual(self._value(self.cash_flow, "J25"), 15_000.0)
        self.assertAlmostEqual(self._value(self.cash_flow, "B25"), 0.0)
        self.assertAlmostEqual(
            self._value(self.cash_flow, "B32"),
            forecast.npv_of_net_cash_flows,
            places=6,
        )

    def test_portfolio_risk_uses_zero_correlation_volatility(self):
        self.assertIn(
            "SQRT(SUMPRODUCT(D11:D16,D11:D16,C11:C16,C11:C16))",
            self.portfolio,
        )
        self.assertNotIn("SQRT(SUMPRODUCT(D11:D16,C11:C16,C11:C16))", self.portfolio)
        metrics = portfolio_metrics(workbook_balanced_portfolio())
        self.assertAlmostEqual(self._value(self.portfolio, "B21"), metrics.volatility, places=8)
        self.assertAlmostEqual(self._value(self.portfolio, "B22"), metrics.sharpe_ratio, places=8)

    def test_documentation_sheet_matches_the_models(self):
        with zipfile.ZipFile("Advanced_Financial_Models.xlsx") as workbook:
            shared = workbook.read("xl/sharedStrings.xml").decode("utf-8")
            documentation = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
        self.assertIn('r="A26"', documentation)
        self.assertIn("NPV of net cash flows", shared)
        self.assertNotIn("NPV and IRR", shared)
        self.assertIn("zero-correlation volatility", shared)
        self.assertNotIn("weighted standard deviation", shared)
        self.assertIn("equivalent monthly rate", shared)
        self.assertNotIn("IRR", shared)


if __name__ == "__main__":
    unittest.main()
