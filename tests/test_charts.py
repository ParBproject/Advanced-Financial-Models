import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib.colors import to_rgba

from financial_models.cash_flow import forecast_cash_flow
from financial_models.charts import cash_flow_matplotlib, cash_flow_plotly, save_figure
from financial_models.theme import BG, EMERALD, plotly_layout


class ChartThemeTests(unittest.TestCase):
    def test_matplotlib_figure_uses_the_shared_background(self):
        figure = cash_flow_matplotlib(forecast_cash_flow())
        self.assertEqual(tuple(figure.get_facecolor()), to_rgba(BG))
        with TemporaryDirectory() as directory:
            path = save_figure(figure, Path(directory) / "cash-flow.png")
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_plotly_layout_uses_the_emerald_accent(self):
        layout = plotly_layout("Cash", x_title="Month", y_title="US dollars")
        self.assertEqual(layout["paper_bgcolor"], BG)
        self.assertEqual(layout["colorway"][0], EMERALD)
        figure = cash_flow_plotly(forecast_cash_flow())
        self.assertEqual(figure.layout.paper_bgcolor, BG)


if __name__ == "__main__":
    unittest.main()
