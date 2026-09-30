import unittest
from importlib.util import find_spec
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

    @unittest.skipUnless(find_spec("plotly") is not None, "plotly is not installed")
    def test_plotly_layout_uses_the_emerald_accent(self):
        layout = plotly_layout("Cash", x_title="Month", y_title="US dollars")
        self.assertEqual(layout["paper_bgcolor"], BG)
        self.assertEqual(layout["colorway"][0], EMERALD)
        figure = cash_flow_plotly(forecast_cash_flow())
        self.assertEqual(figure.layout.paper_bgcolor, BG)

    @unittest.skipUnless(find_spec("plotly") is not None, "plotly is not installed")
    def test_wealth_chart_keeps_series_past_the_palette_length(self):
        from financial_models.charts import wealth_plotly

        series = {f"S{i}": [1.0, 1.01] for i in range(8)}
        figure = wealth_plotly([0, 1], series, title="Growth of $1")
        self.assertEqual(len(figure.data), 8)
        self.assertEqual([trace.name for trace in figure.data], list(series))


if __name__ == "__main__":
    unittest.main()
