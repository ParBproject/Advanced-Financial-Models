"""Shared dark theme for matplotlib, Plotly, and the Streamlit dashboard.

Background ``#0B0F14``, slate panels, and emerald ``#10B981``. Chart code
should take colors from this module instead of setting them inline.
"""

from __future__ import annotations

from pathlib import Path

BG = "#0B0F14"
PANEL = "#111827"
INK = "#E5E7EB"
MUTED = "#94A3B8"
GRID = "#1F2937"
EMERALD = "#10B981"
EMERALD_DARK = "#047857"
SKY = "#38BDF8"
AMBER = "#F59E0B"
VIOLET = "#A78BFA"
ROSE = "#FB7185"
WHITE = "#F8FAFC"

SERIES = (EMERALD, SKY, AMBER, VIOLET, ROSE, WHITE)
RATING_COLORS = {"Low": EMERALD, "Medium": AMBER, "High": ROSE}

_FONT_FILES = (
    "/usr/share/fonts/truetype/macos/Inter-Regular.ttf",
    "/usr/share/fonts/truetype/macos/Inter-Medium.ttf",
    "/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf",
    "/usr/share/fonts/truetype/macos/Inter-Bold.ttf",
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
)

_matplotlib_ready = False


def register_fonts() -> tuple[str, str]:
    """Register Inter and JetBrains Mono when the font files are installed."""
    sans = "DejaVu Sans"
    mono = "DejaVu Sans Mono"
    try:
        from matplotlib import font_manager
    except ImportError:  # pragma: no cover - matplotlib is a core dependency
        return sans, mono

    for raw_path in _FONT_FILES:
        path = Path(raw_path)
        if not path.is_file():
            continue
        font_manager.fontManager.addfont(str(path))
        name = font_manager.FontProperties(fname=str(path)).get_name()
        if "mono" in name.lower() or "jetbrains" in name.lower():
            mono = name
        else:
            sans = name
    return sans, mono


def apply_matplotlib_theme() -> None:
    """Apply the shared theme to matplotlib's global style."""
    global _matplotlib_ready
    if _matplotlib_ready:
        return
    import matplotlib.pyplot as plt

    sans, mono = register_fonts()
    plt.rcParams.update(
        {
            "font.family": sans,
            "font.size": 11,
            "text.color": INK,
            "axes.facecolor": PANEL,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 15,
            "axes.titleweight": "medium",
            "axes.labelsize": 11,
            "axes.grid": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.labelcolor": INK,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.unicode_minus": False,
        }
    )
    plt.rcParams["font.monospace"] = [mono, "DejaVu Sans Mono"]
    _matplotlib_ready = True


def plotly_layout(
    title: str,
    *,
    x_title: str,
    y_title: str,
) -> dict[str, object]:
    """Return a Plotly layout dict in the shared theme."""
    axis = {
        "gridcolor": GRID,
        "zerolinecolor": GRID,
        "linecolor": GRID,
        "tickfont": {"color": MUTED, "family": "Inter, sans-serif", "size": 12},
        "title": {"font": {"color": MUTED, "family": "Inter, sans-serif", "size": 12}},
    }
    return {
        "title": {
            "text": title,
            "x": 0.0,
            "xanchor": "left",
            "font": {"family": "Inter, sans-serif", "size": 18, "color": INK},
        },
        "paper_bgcolor": BG,
        "plot_bgcolor": PANEL,
        "font": {"family": "Inter, sans-serif", "color": INK, "size": 13},
        "colorway": list(SERIES),
        "xaxis": {**axis, "title": {**axis["title"], "text": x_title}},
        "yaxis": {**axis, "title": {**axis["title"], "text": y_title}},
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0.0,
        },
        "margin": {"l": 72, "r": 28, "t": 72, "b": 56},
        "hoverlabel": {"bgcolor": PANEL, "font": {"color": INK, "family": "Inter, sans-serif"}},
    }


def streamlit_css() -> str:
    """CSS for the dashboard. Colors match the chart theme."""
    return f"""
    <style>
      html, body, [class*="css"], .stApp {{
        font-family: Inter, "Segoe UI", sans-serif;
        background-color: {BG};
        color: {INK};
      }}
      code, kbd, pre, samp,
      [data-testid="stMetricValue"],
      [data-testid="stCaptionContainer"] {{
        font-family: "JetBrains Mono", "Cascadia Mono", ui-monospace, monospace;
      }}
      [data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid {GRID};
        border-left: 3px solid {EMERALD};
        padding: 0.75rem 0.9rem 0.55rem 0.9rem;
        border-radius: 10px;
      }}
      [data-testid="stMetricLabel"] {{
        color: {MUTED};
      }}
      [data-testid="stMetricValue"] {{
        color: {INK};
      }}
      h1 {{
        font-weight: 600;
        letter-spacing: -0.03em;
      }}
      h2, h3 {{
        font-weight: 560;
        letter-spacing: -0.02em;
      }}
      .afm-kicker {{
        color: {EMERALD};
        font-family: "JetBrains Mono", ui-monospace, monospace;
        font-size: 0.78rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
      }}
      .afm-rule {{
        height: 2px;
        width: 72px;
        background: {EMERALD};
        border-radius: 99px;
        margin: 0.35rem 0 0.8rem 0;
      }}
    </style>
    """
