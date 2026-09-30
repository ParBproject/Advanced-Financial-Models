"""Build the static stlite site published to GitHub Pages.

The site mounts the repository dashboard and ``financial_models`` package in
the browser. NumPy, pandas, and matplotlib come from the Pyodide build that
stlite ships. Plotly is installed from PyPI. ``yfinance`` is not installed:
it cannot run under Pyodide, and the Market risk tab already falls back to
the worked example and CSV upload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STLITE_VERSION = "1.9.2"
PLOTLY_REQUIREMENT = "plotly==5.24.1"
REQUIREMENTS = ("matplotlib", PLOTLY_REQUIREMENT)
ENTRYPOINT = "dashboard/app.py"
DEMO_URL = "https://parbproject.github.io/Advanced-Financial-Models/"

# Inserted into the published dashboard only, immediately after its future import.
BROWSER_PATH_BOOTSTRAP = """
# Published by demo/build_site.py so this file can import the mounted package.
import sys
from pathlib import Path

_AFM_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _AFM_SRC not in sys.path:
    sys.path.insert(0, _AFM_SRC)
"""

_FUTURE_IMPORT = "from __future__ import annotations\n"


def package_python_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Return the package modules that the dashboard imports."""
    package = repo_root / "src" / "financial_models"
    return sorted(
        path
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    )


def published_dashboard(source: str) -> str:
    """Return ``dashboard/app.py`` with the browser path bootstrap inserted."""
    if _FUTURE_IMPORT not in source:
        raise ValueError("dashboard/app.py must keep its annotations future import")
    if source.count(_FUTURE_IMPORT) != 1:
        raise ValueError("dashboard/app.py must contain one annotations future import")
    return source.replace(_FUTURE_IMPORT, _FUTURE_IMPORT + BROWSER_PATH_BOOTSTRAP, 1)


def _site_files(repo_root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    dashboard = (repo_root / "dashboard" / "app.py").read_text(encoding="utf-8")
    files[ENTRYPOINT] = published_dashboard(dashboard).encode("utf-8")
    for path in package_python_files(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        files[relative] = path.read_bytes()
    credit_book = repo_root / "data" / "portfolio_data.csv"
    files["data/portfolio_data.csv"] = credit_book.read_bytes()
    return files


def _index_html(file_names: list[str]) -> str:
    file_map = {name: {"url": name} for name in file_names}
    files_json = json.dumps(file_map, indent=2)
    requirements_json = json.dumps(list(REQUIREMENTS))
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Advanced Financial Models</title>
    <meta
      name="description"
      content="Cash-flow, credit, and portfolio dashboard running in the browser."
    />
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build/stlite.css"
    />
    <style>
      html, body {{
        margin: 0;
        background: #0B0F14;
        color: #E5E7EB;
        font-family: Inter, "Segoe UI", sans-serif;
      }}
      #root {{
        min-height: 100vh;
      }}
      .boot {{
        max-width: 40rem;
        margin: 0 auto;
        padding: 18vh 1.5rem 2rem;
      }}
      .kicker {{
        margin: 0 0 0.4rem;
        color: #10B981;
        font-family: "JetBrains Mono", ui-monospace, monospace;
        font-size: 0.78rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }}
      h1 {{
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.03em;
      }}
      .rule {{
        height: 2px;
        width: 72px;
        margin: 0.9rem 0 1rem;
        background: #10B981;
        border-radius: 99px;
      }}
      p {{
        color: #94A3B8;
        line-height: 1.5;
      }}
    </style>
  </head>
  <body>
    <div id="root">
      <div class="boot">
        <p class="kicker">Liquidity · Credit · Portfolio risk</p>
        <h1>Advanced Financial Models</h1>
        <div class="rule"></div>
        <p>Loading the dashboard in this browser. No sign-in and no server.</p>
      </div>
    </div>
    <noscript>This demo needs JavaScript to run the Python dashboard in the browser.</noscript>
    <script type="module">
      import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build/stlite.js";

      mount(
        {{
          entrypoint: {json.dumps(ENTRYPOINT)},
          requirements: {requirements_json},
          files: {files_json},
          streamlitConfig: {{
            "theme.base": "dark",
            "theme.primaryColor": "#10B981",
            "theme.backgroundColor": "#0B0F14",
            "theme.secondaryBackgroundColor": "#111827",
            "theme.textColor": "#E5E7EB",
            "client.toolbarMode": "viewer",
          }},
        }},
        document.getElementById("root"),
      );
    </script>
  </body>
</html>
"""


def build_site(destination: Path, repo_root: Path = REPO_ROOT) -> list[str]:
    """Write the static site and return the mounted file paths."""
    files = _site_files(repo_root)
    destination.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (destination / ".nojekyll").write_text("", encoding="utf-8")
    (destination / "index.html").write_text(_index_html(sorted(files)), encoding="utf-8")
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GitHub Pages stlite demo.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "site",
        help="Directory to write the static site into.",
    )
    args = parser.parse_args()
    mounted = build_site(args.output)
    print(f"Wrote {args.output} ({len(mounted)} mounted files)")


if __name__ == "__main__":
    main()
