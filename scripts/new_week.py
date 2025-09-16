"""Utility to scaffold a new week notebook from the course template."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "templates" / "notebook_template.py"
TARGET_DIR = REPO_ROOT / "notebooks"


def run_jupytext(target_py: Path) -> None:
    """Convert the generated .py file into a notebook if jupytext is available."""
    try:
        subprocess.run(
            ["jupytext", "--to", "notebook", str(target_py)],
            check=True,
        )
    except FileNotFoundError:
        print("jupytext is not installed. Install it or open the .py file directly in JupyterLab.")
    except subprocess.CalledProcessError as exc:
        print("jupytext failed to convert the template:", exc)
        print("You can still open the .py file (paired format) in JupyterLab.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new week notebook from the template.")
    parser.add_argument("week_name", help="Week identifier, e.g. W15_topic_name")
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    target_py = TARGET_DIR / f"{args.week_name}.py"
    if target_py.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {target_py}")

    if not TEMPLATE.exists():
        raise SystemExit(f"Template missing at {TEMPLATE}")

    shutil.copy(TEMPLATE, target_py)
    print(f"Created {target_py.relative_to(REPO_ROOT)}")

    run_jupytext(target_py)

    print("Done! Update the title, objectives, and exercises before sharing with students.")


if __name__ == "__main__":
    main()
