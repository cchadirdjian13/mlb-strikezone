# ABOUTME: Repo-root entry point for Streamlit hosting, which runs a script rather
# ABOUTME: than an installed package, so src/ has to be put on the path first.
import sys
from pathlib import Path

# Hosted Streamlit runs this file directly, which puts its own directory on the
# path and not src/, so `import mlb_strikezone` would fail without this. Locally
# the package is installed and this line is a harmless duplicate.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mlb_strikezone.app import main  # noqa: E402

main()
