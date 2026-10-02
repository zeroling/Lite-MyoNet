"""Convert a Generative GaitNet position capture to Lite-MyoNet NCTV arrays."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lite_myonet.data.generative_gaitnet import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
