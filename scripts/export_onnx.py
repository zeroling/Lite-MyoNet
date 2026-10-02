"""Export a self-contained checkpoint to ONNX directly from the source tree."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lite_myonet.export.onnx_export import main


if __name__ == "__main__":
    raise SystemExit(main())
