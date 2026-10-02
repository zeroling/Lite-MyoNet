"""Run the standalone public student trainer from a source checkout."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lite_myonet.training.student_trainer import main


if __name__ == "__main__":
    raise SystemExit(main())
