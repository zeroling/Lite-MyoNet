"""Build the deterministic target schedule for external Generative GaitNet rollouts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lite_myonet.data.gaitnet_conditions import main


if __name__ == "__main__":
    raise SystemExit(main())
