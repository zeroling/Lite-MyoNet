"""Run the main teacher and five-seed D=2 student plan."""
import sys
from pathlib import Path
from run_experiment_matrix import main
if __name__ == "__main__":
    if "--plan" not in sys.argv:
        sys.argv.extend(["--plan", str(Path(__file__).resolve().parents[1] / "configs/main.yaml")])
    main()
