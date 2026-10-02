"""Check that incomplete logs never replace the checkpoint as resume authority."""

import csv
import json

from lite_myonet.training.engine import HISTORY_FIELDS, reconcile_history


def _epoch_record(epoch):
    """Return one complete finite synthetic metric record for log-repair tests."""
    return {field: epoch if field == "epoch" else 0.25 for field in HISTORY_FIELDS}


def test_missing_history_restores_only_the_checkpoint_record(tmp_path):
    """Resume a later checkpoint without inventing rows for lost earlier epochs."""
    reconcile_history(tmp_path, 4, _epoch_record(4))
    with (tmp_path / "history.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [4]
    note = json.loads((tmp_path / "history_recovery.json").read_text(encoding="utf-8"))
    assert note["missing_epoch_count"] == 3
    assert note["historical_metrics_fabricated"] is False
    reconcile_history(tmp_path, 4, _epoch_record(4))


def test_corrupt_history_keeps_valid_rows_and_discards_future_rows(tmp_path):
    """Recover usable JSONL rows despite CSV damage and a truncated final line."""
    (tmp_path / "history.csv").write_text("epoch,unknown\nbroken,value\n", encoding="utf-8")
    lines = [json.dumps(_epoch_record(epoch)) for epoch in (1, 3, 8)]
    (tmp_path / "history.jsonl").write_text("\n".join(lines) + "\n{truncated", encoding="utf-8")
    reconcile_history(tmp_path, 4, _epoch_record(4))
    rows = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["epoch"] for row in rows] == [1, 3, 4]
    note = json.loads((tmp_path / "history_recovery.json").read_text(encoding="utf-8"))
    assert note["missing_epoch_count"] == 1
    assert note["invalid_rows_discarded"] >= 2
    assert note["retained_epochs"] == [1, 3, 4]
