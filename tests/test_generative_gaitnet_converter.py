"""Unit tests for Generative GaitNet capture conversion."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lite_myonet.data.generative_gaitnet import (  # noqa: E402
    ConversionError,
    convert_gaitnet_archive,
    convert_positions,
    load_conversion_config,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "conversion" / "generative_gaitnet.yaml"


class GenerativeGaitNetConverterTests(unittest.TestCase):
    """Check format normalization and metadata-safe output behavior."""

    def setUp(self):
        """Prepare a fixture with named bodies and timestamps."""
        self.config = load_conversion_config(CONFIG_PATH)
        self.frame_count = 120
        self.body_names = list(reversed(self.config["nodes"]))
        self.timestamps = np.arange(self.frame_count, dtype=np.float64) / 30.0
        self.positions = np.zeros(
            (1, self.frame_count, len(self.body_names), 3), dtype=np.float64
        )
        for input_index, body_name in enumerate(self.body_names):
            target_index = self.config["nodes"].index(body_name)
            self.positions[0, :, input_index, 0] = target_index + 1.0

    def test_conversion_reorders_named_bodies_and_returns_nctv_float32(self):
        """Verify body reordering and the resulting NCTV float32 layout."""
        joints, labels, label_names, metadata = convert_positions(
            self.positions,
            self.timestamps,
            self.body_names,
            ["normal"],
            [{"sample_id": "sample-001", "window_start_seconds": 0.0}],
            self.config,
        )

        self.assertEqual(joints.shape, (1, 3, 120, 23))
        self.assertEqual(joints.dtype, np.float32)
        self.assertEqual(labels.tolist(), [0])
        self.assertEqual(label_names.tolist(), ["normal"])
        self.assertTrue(np.all(joints[0, 0, :, 0] == 1.0))
        self.assertEqual(metadata["array_layout"], "NCTV")
        self.assertFalse(metadata["bones_included"])

    def test_missing_window_start_is_rejected(self):
        """Reject conversion records without an explicit window start."""
        with self.assertRaisesRegex(ConversionError, "window_start_seconds"):
            convert_positions(
                self.positions,
                self.timestamps,
                self.body_names,
                ["normal"],
                [{}],
                self.config,
            )

    def test_missing_required_body_is_rejected(self):
        """Reject inputs that omit a required body node."""
        with self.assertRaisesRegex(ConversionError, "missing required bodies"):
            convert_positions(
                self.positions[:, :, :-1, :],
                self.timestamps,
                self.body_names[:-1],
                ["normal"],
                [{"window_start_seconds": 0.0}],
                self.config,
            )

    def test_npz_conversion_writes_only_joints_labels_and_metadata(self):
        """Verify NPZ output contains only joints, labels, and metadata."""
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "capture.npz"
            output_path = Path(directory) / "converted.npz"
            metadata = json.dumps(
                {"samples": [{"sample_id": "sample-001", "window_start_seconds": 0.0}]}
            ).encode("utf-8")
            np.savez_compressed(
                input_path,
                positions=self.positions,
                timestamps=self.timestamps,
                body_names=np.asarray(self.body_names),
                label_names=np.asarray(["normal"]),
                metadata_json_utf8=np.frombuffer(metadata, dtype=np.uint8),
            )

            report = convert_gaitnet_archive(
                input_path,
                output_path,
                CONFIG_PATH,
            )

            self.assertEqual(report["array_layout"], "NCTV")
            self.assertEqual(report["joints_shape"], [1, 3, 120, 23])
            self.assertEqual(len(report["outputs"]), 2)
            with np.load(output_path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"joints", "labels"})
                self.assertEqual(archive["joints"].dtype, np.float32)
                self.assertEqual(archive["labels"].tolist(), [0])
                self.assertEqual(archive["labels"].dtype, np.int64)
            metadata_path = output_path.with_name("converted.metadata.json")
            self.assertTrue(metadata_path.is_file())
            result_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(result_metadata["samples"][0]["sample_id"], "sample-001")

    def test_npy_conversion_writes_labels_and_metadata_sidecars(self):
        """Verify NPY output writes label and metadata sidecars."""
        from lite_myonet.data.generative_gaitnet import write_conversion_output

        joints, labels, label_names, metadata = convert_positions(
            self.positions,
            self.timestamps,
            self.body_names,
            ["normal"],
            [{"window_start_seconds": 0.0}],
            self.config,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "converted.npy"
            outputs = write_conversion_output(
                output_path,
                joints,
                labels,
                label_names,
                metadata,
                expected_sample_shape=(3, 120, 23),
            )
            self.assertEqual(len(outputs), 3)
            self.assertEqual(np.load(output_path, allow_pickle=False).shape, (1, 3, 120, 23))
            labels_path = output_path.with_name("converted.labels.npy")
            self.assertEqual(np.load(labels_path, allow_pickle=False).dtype, np.int64)
            metadata_path = output_path.with_name("converted.metadata.json")
            self.assertTrue(metadata_path.is_file())


if __name__ == "__main__":
    unittest.main()
