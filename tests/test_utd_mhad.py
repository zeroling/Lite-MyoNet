"""Tests for UTD-MHAD skeleton conversion, splits, and archive safety."""

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lite_myonet.data.utd_mhad import (  # noqa: E402
    UTDMHADDataError,
    UTDRecord,
    _audit_subject_split,
    _write_npz,
    convert_d_skel,
    create_split_assignments,
    parse_skeleton_filename,
    safe_extract_zip,
)
from lite_myonet.models.skeleton import load_skeleton  # noqa: E402


class UTDMHADTests(unittest.TestCase):
    """Verify the source topology contract and deterministic preprocessing."""

    def test_official_joint_order_config_has_hip_center_root(self):
        """Check that the UTD-MHAD skeleton uses hip center as its root."""
        skeleton = load_skeleton(PROJECT_ROOT / "configs" / "skeletons" / "utd_mhad20.yaml")
        self.assertEqual(skeleton.point_count, 20)
        self.assertEqual(skeleton.root_index, 3)
        self.assertEqual(skeleton.node_ids[:5], ("head", "shoulder_center", "spine", "hip_center", "left_shoulder"))
        self.assertEqual(len(skeleton.bone_vector_edges), 19)

    def test_filename_parser_uses_one_based_official_ids(self):
        """Validate one-based action, subject, and trial identifiers."""
        self.assertEqual(parse_skeleton_filename("a27_s8_t4_skeleton.mat"), (27, 8, 4))
        with self.assertRaises(UTDMHADDataError):
            parse_skeleton_filename("a28_s8_t4_skeleton.mat")
        with self.assertRaises(UTDMHADDataError):
            parse_skeleton_filename("a1_s1_t5_skeleton.mat")

    def test_conversion_centers_hip_and_linearly_resamples_without_scaling(self):
        """Check hip centering and resampling preserve coordinate scale."""
        source = np.zeros((20, 3, 2), dtype=np.float64)
        source[3, :, 0] = (0.1, 0.2, 0.3)
        source[3, :, 1] = (0.2, 0.4, 0.6)
        source[0, :, 0] = (0.4, 0.6, 0.9)
        source[0, :, 1] = (0.6, 1.0, 1.5)

        output = convert_d_skel(source, target_frames=120, root_index=3)

        self.assertEqual(output.shape, (3, 120, 20))
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_allclose(output[:, :, 3], 0.0, atol=1e-7)
        np.testing.assert_allclose(output[:, 0, 0], (0.3, 0.4, 0.6), atol=1e-7)
        np.testing.assert_allclose(output[:, -1, 0], (0.4, 0.6, 0.9), atol=1e-7)
        midpoint = 60.0 / 119.0
        expected_midpoint = np.asarray((0.3, 0.4, 0.6)) + midpoint * np.asarray(
            (0.1, 0.2, 0.3)
        )
        np.testing.assert_allclose(output[:, 60, 0], expected_midpoint, atol=1e-6)

    def test_split_is_deterministic_stratified_and_subject_disjoint(self):
        """Verify deterministic stratification and subject-disjoint splits."""
        placeholder = np.empty((3, 120, 20), dtype=np.float32)
        records = []
        labels = []
        for action_id, label in enumerate(range(27), start=1):
            for subject_id in (1, 3, 5, 7, 2, 4, 6, 8):
                for trial_id in range(1, 3):
                    records.append(
                        UTDRecord(
                            action_id,
                            subject_id,
                            trial_id,
                            f"a{action_id}_s{subject_id}_t{trial_id}_skeleton.mat",
                            50,
                            placeholder,
                        )
                    )
                    labels.append(label)
        labels = np.asarray(labels, dtype=np.int64)

        first = create_split_assignments(records, labels, random_state=2026)
        second = create_split_assignments(records, labels, random_state=2026)

        self.assertEqual(first, second)
        self.assertEqual({name: first.count(name) for name in ("train", "val", "test")}, {
            "train": 172,
            "val": 44,
            "test": 216,
        })
        for record, split in zip(records, first):
            if record.subject_id % 2 == 0:
                self.assertEqual(split, "test")
            else:
                self.assertIn(split, ("train", "val"))
        for label in range(27):
            indices = [index for index, value in enumerate(labels) if value == label]
            self.assertEqual({first[index] for index in indices}, {"train", "val", "test"})
        audit = _audit_subject_split(records, first)
        self.assertTrue(audit["test_subject_disjoint_from_development"])
        self.assertEqual(audit["test_train_subject_overlap"], [])
        self.assertEqual(audit["test_val_subject_overlap"], [])
        self.assertTrue(audit["validation_is_sample_level_within_development"])

    def test_npz_contains_only_nctv_joints_and_integer_labels(self):
        """Confirm NPZ output contains only NCTV joints and integer labels."""
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "train.npz"
            joints = np.zeros((2, 3, 120, 20), dtype=np.float64)
            labels = np.asarray([0, 26], dtype=np.int32)
            _write_npz(destination, joints, labels, overwrite=False)
            with np.load(destination, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"joints", "labels"})
                self.assertEqual(archive["joints"].shape, (2, 3, 120, 20))
                self.assertEqual(archive["joints"].dtype, np.float32)
                self.assertEqual(archive["labels"].dtype, np.int64)
                self.assertEqual(archive["labels"].tolist(), [0, 26])

    def test_safe_extraction_rejects_path_traversal_before_writing(self):
        """Reject archive paths that escape the extraction destination."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "malicious.zip"
            destination = root / "extracted"
            escaped = root / "escape.txt"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "not allowed")

            with self.assertRaisesRegex(UTDMHADDataError, "Unsafe ZIP member"):
                safe_extract_zip(archive_path, destination)

            self.assertFalse(escaped.exists())


if __name__ == "__main__":
    unittest.main()
