"""Pack local joints/labels split NPZ files into the native training archive.

Optional group JSON: {"train": [group_id, ...], "val": [...], "test": [...]}.
Without real group metadata physiological grouping is marked unverified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lite_myonet.data.bones import compute_project_bone_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument("--" + split, type=Path, required=True)
    parser.add_argument("--label-map", type=Path, default=ROOT / "configs/label_map.json")
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = json.loads(args.label_map.read_text(encoding="utf-8-sig"))
    if sorted(labels.values()) != list(range(33)):
        raise ValueError("Use the 33-class label map.")
    inverse = {index: name for name, index in labels.items()}
    groups = json.loads(args.groups.read_text(encoding="utf-8-sig")) if args.groups else None
    arrays, records, counts, class_counts, group_splits = {}, {}, {}, {}, {}
    for split in ("train", "val", "test"):
        with np.load(getattr(args, split), allow_pickle=False) as source:
            x = np.ascontiguousarray(source["joints"], dtype=np.float32)
            raw_labels = source["labels"]
        if x.ndim != 4 or x.shape[1:] != (3, 120, 23) or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("Expected finite nonempty [N,3,120,23] joints.")
        if not np.issubdtype(raw_labels.dtype, np.integer) or raw_labels.shape != (len(x),):
            raise ValueError("Expected an integer label vector of length N.")
        if raw_labels.min() < 0 or raw_labels.max() >= 33:
            raise ValueError("Labels must lie in [0,32].")
        y = raw_labels.astype(np.int16)
        ids = [f"{split}:{i}" for i in range(len(x))]
        split_groups = [str(g) for g in groups[split]] if groups else ["unverified:" + i for i in ids]
        if len(split_groups) != len(x):
            raise ValueError("Group vector length differs from split size.")
        for group in split_groups:
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f"Group occurs in multiple splits: {group}")
            group_splits[group] = split
        names = [inverse[int(v)] for v in y]
        source_paths = [f"{split}.npz:joints[{i}]" for i in range(len(x))]
        bone_paths = [f"derived:{name}" for name in source_paths]
        arrays.update({f"{split}_joints": x, f"{split}_bones": compute_project_bone_vectors(x),
                       f"{split}_labels": y, f"{split}_simulation_parameter_milli": np.full(len(x), -1, np.int16),
                       f"{split}_sample_ids": np.array(ids), f"{split}_label_names": np.array(names),
                       f"{split}_split_group_ids": np.array(split_groups),
                       f"{split}_source_joint_paths": np.array(source_paths),
                       f"{split}_source_bone_paths": np.array(bone_paths)})
        records[split] = [dict(sample_id=ids[i], label=int(y[i]), label_name=names[i],
                               joints=source_paths[i], bones=bone_paths[i], split_group_id=split_groups[i],
                               simulation_parameter_milli=None) for i in range(len(x))]
        counts[split] = len(x)
        class_counts[split] = {inverse[int(k)]: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    manifest = dict(schema_version=1, splits=records,
                    group_isolation="provided_group_ids_disjoint" if groups else "not_verified_no_physiological_group_ids")
    manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    metadata = dict(schema_version=1, array_layout="NCTV", array_dtype="float32", sample_shape=[3, 120, 23],
                    total_sequences=sum(counts.values()), split_counts=counts, class_counts=class_counts,
                    label_map=labels, source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    group_isolation=manifest["group_isolation"],
                    simulation_parameter_storage="not provided; -1 is missing, not a physiological value")
    arrays["metadata_json_utf8"] = np.frombuffer(json.dumps(metadata, ensure_ascii=False).encode("utf-8"), np.uint8)
    arrays["source_manifest_json_utf8"] = np.frombuffer(manifest_bytes, np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps(dict(output=str(args.output), counts=counts, grouping=manifest["group_isolation"]), indent=2))


if __name__ == "__main__":
    main()
