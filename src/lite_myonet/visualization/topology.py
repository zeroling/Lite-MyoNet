"""Infer fixed teacher/D=2 students on local clinical windows and draw affinities.

The user supplies all checkpoints, coordinates, and optional descriptions.
No training, BN recalibration, class predictions, or diagnostic metrics.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.bones import compute_project_bone_vectors
from ..models.teacher import MyoFormerTeacher
from ..models.student import MyoFormerStudent
from ..models.topology import bones as EDGES
from ..training.distillation import FEATURE_MATCHES, TEACHER_FEATURE_NAMES
from ..training.student_trainer import rename_student_features
from .adjacent_plot import analyze, render, table, write


def affinity(feature, dynamic=False):
    descriptor = ((feature[:, :, 1:] - feature[:, :, :-1]).abs().mean(2)
                  if dynamic else feature.mean(2))
    normalized = F.normalize(descriptor, p=2, dim=1, eps=1e-8).transpose(1, 2)
    distance = torch.cdist(normalized, normalized, p=2).square()
    return F.softmax(-distance / (1.0 + 1e-8), dim=-1), distance


def restore(path, teacher=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if teacher:
        cfg = ck["model_config"]
        allowed = ("num_points", "in_channels", "dropout", "use_spatial_attention", "use_symmetry_attention",
                   "use_scale_attention", "use_joint_spatial_stream", "use_joint_temporal_stream",
                   "use_bone_vector_stream", "width_ratio", "input_scheme", "skeleton_edges", "symmetry_pairs", "root_index")
        kw = {k: cfg[k] for k in allowed if k in cfg}
        kw.update(spatial_attention_config=cfg.get("spatial_attention", {}),
                  symmetry_attention_config=cfg.get("symmetry_attention", {}),
                  scale_attention_config=cfg.get("scale_attention", {}), head_config=cfg.get("head", {}))
        model = MyoFormerTeacher(num_classes=33, **kw)
        if not all(cfg.get(k, True) for k in ("use_joint_spatial_stream", "use_joint_temporal_stream", "use_bone_vector_stream")):
            raise ValueError("Topology route requires all three teacher streams.")
        model.load_state_dict(ck["model_state_dict"], strict=True)
    else:
        cfg = ck["student_config"]
        if cfg["temporal_basis"] != "fourier" or cfg["fourier_degree"] != 2:
            raise ValueError("Both student checkpoints must use Fourier D=2.")
        model = MyoFormerStudent(**cfg)
        model.load_state_dict(ck["student_state_dict"], strict=True)
    info = dict(checkpoint=str(Path(path).resolve()), seed=int(ck["seed"]),
                checkpoint_epoch=int(ck["epoch"]), model_config=cfg)
    return model.float().eval(), info


def infer(args):
    out = args.output.resolve()
    for name in ("data", "patients"):
        (out / name).mkdir(parents=True, exist_ok=True)
    joints, windows = [], []
    for directory in sorted(args.data_root.glob("Person_*")):
        if not directory.is_dir():
            continue
        patient = int(directory.name.split("_")[1])
        ordinal = 0
        for path in sorted(directory.glob("*.npy")):
            if path.name.startswith("skeleton23_"):
                continue
            array = np.load(path, allow_pickle=False).astype(np.float32)
            if array.shape != (3, 120, 23) or not np.isfinite(array).all():
                print(f"Skipped invalid window: {path}", flush=True)
                continue
            joints.append(array)
            windows.append(dict(patient=patient, window_index=len(windows), patient_window_index=ordinal,
                                main_window=ordinal == 0, source_file=str(path.resolve()),
                                start_frame_zero_based=0, end_frame_inclusive_zero_based=119,
                                display_frame_zero_based=60))
            ordinal += 1
    if not windows:
        raise ValueError("No finite [3,120,23] windows found under Person_XX directories.")
    joints = np.stack(joints)
    bones = compute_project_bone_vectors(joints)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    features, matrices, models = {}, {}, {}
    matches = [m for m in FEATURE_MATCHES if m.student == "joint.stage3"]
    if [(m.teacher, m.weight) for m in matches] != [("joint_spatial.stage3", .5), ("joint_temporal.stage3", .5)]:
        raise ValueError("Unexpected final-joint distillation route.")
    for key, path in (("teacher", args.teacher), ("logit_kd", args.logit_student), ("full_st_gtp", args.full_student)):
        model, models[key] = restore(path, teacher=key == "teacher")
        model.to(device)
        kept = {}
        with torch.inference_mode():
            for start in range(len(windows)):
                j = torch.from_numpy(joints[start:start + 1]).to(device)
                b = torch.from_numpy(bones[start:start + 1]).to(device)
                _, returned = model(j, b, return_features=True)
                named = dict(zip(TEACHER_FEATURE_NAMES, returned)) if key == "teacher" else rename_student_features(returned)
                wanted = [m.teacher for m in matches] if key == "teacher" else ["joint.stage3"]
                for name in wanted:
                    kept.setdefault(name, []).append(named[name].cpu())
            for name, parts in kept.items():
                f = torch.cat(parts)
                alias = name.split(".")[0] if key == "teacher" else key
                features[alias] = f.numpy()
                for dynamic, typ in ((False, "static"), (True, "dynamic")):
                    p, d = affinity(f, dynamic)
                    matrices[f"{alias}_{typ}"] = p.numpy()
                    matrices[f"{alias}_{typ}_squared_distances"] = d.numpy()
        del model, kept, named, returned
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if models["logit_kd"]["seed"] != models["full_st_gtp"]["seed"]:
        raise ValueError("Student checkpoints must share a seed.")
    for typ in ("static", "dynamic"):
        matrices[f"teacher_{typ}"] = .5 * matrices[f"joint_spatial_{typ}"] + .5 * matrices[f"joint_temporal_{typ}"]
    np.savez_compressed(out / "data/affinity_matrices.npz", **matrices)
    np.savez_compressed(out / "data/last_joint_features.npz", **features)
    np.savez_compressed(out / "data/display_coordinates.npz", coordinates=joints[:, :, 60, :].transpose(0, 2, 1))
    diagnoses = json.loads(args.descriptions.read_text(encoding="utf-8")) if args.descriptions else {}
    result = dict(models=models, windows=windows, bones=EDGES,
                  diagnoses={str(w["patient"]): diagnoses.get(str(w["patient"]), "Clinical skeleton") for w in windows},
                  formula=dict(rho=0., positive_bandwidth=1., epsilon=1e-8,
                               static="mean_t(F)", dynamic="mean_t(abs(F[t+1]-F[t]))",
                               relation="softmax(-squared_L2_distance/(exp(rho)+eps)); channel normalization",
                               teacher_branch_inputs={"joint_spatial": "X1", "joint_temporal": models['teacher']['model_config'].get('input_scheme', 'X1/X2/X2').split('/')[1]}))
    table(out / "windows.csv", windows)
    write(out / "model_manifest.json", models)
    write(out / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--logit-student", type=Path)
    parser.add_argument("--full-student", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--descriptions", type=Path, help="Optional private patient-ID to English description JSON.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        result = json.loads((args.output / "results.json").read_text(encoding="utf-8"))
    else:
        if any(getattr(args, name) is None for name in ("teacher", "logit_student", "full_student", "data_root")):
            parser.error("Provide --teacher, --logit-student, --full-student and --data-root.")
        result = infer(args)
    weights = analyze(args.output, result)
    render(args.output, result, weights)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
