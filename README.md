# Lite-MyoNet

[English](README.md) | [简体中文](README.zh-CN.md)

Checkpoint-compatible source for the current MyoFormer teacher, Lite-MyoNet student, and ST-GTP distillation. The source includes training, the experiment matrix, data conversion, ONNX export, and feature-affinity skeleton plots. Datasets, patient records, checkpoints, generated results, and environments are not distributed.

## Install

Use Python 3.10+ and a device-compatible PyTorch build (2.4+). The portable Windows runtime used PyTorch 2.7.1 with CUDA 12.8; install PyTorch separately for your driver, then:

```powershell
python -m pip install -e ".[data,export,visualization,test]"
```

## Models and training

- Teacher: full width, eight heads, three streams with X1/X2/X2 inputs, seed 2026.
- Student: dual joint/bone streams, channels 32/64/128, Fourier D=2, standard three-part skeletal adjacency, learnable bias, dynamic graph, and cosine head.
- Student seeds: 11, 21, 31, 41, 51. Both trainers default to 300 epochs, FP32, batch 16. Optimizer and scheduler values are explicit in `configs/teacher.yaml` and `configs/student.yaml`.
- ST-GTP uses static time averages and dynamic adjacent-time absolute differences, channel normalization, squared distances, row-normalized RBF affinity, and forward teacher-to-student KL. Teacher branch losses are computed separately, weighted 0.5/0.5 for each joint stage, and summed across stages; the bone stream is aligned one-to-one.
- The teacher's joint-temporal *named* branch receives bone vectors under X1/X2/X2. Final features are not inferred from stream names.
- The main configuration saves per-epoch weights as well as selected weights. Resume accepts readable compatible checkpoints without hash-verification gates.

Place your own packed archive at `data/synthetic.npz` (or set `LITE_MYONET_DATA_ARCHIVE`). See [data schemas](docs/DATA_FORMAT.md), [simulation and conversion](docs/DATA_GENERATION.md), and [clinical mapping](docs/CLINICAL_MAPPING.md). Then run:

```powershell
python scripts/train_all.py --device cuda --student-jobs 1
```

The teacher runs serially. Increase student concurrency only after checking memory. Main training fixes D=2. To run the configured degree selection and ablation matrix:

```powershell
python scripts/run_experiment_matrix.py --plan configs/experiment_matrix.yaml --device cuda --student-jobs 1
```

Add `--resume` to the matrix command to continue readable existing runs. `--dry-run` checks the plan without training; it still requires a readable local archive. Standalone training uses `scripts/train_teacher.py --config configs/teacher.yaml` and `scripts/train_student.py --config configs/student.yaml`. For standalone student training, set the checkpoint path to your teacher run; keep its `resolved_config.yaml` next to the checkpoint. Standalone resume takes a checkpoint or run directory: `--resume path/to/run`.

`configs/baseline_optimizers.yaml` records the eight comparison methods' optimization recipes. Third-party model implementations and dependencies are obtained separately; they are not bundled with this source publication.

## ONNX and topology plots

```powershell
python scripts/export_onnx.py --checkpoint checkpoints/best_student.pth --output outputs/student.onnx
```

The current ONNX interface has two FP32 inputs, `joint_coordinates` and `bone_vectors`, both `[B,3,120,23]`, and one `logits` output. Earlier one-input exports and earlier public model checkpoints are not interchangeable with this revision.

[Topology visualization instructions](docs/TOPOLOGY.md) describe how to infer the teacher and two same-seed D=2 students on your own clinical files and color the 22 real skeletal connections. The repository contains no clinical examples. Plot values represent feature affinity, not muscle activation or anatomical causal links.

## Source map

- `src/lite_myonet/models/teacher.py`, `student.py`: actual revision model implementations.
- `src/lite_myonet/training/`: trainers, distillation, checkpoint recovery, teacher-target cache, schedulers, and final BN recalibration support.
- `scripts/run_experiment_matrix.py`, `configs/ablations/`: portable experiment plans.
- `src/lite_myonet/visualization/`: reusable affinity inference and vector drawing.
- `src/lite_myonet/data/`, `integrations/`: input packing, Kinect mapping, and simulator capture conversion.

The public names `MyoFormer` and `LiteMyoNet` alias `MyoFormerTeacher` and `MyoFormerStudent`. The prior publication's alternate model/engine is replaced, not mixed with the checkpoint-compatible revision.

## References and license

Teacher: [MyoFormer: Anatomy-guided transformer for muscle-level pathological gait assessment](https://doi.org/10.1007/s11517-026-03608-x).
Synthetic simulator: [Generative GaitNet](https://github.com/namjohn10/GenerativeGaitNet/tree/eaf3e256925c8b0acc3e95b539535beba105b511).
See `references.bib` and [third-party notices](THIRD_PARTY_NOTICES.md). Project-owned code retains the existing MIT license. Upstream software, datasets and weights retain their respective terms.
