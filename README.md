# Lite-MyoNet

[English](README.md) | [简体中文](README.zh-CN.md)

Lite-MyoNet is a teacher–student framework for skeleton-sequence classification. MyoFormer is the teacher and Lite-MyoNet is the student. The project provides model training, public-dataset preparation, data conversion, test-set evaluation, and ONNX export.

The teacher follows the published paper [MyoFormer: Anatomy-guided transformer for muscle-level pathological gait assessment](https://doi.org/10.1007/s11517-026-03608-x). The student uses lightweight spatiotemporal encoding, Fourier layers, and teacher–student distillation. The default fourier_degree: 2 is a configuration value and can be changed in YAML.

## Installation

Python 3.10 or later is required. Install a PyTorch build suitable for your device, then install the project and optional feature dependencies from the repository root:

~~~powershell
python -m pip install -e ".[data,export,test]"
~~~

## Input data

Training data consists of joint-position arrays in NCTV order: sample, coordinate channel, time frame, and skeleton node. The shape is [N,C,T,V], and the dtype is float32.

An NPZ file must contain exactly the joints and labels arrays. For NPY input, provide a .labels.npy file with the same prefix as the joint file. labels is an integer vector of length N, with class IDs numbered consecutively from 0.

~~~python
import numpy as np

np.savez_compressed(
    "train.npz",
    joints=joints.astype(np.float32),
    labels=labels.astype(np.int64),
)
~~~

The node count, frame count, and coordinate channels must match the skeleton YAML. The default UTD-MHAD input shape is [N,3,120,20]; the 23-node gait configuration uses [N,3,120,23]. The model derives bone vectors from joint positions using bone_vector_edges.

To use a custom skeleton, set point_count, nodes in data order, root_index, model_graph_edges, bone_vector_edges, symmetry_pairs, channels, and frame_count. Configure class names and IDs in the corresponding JSON label map.

## UTD-MHAD: preparation and training

This project uses the [official UTD-MHAD skeleton data](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html), with 27 action classes and a 20-node skeleton. The preparation script downloads the official skeleton and sample-code archives, converts the data, and creates train, validation, and test splits:

~~~powershell
python scripts/prepare_utd_mhad.py
~~~

The default output directory is data/utd_mhad/processed/. The cross-subject split contains 344 training, 87 validation, and 430 test sequences. Sequences are hip-centered and interpolated to 120 frames.

The end-to-end command runs one teacher followed by the five student seeds in the configuration:

~~~powershell
python scripts/train_all.py --config configs/experiments/utd_mhad.yaml --device cuda
~~~

Resume unfinished stages after an interruption with:

~~~powershell
python scripts/train_all.py --config configs/experiments/utd_mhad.yaml --device cuda --resume
~~~

The teacher and student can also be trained separately, in that order:

~~~powershell
python scripts/train_teacher.py --config configs/utd_mhad/teacher.yaml --device cuda
python scripts/train_student.py --config configs/utd_mhad/student.yaml --device cuda
~~~

The student configuration uses the best teacher checkpoint from the same output directory by default. Set the device to cuda or cpu; the default output directory is runs/utd_mhad/. Checkpoints are selected by validation Macro-F1, and the test set is evaluated after training.

## Using your own data

Prepare train, val, and test splits in the format described above. Copy configs/teacher.yaml and configs/student.yaml, then update the data paths, skeleton YAML, label map, teacher checkpoint, and output directory in the copies.

Update teacher_config and student_config in configs/train_all.yaml to point to those copies, and set an independent output_root. Enter each path relative to its YAML file.

~~~powershell
python scripts/train_all.py --config configs/train_all.yaml --device cuda
~~~

For separate training, use the corresponding configuration with train_teacher.py, then run train_student.py. Set one or more training seeds in the student configuration’s seeds field.

## Synthetic gait data with Generative GaitNet

Synthetic data is generated with the musculoskeletal simulation in Generative GaitNet (generative gait network). This project pins the upstream commit [eaf3e256925c8b0acc3e95b539535beba105b511](https://github.com/namjohn10/GenerativeGaitNet/tree/eaf3e256925c8b0acc3e95b539535beba105b511).

Each pathological condition changes the maximum isometric muscle-force ratio of one target muscle group: r = 1 - w, where w = 0.05–0.40. Labels cover 16 groups on each side (32 pathological classes) plus one normal class. The maximum-force ratio of every other muscle group remains 1.0.

The current condition configuration selects 11,200 pathological conditions from r=0.600–0.950 at 0.001 resolution. Normal conditions cover r=0.951–1.000 and total 1,600. The 12,800 planned conditions are allocated across training, validation, and test at a target ratio of 2:1:1.

This is a simulation schedule defined by the public configuration; it is not a set of generated trajectories.

### Build the upstream environment and create conditions

Deploy the simulator and walking policy in a separate Generative GaitNet environment. Build the pinned commit:

~~~bash
git clone https://github.com/namjohn10/GenerativeGaitNet.git
cd GenerativeGaitNet
git checkout eaf3e256925c8b0acc3e95b539535beba105b511
./install.sh
./pc_build.sh
cd build
make -j32
~~~

The upstream instructions specify Python 3.6 and Ray 1.8.0. Keep this environment separate from the classifier-training environment. The upstream policy entry point is imgui_render/imgui_render; follow the README for this pinned commit to provide the network paths for a single network or a four-network cascade.

From the Lite-MyoNet repository root, create the condition table:

~~~powershell
python scripts/create_gaitnet_conditions.py --config configs/conversion/generative_gaitnet_conditions.yaml --output data/gaitnet/conditions.json
~~~

Each JSON entry records condition_id, label_name, target_force_scale, split, and simulation_seed. When running the policy for a condition, change only the exact muscle unit mapped to that label.

Example entry point for a single policy; replace the network path with the policy matching this upstream version:

~~~bash
./imgui_render/imgui_render path/to/network
~~~

### Apply force conditions and run the gait simulation

Begin each trajectory from a baseline in which all non-target muscle groups have a maximum-force ratio of 1.0. The upstream python/ray_env.py reset() wrapper resamples and writes current_paramstate. When using this wrapper, apply the target ratio again after every reset, or disable its automatic parameter sampling. The lower-level Environment::Reset does not change the force ratio automatically.

Use GetParamState to read the baseline parameter vector. Based on the loaded metadata, locate the maximum-force parameter segment, which follows the muscle-length segment; set the target group’s ratio to r, apply the vector with SetParamState, and read it back to confirm. Assign ratios directly without further normalization. Map left and right labels to their corresponding units; for example, the lateral gastrocnemius head labels must map to L_Gastrocnemius_Lateral_Head and R_Gastrocnemius_Lateral_Head respectively. The broader group name group L_Gastrocnemius may match both medial and lateral heads.

metadata.txt uses a 30 Hz control rate and 480 Hz physics rate. metadata_fullbody.txt uses a 30 Hz control rate and 600 Hz physics rate. Record the metadata file, policy version, and control and physics rates used for each run.

Run the same control policy for each condition on a 20 m straight walk. Following the MyoFormer paper, use actual displacement to identify the central 10 m steady-state region from 5 m to 15 m. This 10 m defines the spatial sampling region; its traversal time depends on each trajectory’s actual speed.

Then select a fixed-duration window of 120 frames within that region and resample by timestamp at 30 Hz. The interval from the first to the last frame is about 3.97 seconds. Keep both window endpoints inside the steady-state region; the window is only a segment of the central 10 m, not the time required to walk the full 10 m. Record its start as window_start_seconds for conversion.

Capture the 3D positions of all 23 bodies at each frame, increasing timestamps, unique body names in position-axis order, and class names. The upstream GetState() call returns a flattened state vector, not named body positions. This project does not include the simulator or a native capture hook; add the per-frame position capture on the simulator side.

The capture archive interface takes these fields:

| Field | Shape | Contents |
|---|---|---|
| positions | [N,F,B,3] or one sample [F,B,3] | Per-frame 3D positions for named bodies |
| timestamps | [N,F] or one sample [F] | Strictly increasing timestamps in seconds |
| body_names | [B] | Unique names matching the body axis in positions |
| label_names | [N] | Class names from the project label map |
| sample_metadata | One record per sample | Condition ID, split, seed, rates, and window start |

Use write_capture_archive in integrations/generative_gaitnet/capture_archive.py to save the archive. Pass a list of sample_metadata dictionaries; the function encodes them as the metadata_json_utf8 byte array:

~~~python
from integrations.generative_gaitnet.capture_archive import write_capture_archive

write_capture_archive(
    "data/gaitnet/raw/train.npz",
    positions,
    timestamps,
    body_names,
    label_names,
    sample_metadata,
)
~~~

Assign each sample to train, val, or test according to the condition table’s split and retain its condition_id.

### Convert to NCTV and train

Convert train, val, and test separately. If each sample’s window start is in its metadata, the command-line window argument can be omitted:

~~~powershell
python scripts/convert_generative_gaitnet.py --input data/gaitnet/raw/train.npz --output data/gaitnet/processed/train.npz --config configs/conversion/generative_gaitnet.yaml
~~~

Replace train with val and test for the other splits. The converter reorders the 23 nodes by body name and linearly interpolates by timestamp to 120 frames at 30 Hz. It writes float32 NCTV arrays of shape [N,3,120,23] and integer labels. The NPZ contains only joints and labels; class names and conversion records are written to a matching .metadata.json file.

Copy configs/teacher.yaml and configs/student.yaml, then update the three data splits, skeleton, label map, teacher checkpoint, and output directory. Point configs/train_all.yaml to the new configurations, then train as described in “Using your own data.”

## ONNX export and testing

Export the best student checkpoint after training:

~~~powershell
python scripts/export_onnx.py --checkpoint runs/utd_mhad/student_seed_2026/best_student.pth --output outputs/litemyonet_utd_mhad.onnx
~~~

The ONNX model takes joint coordinates as joints input and returns logits.

Testing dependencies are included in the installation command. Run the tests with:

~~~powershell
python -m pytest
~~~

## Modules, references, and license

Main implementation files:

- src/lite_myonet/models/myoformer.py: MyoFormer teacher;
- src/lite_myonet/models/lite_myonet.py: Lite-MyoNet student;
- src/lite_myonet/training/teacher_trainer.py and student_trainer.py: teacher and student training;
- src/lite_myonet/training/engine.py: training engine;
- src/lite_myonet/export/onnx_export.py: ONNX export.

Teacher paper: Fang, Zheng; Li, Jia; Zhao, Xiaoheng; Hua, Chunsheng. [MyoFormer: Anatomy-guided transformer for muscle-level pathological gait assessment](https://doi.org/10.1007/s11517-026-03608-x). Medical & Biological Engineering & Computing, 64(7), 2613–2632, 2026.

Synthetic-data source: Park, Jungnam; Min, Sehee; Chang, Phil Sik; Lee, Jaedong; Park, Moon Seok; Lee, Jehee. “Generative GaitNet.” ACM SIGGRAPH 2022 Conference Proceedings, 2022. [DOI](https://doi.org/10.1145/3528233.3530717); [pinned upstream commit](https://github.com/namjohn10/GenerativeGaitNet/tree/eaf3e256925c8b0acc3e95b539535beba105b511).

UTD-MHAD: [official data page](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html). See references.bib for BibTeX entries.

This project’s code is licensed under the MIT License. Third-party code and data remain subject to their respective terms; see THIRD_PARTY_NOTICES.md.
