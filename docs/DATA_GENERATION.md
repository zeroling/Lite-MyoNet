# Synthetic data generation

## Generative GaitNet

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

The converted split files contain only joints and labels. Pack them into the native training archive with scripts/pack_splits.py before using the teacher/student configurations; see DATA_FORMAT.md.
