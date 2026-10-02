[English](README.md) | [简体中文](README.zh-CN.md)

# Lite-MyoNet

Lite-MyoNet 是用于骨架序列分类的教师—学生框架：教师为 MyoFormer，学生为 Lite-MyoNet。项目提供模型训练、公开数据准备、数据转换、测试集评估和 ONNX（开放神经网络交换格式）导出入口。

教师依据已发表论文 [MyoFormer: Anatomy-guided transformer for muscle-level pathological gait assessment](https://doi.org/10.1007/s11517-026-03608-x)（MyoFormer：解剖引导的肌肉层面病理步态评估 Transformer）。学生采用轻量时空编码、傅里叶层和教师蒸馏；默认 fourier_degree: 2 是配置值，可在 YAML（配置文件）中调整。

## 安装

需要 Python（编程语言）3.10 或更新版本。按设备安装适用的 PyTorch（深度学习框架）后，在仓库根目录运行：

~~~powershell
python -m pip install -e ".[data,export,test]"
~~~

## 输入数据

训练数据为 NCTV（样本、坐标通道、时间帧、骨架节点）格式的关节位置数组，形状为 [N,C,T,V]，类型为 float32（32 位浮点数）。

NPZ（NumPy 压缩数组文件）须恰含 joints 和 labels 两个数组；NPY（NumPy 数组文件）关节输入须配同名前缀的 .labels.npy 文件。labels 为长度 N 的整数向量，类别编号从 0 连续递增。

~~~python
import numpy as np

np.savez_compressed(
    "train.npz",
    joints=joints.astype(np.float32),
    labels=labels.astype(np.int64),
)
~~~

节点数、帧数和通道须与骨架 YAML 配置一致。UTD-MHAD 默认输入为 [N,3,120,20]；23 节点步态配置为 [N,3,120,23]。模型根据 bone_vector_edges 从关节位置计算骨向量。

自定义骨架时，设置 point_count、按序排列的 nodes、root_index、model_graph_edges、bone_vector_edges、symmetry_pairs、channels 和 frame_count；类别名称与编号在 JSON（结构化文本）标签表中配置。

## UTD-MHAD：准备与训练

本项目使用 [UTD-MHAD 官方骨架数据](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)，支持 27 个动作类别和 20 节点骨架。准备脚本下载官方骨架与示例代码包，生成训练、验证和测试数据：

~~~powershell
python scripts/prepare_utd_mhad.py
~~~

数据默认写入 data/utd_mhad/processed/。跨受试者拆分为 344 条训练、87 条验证、430 条测试序列；处理时以髋中心对齐并插值为 120 帧。

端到端训练依次运行一个教师和配置中的五个学生种子：

~~~powershell
python scripts/train_all.py --config configs/experiments/utd_mhad.yaml --device cuda
~~~

中断后继续未完成阶段：

~~~powershell
python scripts/train_all.py --config configs/experiments/utd_mhad.yaml --device cuda --resume
~~~

也可分别训练教师和学生，按顺序运行：

~~~powershell
python scripts/train_teacher.py --config configs/utd_mhad/teacher.yaml --device cuda
python scripts/train_student.py --config configs/utd_mhad/student.yaml --device cuda
~~~

学生配置默认读取同一输出目录下的最佳教师权重。设备可设为 cuda 或 cpu，默认输出目录为 runs/utd_mhad/。训练用验证集 Macro-F1（宏平均 F1）选择权重，完成后评估测试集。

## 使用自备数据

按上文格式准备 train、val、test 三个数据拆分。复制 configs/teacher.yaml、configs/student.yaml，并在副本中更新数据路径、骨架 YAML、标签表、教师权重和输出目录。

修改 configs/train_all.yaml 中的 teacher_config、student_config，使流水线指向上述配置副本，并设置独立 output_root。各路径按对应 YAML 文件的位置填写。

~~~powershell
python scripts/train_all.py --config configs/train_all.yaml --device cuda
~~~

分别训练时，使用对应配置运行 train_teacher.py，再运行 train_student.py。学生配置的 seeds 字段可指定一个或多个种子。

## Generative GaitNet 合成步态数据

合成数据基于 Generative GaitNet（生成式步态网络）的肌骨仿真，固定使用上游提交 [eaf3e256925c8b0acc3e95b539535beba105b511](https://github.com/namjohn10/GenerativeGaitNet/tree/eaf3e256925c8b0acc3e95b539535beba105b511)。

每条病理条件只设置一个目标肌肉组的最大等长肌力比例 r = 1 - w，其中 w = 0.05–0.40。标签包括左右侧各 16 个肌肉组（32 个病理类别）及一个正常类别；其余肌肉组的最大肌力比例保持 1.0。

病理条件在 r=0.600–0.950 范围内，以 0.001 为参数分辨率选取，共 11,200 条。正常条件覆盖 r=0.951–1.000，共 1,600 条。仿真计划共 12,800 条条件，训练、验证和测试集的目标划分比例为 2∶1∶1。

这是公开配置定义的仿真计划，不代表轨迹已生成。

### 构建上游环境与生成条件表

仿真器与策略在 Generative GaitNet 上游环境中独立部署。按固定提交构建：

~~~bash
git clone https://github.com/namjohn10/GenerativeGaitNet.git
cd GenerativeGaitNet
git checkout eaf3e256925c8b0acc3e95b539535beba105b511
./install.sh
./pc_build.sh
cd build
make -j32
~~~

上游说明的依赖为 Python 3.6 和 Ray 1.8.0（分布式训练框架），应与本项目的分类训练环境分开配置。上游策略入口为 imgui_render/imgui_render；启动单网络或四网络级联策略时，按该固定提交的 README 提供相应网络文件路径。

在 Lite-MyoNet 仓库根目录创建条件表：

~~~powershell
python scripts/create_gaitnet_conditions.py --config configs/conversion/generative_gaitnet_conditions.yaml --output data/gaitnet/conditions.json
~~~

JSON 条目记录 condition_id（条件编号）、label_name（标签名）、target_force_scale（目标肌力比例）、split（数据拆分）和 simulation_seed（仿真种子）。逐条运行策略时，每个标签只改变对应的精确肌肉单元。

单策略启动入口示例；网络文件路径须替换为该上游版本对应的策略：

~~~bash
./imgui_render/imgui_render path/to/network
~~~

### 应用肌力条件并运行步行

每条轨迹从目标肌肉以外各组最大肌力比例均为 1.0 的基准状态开始。上游 python/ray_env.py 的 reset() 包装器会重新抽取并写入 current_paramstate；使用该包装器时，每次 reset 后都要重新应用目标比例，或关闭自动参数采样。底层 Environment::Reset 不会自动改变肌力比例。

通过 GetParamState（读取参数状态）取得基准向量，按实际元数据顺序定位最大肌力参数段（位于肌肉长度参数段之后），把目标组比例设为 r，再用 SetParamState（设置参数状态）写入并回读确认。比例直接赋值，无需再次归一化。左右外侧腓肠肌头应分别匹配 L_Gastrocnemius_Lateral_Head 和 R_Gastrocnemius_Lateral_Head；宽泛的 group L_Gastrocnemius 会同时匹配内、外侧头。

metadata.txt 的控制频率为 30 Hz、物理仿真频率为 480 Hz；metadata_fullbody.txt 的控制频率为 30 Hz、物理仿真频率为 600 Hz。每次运行均记录实际加载的元数据、策略版本与控制和仿真频率。

按同一控制策略逐条件模拟 20 m 直线行走。依据 MyoFormer 论文，先根据实际位移确定从起点 5 m 至 15 m 的中央 10 m 稳态区间。这 10 m 用于限定采样的空间范围，其行走时间由每条轨迹的实际步速决定。

随后在稳态区间内截取一个完整的固定时长窗口，按时间戳以 30 Hz（赫兹）重采样为 120 帧，首帧至末帧间隔约 3.97 秒。窗口的起点和终点均须位于稳态区间内；它只是中央 10 m 轨迹中的一段，不对应走完这 10 m 所需的时间。将窗口起始时间记录为 window_start_seconds，供转换器提取对应序列。

采集每帧 23 个刚体的位置、递增时间戳、与位置轴对应的唯一刚体名称及类别名。上游 GetState() 返回扁平状态向量，不是按刚体命名的位置数组。本项目不运行模拟器或策略，逐帧位置采集需由模拟器侧接入钩子完成。

采集归档接口接收以下数据：

| 字段 | 形状 | 内容 |
|---|---|---|
| positions | [N,F,B,3] 或单条 [F,B,3] | 逐帧命名刚体三维位置 |
| timestamps | [N,F] 或单条 [F] | 秒级时间戳，严格递增 |
| body_names | [B] | 与位置数组刚体轴对应的唯一名称 |
| label_names | [N] | 来自项目标签表的类别名称 |
| sample_metadata | 每个样本一条记录 | 条件编号、拆分、种子、频率和窗口起点 |

通过 integrations/generative_gaitnet/capture_archive.py 中的 write_capture_archive 归档。传入 sample_metadata 字典列表后，函数会编码并写入 metadata_json_utf8（UTF-8 字节数组）：

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

每个样本按条件表中的 split 归入 train、val 或 test，并保留 condition_id。

### 转换为 NCTV 并训练

train、val、test 三个拆分分别转换。每条轨迹的窗口起点写入样本元数据后，可省略命令行窗口参数：

~~~powershell
python scripts/convert_generative_gaitnet.py --input data/gaitnet/raw/train.npz --output data/gaitnet/processed/train.npz --config configs/conversion/generative_gaitnet.yaml
~~~

将命令中的 train 分别替换为 val 和 test。转换器按刚体名称重排 23 个节点，使用时间戳线性插值为 120 帧、30 Hz，输出 NCTV [N,3,120,23] 的 float32 数组和整数标签。NPZ 仅含 joints 和 labels；类别名和转换记录保存在同名 .metadata.json 文件中。

复制 configs/teacher.yaml、configs/student.yaml，更新三个数据拆分、骨架、标签表、教师权重和输出目录；同时让 configs/train_all.yaml 指向新配置，然后按“使用自备数据”一节训练。

## ONNX 导出与测试

训练后导出学生最佳权重：

~~~powershell
python scripts/export_onnx.py --checkpoint runs/utd_mhad/student_seed_2026/best_student.pth --output outputs/litemyonet_utd_mhad.onnx
~~~

ONNX 模型接收 joints 关节坐标并输出 logits（分类得分）。

安装部分提供测试依赖。运行测试：

~~~powershell
python -m pytest
~~~

## 模块、引用与许可

主要实现文件：

- src/lite_myonet/models/myoformer.py：MyoFormer 教师；
- src/lite_myonet/models/lite_myonet.py：Lite-MyoNet 学生；
- src/lite_myonet/training/teacher_trainer.py 和 student_trainer.py：教师与学生训练；
- src/lite_myonet/training/engine.py：训练引擎；
- src/lite_myonet/export/onnx_export.py：ONNX 导出。

教师论文：Fang, Zheng; Li, Jia; Zhao, Xiaoheng; Hua, Chunsheng. [MyoFormer: Anatomy-guided transformer for muscle-level pathological gait assessment](https://doi.org/10.1007/s11517-026-03608-x)（MyoFormer：面向肌肉层级病理步态评估的解剖引导变换器）. Medical & Biological Engineering & Computing（医学与生物工程及计算）, 64(7), 2613–2632, 2026.

合成数据来源：Park, Jungnam; Min, Sehee; Chang, Phil Sik; Lee, Jaedong; Park, Moon Seok; Lee, Jehee. “Generative GaitNet”（生成式步态网络）. ACM SIGGRAPH 2022 Conference Proceedings（ACM SIGGRAPH 2022 会议论文集）, 2022. [DOI](https://doi.org/10.1145/3528233.3530717)；[固定上游提交](https://github.com/namjohn10/GenerativeGaitNet/tree/eaf3e256925c8b0acc3e95b539535beba105b511)。

UTD-MHAD：[官方数据页面](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)。BibTeX（文献条目）见 references.bib。

项目代码采用 MIT License（MIT 许可证）。第三方代码与数据按各自条款使用，详情见 THIRD_PARTY_NOTICES.md。
