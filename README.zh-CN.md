# Lite-MyoNet

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库提供本轮实际使用的MyoFormer教师、Lite-MyoNet学生和ST-GTP蒸馏代码，包括训练、消融矩阵、数据转换、ONNX导出及特征亲和骨架图。数据集、患者记录、学习权重、实验输出和本机环境不随源码发布。

## 安装

使用Python 3.10或更新版本，先根据设备驱动安装兼容的PyTorch（2.4或更新版本）。已整理的Windows便携环境使用PyTorch 2.7.1、CUDA 12.8；不需要照搬开发机的解释器路径。

```powershell
python -m pip install -e ".[data,export,visualization,test]"
```

## 当前模型与训练

- 教师为全宽、8头、三流X1/X2/X2，种子2026。
- 学生为关节/骨双流，通道32/64/128、傅里叶D=2、标准三分区骨架邻接、可学习偏置、动态图与余弦分类头。
- 学生种子11、21、31、41、51；默认300轮、FP32、批量16。优化器、预热和后程曲线的实际数值见 `configs/teacher.yaml` 与 `configs/student.yaml`。
- ST-GTP按静态时间均值和动态相邻帧绝对差构造关系；通道归一化后计算平方距离与逐行归一化RBF，使用教师到学生的前向KL。关节阶段分别对两个教师分支计算后按0.5/0.5加权，再累加各阶段；骨流一对一对齐。
- X1/X2/X2下，教师第二个关节命名分支实际接收骨向量，不能根据名称推断输入。
- 主配置保存每轮与最佳权重。续跑只要求断点可读且能恢复，不以哈希验证作为门槛。

数据只提供 [格式说明](docs/DATA_FORMAT.md)、[仿真生成与转换](docs/DATA_GENERATION.md)、[Kinect兼容映射](docs/CLINICAL_MAPPING.md)。自行准备 `data/synthetic.npz`，或用环境变量 `LITE_MYONET_DATA_ARCHIVE` 指定本地归档：

```powershell
python scripts/train_all.py --device cuda --student-jobs 1
```

教师严格串行；学生并行数先设1，自行确认显存后再增加。主训练固定D=2。需要傅里叶阶数选择及完整消融矩阵时运行：

```powershell
python scripts/run_experiment_matrix.py --plan configs/experiment_matrix.yaml --device cuda --student-jobs 1
```

加 `--resume` 继续已有任务；加 `--dry-run` 只检查计划，不训练，但仍需本地归档可读取。单独训练使用 `scripts/train_teacher.py --config configs/teacher.yaml` 和 `scripts/train_student.py --config configs/student.yaml`；后者需在配置中填入自行训练的教师权重路径，并在权重同目录保留 `resolved_config.yaml`。单个训练器续跑用 `--resume 断点或运行目录`。

`configs/baseline_optimizers.yaml` 保留八个对比方法的优化器方案说明；第三方模型源码与依赖按其原项目获取，本次源码包不捆绑分发。

## 导出与可视化

```powershell
python scripts/export_onnx.py --checkpoint checkpoints/best_student.pth --output outputs/student.onnx
```

当前ONNX是双输入：`joint_coordinates`（关节坐标）、`bone_vectors`（骨向量），均为 `[B,3,120,23]`，输出 `logits`（分类分数）。不要把较早的单输入导出或旧发布目录的另一套权重混用。

[拓扑绘图说明](docs/TOPOLOGY.md) 提供自行传入教师、两个同种子D=2学生和临床窗口的命令，直接在22条真实骨架边上标色。仓库没有内嵌患者样本；亲和值不解释为肌肉激活、肌力或解剖因果连接。

## 源码位置

- `src/lite_myonet/models/teacher.py`、`student.py`：与本轮权重兼容的模型实现。
- `src/lite_myonet/training/`：训练器、蒸馏、断点、教师缓存、调度和最终BN统计重估支持。
- `scripts/run_experiment_matrix.py`、`configs/ablations/`：实验矩阵及消融配置。
- `src/lite_myonet/visualization/`：真实特征提取与矢量绘图。
- `src/lite_myonet/data/`、`integrations/`：打包、映射、仿真采集与转换。

公开类名 `MyoFormer`、`LiteMyoNet` 分别是 `MyoFormerTeacher`、`MyoFormerStudent` 的别名，避免维护两套不同结构。

教师论文：[MyoFormer](https://doi.org/10.1007/s11517-026-03608-x)；仿真来源：[Generative GaitNet](https://github.com/namjohn10/GenerativeGaitNet/tree/eaf3e256925c8b0acc3e95b539535beba105b511)。引用见 `references.bib`。自有代码保留仓库既有MIT许可，第三方软件、数据和权重遵循原有条款，详见 [第三方说明](THIRD_PARTY_NOTICES.md)。
