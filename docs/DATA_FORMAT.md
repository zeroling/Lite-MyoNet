# 数据与输入格式 / Data format

本仓库不提供数据集、临床骨架、患者记录、权重或实验输出。这里只提供格式、处理代码与生成流程说明。

## 模型输入

关节和骨向量都采用 `[B,3,120,23]`、FP32。节点顺序见 `configs/skeletons/fullbody23.yaml`，33类编号见 `configs/label_map.json`。骨向量按22条 `(child,parent)` 边计算为子坐标减父坐标，存入child槽位；根槽位0为零。已有23节点数组不应重复执行Kinect映射。

## 训练归档

训练器读取一个本地NPZ，包含三个划分的 `train_joints/train_bones/train_labels`、`val_*`、`test_*`；标签为int16整数向量。归档还包括逐样本ID、标签名、分组信息和UTF-8 JSON元数据。从三个仅含 `joints`、`labels` 的NCTV NPZ生成归档：

```powershell
python scripts/pack_splits.py --train data/train.npz --val data/val.npz --test data/test.npz --output data/synthetic.npz
```

真实生理参数组或受试者组ID通过 `--groups data/groups.json` 提供三组等长列表：`{"train":["group-a"],"val":["group-b"],"test":["group-c"]}`。同组不得跨划分。缺少组信息时明确标注分组独立性未核实，不把自动样本ID当作生理参数组，也不从文件排列臆测分组。打包器不恢复缺失的仿真参数，不改变已有数据划分。

合成轨迹生成和转换见 [DATA_GENERATION.md](DATA_GENERATION.md)。生成配置是重建计划，不声称与已完成研究数据逐条相同。

## 临床可视化

自行准备 `data/clinical/Person_01/*.npy` 等目录，关节数组为 `[3,120,23]`。可视化入口读取关节文件并按训练22边重建骨向量。每人按文件名排序取首个有效窗口绘图，保留处理后文件内帧号；缺失的原始录像偏移不作推断。输入需事先完成一致的坐标规范化，绘图器不再次改变模型输入。

兼容映射见 [CLINICAL_MAPPING.md](CLINICAL_MAPPING.md)。本仓库不包含病例描述或受试者标签表。

## English

No datasets, clinical records, checkpoints, or generated results are distributed. Both streams are FP32 `[B,3,120,23]`; bones are child-minus-parent vectors with root slot zero. `pack_splits.py` creates the native archive from three NCTV `joints/labels` NPZ files. Optional real group IDs are checked for cross-split overlap; grouping is explicitly unverified when absent. Clinical visualization reads user-supplied `Person_XX` joint files and derives bones consistently. Input coordinates must already follow your normalization protocol. Generation instructions describe a reconstruction protocol, not an included historical dataset.
