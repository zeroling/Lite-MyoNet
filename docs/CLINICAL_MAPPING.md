# Kinect兼容映射 / Kinect compatibility mapping

Kinect v2接口有25点；保留的历史映射使用其中前21点全身骨架，另4个手指尖/拇指点不进入模型。`lite_myonet.data.kinect23.map_kinect_to_synthetic23` 接收 `[...,21,3]` 或 `[...,25,3]`，输出 `[...,23,3]`，仅做取点、固定中点插值和重排。

它不执行坐标反射、归一化、根平移或时间重采样。输出槽位不是23个独立实测关节：头槽位采用颈点代理，手槽位采用腕点代理，足部和肩臂部分槽位为插值，不能将其描述为相应骨骼中心的直接测量。

历史映射将Kinect左侧来源放入项目右侧命名槽位，反之亦然，为兼容既有权重予以保留。骨架图的临床解剖L/R分别对应槽位1–5/15–18及6–10/19–22。解释分类输出时应核对类别侧别转换，避免同时交换输入和输出。

```python
from lite_myonet.data.kinect23 import map_kinect_to_synthetic23
mapped = map_kinect_to_synthetic23(kinect_sequence)  # [T,23,3]
model_joints = mapped.transpose(2, 0, 1)             # [3,T,23]
```

逐点公式见模块中的 `process_points`、`SOURCE_TO_TARGET` 和 `reorder_array`。节点号全部从0开始。输入已为 `[3,120,23]` 时不再次映射。

## English

The compatibility mapping uses the first 21 Kinect joints and constructs 23 model slots by fixed midpoint interpolation and reordering. It does not normalize, reflect, resample, or translate coordinates. Several slots are proxies rather than independent anatomical measurements. Historical left/right slot naming is retained for checkpoint compatibility; anatomical plotting labels and any output-side correction must avoid a second input swap. No subject data are included.
