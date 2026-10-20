# 特征亲和拓扑可视化 / Feature-affinity visualization

只读取使用者提供的教师、两个D=2学生权重与临床23节点窗口，不训练、不调参、不重估临床BN。仓库不包含任何患者坐标、描述、预测、图像或学习权重。

```powershell
python scripts/visualize_topology.py --teacher checkpoints/best_teacher.pth --logit-student checkpoints/logit_seed11.pth --full-student checkpoints/full_seed11.pth --data-root data/clinical --output outputs/topology --device cpu
```

两个学生必须共用种子，推荐固定11。教师为完整三流模型，使用权重内的配置恢复。工具不会根据图形更换种子。节点映射、预归一化和左右约定见 [DATA_FORMAT.md](DATA_FORMAT.md) 与 [CLINICAL_MAPPING.md](CLINICAL_MAPPING.md)。

最后关节阶段的学生`joint.stage3`分别对应教师`joint_spatial.stage3`及`joint_temporal.stage3`，权重各0.5；教师第二个关节命名流在X1/X2/X2下实际输入骨向量。静态描述符取时间均值；动态描述符取相邻帧差绝对值的时间均值；沿通道归一化。固定rho=0、带宽1、epsilon=1e-8，以平方欧氏距离构造行归一化RBF亲和矩阵。教师分别构图后再平均，不能先平均不同通道空间的特征。

输出原始23×23静态/动态矩阵、教师单分支矩阵、窗口与权重配置、22条实际骨架边权以及逐窗口差异。骨架图在22条真实连接上着色，权重为(Pij+Pji)/2，不额外对边重归一化。同病例的三个模型共享坐标、视角、颜色和线宽范围；首个有效窗口的第61帧作为底图。

前三病例按完整学生对教师的22条静态边权RMSE升序选择，并保留全病例排名和所有窗口均值。该排序只衡量相邻静态边，不替代完整23×23分布KL，也不代表真实肌力、肌肉激活或解剖因果关系。`--plot-only`重用可读数值文件，不检验哈希。

## English

Supply your own checkpoints and `Person_XX` coordinate files. Both students must be Fourier D=2 with the same seed. Inference uses frozen BN and no optimization. Final-joint affinities are constructed separately for the teacher branches and then averaged. The fixed visualization bandwidth is one, not the learned training bandwidth. Each figure colors only 22 real skeletal connections, shares scales across models, and ranks cases by full-student-to-teacher adjacent static edge RMSE. These are selected qualitative examples, not full-topology or physiological claims. No private data are bundled.
