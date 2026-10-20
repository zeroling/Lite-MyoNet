import torch


# 当前模型使用的 23 点骨架连接关系。
bones = [
    # 下躯干
    (11, 0), (1, 0), (6, 0),
    # 右腿
    (2, 1), (3, 2), (4, 3), (5, 4),
    # 左腿
    (7, 6), (8, 7), (9, 8), (10, 9),
    # 上躯干到头部
    (12, 11), (13, 12), (14, 13),
    # 右臂
    (15, 13), (16, 15), (17, 16), (18, 17),
    # 左臂
    (19, 13), (20, 19), (21, 20), (22, 21)
]

symmetry_pairs = [
    (18, 22), (17, 21), (16, 20), (15, 19),
    (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
]

def build_adjacency_matrix(num_points, bones, center=0):
    A = torch.zeros((3, num_points, num_points), dtype=torch.float32)
    A[0] = torch.eye(num_points)

    if bones is not None and len(bones) > 0:
        bones_tensor = torch.tensor(bones, dtype=torch.long)
        i_idx, j_idx = bones_tensor[:, 0], bones_tensor[:, 1]

        # 构建无向邻接矩阵
        adj = torch.zeros((num_points, num_points), dtype=torch.float32)
        adj[i_idx, j_idx] = 1
        adj[j_idx, i_idx] = 1

        # BFS 计算距离中心
        dist = torch.full((num_points,), float('inf'))
        dist[center] = 0
        queue = [center]
        while queue:
            node = queue.pop(0)
            neighbors = (adj[node] > 0).nonzero(as_tuple=False).squeeze().tolist()
            if isinstance(neighbors, int):
                neighbors = [neighbors]
            for n in neighbors:
                if dist[n] == float('inf'):
                    dist[n] = dist[node] + 1
                    queue.append(n)
        dist = dist.long()

        # 按节点到中心点的拓扑距离填充近心和远心方向分支。
        d_i, d_j = dist[i_idx], dist[j_idx]

        # A[1, source, target]：从较近中心的节点向较远中心的节点传递。
        mask_close = (d_i <= d_j)
        A[1, i_idx[mask_close], j_idx[mask_close]] = 1.0
        mask_close_rev = (d_i > d_j)
        A[1, j_idx[mask_close_rev], i_idx[mask_close_rev]] = 1.0

        # A[2, source, target]：从较远中心的节点向较近中心的节点传递。
        mask_far = (d_i > d_j)
        A[2, i_idx[mask_far], j_idx[mask_far]] = 1.0
        mask_far_rev = (d_i < d_j)
        A[2, j_idx[mask_far_rev], i_idx[mask_far_rev]] = 1.0

    # 归一化
    D = A.sum(dim=2, keepdim=True)
    D[D == 0] = 1
    A = A / D
    return A
