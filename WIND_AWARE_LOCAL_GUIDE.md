# wind_aware_local 图构建方法说明

## 1. 改动了什么

在 `data/graph_generation.py` 中新增函数 `generate_wind_aware_local_adjacency_matrix`。

该函数与原有 `generate_wind_aware_adjacency_matrix` **完全相同**，唯一区别在于 **Step 3（上游强度计算）** 中角度差的计算方式：

| 版本 | 角度差计算 | theta 含义 |
|------|-----------|------------|
| `wind_aware`（原版） | `theta_t[:, None, None] - phi_mat[None, :, :]` | `theta_t` 是**全场所有节点的循环均值方向**（标量，每个时刻一个值） |
| `wind_aware_local`（新版） | `theta_mt[:, :, None] - phi_mat[None, :, :]` | `theta_mt[:, j]` 是**节点 j 自身在 t 时刻的实测风向**（每节点、每时刻独立） |

核心代码变化（仅 Step 3）：

```python
# 原版：全场均值风向判断上游关系
angle_diff = theta_t[:, np.newaxis, np.newaxis] - phi_mat[np.newaxis, :, :]  # (C, N, N)

# 新版：节点 j 自身风向判断 j 是否朝向 i
angle_diff = theta_chunk[:, :, np.newaxis] - phi_mat[np.newaxis, :, :]  # (C, N, N)
# angle_diff[t, j, i] = theta_jt - phi_ji
```

Step 1、2、4、5、6 与原版完全一致，未做任何改动。

---

## 2. 为什么这样改更合理

### 2.1 原版的问题

原版 `wind_aware` 在计算 p_ji 时使用的是**全场循环均值风向** theta_t，即把风场内所有 N 个节点在时刻 t 的风向做圆形均值，得到一个代表整个风场的方向标量。随后用这一个标量对所有节点对 (j, i) 统一判断上游关系。

这带来两个缺陷：
1. **信息损失**：圆形均值会平滑掉各节点之间的风向差异，尤其在大型风场中不同区域风向可能相差较大。
2. **对称性不足**：由于所有节点对共享同一个 theta_t，p_ji 与 p_ij 的差异仅来源于方位角 phi_ji 和 phi_ij 的对称性，非对称程度有限，有向图结构不够显著。

### 2.2 新版的改进

新版 `wind_aware_local` 直接使用**节点 j 自身在时刻 t 的实测绝对风向** theta_jt = deg2rad(Ndir_jt + Wdir_jt) 来判断"j 的风是否吹向 i"：

```
angle_diff[t, j, i] = theta_jt - phi_ji
p_ji = mean_t  max(0, cos(theta_jt - phi_ji))
```

物理含义：p_ji 衡量的是"节点 j 处的风在多大程度上朝向 i 的方位"，而非"全场平均风是否朝向 i"。

改进效果：
- **逐节点差异化**：每个节点 j 用自己的实测风向判断上游关系，保留了节点间风向的空间异质性。
- **更强的非对称性**：p_ji 与 p_ij 现在分别由 j 和 i 各自的风向决定，当 j 和 i 风向不同时，有向边权重差异更显著，有向图结构更清晰。
- **物理依据充分**：尾流效应本质上是局部现象，用局部风向判断局部上游关系比用全场均值更贴近实际物理过程。
- **创新性**：现有大多数风向感知图构建方法均使用单一风向（场均值或某一代表值），逐节点风向加权是对现有方法的实质性改进。

---

## 3. 如何生成新的图（不覆盖已有结果）

`main.py` 中 `--adj_type` 参数控制使用哪种邻接矩阵类型，不同类型的数据和图分别保存在独立目录下，**互不干扰**：

| adj_type | 处理数据保存路径 | 图文件保存路径 |
|----------|----------------|---------------|
| `distance` | `data/processed/distance/` | `data/graph/distance/` |
| `wind_aware` | `data/processed/wind_aware/` | `data/graph/wind_aware/` |
| `wind_aware_local` | `data/processed/wind_aware_local/` | `data/graph/wind_aware_local/` |

### 生成 wind_aware_local 图及处理数据

```bash
cd /home/chenyujie/gyf/STCWPF_final
python main.py \
    --data_process 1 \
    --adj_type wind_aware_local \
    --wind_top_k 16
```

运行完成后会在 `data/graph/wind_aware_local/` 下生成 `stage_0_adj.npz` ~ `stage_5_adj.npz`，在 `data/processed/wind_aware_local/` 下生成 `unified_data.npz`。原有的 `distance` 和 `wind_aware` 目录**完全不受影响**。

---

## 4. 如何跑模型

以 EAC 模型为例（`--data_process 0` 表示跳过数据处理，直接用已生成的图）：

```bash
# 训练 + 测试
python main.py --method EAC --adj_type wind_aware_local --data_process 0 --train 1 --logname eac_wind_local --seed 42

python main.py --method EAC --adj_type wind_aware_local --data_process 0 --train 1 --logname eac_rank16_wind_local --seed 42 --gpuid 0
python main.py --method EAC --adj_type distance --data_process 0 --train 1 --logname eac_rank16_dist --seed 42 --gpuid 0

python main.py --method EAC --adj_type wind_aware_local --data_process 0 --train 1 --logname eac_rank10_wind_local --seed 42 --gpuid 1
python main.py --method EAC --adj_type distance --data_process 0 --train 1 --logname eac_rank10_dist --seed 42 --gpuid 1

python main.py --method EAC --adj_type wind_aware_local --data_process 0 --train 1 --logname eac_rank12_wind_local --seed 42 --gpuid 0
python main.py --method EAC --adj_type distance --data_process 0 --train 1 --logname eac_rank12_dist --seed 42 --gpuid 0
# 仅测试（加载已训练模型）
python main.py \
    --method EAC \
    --adj_type wind_aware_local \
    --data_process 0 \
    --train 0 \
    --logname eac_wind_local \
    --seed 42
```

其他模型（`ScaleShift`、`VariationalScaleShift`、`PatchTST`）同理，只需修改 `--method` 参数。

---

## 5. 消融实验方案（不覆盖已有结果）

三种图类型彼此独立，可直接进行三组对比实验：

```bash
# Baseline: distance
python main.py --method EAC --adj_type distance --train 1 --logname eac_dist

# Ablation 1: wind_aware（全场均值风向）
python main.py --method EAC --adj_type wind_aware --train 1 --logname eac_wind

# Proposed: wind_aware_local（逐节点风向）
python main.py --method EAC --adj_type wind_aware_local --train 1 --logname eac_wind_local
```

对 `ScaleShift`、`VariationalScaleShift`、`PatchTST` 重复以上三组，即可得到完整的三模型 × 三图类型消融矩阵。

---

## 6. 你可能没有想到的补充建议

### 6.1 验证 p_mat 的非对称性是否确实增强

在生成图后，可以用以下代码快速比较两种方法的非对称程度：

```python
import numpy as np

adj_wind       = np.load('data/graph/wind_aware/stage_0_adj.npz')['x']
adj_wind_local = np.load('data/graph/wind_aware_local/stage_0_adj.npz')['x']

def asymmetry_score(A):
    """Mean absolute difference between A[i,j] and A[j,i]"""
    return np.mean(np.abs(A - A.T))

print('wind_aware asymmetry:      ', asymmetry_score(adj_wind))
print('wind_aware_local asymmetry:', asymmetry_score(adj_wind_local))
```

如果 `wind_aware_local` 的非对称分数更高，说明改进有效，这一数值可以直接写入论文作为定量依据。

### 6.2 可视化有向图结构差异

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, path, title in zip(axes,
    ['data/graph/wind_aware/stage_0_adj.npz',
     'data/graph/wind_aware_local/stage_0_adj.npz'],
    ['wind_aware', 'wind_aware_local']):
    A = np.load(path)['x']
    im = ax.imshow(A, cmap='hot', aspect='auto')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('adj_comparison.png', dpi=150)
```

### 6.3 关于 valid_t 的计算

原版 `wind_aware` 中 `valid_t` 统计的是 `theta_t`（均值后的向量）中有限值的数量；新版 `wind_aware_local` 改为直接用 `theta_mt.shape[0]`（训练集时间步总数），因为 `theta_mt` 是逐节点的，其有效性由 `nansum` 自动处理，分母取总时间步数更合理。若你希望严格对齐，也可改为 `np.sum(np.any(np.isfinite(theta_mt), axis=1))` 统计至少有一个节点有效的时间步数。

### 6.4 记录实验结果

建议在 `log/` 目录下为每次实验保存结果摘要，便于后续对比：

```bash
python main.py --method EAC --adj_type wind_aware_local --logname eac_wind_local 2>&1 | tee log/eac_wind_local_result.txt
```
