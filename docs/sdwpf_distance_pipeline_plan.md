# SDWPF Distance Pipeline Refactor

## 1. 范围

这版重构只支持 `sdwpf` 作为运行时数据集。

保留但不启用：

- 其他数据集的原始代码和数据文件

当前运行时支持：

- 邻接矩阵类型：`baseline`
- 邻接矩阵类型：`local_upstream`
- 数据入口：`data/raw/sdwpf/sdwpf_2001_2112_full.csv`
- 风机静态坐标：`data/raw/sdwpf/sdwpf_turb_location_elevation.csv`

## 2. 当前设计结论

### 2.1 数据切分

整个时间轴按时间顺序切成：

- 训练：前 `20%`
- 验证：接下来的 `10%`
- 测试：最后 `70%`

也就是 `2:1:7`。

对应缓存字段：

- `pretrain_end_idx`
- `val_end_idx`

### 2.2 邻接图方案

当前运行时有两种图。

生成逻辑在：

- [/home/chenyujie/gyf/STCWPF_final/data/graph_generation.py](/home/chenyujie/gyf/STCWPF_final/data/graph_generation.py)
- [/home/chenyujie/gyf/STCWPF_final/util/distance_utils.py](/home/chenyujie/gyf/STCWPF_final/util/distance_utils.py)

`baseline`：

1. 用风机 `x, y` 计算两两欧式距离，单位转成 `km`
2. 用阶段内所有节点对距离的标准差作为 `sigma`
3. 用 `exp(-d^2 / sigma^2)` 计算权重
4. 用阈值 `r` 做稀疏化
5. 去自环
6. 转成 `adj @ x` 可直接使用的消息传递布局

当前运行配置：

- `baseline_weight_threshold = 0.95`

`local_upstream`：

1. 用 `SDWPF` 的 `Ndir + Wdir` 重构绝对来风方向
2. 转成与 `atan2` 一致的数学角
3. 计算长期局部上游强度 `p_{ji}`
4. 与距离高斯核相乘
5. 对每个目标节点保留 Top-K 入边
6. 按源节点出边归一化
7. 转成 `adj @ x` 可直接使用的消息传递布局

当前运行配置：

- `local_upstream_top_k = 16`

### 2.3 不再做训练/验证多频率采样

已经删除：

- 人造 `5min` 基础时间栅格
- 训练/验证阶段的 `5/10/15min` 多频率预训练采样
- 与之相关的缓存字段和运行时依赖

现在训练和验证都只使用原始时间戳上的真实窗口。

### 2.4 频率信息仍然保留

虽然训练/验证不再做人造多频率采样，但频率信号仍然保留。

原因是 `sdwpf` 的测试阶段确实会出现分辨率变化。

当前 `sdwpf` 的真实频率特征是：

- `2020` 年主体是 `15min`
- `2021` 年后主体是 `10min`

另外数据中还存在两个不连续缺口：

- `2020-03-31 23:55:00 -> 2020-05-01 00:10:00`，间隔 `43215min`
- `2020-12-31 15:55:00 -> 2021-01-01 00:10:00`，间隔 `495min`

这些缺口现在不会再被滑窗跨过去拼样本。

## 3. 数据处理产物

数据处理入口：

- [/home/chenyujie/gyf/STCWPF_final/data/data_processing.py](/home/chenyujie/gyf/STCWPF_final/data/data_processing.py)

主缓存文件：

- `data/processed/sdwpf/final_xy_16feat_origfreq_freqproj_maskv1/{graph_variant}/unified_data.npz`

图文件：

- `data/graph/sdwpf/final_xy_16feat_origfreq_freqproj_maskv1/{graph_variant}/stage_{k}_adj.npz`

当前 `unified_data.npz` 里保留的核心字段：

- `raw_data`
- `patv_mask`
- `raw_timestamps`
- `static_data`
- `static_data`
- `static_mean`
- `static_std`
- `x_mean`, `x_std`
- `y_mean`, `y_std`
- `pretrain_end_idx`
- `val_end_idx`
- `initial_cols`
- `turbine_schedule_*`
- `supported_frequency_minutes`

已删除的旧字段：

- `raw_data_base`
- `patv_mask_base`
- `base_timestamps`
- `pretrain_freq_minutes`
- `base_resolution_minutes`

## 4. 训练/验证样本怎么构造

训练和验证的窗口构造逻辑在：

- [/home/chenyujie/gyf/STCWPF_final/trainer.py](/home/chenyujie/gyf/STCWPF_final/trainer.py)

当前规则如下。

### 4.1 先按“连续且单一频率”切段

时间轴不会直接整体滑窗。

会先找出所有满足以下条件的连续片段：

- 时间差属于允许频率集合
- 片段内所有相邻时间差完全一致

对于 `sdwpf`，允许频率集合是：

- `10`
- `15`

因此：

- `15min` 连续段单独建样本
- `10min` 连续段单独建样本
- 任何大缺口都直接断开
- 频率切换点也直接断开

### 4.2 训练/验证窗口

设：

- `x_len = 12`
- `y_len = 12`

对每个合法连续片段，按步长 `1` 做标准滑窗：

- 输入：过去 `12` 步
- 目标：未来 `12` 步

也就是说训练/验证仍然是一格一格滑窗。

只是窗口必须完全落在同一连续、同一频率片段里。

## 5. 流式测试怎么构造

流式测试也在：

- [/home/chenyujie/gyf/STCWPF_final/trainer.py](/home/chenyujie/gyf/STCWPF_final/trainer.py)

### 5.1 预测节拍

当前流式测试不是一步一步滚动，而是按 `y_len` 前进。

当 `x_len = 12`, `y_len = 12` 时：

1. 在时刻 `t` 用过去 `12` 步预测未来 `12` 步
2. 时间向前推进 `12` 步
3. 新获得的这 `12` 步真实值形成一个新样本
4. 如果发生了新风机加入，并且方法支持 warmup，就先做更新
5. 再进行下一次预测

这和你前面要求的“每 `outputstep` 预测一次，每 `inputstep` 收到一批新数据后再决定是否更新”是一致的。

### 5.2 也只在合法连续片段内测试

测试窗口也不会跨越：

- 大缺口
- 频率变化边界

当前测试计划是先生成所有合法的 `start_idx`，再按这些合法起点做流式预测。

### 5.3 风机扩容和 warmup

当新风机加入时：

- 模型先扩展自适应参数
- 切换到新的 stage 邻接矩阵
- 对允许 warmup 的模型做局部更新

当前 warmup 只开放给：

- `ScaleShift`
- `VariationalScaleShift`
- `EAC`

`PatchTST` 不做 warmup 更新。

## 6. 静态特征和频率特征

### 6.1 当前真正可共同使用的静态特征

现在跨数据集最稳妥、最一致、当前代码里真正可依赖的静态特征只有：

- `x`
- `y`

本次 `sdwpf` 运行时也只使用这两个。

### 6.2 静态特征怎么进模型

对我们自己的模型：

- `ScaleShift`
- `VariationalScaleShift`

会把 `x, y` 先离散到 bin：

- `static_data`
- `static_mean`
- `static_std`

然后做位置 embedding。

### 6.3 频率特征怎么进模型

对我们自己的模型，不再使用“离散频率 id 对应独立 embedding 行”的方案。

当前实现改成了共享的频率投影：

- 先读取窗口对应的真实频率分钟数
- 用 `supported_frequency_minutes = [10, 15]` 做标准化
- 再通过一层线性投影得到频率偏置

实现位置：

- [/home/chenyujie/gyf/STCWPF_final/model/static_embedding.py](/home/chenyujie/gyf/STCWPF_final/model/static_embedding.py)

这样做的原因很直接：

- 训练/验证基本都在 `2020`，主要只见到 `15min`
- 如果用离散 embedding，`10min` 那一行在训练阶段几乎学不到东西
- 共享投影至少能让 `10` 和 `15` 共享一套连续映射，而不是让测试时出现一行随机参数

### 6.4 哪些模型不用静态/频率嵌入

以下模型保持原有结构，不加我们的创新点：

- `EAC`
- `PatchTST`

它们只共享同一套数据切分、邻接矩阵和测试流程。

## 7. 当前各模型的职责边界

### 7.1 我们自己的模型

- `ScaleShift`
- `VariationalScaleShift`

保留：

- 静态特征嵌入
- 频率信号注入
- warmup 更新

### 7.2 EAC

保留：

- 原有模型结构
- 同样的流式测试框架
- 支持 warmup 更新

不加入：

- 静态特征嵌入
- 我们的方法特有的频率嵌入

### 7.3 PatchTST

保留：

- 原有 PatchTST 结构
- 同样的训练/验证/流式测试切分方式

不加入：

- 静态特征嵌入
- 频率嵌入
- warmup 更新

## 8. 运行方式

### 8.1 先做数据处理

命令：

```bash
conda run -n gyf python main.py --method ScaleShift --data_process 1 --train 0 --logname smoke_process
```

`--data_process 1` 的意思是：

- 读取原始 `sdwpf` CSV
- 生成新的 `unified_data.npz`
- 生成各 stage 的距离邻接矩阵
- 处理完成后直接退出，不进入训练和测试

### 8.2 再训练和测试

数据处理完成后，再用：

```bash
conda run -n gyf python main.py --method ScaleShift --data_process 0 --train 1 --logname run
```

如果只想加载已有 checkpoint 做测试：

```bash
conda run -n gyf python main.py --method ScaleShift --data_process 0 --train 0 --logname run
```

前提是对应 `log/.../pretrain/*.pkl` 已存在。

## 9. 已完成的验证

在 `gyf` 环境里已经做过以下 smoke test：

1. 重新生成了新的 `sdwpf` 缓存和距离图
2. 确认新缓存不再包含旧的 `base grid` 字段
3. 确认测试集被切成一个 `15min` 连续段和一个 `10min` 连续段，中间没有跨 gap 采样
4. 确认 `ScaleShift / VariationalScaleShift / EAC / PatchTST` 都能前向跑通
5. 确认 `ScaleShift` 和 `EAC` 的短程流式测试加 warmup 可以跑通

## 10. 当前你需要记住的操作顺序

如果缓存还是旧的 `final_xy_16feat_multifreq_maskv1`，不要直接训练。

先重新处理数据，再训练。

标准顺序是：

1. `--data_process 1`
2. 确认生成的是 `final_xy_16feat_origfreq_freqproj_maskv1`
3. 再跑训练或测试
