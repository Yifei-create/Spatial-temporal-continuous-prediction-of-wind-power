# 框架修改最终实施方案

> 目标：
>
> 1. 接入 `Penmanshiel` 与 `Towie`
> 2. 为每个风机加入静态位置嵌入
> 3. 为 `sdwpf` 加入 5/10/15min 混合预训练与频率 embedding
>
> 约束：
>
> - 方案必须是确定性的
> - 不做危险猜测
> - 不补不存在的特征列
> - 不把框架改乱
> - 只做最有用的静态嵌入

---

# 1. 最终确定的静态特征

## 1.1 三个数据集统一只用这三个静态特征

最终确定：

- `x`
- `y`
- `elevation`


## 1.2 为什么就这三个

原因很简单：

- `x/y` 直接对应风机空间位置
- `elevation` 直接对应地形高度差
- 这三个最稳定、最通用、最接近风功率预测的空间先验
- 再加更多静态特征，第一版很容易引入噪声，得不偿失

## 1.3 各数据集的确定性读取方式

### SDWPF
直接读取：

- `x` <- `data/raw/sdwpf_turb_location_elevation.csv` 中的 `x`
- `y` <- 同上 `y`
- `elevation` <- 同上 `Ele`

### Penmanshiel
直接读取：

- `Latitude`
- `Longitude`
- `Elevation (m)`

再确定性转换为：

- `x`
- `y`
- `elevation`

其中：

- `x/y` 由 `Latitude/Longitude` 转局部米制坐标得到
- `elevation` 直接取 `Elevation (m)`

### Towie
直接读取：

- `Latitude`
- `Longitude`

再确定性转换为：

- `x`
- `y`

`Towie` 的 metadata 里当前没有单独 elevation 列，所以最终静态特征确定为：

- `x`
- `y`

为了统一接口：

- `static_feature_names` 仍然统一定义为 `['x', 'y', 'elevation']`
- `Towie` 的 `elevation` 固定置为 0

这不是“补动态特征”，而是统一静态矩阵维度；这个值来自“数据中确实没有海拔列，因此静态海拔维设为常数0”。

如果你不想这样统一，也可以让 `Towie` 只保留 2 维静态特征，但那会让静态编码层也动态变化。为了代码规范和干净，**推荐统一成 3 维静态特征**。

---

# 2. 最终确定的动态特征

这里不再模棱两可，直接给最终方案。

## 2.1 SDWPF
保持现有 16 维，不改：

- `Wspd`
- `Wdir`
- `Etmp`
- `Itmp`
- `Ndir`
- `Pab1`
- `Pab2`
- `Pab3`
- `Prtv`
- `T2m`
- `Sp`
- `RelH`
- `Wspd_w`
- `Wdir_w`
- `Tp`
- `Patv`

---

## 2.2 Penmanshiel
最终确定使用这 10 列：

- `Wspd` <- `Wind speed (m/s)`
- `Wdir` <- `Wind direction (°)`
- `Etmp` <- `Nacelle ambient temperature (°C)`
- `Itmp` <- `Nacelle temperature (°C)`
- `Ndir` <- `Nacelle position (°)`
- `Pab1` <- `Blade angle (pitch position) A (°)`
- `Pab2` <- `Blade angle (pitch position) B (°)`
- `Pab3` <- `Blade angle (pitch position) C (°)`
- `Prtv` <- `Reactive power (kvar)`
- `Patv` <- `Power (kW)`

### 为什么就这 10 列

因为这 10 列已经从真实 CSV 中确认存在，而且：

- 对构图足够（`Wspd/Wdir/Ndir`）
- 对功率预测最关键
- 与现有 SDWPF 框架语义最接近
- 不引入额外不稳定列

---

## 2.3 Towie
基于当前真实月文件列名，最终第一版确定使用这 7 列：

- `Wspd` <- `wtc_AcWindSp_mean`
- `Wdir` <- `wtc_ActualWindDirection_mean`
- `Ndir` <- `wtc_NacelPos_mean`
- `Pab1` <- `wtc_PitcPosA_mean`
- `Pab2` <- `wtc_PitcPosB_mean`
- `Pab3` <- `wtc_PitcPosC_mean`
- `Patv` <- `wtc_PowerRef_endvalue`


# 3. 新数据集清洗规则

## 3.1 SDWPF
继续沿用现有严格规则，不改。

## 3.2 Penmanshiel / Towie
统一采用轻量规则，只做绝对不合理值过滤。

最终规则：

1. `Patv` 缺失 -> invalid
2. 关键输入缺失 -> invalid
   - Penmanshiel：`Wspd/Wdir/Ndir/Pab1/Pab2/Pab3`
   - Towie：`Wspd/Wdir/Ndir/Pab1/Pab2/Pab3`
3. `Wspd < 0` -> invalid
4. `Patv < 0` -> clip 为 0
5. `Patv > rated_power * 1.05` -> invalid
6. `Wdir` / `Ndir` 超出设备合理角度范围 -> invalid

### 为什么不用更复杂规则

因为：

- 新数据集没有像 SDWPF 那样明确论文规则支撑
- 过于激进的过滤会误删有效样本
- 第一版应该保守、稳妥

---

# 4. 静态嵌入最终怎么做

- 对 `x/y/elevation` 先做标准化
- 再用一个**单层线性映射**投影到 `gcn.in_channel`

> **静态信息 = 标准化后的 `x/y/elevation` + 一个线性投影层 + 加法注入**
---

# 5. 多频率混合训练最终怎么做

## 5.1 只在 pretrain 阶段做

最终确定：

- **只在 pretrain 阶段做 5/10/15min 混合训练**
- `warmup_update()` 和 `streaming_test()` 不做人为混频

### 为什么

因为你要讲的故事是：

- backbone 在预训练阶段学会多频率适应
- streaming 阶段继续按真实数据流程进行

这样：

- 逻辑清晰
- 框架改动最小
- 实验解释最合理

---

## 5.2 训练阶段怎么混

最终确定：

1. 先把 SDWPF 原始数据插值到 5min 统一时间轴
2. 从这条 5min 时间轴中构造三类样本：
   - 5min 样本
   - 10min 样本
   - 15min 样本
3. 三类样本拼接成一个训练集
4. 每个样本增加 `freq_id`
5. 训练时混合 shuffle

## 5.3 三类样本的定义

继续保持当前框架：

- `x_len = 12`
- `y_len = 12`

也就是：

### 5min 样本
- 从 5min 时间轴直接取点
- 输入 12 步，预测 12 步

### 10min 样本
- 从 5min 时间轴每隔 2 个点取 1 个点
- 输入 12 步，预测 12 步

### 15min 样本
- 从 5min 时间轴每隔 3 个点取 1 个点
- 输入 12 步，预测 12 步

这三类样本一起训练。

---

## 5.4 频率 embedding 最终怎么做

最终确定：

- `freq_vocab = {5: 0, 10: 1, 15: 2}`
- 用一个 `nn.Embedding(3, freq_emb_dim)`
- 再通过一个线性层投影到 `gcn.in_channel`
- 然后加到输入上

即：

```python
freq_bias = Linear(freq_emb_dim, gcn_in_channel)(Embedding(freq_id))
x = x + freq_bias
```

### 为什么这里可以用 embedding

因为 `freq_id` 是离散类别：

- 5min
- 10min
- 15min

这正适合 `nn.Embedding`。

---

# 6. 真实预测阶段怎么处理

这是你特别要求我按最合理方式设计的地方，直接给最终方案。

## 6.1 SDWPF
### pretrain 阶段
- 用 5/10/15 三频混合训练

### streaming / 推理阶段
- 仍按真实原始时序协议进行
- 不做额外三频混合推理
- 用真实测试窗口构造样本

也就是说：

> 多频率是预训练增强策略，不是推理阶段协议。

### 为什么这是最合理的

因为：

- 你现在的 streaming 框架已经有节点扩展和 warmup 机制
- 如果预测阶段再混频，协议会变复杂，结果难解释
- 预训练混频、推理真实协议，是最稳妥、最论文友好的方案

---

## 6.2 Penmanshiel / Towie
这两个新数据集不做多频率训练。

最终确定：

- 按它们自己的原生频率训练和测试
- `freq_id` 固定为该数据集原生频率
---

# 7. 框架最终怎么改

这里只说最终要怎么改，不说废话。

## 7.1 `config/config.py`

新增：

- `dataset`
- `use_static_embedding`
- `use_freq_embedding`
- `freq_emb_dim`
- `base_resolution_minutes`
- `train_freqs`

作用：

- 支持多数据集
- 控制静态嵌入
- 控制 SDWPF 多频率预训练

---

## 7.2 `config/dataset_registry.py`（新增）

集中写死三个数据集的：

- raw 路径
- location 路径
- 动态特征映射
- 静态特征映射
- 额定功率
- 默认频率
- 默认 schedule

作用：

- 去掉大量 if/else
- 所有数据集配置集中管理

---

## 7.3 `config/node_schedule.py`

改成 dataset-aware。

最终确定：

### SDWPF
沿用现有逻辑。

### Penmanshiel（15台）
- initial: 1-8
- expansion1: 9-11
- expansion2: 12-15

### Towie（21台）
- initial: 1-11
- expansion1: 12-16
- expansion2: 17-21

---

## 7.4 `data/adapters/`（新增）

新增：

- `sdwpf_adapter.py`
- `penmanshiel_adapter.py`
- `towie_adapter.py`

作用：

- 每个数据集只负责把原始数据读取成统一格式
- 主框架不再写满数据集分支逻辑

---

## 7.5 `data/data_processing.py`

最终改法：

- `process_unified_dataset(dataset=...)`
- 从 adapter 读取：
  - `df_data`
  - `df_location`
  - `feature_cols`
  - `static_features`
- 统一生成：
  - `raw_data`
  - `patv_mask`
  - `static_data`
  - `unified_data.npz`

对 SDWPF 额外增加：

- `build_sdwpf_5min_base()`
- `build_multifreq_pretrain_inputs()`

---

## 7.6 `data/dataset.py`

扩展 Data 返回内容：

- `x`
- `y`
- `y_mask`
- `static_x`
- `freq_id`

作用：

- 模型统一接收动态、静态和频率信息

---

## 7.7 `trainer.py`

最终改法：

### `pretrain()`
- 若 `dataset == sdwpf`：走多频率样本构建
- 否则：走普通单频样本构建

### `warmup_update()` / `streaming_test()`
- 构造 Data 时加入 `static_x`
- `freq_id` 对新数据集固定为原生频率
- `sdwpf` 推理阶段按真实频率窗口给固定 `freq_id`

---

## 7.8 模型文件

修改：

- `model/EAC_model.py`
- `model/scale_shift.py`
- `model/variational_scale_shift.py`
- `model/patchtst.py`

统一增加两部分：

1. `static_proj = nn.Linear(static_dim, gcn_in_channel)`
2. `freq_embedding + freq_proj`

forward 中统一：

```python
x = x + static_bias + freq_bias
```

然后再走原有 backbone。

---

# 8. 最终推荐的实现顺序

1. 先写 `dataset_registry.py`
2. 写 `Penmanshiel` / `Towie` adapter
3. 改 `data_processing.py` 统一输出三数据集 processed 数据
4. 加入 `static_x`
5. 给四个模型加静态投影
6. 再做 SDWPF 的 5/10/15 混合预训练
7. 最后接入频率 embedding

---

# 9. 最终方案一句话总结

最终方案就是：

- **新数据集只用真实存在的动态特征，不补齐**
- **静态特征统一只用 `x/y/elevation`**
- **静态信息用线性投影加到输入上**
- **SDWPF 只在预训练阶段做 5/10/15 三频混合训练**
- **频率用 `nn.Embedding` 表示，预测阶段仍按真实协议进行**

这套方案最简单、最确定、最规范，也最符合你现在要的效果导向。
