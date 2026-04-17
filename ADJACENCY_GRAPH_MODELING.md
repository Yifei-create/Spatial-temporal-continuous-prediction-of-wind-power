# 邻接矩阵建模说明

本文档用于统一记录本项目中邻接矩阵的建模方式，方便后续实现、实验记录与论文写作。

## 1. 总览

当前保留两类建图方式：

1. 默认距离图（baseline）
2. 局部上游风向图（最终确定方案）

其中：

- 默认距离图用于提供与常规时空图学习方法一致的空间邻接基线。
- 局部上游风向图用于刻画风机之间的方向性影响关系，适用于持续学习场景下的动态扩图。

---

## 2. 默认距离图（Baseline）

默认邻接矩阵 $A_\tau$ 的元素定义为：

$$
A_{\tau, ij} =
\begin{cases}
\exp\left(-\frac{d_{ij}^2}{\sigma^2}\right) & \text{if } \exp\left(-\frac{d_{ij}^2}{\sigma^2}\right) \ge r \text{ and } i \neq j \\
0 & \text{otherwise}
\end{cases}
$$

变量说明：

- $d_{ij}$：传感器或风机 $i$ 与 $j$ 之间的实际地理距离。
- $\sigma$：所有节点对距离的标准差，用于缩放距离影响。
- $r$：阈值，用于保证图的稀疏性，只有权重大于等于该阈值的边被保留。

建议设置：

- 通用公式里 `r` 是显式阈值超参数。
- 对 `SDWPF`，若严格取 `r = 0.99`，按当前坐标尺度会退化成空图；当前运行时配置显式使用 `r = 0.95`。
- 去自环，即 $A_{\tau, ii} = 0$。

说明：

- 该图为无向距离图，可在得到矩阵后按需要做对称化。
- 若训练阶段需要数值稳定性，可在载入训练前再进行归一化。

---

## 3. 局部上游风向图（最终确定方案）

该方案用于建模风机之间的有向影响关系，核心思想是：若风机 $j$ 的风场流向长期朝向风机 $i$，则认为 $j \to i$ 存在更强的上游影响。

### 3.1 确定参与建图的节点集合

只对当前阶段的风机集合 `current_cols` 建图，共有

$$
N = |current\_cols|
$$

个节点。

在持续学习场景中，网络会随新风机接入而动态增长：

$$
\mathcal{G}_{\tau} = \mathcal{G}_{\tau-1} + \Delta \mathcal{G}_{\tau}
$$

因此每个阶段 $\tau$ 都需要基于当前累计节点重新生成图结构。

### 3.2 计算风机两两之间的空间关系

设第 $j$ 台风机坐标为 $(x_j, y_j)$，第 $i$ 台风机坐标为 $(x_i, y_i)$。

位移向量：

$$
\Delta x_{ji} = x_i - x_j,\qquad \Delta y_{ji} = y_i - y_j
$$

空间方位角：

$$
\phi_{ji} = \operatorname{atan2}(\Delta y_{ji}, \Delta x_{ji})
$$

欧氏距离：

$$
d_{ji} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

说明：

- $\phi_{ji}$ 使用 `atan2` 的数学坐标系定义。
- 其参考方向为正 $x$ 轴，逆时针为正方向。

### 3.3 风向方向与坐标系统一

这一步是整个建图里最容易写错的部分。

在当前项目涉及的四个数据集中，原始风向都统一按“来向”处理，即表示风从哪个方向吹来，而不是风吹向哪个方向。

因此，为了和节点间空间方位角 $\phi_{ji}$ 比较，必须依次做两步转换：

1. 来向转换为去向
2. 气象角转换为 `atan2` 使用的数学角

先定义绝对来风方向 $\alpha^{from}_{jt}$，单位为度。其具体构造方式依赖数据集，见第 4 节。

第一步，来向转去向：

$$
\alpha^{to}_{jt} = (\alpha^{from}_{jt} + 180^\circ) \bmod 360^\circ
$$

第二步，将气象角转成数学角：

$$
\theta_{jt} = \operatorname{deg2rad}\big((90^\circ - \alpha^{to}_{jt}) \bmod 360^\circ\big)
$$

等价写法为：

$$
\theta_{jt} = \operatorname{deg2rad}\big((270^\circ - \alpha^{from}_{jt}) \bmod 360^\circ\big)
$$

其中：

- $\alpha^{from}_{jt}$：绝对来风方向，按气象定义记录。
- $\alpha^{to}_{jt}$：风的实际吹向。
- $\theta_{jt}$：与 $\phi_{ji}$ 同坐标系的数学角，可直接代入余弦计算。

说明：

- `deg2rad` 只表示把角度从度转换为弧度，以便传给 `cos`、`atan2` 等数学函数。
- 不能直接用原始 `Wdir` 或 `Ndir + Wdir` 去和 $\phi_{ji}$ 相减，因为前者是气象角，后者是数学角。

### 3.4 计算局部上游强度

对于有向边 $j \to i$，定义角度差：

$$
\Delta \theta_{tji} = \theta_{jt} - \phi_{ji}
$$

用余弦刻画风向与空间连线的一致程度，并截断负值：

$$
s_{tji} = \max\bigl(0,\cos(\theta_{jt} - \phi_{ji})\bigr)
$$

对所有训练时刻 $T$ 求平均，得到长期上游概率强度：

$$
p_{ji} = \frac{1}{T}\sum_{t=1}^{T}\max\bigl(0,\cos(\theta_{jt} - \phi_{ji})\bigr)
$$

于是得到有向强度矩阵：

$$
\mathbf{P} = [p_{ji}] \in \mathbb{R}^{N \times N}
$$

其中一般有 $p_{ji} \neq p_{ij}$。

### 3.5 结合高斯核得到原始边权

为了对齐论文中对能源数据的处理方式，使用高斯核代替简单的反比函数，使近距离影响更加平滑：

$$
A_{ji} = p_{ji} \cdot \exp\left(-\frac{d_{ji}^2}{\sigma^2}\right)
$$

其中：

- $\sigma$ 为当前阶段所有节点对距离的标准差。

并去除自环：

$$
A_{ii} = 0
$$

### 3.6 对每个目标节点做 Top-K 稀疏化

为防止图神经网络过度平滑，对每个目标风机 $i$，仅保留指向它的权重最大的 $K$ 个来源节点：

$$
\mathcal{N}^{in}_i = \operatorname{TopK}\{A_{1i}, A_{2i}, \dots, A_{Ni}\}
$$

得到稀疏矩阵 $\mathbf{A}^{sparse}$。

### 3.7 行归一化

最后对每个源节点 $j$ 的所有出边进行归一化：

$$
\tilde{A}_{ji} =
\frac{A^{sparse}_{ji}}{\sum_{k=1}^{N} A^{sparse}_{jk} + 10^{-6}}
$$

这样可以提高消息传递时的数值稳定性。

---

## 4. 各数据集的处理方式

### 4.1 SDWPF

特征情况：

- 有 `Wdir`
- 有 `Ndir`
- `Wdir` 为相对机舱方向的风向角
- `Ndir` 为机舱朝向

方向语义：

- `Ndir + Wdir` 恢复的是绝对来风方向。

处理方式：

$$
\alpha^{from}_{jt} = (Ndir_{jt} + Wdir_{jt}) \bmod 360^\circ
$$

$$
\theta_{jt} = \operatorname{deg2rad}\big((270^\circ - \alpha^{from}_{jt}) \bmod 360^\circ\big)
$$

说明：

- `SDWPF` 是当前项目中最适合使用该局部上游重构公式的数据集。
- 若论文中只讨论绝对风向，可写为 `Ndir + Wdir`。
- 若实现中需要与 `atan2` 比较，则必须进一步转换为 $\theta_{jt}$。

### 4.2 Penmanshiel

特征情况：

- 有 `Wdir`
- 有 `Ndir`
- `Wdir` 对应原始 SCADA 字段 `Wind direction (°)`
- `Ndir` 对应原始 SCADA 字段 `Nacelle position (°)`

方向语义：

- `Wdir` 作为绝对来风方向处理。
- `Ndir` 为机舱朝向，不参与绝对风向重构。

处理方式：

$$
\alpha^{from}_{jt} = Wdir_{jt}
$$

$$
\theta_{jt} = \operatorname{deg2rad}\big((270^\circ - \alpha^{from}_{jt}) \bmod 360^\circ\big)
$$

说明：

- 该数据集中的 `Wdir` 不再与 `Ndir` 相加，否则会重复编码方向信息。

### 4.3 HLRS

特征情况：

- 有 `Wdir`
- 无 `Ndir`
- `Wdir` 来自 ERA5 风向变量

方向语义：

- `Wdir` 作为绝对来风方向处理。

处理方式：

$$
\alpha^{from}_{jt} = Wdir_{jt}
$$

$$
\theta_{jt} = \operatorname{deg2rad}\big((270^\circ - \alpha^{from}_{jt}) \bmod 360^\circ\big)
$$

说明：

- 由于缺少 `Ndir`，不能使用 `Ndir + Wdir` 的重构公式。

### 4.4 Norrekaer Enge

特征情况：

- 有 `Wdir`
- 有 `Ndir`
- `Wdir` 对应场级 `wind_from_direction`
- `Ndir` 对应机组 `yaw_angle`

方向语义：

- `Wdir` 明确为绝对来风方向。
- `Ndir` 表示机组偏航位置，不用于恢复绝对风向。

处理方式：

$$
\alpha^{from}_{jt} = Wdir_{jt}
$$

$$
\theta_{jt} = \operatorname{deg2rad}\big((270^\circ - \alpha^{from}_{jt}) \bmod 360^\circ\big)
$$

---

## 5. 推荐写法

若论文中需要统一描述，可采用如下表述：

- 当前运行时代码同时支持两类图：`baseline` 与 `local_upstream`。
- `baseline` 使用距离高斯核加阈值稀疏化。
- `local_upstream` 使用局部上游强度与距离高斯核的乘积，并对每个目标节点做 Top-K 稀疏化。
- 当前项目中各数据集原始风向统一按来风方向处理，而非吹向方向。
- 为与节点间空间方位角进行比较，需先将来向转换为去向，再将气象角转换到数学坐标系。
- 对 `SDWPF`，绝对来风方向由 `Ndir + Wdir` 重构。
- 对 `Penmanshiel`、`HLRS`、`Norrekaer Enge`，绝对来风方向直接由 `Wdir` 提供。

---

## 6. 实现备注

实现时建议注意以下事项：

- `current_cols` 必须表示当前阶段累计可见的节点集合，而不是仅新增节点集合。
- 距离标准差 $\sigma$ 应在当前阶段节点对上重新计算。
- `Top-K` 建议对每个目标节点的入边执行。
- 代码里若使用 `adj @ x` 进行消息传递，需确认矩阵存储方向与 $j \to i$ 的定义一致，必要时做转置。
- 风向增强图的实现不要直接拿原始 `Wdir` 或 `Ndir + Wdir` 与 `atan2` 输出比较，必须先完成来向/去向转换和坐标系统一。
- 当前工程运行时要求 `adj_type` 显式取 `baseline` 或 `local_upstream`，不再使用旧的 `distance` 命名。
