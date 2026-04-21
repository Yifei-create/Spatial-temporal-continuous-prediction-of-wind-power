# 基于 STGNN Backbone 的当前保留变体

## 当前保留的方法

当前只保留 3 个和 STGNN 相关的方法：

- `STGNN`
- `STGNNInputBias`
- `STGNNStageResidual`

其中：

- `STGNN` 是纯共享 backbone
- `STGNNInputBias` 是最简单的节点输入偏置版本
- `STGNNStageResidual` 是阶段偏置 + 节点残差版本

其余轻量试验变体已经从运行代码中移除，不再作为可选方法。

## 共享主干

这 3 个方法共享同一个主干：

- `GCN -> TCN -> GCN -> residual add -> GELU -> FC`

输入是每个节点展平后的历史窗口：

- `D = x_len * num_features`

输出是未来 `y_len` 步的 `Patv` 预测。

## 1. STGNN

### 形式

纯 backbone，不带任何节点自适应参数。

### 特点

- 参数最少
- 没有 warmup 依赖
- 扩容后直接在新节点集合上联合预测

## 2. STGNNInputBias

### 形式

对每个节点维护一个输入偏置向量：

`x_i' = x_i + b_i`

其中：

- `b_i ∈ R^D`

### 特点

- 最简单的节点个性化方式
- 只做加法，不做缩放
- 新节点参数初始化为 0，warmup 时再学习

## 3. STGNNStageResidual

### 形式

在输入端同时加入阶段共享偏置和节点残差：

`x_i' = x_i + e_stage + r_i`

其中：

- `e_stage ∈ R^D`
- `r_i ∈ R^D`

### 特点

- `e_stage` 用来建模扩容阶段带来的整体分布变化
- `r_i` 用来建模节点个体偏移
- 两者初始化都为 0，所以模型初始行为和纯 `STGNN` 一致

## warmup 约定

当前代码里：

- `STGNN` 不使用 warmup
- `STGNNInputBias` 支持 warmup
- `STGNNStageResidual` 支持 warmup

## 适用理解

如果只看复杂度和解释性：

- `STGNN`：最干净的共享基线
- `STGNNInputBias`：最轻的节点适配
- `STGNNStageResidual`：在轻量前提下，额外显式考虑“阶段漂移”
