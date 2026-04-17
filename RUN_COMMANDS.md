# Run Commands

当前重构后的主线只支持：

- 数据集：`sdwpf`
- 邻接矩阵：`baseline`、`local_upstream`
- 方法：`ScaleShift`、`VariationalScaleShift`、`EAC`、`PatchTST`

## 1. 缓存与预处理规则

当前行为固定如下：

- `--data_process 1 --train 0`：强制重建缓存，然后结束。
- `--data_process 1 --train 1`：强制重建缓存，然后继续训练和流式测试。
- `--data_process 0 --train 1`：优先读取现有缓存；如果没检测到缓存，会自动先重建，再继续训练和流式测试。
- `--data_process 0 --train 0`：不重建缓存，直接做流式测试；前提是实验权重已经存在。

如果你只是想单独重建缓存，可以运行：

```bash
python main.py --dataset sdwpf --graph_variant baseline --data_process 1 --train 0 --logname preprocess_sdwpf_baseline
```

缓存与预处理产物会写到：

- `data/processed/sdwpf/preprocess__graph-baseline__seed-42__x12_y12__exp-auto/`
- `data/graph/sdwpf/preprocess__graph-baseline__seed-42__x12_y12__exp-auto/`
- `results/sdwpf/preprocess__graph-baseline__seed-42__x12_y12__exp-auto/`

如果要强制重建 `local_upstream` 图缓存：

```bash
python main.py --dataset sdwpf --graph_variant local_upstream --data_process 1 --train 0 --logname preprocess_sdwpf_local_upstream
```

如果你想“一条命令直接跑”，最常用的是：

```bash
python main.py --dataset sdwpf --graph_variant baseline --method ScaleShift --train 1 --seed 42 --logname scaleshift_sdwpf
```

如果缓存已经存在，会直接训练和测试；如果缓存不存在，会自动先处理再训练。

训练和测试读取的缓存目录同样是对应的 `preprocess__...` 目录，例如：

- `data/processed/sdwpf/preprocess__graph-baseline__seed-42__x12_y12__exp-auto/unified_data.npz`
- `data/graph/sdwpf/preprocess__graph-baseline__seed-42__x12_y12__exp-auto/stage_*.npz`

如果你明确要强制重建后再训练：

```bash
python main.py --dataset sdwpf --graph_variant baseline --data_process 1 --method ScaleShift --train 1 --seed 42 --logname scaleshift_sdwpf
```

如果你要做 no-warmup 消融：

```bash
python main.py --dataset sdwpf --graph_variant baseline --method ScaleShift --train 0 --seed 42 --no_warmup 1 --logname scaleshift_sdwpf_no_warmup
```

这会关闭 streaming 阶段的 warmup，并把结果写到单独的 `warmup-off` 实验目录。

## 2. 训练并测试

### ScaleShift

```bash
python main.py --dataset sdwpf --graph_variant baseline --method ScaleShift --train 1 --seed 42 --logname scaleshift_sdwpf
```

### VariationalScaleShift

```bash
python main.py --dataset sdwpf --graph_variant baseline --method VariationalScaleShift --train 1 --seed 42 --logname variational_scaleshift_sdwpf
```

### EAC

```bash
python main.py --dataset sdwpf --graph_variant baseline --method EAC --train 1 --seed 42 --logname eac_sdwpf
```

### PatchTST

```bash
python main.py --dataset sdwpf --graph_variant baseline --method PatchTST --train 1 --seed 42 --logname patchtst_sdwpf
```

## 3. 只做流式测试

如果已经有对应方法的 `best.pt` checkpoint，可以直接运行：

### ScaleShift

```bash
python main.py --dataset sdwpf --graph_variant baseline --method ScaleShift --train 0 --seed 42 --logname scaleshift_sdwpf
```

### VariationalScaleShift

```bash
python main.py --dataset sdwpf --graph_variant baseline --method VariationalScaleShift --train 0 --seed 42 --logname variational_scaleshift_sdwpf
```

### EAC

```bash
python main.py --dataset sdwpf --graph_variant baseline --method EAC --train 0 --seed 42 --logname eac_sdwpf
```

### PatchTST

```bash
python main.py --dataset sdwpf --graph_variant baseline --method PatchTST --train 0 --seed 42 --logname patchtst_sdwpf
```

## 4. 当前行为说明

- 数据切分固定为 `2:1:7` 的 train/val/test。
- 测试阶段使用流式预测：每次预测未来 `12` 步，每过 `12` 步接收一个新样本，再决定是否 warmup 后继续预测。
- `ScaleShift` 和 `VariationalScaleShift` 使用原始频率窗口、静态位置嵌入和频率信号注入。
- `EAC` 和 `PatchTST` 保留各自结构，不使用多频训练采样和静态位置嵌入。
- warmup 只对 `ScaleShift`、`VariationalScaleShift`、`EAC` 生效；`PatchTST` 不做 warmup。

## 5. 结果目录结构

训练/测试结果产物统一写到：

```text
results/{dataset}/{experiment_dir}/
  run.log
  config.json
  metrics.json
  checkpoints/
    best.pt
    last.pt
  predictions/
    streaming_predictions.npz
```

其中 `experiment_dir` 使用主要参数命名，例如：

```text
ScaleShift__graph-baseline__seed-42__x12_y12__bs64__lr0p001__drop0__warmup-on__exp-auto
```
