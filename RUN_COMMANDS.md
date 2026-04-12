# 运行与实验命令说明

本文档给出当前框架在 `sdwpf`、`penmanshiel`、`hlrs`、`norrekaer_enge` 四个数据集上的完整运行命令。

本文档已经基于你当前最新代码更新，主要变化如下：

1. 已删除 `Towie` 数据集，不再提供任何 `Towie` 命令。
2. 处理结果、图结构、日志与模型目录都带有 `processing_tag`，不同数据集和不同处理版本不会互相覆盖。
3. 静态特征已经统一为 `x/y`。
4. `HLRS` 与 `Norrekaer Enge` 已纳入运行入口。

---

# 一、当前支持的数据集

当前 `main.py --dataset` 只支持以下四个数据集：

- `sdwpf`
- `penmanshiel`
- `hlrs`
- `norrekaer_enge`

---

# 二、处理后目录说明

## 1. 处理数据目录

### SDWPF

- `data/processed/sdwpf/final_xy_16feat_multifreq_maskv1/distance/`
- `data/processed/sdwpf/final_xy_16feat_multifreq_maskv1/wind_aware_local/`
- `data/processed/sdwpf/final_xy_16feat_multifreq_maskv1/wind_aware/`

### Penmanshiel

- `data/processed/penmanshiel/final_xy_10feat_10min_maskv1/distance/`
- `data/processed/penmanshiel/final_xy_10feat_10min_maskv1/wind_aware_local/`
- `data/processed/penmanshiel/final_xy_10feat_10min_maskv1/wind_aware/`

### HLRS

- `data/processed/hlrs/final_xy_5feat_hourly_maskv1/distance/`
- `data/processed/hlrs/final_xy_5feat_hourly_maskv1/wind_aware_local/`
- `data/processed/hlrs/final_xy_5feat_hourly_maskv1/wind_aware/`

### Norrekaer Enge

- `data/processed/norrekaer_enge/final_xy_4feat_fullrange10min_maskv1/distance/`
- `data/processed/norrekaer_enge/final_xy_4feat_fullrange10min_maskv1/wind_aware_local/`
- `data/processed/norrekaer_enge/final_xy_4feat_fullrange10min_maskv1/wind_aware/`

---

## 2. 图目录

### SDWPF

- `data/graph/sdwpf/final_xy_16feat_multifreq_maskv1/distance/`
- `data/graph/sdwpf/final_xy_16feat_multifreq_maskv1/wind_aware_local/`
- `data/graph/sdwpf/final_xy_16feat_multifreq_maskv1/wind_aware/`

### Penmanshiel

- `data/graph/penmanshiel/final_xy_10feat_10min_maskv1/distance/`
- `data/graph/penmanshiel/final_xy_10feat_10min_maskv1/wind_aware_local/`
- `data/graph/penmanshiel/final_xy_10feat_10min_maskv1/wind_aware/`

### HLRS

- `data/graph/hlrs/final_xy_5feat_hourly_maskv1/distance/`
- `data/graph/hlrs/final_xy_5feat_hourly_maskv1/wind_aware_local/`
- `data/graph/hlrs/final_xy_5feat_hourly_maskv1/wind_aware/`

### Norrekaer Enge

- `data/graph/norrekaer_enge/final_xy_4feat_fullrange10min_maskv1/distance/`
- `data/graph/norrekaer_enge/final_xy_4feat_fullrange10min_maskv1/wind_aware_local/`
- `data/graph/norrekaer_enge/final_xy_4feat_fullrange10min_maskv1/wind_aware/`

---

## 3. 日志与模型目录

日志、checkpoint、预测结果按下面规则保存：

- `log/{dataset}-{processing_tag}-{adj_type}-{logname}-{seed}/`

例如：

- `log/sdwpf-final_xy_16feat_multifreq_maskv1-distance-eac-42/`
- `log/penmanshiel-final_xy_10feat_10min_maskv1-wind_aware_local-eac-42/`
- `log/hlrs-final_xy_5feat_hourly_maskv1-distance-eac-42/`
- `log/norrekaer_enge-final_xy_4feat_fullrange10min_maskv1-wind_aware_local-eac-42/`

目录中会保留：

- 日志文件
- `pretrain/` checkpoint
- `results/` 预测结果

---

# 三、建议的实验顺序

建议严格按照下面顺序进行：

1. 先处理数据：`--data_process 1 --train 0`
2. 再正式训练与测试：`--data_process 0 --train 1`
3. 如果已有 checkpoint，只做 streaming 测试：`--data_process 0 --train 0`
4. 如果做 warmup 消融，加上：`--no_warmup 1`

---

# 四、先重新处理数据

说明：

- 首次运行某个 `dataset × adj_type` 组合时，必须先处理数据。
- 如果你修改了特征列、mask 规则、processing tag、时间范围，也必须重新处理数据。

---

## 1. SDWPF 数据处理

### distance

```bash
python main.py --dataset sdwpf --adj_type distance --data_process 1 --train 0 --logname preprocess_sdwpf_distance
```

### wind_aware_local

```bash
python main.py --dataset sdwpf --adj_type wind_aware_local --data_process 1 --train 0 --logname preprocess_sdwpf_wal
```

### wind_aware

```bash
python main.py --dataset sdwpf --adj_type wind_aware --data_process 1 --train 0 --logname preprocess_sdwpf_wa
```

---

## 2. Penmanshiel 数据处理

### distance

```bash
python main.py --dataset penmanshiel --adj_type distance --data_process 1 --train 0 --logname preprocess_penmanshiel_distance
```

### wind_aware_local

```bash
python main.py --dataset penmanshiel --adj_type wind_aware_local --data_process 1 --train 0 --logname preprocess_penmanshiel_wal
```

### wind_aware

```bash
python main.py --dataset penmanshiel --adj_type wind_aware --data_process 1 --train 0 --logname preprocess_penmanshiel_wa
```

---

## 3. HLRS 数据处理

### distance

```bash
python main.py --dataset hlrs --adj_type distance --data_process 1 --train 0 --logname preprocess_hlrs_distance
```

### wind_aware_local

```bash
python main.py --dataset hlrs --adj_type wind_aware_local --data_process 1 --train 0 --logname preprocess_hlrs_wal
```

### wind_aware

```bash
python main.py --dataset hlrs --adj_type wind_aware --data_process 1 --train 0 --logname preprocess_hlrs_wa
```

---

## 4. Norrekaer Enge 数据处理

### distance

```bash
python main.py --dataset norrekaer_enge --adj_type distance --data_process 1 --train 0 --logname preprocess_norre_distance
```

### wind_aware_local

```bash
python main.py --dataset norrekaer_enge --adj_type wind_aware_local --data_process 1 --train 0 --logname preprocess_norre_wal
```

### wind_aware

```bash
python main.py --dataset norrekaer_enge --adj_type wind_aware --data_process 1 --train 0 --logname preprocess_norre_wa
```

---

# 五、正式训练与测试命令

下面给的是最常用的正式训练命令。

默认会执行：

1. 读取已经处理好的数据
2. 预训练
3. streaming 测试
4. 保存日志、checkpoint、预测结果

---

## 1. SDWPF 训练与测试

### distance

```bash
python main.py --dataset sdwpf --adj_type distance --method EAC --train 1 --seed 42 --logname eac_dist_sdwpf
python main.py --dataset sdwpf --adj_type distance --method ScaleShift --train 1 --seed 42 --logname ss_dist_sdwpf
python main.py --dataset sdwpf --adj_type distance --method VariationalScaleShift --train 1 --seed 42 --logname vss_dist_sdwpf
python main.py --dataset sdwpf --adj_type distance --method PatchTST --train 1 --seed 42 --logname patchtst_dist_sdwpf
```

### wind_aware_local

```bash
python main.py --dataset sdwpf --adj_type wind_aware_local --method EAC --train 1 --seed 42 --logname eac_wal_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware_local --method ScaleShift --train 1 --seed 42 --logname ss_wal_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware_local --method VariationalScaleShift --train 1 --seed 42 --logname vss_wal_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware_local --method PatchTST --train 1 --seed 42 --logname patchtst_wal_sdwpf
```

### wind_aware

```bash
python main.py --dataset sdwpf --adj_type wind_aware --method EAC --train 1 --seed 42 --logname eac_wa_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware --method ScaleShift --train 1 --seed 42 --logname ss_wa_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware --method VariationalScaleShift --train 1 --seed 42 --logname vss_wa_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware --method PatchTST --train 1 --seed 42 --logname patchtst_wa_sdwpf
```

---

## 2. Penmanshiel 训练与测试

### distance

```bash
python main.py --dataset penmanshiel --adj_type distance --method EAC --train 1 --seed 42 --logname eac_dist_penmanshiel
python main.py --dataset penmanshiel --adj_type distance --method ScaleShift --train 1 --seed 42 --logname ss_dist_penmanshiel
python main.py --dataset penmanshiel --adj_type distance --method VariationalScaleShift --train 1 --seed 42 --logname vss_dist_penmanshiel
python main.py --dataset penmanshiel --adj_type distance --method PatchTST --train 1 --seed 42 --logname patchtst_dist_penmanshiel
```

### wind_aware_local

```bash
python main.py --dataset penmanshiel --adj_type wind_aware_local --method EAC --train 1 --seed 42 --logname eac_wal_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware_local --method ScaleShift --train 1 --seed 42 --logname ss_wal_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware_local --method VariationalScaleShift --train 1 --seed 42 --logname vss_wal_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware_local --method PatchTST --train 1 --seed 42 --logname patchtst_wal_penmanshiel
```

### wind_aware

```bash
python main.py --dataset penmanshiel --adj_type wind_aware --method EAC --train 1 --seed 42 --logname eac_wa_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware --method ScaleShift --train 1 --seed 42 --logname ss_wa_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware --method VariationalScaleShift --train 1 --seed 42 --logname vss_wa_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware --method PatchTST --train 1 --seed 42 --logname patchtst_wa_penmanshiel
```

---

## 3. HLRS 训练与测试

### distance

```bash
python main.py --dataset hlrs --adj_type distance --method EAC --train 1 --seed 42 --logname eac_dist_hlrs
python main.py --dataset hlrs --adj_type distance --method ScaleShift --train 1 --seed 42 --logname ss_dist_hlrs
python main.py --dataset hlrs --adj_type distance --method VariationalScaleShift --train 1 --seed 42 --logname vss_dist_hlrs
python main.py --dataset hlrs --adj_type distance --method PatchTST --train 1 --seed 42 --logname patchtst_dist_hlrs
```

### wind_aware_local

```bash
python main.py --dataset hlrs --adj_type wind_aware_local --method EAC --train 1 --seed 42 --logname eac_wal_hlrs
python main.py --dataset hlrs --adj_type wind_aware_local --method ScaleShift --train 1 --seed 42 --logname ss_wal_hlrs
python main.py --dataset hlrs --adj_type wind_aware_local --method VariationalScaleShift --train 1 --seed 42 --logname vss_wal_hlrs
python main.py --dataset hlrs --adj_type wind_aware_local --method PatchTST --train 1 --seed 42 --logname patchtst_wal_hlrs
```

### wind_aware

```bash
python main.py --dataset hlrs --adj_type wind_aware --method EAC --train 1 --seed 42 --logname eac_wa_hlrs
python main.py --dataset hlrs --adj_type wind_aware --method ScaleShift --train 1 --seed 42 --logname ss_wa_hlrs
python main.py --dataset hlrs --adj_type wind_aware --method VariationalScaleShift --train 1 --seed 42 --logname vss_wa_hlrs
python main.py --dataset hlrs --adj_type wind_aware --method PatchTST --train 1 --seed 42 --logname patchtst_wa_hlrs
```

---

## 4. Norrekaer Enge 训练与测试

### distance

```bash
python main.py --dataset norrekaer_enge --adj_type distance --method EAC --train 1 --seed 42 --logname eac_dist_norre
python main.py --dataset norrekaer_enge --adj_type distance --method ScaleShift --train 1 --seed 42 --logname ss_dist_norre
python main.py --dataset norrekaer_enge --adj_type distance --method VariationalScaleShift --train 1 --seed 42 --logname vss_dist_norre
python main.py --dataset norrekaer_enge --adj_type distance --method PatchTST --train 1 --seed 42 --logname patchtst_dist_norre
```

### wind_aware_local

```bash
python main.py --dataset norrekaer_enge --adj_type wind_aware_local --method EAC --train 1 --seed 42 --logname eac_wal_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware_local --method ScaleShift --train 1 --seed 42 --logname ss_wal_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware_local --method VariationalScaleShift --train 1 --seed 42 --logname vss_wal_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware_local --method PatchTST --train 1 --seed 42 --logname patchtst_wal_norre
```

### wind_aware

```bash
python main.py --dataset norrekaer_enge --adj_type wind_aware --method EAC --train 1 --seed 42 --logname eac_wa_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware --method ScaleShift --train 1 --seed 42 --logname ss_wa_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware --method VariationalScaleShift --train 1 --seed 42 --logname vss_wa_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware --method PatchTST --train 1 --seed 42 --logname patchtst_wa_norre
```

---

# 六、如果只想测试，不重新训练

如果你已经有对应目录下的 `pretrain` checkpoint，可以直接只做 streaming 测试。

说明：

- `--train 0` 表示不重新预训练
- 默认仍会执行 streaming test

---

## 1. SDWPF

```bash
python main.py --dataset sdwpf --adj_type distance --method EAC --train 0 --seed 42 --logname eac_dist_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware_local --method EAC --train 0 --seed 42 --logname eac_wal_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware --method EAC --train 0 --seed 42 --logname eac_wa_sdwpf
```

## 2. Penmanshiel

```bash
python main.py --dataset penmanshiel --adj_type distance --method EAC --train 0 --seed 42 --logname eac_dist_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware_local --method EAC --train 0 --seed 42 --logname eac_wal_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware --method EAC --train 0 --seed 42 --logname eac_wa_penmanshiel
```

## 3. HLRS

```bash
python main.py --dataset hlrs --adj_type distance --method EAC --train 0 --seed 42 --logname eac_dist_hlrs
python main.py --dataset hlrs --adj_type wind_aware_local --method EAC --train 0 --seed 42 --logname eac_wal_hlrs
python main.py --dataset hlrs --adj_type wind_aware --method EAC --train 0 --seed 42 --logname eac_wa_hlrs
```

## 4. Norrekaer Enge

```bash
python main.py --dataset norrekaer_enge --adj_type distance --method EAC --train 0 --seed 42 --logname eac_dist_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware_local --method EAC --train 0 --seed 42 --logname eac_wal_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware --method EAC --train 0 --seed 42 --logname eac_wa_norre
```

---

# 七、warmup 消融实验命令

当前框架中最直接的 warmup 消融开关是：

- 正常版本：`--no_warmup 0`
- 去掉 warmup：`--no_warmup 1`

建议做消融时：

1. 先跑一遍正常版本，生成对应 checkpoint
2. 再在相同设置下使用 `--train 0 --no_warmup 1` 只做 streaming 消融

---

## 1. SDWPF warmup 消融

```bash
python main.py --dataset sdwpf --adj_type distance --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_dist_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware_local --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_wal_sdwpf
python main.py --dataset sdwpf --adj_type wind_aware --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_wa_sdwpf
```

## 2. Penmanshiel warmup 消融

```bash
python main.py --dataset penmanshiel --adj_type distance --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_dist_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware_local --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_wal_penmanshiel
python main.py --dataset penmanshiel --adj_type wind_aware --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_wa_penmanshiel
```

## 3. HLRS warmup 消融

```bash
python main.py --dataset hlrs --adj_type distance --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_dist_hlrs
python main.py --dataset hlrs --adj_type wind_aware_local --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_wal_hlrs
python main.py --dataset hlrs --adj_type wind_aware --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_wa_hlrs
```

## 4. Norrekaer Enge warmup 消融

```bash
python main.py --dataset norrekaer_enge --adj_type distance --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_dist_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware_local --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_wal_norre
python main.py --dataset norrekaer_enge --adj_type wind_aware --method EAC --train 0 --no_warmup 1 --seed 42 --logname eac_wa_norre
```

---

# 八、推荐你现在最先跑的命令

如果你现在要开始正式实验，我建议你先按这个顺序跑：

## 1. 先处理四个数据集的 `distance`

```bash
python main.py --dataset sdwpf --adj_type distance --data_process 1 --train 0 --logname preprocess_sdwpf_distance
python main.py --dataset penmanshiel --adj_type distance --data_process 1 --train 0 --logname preprocess_penmanshiel_distance
python main.py --dataset hlrs --adj_type distance --data_process 1 --train 0 --logname preprocess_hlrs_distance
python main.py --dataset norrekaer_enge --adj_type distance --data_process 1 --train 0 --logname preprocess_norre_distance
```

## 2. 再跑 EAC 的正式实验

```bash
python main.py --dataset sdwpf --adj_type distance --method EAC --train 1 --seed 42 --logname eac_dist_sdwpf
python main.py --dataset penmanshiel --adj_type distance --method EAC --train 1 --seed 42 --logname eac_dist_penmanshiel
python main.py --dataset hlrs --adj_type distance --method EAC --train 1 --seed 42 --logname eac_dist_hlrs
python main.py --dataset norrekaer_enge --adj_type distance --method EAC --train 1 --seed 42 --logname eac_dist_norre
```

## 3. 然后再比较 `wind_aware_local`

```bash
python main.py --dataset sdwpf --adj_type wind_aware_local --data_process 1 --train 0 --logname preprocess_sdwpf_wal
python main.py --dataset penmanshiel --adj_type wind_aware_local --data_process 1 --train 0 --logname preprocess_penmanshiel_wal
python main.py --dataset hlrs --adj_type wind_aware_local --data_process 1 --train 0 --logname preprocess_hlrs_wal
python main.py --dataset norrekaer_enge --adj_type wind_aware_local --data_process 1 --train 0 --logname preprocess_norre_wal

python main.py --dataset sdwpf --adj_type wind_aware_local --method EAC --train 1 --seed 42 --logname eac_wal_sdwpf
python main.py --dataset penmanshiel --adj_type wind_aware_local --method EAC --train 1 --seed 42 --logname eac_wal_penmanshiel
python main.py --dataset hlrs --adj_type wind_aware_local --method EAC --train 1 --seed 42 --logname eac_wal_hlrs
python main.py --dataset norrekaer_enge --adj_type wind_aware_local --method EAC --train 1 --seed 42 --logname eac_wal_norre
```
