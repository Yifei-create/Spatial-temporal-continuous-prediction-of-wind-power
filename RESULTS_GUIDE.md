# 结果保存说明

## 训练过程显示

运行 `bash run_all.sh` 后，你会在终端看到：

```
==========================================
Training EAC model on GPU 0...
==========================================
2026-03-01 - params : {...}
2026-03-01 - [*] Year 0 load from data/processed/0.npz
2026-03-01 - [*] Year 0 Dataset load!
2026-03-01 - Total Parameters: 50336
2026-03-01 - Trainable Parameters: 50336
2026-03-01 - [*] Year 0 Training start
2026-03-01 - node number torch.Size([44, 16])
2026-03-01 - epoch:0, training loss:0.5234 validation loss:0.4567
2026-03-01 - epoch:1, training loss:0.4123 validation loss:0.3890
...
2026-03-01 - Finished optimization, total time:120.45 s
2026-03-01 - [*] Year 1 load from data/processed/1.npz
...
```

## 4个阶段进度

训练会依次经过4个period：

- **Period 0**: 44个风机，从头训练
- **Period 1**: 66个风机，冻结backbone
- **Period 2**: 99个风机，冻结backbone
- **Period 3**: 134个风机，冻结backbone

每个period完成后会显示该period的指标。

## 结果保存位置

### 1. 模型文件

```
log/
├── eac-42/
│   ├── eac.log              # 训练日志（包含所有输出）
│   ├── 0/
│   │   └── 0.4567.pkl       # Period 0最佳模型（文件名=验证loss）
│   ├── 1/
│   │   └── 0.3890.pkl       # Period 1最佳模型
│   ├── 2/
│   │   └── 0.3456.pkl       # Period 2最佳模型
│   └── 3/
│       └── 0.3123.pkl       # Period 3最佳模型
├── scaleshift-42/
│   └── ...
└── var_scaleshift-42/
    └── ...
```

### 2. 训练日志

每个模型的完整训练过程保存在：
- `log/eac-42/eac.log`
- `log/scaleshift-42/scaleshift.log`
- `log/var_scaleshift-42/var_scaleshift.log`

### 3. 性能指标

在训练日志和终端输出中，你会看到：

```
3    MAE     10.23    9.87    9.45    9.12    9.67
6    MAE     12.45   11.89   11.34   10.98   11.67
12   MAE     15.67   14.98   14.23   13.89   14.69
Avg  MAE     12.78   12.25   11.67   11.33   12.01

3    RMSE    15.34   14.76   14.12   13.89   14.53
...

year  0    total_time  120.5    average_time  1.205    epoch  100
year  1    total_time   45.3    average_time  0.906    epoch   50
year  2    total_time   52.1    average_time  1.042    epoch   50
year  3    total_time   58.7    average_time  1.174    epoch   50
total time: 276.6
```

说明：
- 每行显示不同预测步长（3, 6, 12步）的指标
- 每列对应一个period的结果
- 最后一列是平均值

### 4. 预测结果

如果需要保存预测值和真实值，可以修改 `trainer.py` 中的 `test_model` 函数，添加：

```python
# 在 test_model 函数最后添加
np.savez(f'results_{args.method}_year{args.year}.npz', 
         predictions=pred_, 
         ground_truth=truth_)
```

## 如何查看结果

### 查看训练进度（实时）

```bash
# 在另一个终端运行
tail -f log/eac-42/eac.log
```

### 查看最终指标

```bash
# 查看日志文件末尾
tail -50 log/eac-42/eac.log
```

### 加载模型进行预测

```python
import torch

# 加载Period 3的最佳模型
model_path = 'log/eac-42/3/0.3123.pkl'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
```

## 运行命令

```bash
# 顺序训练所有模型（在终端直接看到进度）
bash run_all.sh

# 或单独训练
python main.py --method EAC --gpuid 0
python main.py --method ScaleShift --gpuid 0
python main.py --method VariationalScaleShift --gpuid 0
```

