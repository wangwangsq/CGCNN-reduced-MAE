# AI for Science 三周项目工作流

项目：**Efficient Graph Neural Networks for Materials Property Prediction: Accuracy–Efficiency Trade-offs under Quantization**

目标：在 MatBench 的固定划分上，比较「成分特征 baseline → 晶体图神经网络 → 量化后的 GNN」，产出一份可复现的代码、一张精度—时延图、一份 2 页英文报告。

---

## 0. 最终交付物

- [ ] GitHub 仓库：`src/`、`requirements.txt`、`README.md`、`results/`
- [ ] `results/baselines.csv`：均值预测器、XGBoost 的 5 折 MAE / R²
- [ ] `results/gnn.csv`：CGCNN/MEGNet 的 5 折 MAE / R²
- [ ] `results/efficiency.csv`：FP32 vs INT8 的 MAE、模型大小、CPU 时延
- [ ] `figures/accuracy_vs_latency.png`
- [ ] `report/AI4S_report.pdf`（2 页）

---

## 1. 环境（第 1 天）

```bash
conda create -n ai4s python=3.10 -y
conda activate ai4s
pip install matbench matminer pymatgen scikit-learn xgboost pandas numpy matplotlib tqdm
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
# 更省事的 GNN 方案（可选）
pip install matgl
```

检查：

```python
import torch, matbench, matminer, pymatgen
print(torch.__version__, matbench.__version__)
```

目录：

```
ai4s-matbench/
├── README.md
├── requirements.txt
├── src/
│   ├── load_data.py
│   ├── baseline.py
│   ├── train_gnn.py
│   ├── quantize_eval.py
│   └── error_analysis.py
├── results/
└── figures/
```

---

## 2. 数据与 baseline（第 2–5 天）

选 `matbench_dielectric`（结构型数据，4,764 条晶体结构，目标是折射率，无量纲，用 MAE）。

```python
from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False, subset=["matbench_dielectric"])
task = list(mb.tasks)[0]
task.load()

for fold in task.folds:
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    test_inputs = task.get_test_data(fold, include_target=False)
    # train_inputs: list[pymatgen Structure]；train_outputs: pd.Series
```

三个 baseline：

1. **均值预测器**：每折用训练集均值预测，作为最低线。
2. **成分特征 + XGBoost**：用 `matminer.featurizers.composition.ElementProperty` 提取 Magpie 特征（元素比例、电负性、原子半径等）。
3. **只含成分、不含结构**：作为「结构信息带来多少提升」的对照。

```python
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

featurizer = ElementProperty.from_preset("magpie")
X = [featurizer.featurize(Composition(s.composition)) for s in train_inputs]
```

产出：`results/baselines.csv`，列：`model, fold, mae, r2`，最后算 mean ± std。

---

## 3. GNN 模型（第 6–12 天）

两个选择：

- **省事版**：`matgl` 的 MEGNet，按官方 MatBench 示例跑通。
- **自己写**：PyTorch Geometric 实现 CGCNN（原子特征 + 邻居图 + 3 层卷积 + 全局池化 + MLP 回归）。

训练要点：

- 用 MatBench 固定 folds，不要自己随机切分；
- 先只跑 1 折确认流程，再跑 5 折；
- 超参：`lr=1e-3`、`batch=64`、`hidden=64/128`、`layers=3`、early stopping（patience 30）；
- 记录每折 MAE / R² 和训练时间。

消融：

- 成分 baseline vs 结构 GNN；
- 去掉键特征 / 减少层数的 GNN。

产出：`results/gnn.csv`。

---

## 4. 量化与效率（第 13–18 天）

```python
import torch, torch.nn as nn

quantized = torch.quantization.quantize_dynamic(
    model.cpu().eval(),
    {nn.Linear},
    dtype=torch.qint8,
)
```

测量三项：

1. **MAE**：量化前后在同一测试折上比较；
2. **模型大小**：`os.path.getsize` 保存后的 `.pt` / `.pth`；
3. **CPU 时延**：batch=1，先 warm-up 20 次，再测 100 次取中位数；同时记录吞吐（samples/s）。

注意：GNN 的主要耗时可能在消息传递（非 Linear），动态量化只改 Linear 层，**速度提升可能有限**——如实报告，不要硬写「大幅加速」。

产出：`results/efficiency.csv`、`figures/accuracy_vs_latency.png`（横轴时延、纵轴 MAE，标出 FP32 / INT8）。

---

## 5. 误差分析（第 19–21 天）

- 按化学体系分组：含 O、含过渡金属、卤化物、硫化物等；
- 计算每组 MAE，画柱状图；
- 找 5 个误差最大的样本，回看结构 / 元素组成 / 数据量；
- 讨论：数据稀缺、结构复杂、带隙范围的分布偏移。

产出：`figures/error_by_family.png` + 报告中的 Error Analysis 一节。

---

## 6. 写作（第 22–24 天）

按 `REPORT_template.md` 填写。核心原则：

- 不写 SOTA，写清楚「和 baseline 比提升了多少」「量化损失多少」「哪些体系误差大」；
- 所有数字来自 `results/*.csv`；
- 报告里写局限：单一数据集、仅 CPU、动态量化只作用于 Linear。

README 按 `README_template.md` 填写，确保别人能复现：

```bash
pip install -r requirements.txt
python src/baseline.py
python src/train_gnn.py --folds 5
python src/quantize_eval.py
```

---

## 7. 常见坑

- MatBench 的目标单位和指标：`dielectric` 的折射率无量纲，报告 MAE（不要和 RMSE 混用）。
- 标准化/特征拟合只能在训练折上做，避免数据泄漏。
- 5 折结果报告 mean ± std，不要只报最好的一折。
- 量化前后要在同一台机器、同一 batch size 下测时延。
- PyG 与 torch 版本要匹配；装不上就先用 `matgl` 跑通 MEGNet。
- 不要把「成分 baseline」和「结构 GNN」用不同的数据划分。

