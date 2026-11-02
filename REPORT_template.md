# Efficient Graph Neural Networks for Materials Property Prediction: Accuracy–Efficiency Trade-offs under Quantization

Shaoquan Wang

## 1. Introduction

One short paragraph: why materials property prediction matters; why accuracy alone is not enough for deployment; what this project asks.

## 2. Data and Task

- MatBench `matbench_dielectric`
- Number of samples; target (refractive index, unitless); metric (MAE, R²)
- Official 5-fold splits; no custom splitting

## 3. Methods

### 3.1 Baselines

- Mean predictor
- Composition features (Magpie) + XGBoost
- Composition-only model as a control for structural information

### 3.2 Crystal graph neural network

- Node features; edge features; number of layers; pooling; regression head
- Training settings: optimizer, learning rate, batch size, early stopping

### 3.3 Quantization

- Post-training dynamic quantization (`torch.quantization.quantize_dynamic`, `nn.Linear` → `qint8`)
- Evaluation protocol: same test folds, CPU, batch size 1

## 4. Results

Table 1: mean ± std MAE and R² across five folds.

Table 2: model size and CPU latency before/after quantization.

Figure 1: accuracy vs latency.

## 5. Error Analysis

- MAE by chemical family (oxide, halide, sulfide, transition-metal-containing, etc.)
- Five largest errors and possible reasons
- Discussion of data scarcity and distribution shift

## 6. Discussion and Limitations

- What structural information contributes over composition
- What quantization costs and what it saves
- Limitations: single dataset, CPU-only latency, only linear layers quantized
- What I would do next with more time

## References

1. MatBench: Dunn et al., *npj Computational Materials* 6, 138 (2020).
2. CGCNN: Xie & Grossman, *Physical Review Letters* 120, 145301 (2018).
3. PyTorch quantization documentation.

