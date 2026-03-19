# Semantic Loss Fine-Tuning for Causal Reasoning

Code and resources for the paper:  
**"On Semantic Loss Fine-Tuning Approach for Preventing Model Collapse in Causal Reasoning"**  
*Pratik Deshmukh (TU Vienna) · Atirek Gupta (HCLTech)*

## Key Finding

Standard fine-tuning on causal reasoning tasks causes **100% model collapse** — models learn trivial prediction biases (always "Yes" or always "No"). Semantic loss with dynamic lambda scheduling (λ: 0.05→0.30) prevents collapse and achieves **42.7% average improvement** over collapsed baselines.

## Resources

| Resource | Link |
|---|---|
| 📄 Paper | *arXiv link — coming soon* |
| 🤗 Transitivity Semantic V4 | [ludwigw/gemma-transitivity-semantic-v4](https://huggingface.co/ludwigw/gemma-transitivity-semantic-v4) |
| 🤗 D-Separation Semantic V2 | [ludwigw/gemma-dseparation-semantic-v2](https://huggingface.co/ludwigw/gemma-dseparation-semantic-v2) |
| 🤗 Transitivity Baseline | [ludwigw/gemma-transitivity-baseline](https://huggingface.co/ludwigw/gemma-transitivity-baseline) |
| 🤗 D-Separation Baseline | [ludwigw/gemma-dseparation-baseline](https://huggingface.co/ludwigw/gemma-dseparation-baseline) |
| 🤗 Dataset | [ludwigw/causal-reasoning-benchmarks](https://huggingface.co/datasets/ludwigw/causal-reasoning-benchmarks) |

## Architecture

<p align="center">
  <img src="arch.png" alt="System Architecture" width="900">
</p>

## Results

| Model | Std. Accuracy | Adv. Accuracy | Status |
|---|---|---|---|
| Transitivity V1 (baseline) | 27.7% | 70.8% | ❌ Collapsed (always "Yes") |
| D-Separation V1 (baseline) | 73.9% | 43.0% | ❌ Collapsed (always "No") |
| **Transitivity Semantic V4** | **70.4%** | **69.8%** | ✅ Structural reasoning |
| **D-Separation Semantic V2** | **68.6%** | **67.8%** | ✅ Structural reasoning |

Branching task improvement: **1.96% → 97.9%** (Transitivity Semantic V4)

## Repository Structure
```
├── gemma_semantic.ipynb                        # Main training notebook (Colab)
├── docs/
│   ├── comprehensive_evaluation_guide.md       # Benchmarking guide
│   └── archive/                                # Archived early docs
├── evaluations/
│   └── report.md                               # Full evaluation report (4 models)
└── arch.png                                    # Architecture diagram
```

## Quick Start

1. Open `gemma_semantic.ipynb` in Google Colab (T4 GPU)
2. Mount Drive and set `DATA_PATH` to your axiomatic data folder  
   *or load directly from [ludwigw/causal-reasoning-benchmarks](https://huggingface.co/datasets/ludwigw/causal-reasoning-benchmarks)*
3. Follow the [Comprehensive Evaluation Guide](./docs/comprehensive_evaluation_guide.md) to benchmark all models

## Documentation

- [Comprehensive Evaluation Guide](./docs/comprehensive_evaluation_guide.md)
- [Evaluation Report (4 models)](./evaluations/report.md)

## Citation
```bibtex
@article{deshmukh2026semantic,
  title={On Semantic Loss Fine-Tuning Approach for Preventing Model Collapse in Causal Reasoning},
  author={Deshmukh, Pratik and Gupta, Atirek},
  year={2026}
}
```
