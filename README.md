# Adaptive Collapse Dynamics: A General L²/t Principle from Gene Regulation to Transformer KV-Cache

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Official implementation of the paper by Stanislav Usychenko (May 2026)**

ACE (Adaptive Collapse Eviction) is a KV-cache eviction policy for long-context Transformers, based on a universal information-retention principle: **viability = L²/t**. It reduces memory by 4×–8× with <0.3 perplexity increase, outperforming H2O and StreamingLLM.

---

## 🔬 The General Principle

> In self-organizing systems — from DNA regulation to human language to Transformer decoders — elements survive if their **structural density L** is high and their **age t** is low.

**Viability score:**

\[
v = \frac{L^2}{t + \epsilon}
\]

| Domain | L (density) | t (age) |
|--------|-------------|---------|
| Gene regulation | Number of bound transcription factors | Time since last transcription |
| Language evolution | Word frequency (√) | Time since last use |
| Transformer KV-cache | Learned token utility w | Decoding steps since insertion |

---

## 🧠 ACE Method

1. **Learn token utility w** (tiny 2-layer MLP, ~5000 params) from:
   - Normalized key vector
   - Positional encoding
   - Local context (mean of last 5 keys)

2. **Compute viability** `v = w²/(age + ε)`

3. **Evict** tokens with smallest v when cache exceeds budget M

**Complexity:** O(M) attention + O(log M) heap update. Overhead <5% of inference time.

---

## 📊 Results (LLaMA-2 7B, 32k context)

### PG19 Perplexity (lower is better)

| Method | Cache size | Perplexity |
|--------|------------|------------|
| Full cache | 32768 | 14.2 |
| Sliding window | 1024 | 18.7 |
| StreamingLLM | 1024 | 16.3 |
| H2O | 1024 | 15.1 |
| **ACE (ours)** | 1024 | **14.5** |
| **ACE (ours)** | 4096 | **14.2** |

### Needle-in-a-Haystack (depth 31k)

| Method | Accuracy |
|--------|----------|
| Full cache | 100% |
| H2O | 90% |
| StreamingLLM | 75% |
| Sliding window | 20% |
| **ACE** | **95%** |

### LongBench (NarrativeQA, MultiFieldQA, TREC)

| Method | Cache | F1 |
|--------|-------|-----|
| Full cache | 4096 | 0.62 |
| **ACE** | 512 | **0.61** |

### Speed (A100, batch=1)

| Method | tokens/sec | overhead/step |
|--------|------------|----------------|
| Full cache | 12.3 | – |
| H2O | 26.1 | 3.2 ms |
| **ACE** | **25.4** | **1.5 ms** |

---

## 🧪 Ablation Studies

| Variant | PG19 PPL |
|---------|----------|
| ACE (w²/t) | **14.5** |
| w/t (linear instead of quadratic) | 15.0 |
| w²/t² (quadratic age) | 14.9 |
| Linear age decay (1/t) | 14.9 |
| Random utility (no MLP) | 18.1 |

**Conclusion:** The specific form `w²/t` hits the sweet spot.

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/adaptive-collapse-dynamics.git
cd adaptive-collapse-dynamics
pip install -r requirements.txt


from ace import ACEKVCache
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Wrap model with ACE
ace_model = ACEKVCache(model, cache_budget=1024, epsilon=1e-8)


📁 Repository Structure

.
├── ace/
│   ├── __init__.py
│   ├── cache.py          # KV-cache with min-heap eviction
│   ├── utility_mlp.py    # 2-layer MLP for w prediction
│   └── trainer.py        # Training on C4 with cumulative attention
├── experiments/
│   ├── pg19_eval.py
│   ├── needle_test.py
│   └── longbench_eval.py
├── checkpoints/          # Pretrained utility MLP
├── requirements.txt
└── README.md



📝 Limitations

· MLP trained only on C4 (not verified on code/medical text)
· Tested only on LLaMA-2 7B (not on 70B or other architectures)
· Biological/linguistic analogies are suggestive, not experimentally validated

---

📖 Citation

```bibtex
@article{usychenko2026adaptive,
  title={Adaptive Collapse Dynamics: A General L²/t Principle from Gene Regulation to Transformer KV-Cache},
  author={Usychenko, Stanislav},
  year={2026},
  month={May}
}
```

---

📚 References

· [1] Shannon, C.E. (1948). A Mathematical Theory of Communication.
· [2] Prigogine, I. (1977). Time, Structure and Fluctuations (Nobel Lecture).
· [3] Zhang et al. (2023). H2O: Heavy-Hitter Oracle. NeurIPS.
· [4] Xiao et al. (2023). StreamingLLM. arXiv:2309.17453