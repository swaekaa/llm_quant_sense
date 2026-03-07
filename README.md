# 📌 Quantization Sensitivity Analysis for LLMs

This project studies **layer-wise quantization sensitivity** in Large Language Models (LLMs) and evaluates **mixed-precision quantization strategies** to reduce model precision while preserving accuracy.

The goal is to understand which transformer layers are most sensitive to quantization, and how this knowledge can be used to design efficient, deployment-friendly models.

---

## 🧠 Motivation

Quantization is widely used to:

- Reduce model size  
- Improve inference speed  
- Lower deployment cost  

However, **uniform quantization across all layers** often leads to unnecessary accuracy loss.

This project answers:

- Are all transformer layers equally sensitive to quantization?  
- Can we quantize some layers more aggressively than others?
- How do different components (attention vs MLP) respond to quantization?
- What precision formats (8-bit, 4-bit NF4, 4-bit FP4) offer the best trade-offs?

---

## 🏗️ Models & Datasets

### Experiment 1: DistilBERT (Accuracy-Based Analysis)

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Architecture:** 6-layer Transformer (66M parameters)
- **Task:** Sentiment Classification
- **Dataset:** SST-2 (GLUE benchmark)
- **Evaluation Metric:** Accuracy

### Experiment 2: Phi-2 (Perplexity-Based Analysis)

- **Model:** `microsoft/phi-2`
- **Architecture:** 32-layer Transformer (2.7B parameters)
- **Task:** Language Modeling
- **Dataset:** WikiText-2
- **Evaluation Metric:** Perplexity (lower is better)
- **Hardware:** RTX 3060 12GB

> **Note:** Initial experiments attempted Mistral 7B, but it was too memory-intensive for consumer hardware. Phi-2 provides sufficient depth (32 layers) while remaining stable on 12GB VRAM.

---

## ⚙️ Project Structure
```text
llm-quant-sensitivity/
│
├── models/                 # Model loading
├── data_utils/             # Dataset loading
├── evaluation/             # Accuracy evaluation
│
├── quantization/
│   ├── layerwise.py        # Fake quantization utilities
│   └── mixed_precision.py # Mixed-precision quantization
│
├── sensitivity/
│   ├── metrics.py          # Sensitivity metric
│   └── ranking.py          # Layer ranking utility
│
├── visualization/
│   └── plots.py            # Sensitivity plots
│
├── experiments/
│   ├── baseline_eval.py
│   ├── layer_sensitivity.py
│   ├── mixed_precision_eval.py
│   └── ablation_study.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Experimental Pipeline

## Part A: DistilBERT Experiments

### Phase 1 — FP32 Baseline

- Evaluate the full-precision (FP32) model  
- Establish a reference accuracy  

**Result:**

- **FP32 Accuracy:** **91.06%**

---

### Phase 2 — Layer-wise Quantization Sensitivity

Each transformer layer is quantized individually (others kept FP32), and the resulting accuracy drop is measured.

This isolates how sensitive each layer is to quantization noise.

#### Results

| Layer | Accuracy Drop |
|-------|---------------|
| Layer 0 | 0.23% |
| Layer 1 | 0.23% |
| Layer 2 | 0.34% |
| Layer 3 | 0.34% |
| Layer 4 | 0.34% |
| Layer 5 | 0.00% |

#### Key Observations

- Middle layers (**2–4**) are the most sensitive  
- Final layer (**5**) is highly robust to quantization  

---

### Phase 3 — Mixed-Precision Quantization

Using sensitivity results, mixed-precision strategies were evaluated where different layers use different bit-widths.

#### Example strategies tested:

- Aggressive mixed precision (**INT6 + INT4**)  
- Uniform **INT8** with selective **INT4**  
- Targeted single-layer quantization  

---

### Ablation Study (Key Validation)

Only the least sensitive layer (**Layer 5**) was quantized to **INT4**, all others kept FP32.

**Result:**

- **Accuracy:** **90.94%**  
- **Accuracy Drop:** **0.12%**

This confirms that **late transformer layers can be aggressively quantized with negligible impact**.

---

## Part B: Phi-2 Experiments

### Phase 1 — Full Layer Masking Sensitivity

**Baseline Perplexity:** 6.745

Each transformer layer was completely masked (zeroed out) to measure its criticality to model performance.

#### Results Summary

| Sensitivity Tier | Layers | Peak Sensitivity |
|------------------|--------|------------------|
| **Catastrophic** | 23 | 3,579,120.63 |
| **Extreme** | 20 | 140,746.96 |
| **Very High** | 5, 14, 16, 19, 27 | 29,000–50,000 |
| **High** | 2, 6, 7, 8, 10, 11, 12, 18, 21, 24 | 17,000–29,000 |
| **Medium** | 0, 4, 9, 13, 15, 17, 22, 26, 29, 31 | 10,000–17,000 |
| **Low** | 1, 25, 28, 30 | 5,800–20,800 |

#### Critical Insights

- **Layer 23** is catastrophically important (perplexity explodes to 3.5M)
- **Layer 20** shows extreme sensitivity (140K perplexity)
- **Layers 1, 28, 30** are surprisingly robust
- Sensitivity is **non-monotonic** across depth

> **Why full masking?** Residual connections mean each transformer block is essential for information flow. Complete removal reveals true criticality.

---

### Phase 2 — Component-Level Sensitivity (Attention vs MLP)

Individual components within each layer were masked to isolate attention vs MLP sensitivity.

#### Key Findings

**Layer 0 (Early Layer):**
- Attention PPL: 8.77
- MLP PPL: 1,586.47
- **Early MLP is foundational** — removing it collapses representations immediately

**Middle Layers (1–26):**
- Attention PPL: ~6.7–7.3
- MLP PPL: ~6.8–7.4
- **Highly distributed computation** — either component can be masked with minimal impact
- Significant redundancy exists

**Late Layers (27–31):**
- Layer 27 Attention: 7.34
- Layer 31 MLP: 7.89
- **Slightly more sensitive** than middle layers
- Responsible for final output refinement

#### Component-Level Insights

- **Early feature expansion (Layer 0 MLP) is critical**
- **Middle layers are robust and redundant**
- **Late layers refine output probabilities**
- Catastrophic failures only occur with full layer removal, not component-level masking

---

### Phase 3 — Gaussian Noise Injection (Quantization Simulation)

Since the model was loaded in 4-bit format, direct weight modification wasn't possible. Gaussian noise injection simulates quantization error more realistically than masking.

#### Sensitivity Ranking (Normalized PPL Increase)

**Highly Sensitive (Top Risk Group):**
- Layer 31 → 0.88
- Layer 27 → 0.82
- Layer 26 → 0.79
- Layer 5 → 0.71
- Layer 12 → 0.71
- Layer 4 → 0.68

**Medium Sensitivity (0.3–0.6):**
- Layer 6 → 0.58
- Layer 25 → 0.59
- Layer 13 → 0.54
- Layer 11 → 0.51
- Layer 16 → 0.38
- Layer 22 → 0.36

**Low Sensitivity (< 0.25) — Ideal 4-bit Candidates:**
- Layer 10 → 0.09
- Layer 9 → 0.11
- Layer 17 → 0.16
- Layer 24 → 0.17
- Layer 20 → 0.18
- Layer 2 → 0.22

#### Strategic Recommendations

- **Keep in FP16/8-bit:** Layers 4, 5, 12, 26, 27, 31
- **Tolerate 8-bit:** Layers 6, 11, 13, 16, 22, 25
- **Aggressive 4-bit:** Layers 2, 9, 10, 17, 20, 24

---

### Phase 4 — Actual Quantization Evaluation (Phi-1.5)

> **Note:** Due to GPU memory constraints (4GB), this phase was conducted on Phi-1.5 instead of Phi-2.

| Precision | PPL | Δ vs FP16 | Tokens/sec | Peak GPU MB |
|-----------|-----|-----------|------------|-------------|
| **FP16** | 12.5600 | 0.0000 | 3282 | 3061 |
| **8-bit** | 12.6008 | +0.0409 | 1307 | 2522 |
| **4-bit NF4** | 12.7195 | +0.1595 | 2471 | 2581 |
| **4-bit FP4** | 13.2768 | +0.7169 | 2441 | 3160 |

#### Critical Insights

- **8-bit quantization is nearly lossless** (+0.04 PPL)
- **NF4 significantly outperforms FP4** (0.16 vs 0.72 PPL increase)
- **Quantization ≠ guaranteed speedup** on small GPUs
  - FP16 achieved highest throughput (3282 tokens/sec)
  - 8-bit was **2.5× slower** than FP16 despite quantization
- **Peak memory depends on kernel implementation**, not just bit width
- **NF4 preserves model distribution better** due to non-uniform quantization bins

---

## 📊 Visualization

Layer-wise sensitivity can be visualized using bar plots and heatmaps to highlight:

- Peak sensitivity in middle layers (DistilBERT) or specific critical layers (Phi-2)
- Near-zero sensitivity in robust layers
- Component-level (Attention vs MLP) sensitivity patterns
- Quantization format trade-offs (PPL vs throughput vs memory)

---

## 🧠 Key Findings

### General Principles

1. **Transformer layers are not equally sensitive** to quantization
2. **Layer-wise sensitivity analysis is necessary but not sufficient**
3. **Quantization error compounds non-linearly** when multiple layers are quantized
4. **Aggressive quantization is safe for selected layers**
5. **Uniform quantization is often suboptimal** compared to targeted strategies

### Model-Specific Insights

**DistilBERT (6 layers):**
- Middle layers most sensitive
- Final layer extremely robust
- Suitable for aggressive tail quantization

**Phi-2 (32 layers):**
- Specific critical layers exist (Layer 23, Layer 20)
- Early MLP blocks are foundational
- Middle layers exhibit high redundancy
- Late layers require precision for output refinement
- Sensitivity is non-monotonic across depth

### Technical Insights

- **Gaussian noise injection** simulates quantization better than masking
- **Component-level analysis** (Attention vs MLP) reveals architectural insights
- **8-bit quantization** offers near-lossless compression
- **NF4 format** is superior to FP4 for 4-bit quantization
- **Consumer GPU throughput** depends on kernel optimization, not just bit width

---

## 💡 Takeaway

Effective LLM quantization requires **sensitivity-aware, layer-specific precision choices** rather than uniform compression. The optimal strategy depends on:

- Model architecture and depth
- Component-level (Attention vs MLP) sensitivity
- Target hardware and deployment constraints
- Acceptable perplexity/accuracy trade-offs

---

## 🖥️ Hardware & Runtime

- **DistilBERT experiments:** CPU with automatic CUDA fallback
- **Phi-2 experiments:** RTX 3060 12GB (Windows)
- **Phi-1.5 quantization:** 4GB GPU
- No retraining required
- Fake quantization used for fast, hardware-agnostic analysis

---

## 📦 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Running Experiments

### DistilBERT
```bash
# FP32 baseline
python -m experiments.baseline_eval

# Layer-wise sensitivity
python -m experiments.layer_sensitivity

# Mixed-precision evaluation
python -m experiments.mixed_precision_eval

# Ablation study
python -m experiments.ablation_study
```

### Phi-2
```bash
# Full layer masking sensitivity
python -m experiments.phi2_layer_masking

# Component-level sensitivity
python -m experiments.phi2_component_sensitivity

# Gaussian noise injection
python -m experiments.phi2_noise_sensitivity

# Quantization evaluation
python -m experiments.phi2_quantization_eval
```

---

## 🏁 Conclusion

This project demonstrates a **comprehensive quantization sensitivity analysis pipeline for LLMs**, combining:

- **Rigorous multi-phase experimentation** across different model scales
- **Clear empirical insights** from accuracy and perplexity-based evaluation
- **Practical deployment relevance** for consumer hardware constraints
- **Component-level analysis** revealing architectural sensitivity patterns
- **Quantization format comparisons** (8-bit, 4-bit NF4, 4-bit FP4)

The results provide actionable guidance for designing mixed-precision quantization strategies that balance model size, inference speed, and quality preservation.

Suitable as:

- A research prototype for LLM compression
- A portfolio project demonstrating ML systems expertise
- A foundation for further work on hardware-aware quantization
- A reference implementation for sensitivity-based quantization strategies

---

## 🔬 Future Work

- Extend to larger models (7B+ parameters)
- Implement actual mixed-precision inference kernels
- Test on diverse downstream tasks (reasoning, code generation, summarization)
- Explore layer-wise learning rate scaling based on sensitivity
- Investigate quantization-aware fine-tuning for critical layers
- Benchmark on edge devices (mobile, embedded systems)
