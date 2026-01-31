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

---

## 🏗️ Model & Dataset

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Architecture:** 6-layer Transformer (DistilBERT)
- **Task:** Sentiment Classification
- **Dataset:** SST-2 (GLUE benchmark)
- **Evaluation Metric:** Accuracy

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

## 🚀 Experimental Pipeline

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
|------|----------------|
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

## 📊 Visualization

Layer-wise sensitivity can be visualized using a simple bar plot:

**Layer-wise Quantization Sensitivity**

This plot highlights:

- Peak sensitivity in middle layers  
- Near-zero sensitivity in the final layer  

---

## 🧠 Key Findings

- Transformer layers are **not equally sensitive** to quantization  
- Layer-wise sensitivity analysis is necessary but not sufficient  
- Quantization error compounds **non-linearly** when multiple layers are quantized  
- Aggressive quantization is safe for selected layers  
- Uniform quantization is often suboptimal compared to targeted strategies  

---

## 💡 Takeaway

Effective LLM quantization requires **sensitivity-aware, layer-specific precision choices** rather than uniform compression.

---

## 🖥️ Hardware & Runtime

- Runs on CPU with automatic CUDA fallback  
- No retraining required  
- Fake quantization used for fast, hardware-agnostic analysis  

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Running Experiments

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

## 🏁 Conclusion

This project demonstrates a complete **quantization sensitivity analysis pipeline for LLMs**, combining:

- Rigorous experimentation  
- Clear empirical insights  
- Practical deployment relevance  

It is suitable as:

- A research prototype  
- A portfolio project  
- A foundation for further work on LLM compression

