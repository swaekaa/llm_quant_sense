## Model Details

The model used in this experiment is:

**distilbert-base-uncased-finetuned-sst-2-english**

- Number of parameters: **~66M**
- Transformer layers: **6**
- Reason for model choice: Due to limited computational resources, a smaller and faster model is used initially. A more powerful model can be explored later.

---

## Dataset

**SST-2 (Stanford Sentiment Treebank)** from the **GLUE benchmark**

- Task: Binary sentiment classification
- Focus: Fast inference and efficient evaluation

---

## Objective

The objective of this experiment is to analyze **layer-wise sensitivity to quantization** in order to determine which transformer layers are most robust and which are most sensitive.

---

## Methodology

### Pipeline
```
FP32 Model → Baseline Accuracy
        ↓
For each transformer layer:
        ↓
    Quantize layer → Evaluate → Measure Accuracy Drop
        ↓
Layer Sensitivity Ranking
```


### Explanation

1. Start with the FP32 (full precision) model and record baseline accuracy.
2. For each transformer layer:
   - Quantize the layer.
   - Evaluate the model.
   - Measure the resulting accuracy drop.
3. Rank the layers based on how sensitive they are to quantization.

---

## How to Run

```bash
python -m experiments.layer_sensitivity
```
## Results

### Layer Sensitivity Results (6 Transformer Layers)

| Layer  | Accuracy Drop |
|---------|----------------|
| Layer 0 | 0.23% |
| Layer 1 | 0.23% |
| Layer 2 | 0.34% |
| Layer 3 | 0.34% |
| Layer 4 | 0.34% |
| Layer 5 | 0.00% |

---

## Observations

- **Layer 5** is the least sensitive to quantization, showing **no accuracy degradation**.
- **Layers 2, 3, and 4** are the most sensitive, each exhibiting the highest accuracy drop.
- Early layers (0 and 1) show moderate sensitivity.
- These results can guide **mixed-precision or selective quantization strategies** for optimal performance and accuracy trade-offs.


