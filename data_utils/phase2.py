
import pandas as pd

# Baseline PPL
baseline = 6.745146786722399

# Attention PPL values (from your experiment)
attention_ppl = [
8.77, 6.89, 6.66, 6.52, 6.92, 6.89, 6.83, 6.80, 6.71, 6.77, 6.70, 6.80,
6.76, 6.76, 6.89, 6.72, 6.88, 6.88, 6.78, 6.90, 6.95, 6.82, 6.79, 6.84,
6.95, 7.04, 7.01, 7.34, 6.96, 6.66, 6.69, 6.78
]

# MLP PPL values (from your experiment)
mlp_ppl = [
1586.47, 6.97, 7.12, 7.42, 7.18, 7.34, 7.06, 7.10, 7.02, 6.87, 6.89, 6.99,
7.00, 6.95, 6.86, 7.14, 7.02, 6.94, 6.93, 6.80, 6.87, 6.99, 6.90, 7.19,
6.80, 6.84, 7.14, 7.08, 7.01, 6.97, 6.99, 7.89
]

layers = list(range(32))

# Compute sensitivity deltas
attention_delta = [p - baseline for p in attention_ppl]
mlp_delta = [p - baseline for p in mlp_ppl]

# Create DataFrame
df = pd.DataFrame({
    "Layer": layers,
    "Attention_PPL": attention_ppl,
    "Attention_Delta": attention_delta,
    "MLP_PPL": mlp_ppl,
    "MLP_Delta": mlp_delta
})

# Rank layers by sensitivity
df["Attention_Rank"] = df["Attention_Delta"].rank(ascending=False)
df["MLP_Rank"] = df["MLP_Delta"].rank(ascending=False)

# Save CSV
file_path = "component_sensitivity_results.csv"
df.to_csv(file_path, index=False)

df