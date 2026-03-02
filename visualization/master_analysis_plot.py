import matplotlib.pyplot as plt



# From your FP16 sensitivity experiment
layer_sensitivity = [
    14150.0204, 0.4129, 0.2219, 0.4280, 0.6823, 0.7159,
    0.5841, 0.3377, 0.4155, 0.1197, 0.0935, 0.5097,
    0.7100, 0.5389, 0.3311, 0.3344, 0.3802, 0.1669,
    0.3422, 0.2402, 0.1874, 0.2641, 0.3636, 0.4374,
    0.1799, 0.5946, 0.7938, 0.8233, 0.2478, 0.3258,
    0.2885, 0.8849
]

layers = list(range(len(layer_sensitivity)))



quant_models = ["FP16", "8bit", "4bit_nf4", "4bit_fp4"]
ppl = [12.5600, 12.6008, 12.7195, 13.2768]
memory = [3061.27, 2521.64, 2581.07, 3160.47]



plt.figure()
plt.plot(layers, layer_sensitivity)
plt.title("Layer Sensitivity (FP16 Noise Analysis)")
plt.xlabel("Layer Index")
plt.ylabel("Sensitivity (ΔPPL)")
plt.tight_layout()
plt.show()


plt.figure()
plt.scatter(memory, ppl)

for i, model in enumerate(quant_models):
    plt.annotate(model, (memory[i], ppl[i]))

plt.xlabel("Memory (MB)")
plt.ylabel("Perplexity")
plt.title("Quantization Tradeoff: Quality vs Memory")
plt.tight_layout()
plt.show()



sorted_pairs = sorted(zip(layers, layer_sensitivity),
                      key=lambda x: x[1],
                      reverse=True)

sorted_layers = [x[0] for x in sorted_pairs]
sorted_sens = [x[1] for x in sorted_pairs]

plt.figure()
plt.bar(sorted_layers, sorted_sens)
plt.title("Layer Sensitivity Ranking")
plt.xlabel("Layer")
plt.ylabel("Sensitivity (ΔPPL)")
plt.tight_layout()
plt.show()