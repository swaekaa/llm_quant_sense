import matplotlib.pyplot as plt



results = {
    "FP16":      {"ppl": 12.5600, "speed": 3282.10, "memory": 3061.27},
    "8bit":      {"ppl": 12.6008, "speed": 1307.33, "memory": 2521.64},
    "4bit_nf4":  {"ppl": 12.7195, "speed": 2470.85, "memory": 2581.07},
    "4bit_fp4":  {"ppl": 13.2768, "speed": 2440.90, "memory": 3160.47},
}

models = list(results.keys())
ppl = [results[m]["ppl"] for m in models]
speed = [results[m]["speed"] for m in models]
memory = [results[m]["memory"] for m in models]



plt.figure()
plt.bar(models, ppl)
plt.title("Perplexity Comparison")
plt.ylabel("Perplexity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure()
plt.bar(models, speed)
plt.title("Inference Speed (Tokens/sec)")
plt.ylabel("Tokens/sec")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



plt.figure()
plt.bar(models, memory)
plt.title("Peak GPU Memory Usage")
plt.ylabel("Memory (MB)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



plt.figure()
plt.scatter(memory, ppl)

for i, model in enumerate(models):
    plt.annotate(model, (memory[i], ppl[i]))

plt.xlabel("Memory (MB)")
plt.ylabel("Perplexity")
plt.title("Quality vs Memory Tradeoff")
plt.tight_layout()
plt.show()