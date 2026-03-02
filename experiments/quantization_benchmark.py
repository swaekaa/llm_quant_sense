import torch
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from evaluation.evaluate import compute_perplexity


MODEL_NAME = "microsoft/phi-1_5"


def clear_gpu():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def measure_memory():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2  # MB


def benchmark_model(name, quant_config=None, dtype=None):

    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        dtype=dtype,
        device_map={"": 0}
    )

    model.eval()

    
    ppl = compute_perplexity(model, tokenizer, TEXT)

    
    input_ids = tokenizer(TEXT[:1000], return_tensors="pt").to("cuda")

    start = time.time()
    with torch.no_grad():
        for _ in range(20):
            _ = model(**input_ids)
    torch.cuda.synchronize()
    end = time.time()

    total_tokens = input_ids["input_ids"].numel() * 20
    tokens_per_sec = total_tokens / (end - start)

   
    memory = measure_memory()

    del model
    clear_gpu()

    return ppl, tokens_per_sec, memory


print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
TEXT = "\n\n".join(dataset["validation"]["text"][:50])

results = {}

# FP16

print("\nBenchmarking FP16...")
results["FP16"] = benchmark_model(
    "FP16",
    quant_config=None,
    dtype=torch.float16
)

# 8-bit

print("\nBenchmarking 8-bit...")

bnb_8bit = BitsAndBytesConfig(load_in_8bit=True)

results["8bit"] = benchmark_model(
    "8bit",
    quant_config=bnb_8bit,
    dtype=None
)

# 4-bit NF4

print("\nBenchmarking 4-bit NF4...")

bnb_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

results["4bit_nf4"] = benchmark_model(
    "4bit_nf4",
    quant_config=bnb_nf4,
    dtype=None
)

# 4-bit FP4

print("\nBenchmarking 4-bit FP4...")

bnb_fp4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

results["4bit_fp4"] = benchmark_model(
    "4bit_fp4",
    quant_config=bnb_fp4,
    dtype=None
)

# FINAL RESULTS


print("\n=========== FINAL BENCHMARK RESULTS ===========")

for k, (ppl, speed, mem) in results.items():
    print(f"\n{k}")
    print(f"PPL: {ppl:.4f}")
    print(f"Tokens/sec: {speed:.2f}")
    print(f"Peak GPU Memory (MB): {mem:.2f}")

print("\nDegradation vs FP16:")
fp16_ppl = results["FP16"][0]

for k in results:
    if k != "FP16":
        print(f"{k}: +{results[k][0] - fp16_ppl:.4f}")