import torch
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


def evaluate_model(model_name, quant_config=None, dtype=None):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        dtype=dtype,
        device_map={"": 0}
    )

    model.eval()

    ppl = compute_perplexity(model, tokenizer, TEXT)

    del model
    clear_gpu()

    return ppl


print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
TEXT = "\n\n".join(dataset["validation"]["text"][:50])

results = {}

#FP16

print("\nEvaluating FP16...")
results["FP16"] = evaluate_model(
    MODEL_NAME,
    quant_config=None,
    dtype=torch.float16
)
print("FP16 PPL:", results["FP16"])


# 8-bit

print("\nEvaluating 8-bit...")

bnb_8bit = BitsAndBytesConfig(
    load_in_8bit=True
)

results["8bit"] = evaluate_model(
    MODEL_NAME,
    quant_config=bnb_8bit,
    dtype=None
)
print("8-bit PPL:", results["8bit"])


#  4-bit NF4

print("\nEvaluating 4-bit NF4...")

bnb_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

results["4bit_nf4"] = evaluate_model(
    MODEL_NAME,
    quant_config=bnb_nf4,
    dtype=None
)
print("4-bit NF4 PPL:", results["4bit_nf4"])


# 4-bit FP4

print("\nEvaluating 4-bit FP4...")

bnb_fp4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

results["4bit_fp4"] = evaluate_model(
    MODEL_NAME,
    quant_config=bnb_fp4,
    dtype=None
)
print("4-bit FP4 PPL:", results["4bit_fp4"])


# FINAL SUMMARY

print("\n========= FINAL RESULTS =========")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

print("\nDegradation vs FP16:")
for k in results:
    if k != "FP16":
        print(f"{k}: +{results[k] - results['FP16']:.4f}")