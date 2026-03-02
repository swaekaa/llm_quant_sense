import torch
import bitsandbytes as bnb
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


def quantize_mlp_layers(model):
    for i, layer in enumerate(model.model.layers):
        print(f"Quantizing MLP in Layer {i}")

        old_fc1 = layer.mlp.fc1
        new_fc1 = bnb.nn.Linear4bit(
            old_fc1.in_features,
            old_fc1.out_features,
            bias=old_fc1.bias is not None,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type="nf4"
        ).to("cuda")

        new_fc1.load_state_dict(old_fc1.state_dict())
        layer.mlp.fc1 = new_fc1

        old_fc2 = layer.mlp.fc2
        new_fc2 = bnb.nn.Linear4bit(
            old_fc2.in_features,
            old_fc2.out_features,
            bias=old_fc2.bias is not None,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type="nf4"
        ).to("cuda")

        new_fc2.load_state_dict(old_fc2.state_dict())
        layer.mlp.fc2 = new_fc2

    return model


def run():

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(dataset["validation"]["text"][:50])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    

    print("\nLoading FP16 model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map={"": 0}
    )

    model.eval()
    fp16_ppl = compute_perplexity(model, tokenizer, text)
    print("FP16 PPL:", fp16_ppl)

    del model
    clear_gpu()

    

    print("\nLoading FULL 4-bit model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0}
    )

    model.eval()
    full_4bit_ppl = compute_perplexity(model, tokenizer, text)
    print("Full 4-bit PPL:", full_4bit_ppl)

    del model
    clear_gpu()

    
    print("\nLoading FP16 model for structured quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map={"": 0}
    )

    model = quantize_mlp_layers(model)
    model.eval()

    mixed_ppl = compute_perplexity(model, tokenizer, text)
    print("Structured Mixed PPL:", mixed_ppl)

    print("\n====== FINAL SUMMARY ======")
    print("FP16:", fp16_ppl)
    print("Full 4-bit:", full_4bit_ppl)
    print("Structured Mixed:", mixed_ppl)


if __name__ == "__main__":
    run()