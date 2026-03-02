import torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from evaluation.evaluate import compute_perplexity



# Choose which layers to restore to FP16
# (based on sensitivity results)


CRITICAL_LAYERS = [0, 31, 26, 27]


def restore_layer_to_fp16(model, layer_id):
    """
    Converts all 4-bit linear layers inside a specific transformer
    block back to float16.
    """

    layer = model.model.layers[layer_id]

    for module in layer.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            module.to(torch.float16)


def run():

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(dataset["validation"]["text"][:50])

    print("\nLoading FULL 4-bit model...")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.eval()

    print("\nComputing FULL 4-bit perplexity...")
    full_4bit_ppl = compute_perplexity(model, tokenizer, text)
    print("Full 4-bit PPL:", full_4bit_ppl)

    print("\nRestoring critical layers to FP16...")
    for layer_id in CRITICAL_LAYERS:
        restore_layer_to_fp16(model, layer_id)

    print("\nComputing Mixed Precision perplexity...")
    mixed_ppl = compute_perplexity(model, tokenizer, text)

    print("\nMixed Precision PPL:", mixed_ppl)
    print("Improvement over 4-bit:", full_4bit_ppl - mixed_ppl)


if __name__ == "__main__":
    run()