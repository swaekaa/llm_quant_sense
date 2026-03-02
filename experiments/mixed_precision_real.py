import torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.evaluate import compute_perplexity


# CONFIGURE HERE


FP16_LAYERS = [0, 31]

EIGHT_BIT_LAYERS = [4, 5, 6, 11, 12, 13, 25, 26, 27]

FOUR_BIT_LAYERS = [2, 9, 10, 17, 20, 24]


def convert_linear(module, layer_id):

    for name, child in module.named_children():

        if isinstance(child, torch.nn.Linear):

            in_f = child.in_features
            out_f = child.out_features
            bias = child.bias is not None

            if layer_id in FOUR_BIT_LAYERS:

                new_layer = bnb.nn.Linear4bit(
                    in_f,
                    out_f,
                    bias=bias,
                    compute_dtype=torch.float16,
                    compress_statistics=True,
                    quant_type="nf4"
                ).cuda()

            elif layer_id in EIGHT_BIT_LAYERS:

                new_layer = bnb.nn.Linear8bitLt(
                    in_f,
                    out_f,
                    bias=bias
                ).cuda()

            else:
                continue

            new_layer.weight.data = child.weight.data.clone()
            if bias:
                new_layer.bias.data = child.bias.data.clone()

            setattr(module, name, new_layer)

        else:
            convert_linear(child, layer_id)


def apply_mixed_precision(model):

    layers = model.model.layers

    for i, layer in enumerate(layers):

        if i in FP16_LAYERS:
            continue

        convert_linear(layer, i)

    return model


def run_mixed_precision():

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(dataset["validation"]["text"][:50])

    print("Loading FP16 model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        dtype=torch.float16
    ).cuda()

    model.eval()

    baseline = compute_perplexity(model, tokenizer, text)
    print("FP16 Baseline:", baseline)

    print("\nApplying mixed precision...")
    model = apply_mixed_precision(model)

    mixed_ppl = compute_perplexity(model, tokenizer, text)
    print("Mixed Precision PPL:", mixed_ppl)

    print("\nDelta:", mixed_ppl - baseline)


if __name__ == "__main__":
    run_mixed_precision()