import copy
from datasets import load_dataset
from models.load_model_fp16 import load_model_fp16
from evaluation.evaluate import compute_perplexity
from quantization.fake_quant import fake_4bit_quantize


def quantize_layer(layer):

    for name, param in layer.named_parameters():
        param.data = fake_4bit_quantize(param.data)


def run_precision_sensitivity():

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(dataset["validation"]["text"][:50])

    model, tokenizer, layers = load_model_fp16()

    baseline = compute_perplexity(model, tokenizer, text)
    print("FP16 Baseline PPL:", baseline)

    results = []

    for i, layer in enumerate(layers):

        print(f"\nLayer {i}")

        original_state = copy.deepcopy(layer.state_dict())

        quantize_layer(layer)

        ppl = compute_perplexity(model, tokenizer, text)
        sensitivity = ppl - baseline

        print(f"PPL: {ppl:.4f} | Sensitivity: {sensitivity:.4f}")

        results.append((i, sensitivity))

        layer.load_state_dict(original_state)

    return results


if __name__ == "__main__":
    run_precision_sensitivity()