import torch
from datasets import load_dataset
from models.load_model import load_model
from evaluation.evaluate import compute_perplexity


def mask_module(module):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            zero_hidden = torch.zeros_like(hidden)
            return (zero_hidden,) + output[1:]
        return torch.zeros_like(output)

    return module.register_forward_hook(hook)


def run_component_sensitivity():

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(dataset["validation"]["text"][:50])

    model, tokenizer, layers = load_model()

    baseline = compute_perplexity(model, tokenizer, text)
    print("Baseline PPL:", baseline)

    results = []

    for i, layer in enumerate(layers):

        print(f"\nLayer {i}")

        # --- Attention ---
        attn_hook = mask_module(layer.self_attn)
        ppl_attn = compute_perplexity(model, tokenizer, text)
        attn_hook.remove()

        # --- MLP ---
        mlp_hook = mask_module(layer.mlp)
        ppl_mlp = compute_perplexity(model, tokenizer, text)

        mlp_hook.remove()

        print(f"Attention PPL: {ppl_attn:.2f}")
        print(f"MLP PPL: {ppl_mlp:.2f}")

        results.append((i, ppl_attn - baseline, ppl_mlp - baseline))

    return results


if __name__ == "__main__":
    run_component_sensitivity()