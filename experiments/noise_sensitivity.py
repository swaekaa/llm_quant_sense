import torch
from datasets import load_dataset
from models.load_model import load_model
from evaluation.evaluate import compute_perplexity


def mask_layer_output(layer):

    def forward_hook(module, input, output):

        # If output is tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            zero_hidden = torch.zeros_like(hidden_states)
            return (zero_hidden,) + output[1:]

        # If it's tensor (just in case)
        return torch.zeros_like(output)

    return layer.register_forward_hook(forward_hook)

def run_noise_sensitivity():

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(dataset["validation"]["text"][:50])

    model, tokenizer, layers = load_model()

    baseline_ppl = compute_perplexity(model, tokenizer, text)
    print("Baseline PPL:", baseline_ppl)

    results = []

    for i, layer in enumerate(layers):

        print(f"\nTesting Layer {i}")

        hook = mask_layer_output(layer)

        ppl = compute_perplexity(model, tokenizer, text)
        sensitivity = ppl - baseline_ppl

        print(f"Layer {i} PPL: {ppl:.4f} | Sensitivity: {sensitivity:.4f}")

        results.append((i, sensitivity))

        hook.remove()

    return results


if __name__ == "__main__":
    run_noise_sensitivity()