from datasets import load_dataset
from models.load_model import load_model
from evaluation.evaluate import compute_perplexity

print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

text = "\n\n".join(dataset["validation"]["text"][:50])

print("Loading model...")
model, tokenizer, layers = load_model()

print("Computing perplexity...")
ppl = compute_perplexity(model, tokenizer, text)

print("Baseline Perplexity:", ppl)