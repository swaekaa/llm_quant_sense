import torch
from torch.utils.data import DataLoader

from models.load_model import load_model
from data_utils.load_dataset import load_sst2
from evaluation.evaluate import evaluate
from quantization.layerwise import quantize_single_transformer_layer
from sensitivity.metrics import sensitivity_score


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Using device: {DEVICE}")

    dataset = load_sst2(MODEL_NAME)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, layers = load_model(MODEL_NAME, DEVICE)

    baseline_acc = evaluate(model, dataloader, DEVICE)

    print(f"\nBaseline Accuracy: {baseline_acc * 100:.2f}%\n")

    results = {}

    for i in range(len(layers)):
        print(f"\nQuantizing layer {i} ...")

        q_model = quantize_single_transformer_layer(model, i, num_bits=8)

        q_model.to(DEVICE)

        q_acc = evaluate(q_model, dataloader, DEVICE)

        drop = sensitivity_score(baseline_acc, q_acc)

        results[f"Layer {i}"] = drop

        print(f"Layer {i} → Accuracy: {q_acc*100:.2f}% | Drop: {drop*100:.2f}%")

    print("\n====== Sensitivity Summary ======")
    for k, v in results.items():
        print(f"{k}: {v*100:.2f}% drop")


if __name__ == "__main__":
    main()
