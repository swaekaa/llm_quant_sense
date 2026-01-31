import torch
from torch.utils.data import DataLoader

from models.load_model import load_model
from data_utils.load_dataset import load_sst2
from evaluation.evaluate import evaluate
from quantization.mixed_precision import apply_mixed_precision


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


def main():
    print(f"Using device: {DEVICE}")

    dataset = load_sst2(MODEL_NAME)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, _ = load_model(MODEL_NAME, DEVICE)

    # Only quantize the last layer
    layer_bit_map = {5: 4}

    q_model = apply_mixed_precision(model, layer_bit_map)
    q_model.to(DEVICE)

    acc = evaluate(q_model, dataloader, DEVICE)

    print(f"Ablation (Only Layer 5 INT4) Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
