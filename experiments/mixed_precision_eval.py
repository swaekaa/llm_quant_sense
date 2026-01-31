import torch
from torch.utils.data import DataLoader

from models.load_model import load_model
from data_utils.load_dataset import load_sst2
from evaluation.evaluate import evaluate
from quantization.mixed_precision import apply_mixed_precision


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")

    dataset = load_sst2(MODEL_NAME)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, _ = load_model(MODEL_NAME, DEVICE)

    # Your sensitivity-driven mixed precision policy
    layer_bit_map = {
    0: 8,
    1: 8,
    2: 8,
    3: 8,
    4: 8,
    5: 4
}


    mixed_model = apply_mixed_precision(model, layer_bit_map)
    mixed_model.to(DEVICE)

    acc = evaluate(mixed_model, dataloader, DEVICE)

    print(f"\nMixed Precision Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
