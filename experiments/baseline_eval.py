import torch
from torch.utils.data import DataLoader

from models.load_model import load_model
from data_utils.load_dataset import load_sst2
from evaluation.evaluate import evaluate


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Using device: {DEVICE}")

    dataset = load_sst2(MODEL_NAME)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, _ = load_model(MODEL_NAME, DEVICE)

    acc = evaluate(model, dataloader, DEVICE)

    print(f"\nBaseline FP32 Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
