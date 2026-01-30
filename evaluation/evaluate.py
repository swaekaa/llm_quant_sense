import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataloader, device="cpu"):
    """
    Evaluates classification accuracy.
    """
    model.eval()

    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total
