import torch
from transformers import AutoModelForSequenceClassification


def load_model(model_name, device="cpu"):
    """
    Loads pretrained LLM and extracts transformer layers.
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    transformer_layers = None

    if "distilbert" in model_name.lower():
        transformer_layers = model.distilbert.transformer.layer

    return model, transformer_layers
