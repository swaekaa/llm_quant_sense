import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_fp16(model_name="microsoft/phi-2"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    ).cuda()

    model.eval()

    return model, tokenizer, model.model.layers