from datasets import load_dataset
from transformers import AutoTokenizer


def load_sst2(model_name, max_length=128):
    """
    Loads SST-2 dataset and tokenizes it for the given model.
    Returns validation split for fast evaluation.
    """

    dataset = load_dataset("glue", "sst2")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return dataset["validation"]
