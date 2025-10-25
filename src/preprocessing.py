from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "Musixmatch/umberto-commoncrawl-cased-v1"):
    """
    Loads and returns the tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_batch(batch, tokenizer, max_length: int = 128):
    """
    Tokenizes a batch of text samples for use with 🤗 Datasets map().
    """
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=max_length
    )
