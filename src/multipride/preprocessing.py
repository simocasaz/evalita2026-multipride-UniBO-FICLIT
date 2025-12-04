from transformers import AutoTokenizer, PreTrainedTokenizer
import re


# Function to clean the tweets
def clean_text(text: str, transform_lower: bool = False) -> str:
    """
    Clean a tweet-like string.

    Steps performed:
    - Remove user mentions of the form @username.
    - Remove the literal token "url" (case-insensitive), which in this dataset marks anonymized links.
    - Collapse consecutive whitespace into a single space and strip leading/trailing spaces.
    - Optionally convert the text to lowercase.

    Parameters:
        text (str): Input text to clean.
        transform_lower (bool): If True, convert the cleaned text to lowercase. Default: False.

    Returns:
        str: The cleaned (and optionally lowercased) text.
    """
    # Remove user mentions
    text = re.sub(r"@\w+", "", text)

    # Remove url
    text = re.sub(r"\burl\b", "", text, flags=re.I)

    # Remove useless extra space
    text = re.sub(r"\s+", " ", text).strip()

    # Transform everything in lower_case
    if transform_lower:
        text = text.lower()

    return text


def load_tokenizer(
    model_name: str,
) -> PreTrainedTokenizer:
    """
    Loads and returns the tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_batch(batch, tokenizer):
    """
    Tokenizes a batch of text samples for use with 🤗 Datasets map().
    """
    return tokenizer(batch["text"])


def tokenize_batch_with_bios(batch, tokenizer):
    return tokenizer(
        batch["text"], batch["bio"], truncation="only_second", max_length=256
    )
