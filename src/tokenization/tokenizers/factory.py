from .base import EnhancedTokenizer
from .wordpiece import WordPieceTokenizer


def create_tokenizer(
    tokenizer_type: str = "wordpiece",
    **kwargs
) -> EnhancedTokenizer:
    if tokenizer_type == "wordpiece":
        return WordPieceTokenizer(**kwargs)
    elif tokenizer_type == "enhanced":
        return EnhancedTokenizer(**kwargs)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
