from .base import EnhancedTokenizer


class WordPieceTokenizer(EnhancedTokenizer):

    def __init__(self, **kwargs):
        default_kwargs = {
            'vocab_size': 15000,
            'min_frequency': 2,
            'max_length': 100,
            'special_tokens': ["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"]
        }
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def _preprocess_text_for_tokenization(self, text: str) -> str:
        text = super()._preprocess_text_for_tokenization(text)
        return text
