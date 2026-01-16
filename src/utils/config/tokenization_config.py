from dataclasses import dataclass, field


@dataclass
class TokenizationConfig:
    vocab_size: int = 15000
    max_length: int = 100
    min_frequency: int = 2
    special_tokens: list = field(default_factory=lambda: ["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"])
    use_cached_tokenizer: bool = True
    tokenizer_cache_dir: str = "cache/tokenizers"
    # Legacy parameter from old configs (not used but needed for backward compatibility)
    tokenized_cache_dir: str = "cache/tokenized_datasets"
    use_tokenized_cache: bool = False
