import re
import pandas as pd
from .base import TextPreprocessor


class BasicTextPreprocessor(TextPreprocessor):

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_extra_whitespace: bool = True,
        min_word_length: int = 2,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_word_length = min_word_length

    def preprocess_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return ""

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        if self.min_word_length > 1:
            words = text.split()
            words = [word for word in words if len(word) >= self.min_word_length]
            text = " ".join(words)

        return text
