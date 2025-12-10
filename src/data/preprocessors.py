import re
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Set, Union, Optional, Callable
from ..utils import LoggerMixin, SERBIAN_STOP_WORDS


class TextPreprocessor(ABC, LoggerMixin):

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        pass

    def preprocess_dataframe(
        self,
        data: pd.DataFrame,
        text_column: str,
        output_column: str,
        inplace: bool = False
    ) -> pd.DataFrame:
        if not inplace:
            data = data.copy()

        if self.verbose:
            self.logger.info(f"Preprocessing text column '{text_column}' -> '{output_column}'")

        data[output_column] = data[text_column].apply(self.preprocess_text)

        empty_mask = data[output_column].str.strip() == ''
        if empty_mask.any():
            empty_count = empty_mask.sum()
            data = data[~empty_mask]
            if self.verbose:
                self.logger.warning(f"Removed {empty_count} rows with empty preprocessed text")

        if self.verbose:
            self.logger.info(f"Preprocessing complete. Final dataset size: {len(data)} rows")

        return data


class SerbianTextPreprocessor(TextPreprocessor):

    def __init__(
        self,
        transliterate_cyrillic: bool = True,
        lowercase: bool = True,
        remove_stop_words: bool = True,
        stop_words: Optional[Set[str]] = None,
        custom_transformations: Optional[List[Callable[[str], str]]] = None,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.transliterate_cyrillic = transliterate_cyrillic
        self.lowercase = lowercase
        self.remove_stop_words = remove_stop_words
        self.stop_words = stop_words or set(SERBIAN_STOP_WORDS)
        self.custom_transformations = custom_transformations or []

        self.cyrillic_to_latin = {
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Ђ': 'Đ', 'Е': 'E',
            'Ж': 'Ž', 'З': 'Z', 'И': 'I', 'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj',
            'М': 'M', 'Н': 'N', 'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S',
            'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Č',
            'Џ': 'Dž', 'Ш': 'Š',
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e',
            'ж': 'ž', 'з': 'z', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj',
            'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's',
            'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č',
            'џ': 'dž', 'ш': 'š'
        }

    def detect_script(self, text: str) -> str:
        cyrillic_range = range(0x0410, 0x0450)
        latin_range = [
            *range(0x0041, 0x005B),
            *range(0x0061, 0x007B),
            0x0106, 0x010C, 0x0110, 0x0160, 0x017D,
            0x0107, 0x010D, 0x0111, 0x0161, 0x017E
        ]

        cyrillic = any(ord(ch) in cyrillic_range for ch in text if ch.isalpha())
        latin = any(ord(ch) in latin_range for ch in text if ch.isalpha())

        if cyrillic and not latin:
            return "Cyrillic"
        elif latin and not cyrillic:
            return "Latin"
        elif cyrillic and latin:
            return "Mixed"
        else:
            return "Unknown"

    def transliterate_cyrillic_to_latin(self, text: str) -> str:
        return ''.join(self.cyrillic_to_latin.get(char, char) for char in text)

    def extract_words(self, text: str) -> List[str]:
        return re.findall(r'[\w+]+', text)

    def contains_number(self, word: str) -> bool:
        return bool(re.search(r'\d', word))

    def concatenate_i_phone(self, text: str) -> str:
        pattern = r'\bi phone\b'
        return re.sub(pattern, "iphone", text, flags=re.IGNORECASE)

    def split_numbers_on_digits(self, text: str) -> str:
        text = re.sub(r'(\d+)', r' \1 ', text)
        text = re.sub(r'(\d)', r' \1 ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return ""

        if self.transliterate_cyrillic:
            text = self.transliterate_cyrillic_to_latin(text)

        if self.lowercase:
            text = text.lower()

        for transformation in self.custom_transformations:
            text = transformation(text)

        text = self.concatenate_i_phone(text)

        words = self.extract_words(text)

        if self.remove_stop_words:
            words = [word for word in words if word not in self.stop_words]

        return " ".join(words)

    def get_preprocessing_stats(self, data: pd.DataFrame, text_column: str) -> dict:
        stats = {
            'total_texts': len(data),
            'script_distribution': {},
            'avg_word_count_before': 0,
            'avg_word_count_after': 0,
            'empty_after_preprocessing': 0
        }

        scripts = data[text_column].apply(self.detect_script).value_counts()
        stats['script_distribution'] = scripts.to_dict()

        word_counts_before = data[text_column].apply(lambda x: len(self.extract_words(x)) if pd.notna(x) else 0)
        stats['avg_word_count_before'] = word_counts_before.mean()

        preprocessed = data[text_column].apply(self.preprocess_text)
        word_counts_after = preprocessed.apply(lambda x: len(x.split()) if x else 0)
        stats['avg_word_count_after'] = word_counts_after.mean()

        stats['empty_after_preprocessing'] = (preprocessed.str.strip() == '').sum()

        return stats


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


def create_preprocessor(
    preprocessor_type: str = "serbian",
    **kwargs
) -> TextPreprocessor:
    if preprocessor_type == "serbian":
        return SerbianTextPreprocessor(**kwargs)
    elif preprocessor_type == "basic":
        return BasicTextPreprocessor(**kwargs)
    else:
        raise ValueError(f"Unsupported preprocessor type: {preprocessor_type}")