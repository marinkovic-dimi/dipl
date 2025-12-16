import re
import pandas as pd
from typing import List, Set, Optional, Callable
from .base import TextPreprocessor
from ...utils import SERBIAN_STOP_WORDS


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
