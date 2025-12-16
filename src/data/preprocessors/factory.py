from .base import TextPreprocessor
from .serbian_preprocessor import SerbianTextPreprocessor
from .basic_preprocessor import BasicTextPreprocessor


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
