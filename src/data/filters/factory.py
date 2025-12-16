from typing import Optional
from .composite_filter import CompositeFilter
from .text_quality_filter import TextQualityFilter
from .ostalo_filter import OstaloGroupFilter
from .group_size_filter import GroupSizeFilter


def create_default_filters(
    class_column: str,
    text_column: str,
    min_samples_per_class: int = 50,
    max_samples_per_class: Optional[int] = None,
    ostalo_groups_file: Optional[str] = None,
    remove_ostalo: bool = True,
    **kwargs
) -> CompositeFilter:
    filters = []

    text_filter = TextQualityFilter(
        text_column=text_column,
        min_text_length=1,
        min_word_count=1,
        remove_duplicates=True,
        remove_empty=True,
        **kwargs.get('text_filter', {})
    )
    filters.append(text_filter)

    if remove_ostalo and ostalo_groups_file:
        ostalo_filter = OstaloGroupFilter(
            class_column=class_column,
            ostalo_groups_file=ostalo_groups_file,
            **kwargs.get('ostalo_filter', {})
        )
        filters.append(ostalo_filter)

    group_filter = GroupSizeFilter(
        class_column=class_column,
        min_samples=min_samples_per_class,
        max_samples=max_samples_per_class,
        **kwargs.get('group_filter', {})
    )
    filters.append(group_filter)

    return CompositeFilter(filters)
