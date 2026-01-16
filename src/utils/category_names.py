"""
Utility for loading and managing category names.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from .logging import LoggerMixin


class CategoryNameLoader(LoggerMixin):
    """Loads category names from category_name.json and creates combined names."""

    def __init__(self, category_file_path: str = "data/category_name.json"):
        """
        Initialize the category name loader.

        Args:
            category_file_path: Path to category_name.json file
        """
        self.category_file_path = Path(category_file_path)
        self.category_map: Dict[int, dict] = {}
        self.combined_names: Dict[str, str] = {}
        self._load_categories()

    def _load_categories(self):
        """Load categories from JSON file and build combined names."""
        if not self.category_file_path.exists():
            self.logger.warning(f"Category file not found: {self.category_file_path}")
            return

        try:
            with open(self.category_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            items = data.get('category_name', [])
            self.category_map = {item['category_id']: item for item in items}

            # Build combined names: {group_name} | {category_name}
            for item in items:
                category_id = item['category_id']
                group_name = item['name']
                parent_id = item.get('parent', 0)

                if parent_id != 0 and parent_id in self.category_map:
                    # This is a group under a category
                    parent = self.category_map[parent_id]
                    category_name = parent['name']
                    combined = f"{group_name} | {category_name}"
                else:
                    # This is a top-level category
                    combined = group_name

                self.combined_names[str(category_id)] = combined

            self.logger.info(f"Loaded {len(self.combined_names)} category names")

        except Exception as e:
            self.logger.error(f"Failed to load category names: {e}")
            self.combined_names = {}

    def get_name(self, category_id: str) -> str:
        """
        Get combined name for a category ID.

        Args:
            category_id: Category ID as string

        Returns:
            Combined name in format "{group_name} | {category_name}"
            or just group name if it's a top-level category,
            or the category_id itself if not found
        """
        return self.combined_names.get(str(category_id), str(category_id))

    def get_all_names(self) -> Dict[str, str]:
        """Get all category names as a dictionary."""
        return self.combined_names.copy()
