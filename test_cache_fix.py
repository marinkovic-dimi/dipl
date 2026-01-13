#!/usr/bin/env python3
"""
Quick test to verify the cache JSON serialization fix.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.tokenization.cache import TokenizedDatasetCache

def test_cache_fix():
    print("Testing cache JSON serialization fix...")

    # Create cache instance
    cache = TokenizedDatasetCache(cache_dir="cache/test", verbose=False)

    # Create dummy metadata with numpy types (like the real metadata)
    metadata = {
        'num_classes': np.int64(1497),
        'vocab_size': np.int64(15000),
        'class_distribution': {
            np.int64(0): np.int64(1000),
            np.int64(1): np.int64(500),
            np.int64(2): np.int64(750),
        },
        'statistics': {
            'mean': np.float64(123.45),
            'std': np.float64(67.89),
        },
        'flags': {
            'balanced': np.bool_(True),
            'filtered': np.bool_(False),
        }
    }

    print("Original metadata types:")
    print(f"  num_classes: {type(metadata['num_classes'])}")
    print(f"  class_dist key: {type(list(metadata['class_distribution'].keys())[0])}")
    print(f"  mean: {type(metadata['statistics']['mean'])}")

    # Test conversion
    converted = cache._convert_numpy_types(metadata)

    print("\nConverted metadata types:")
    print(f"  num_classes: {type(converted['num_classes'])}")
    print(f"  class_dist key: {type(list(converted['class_distribution'].keys())[0])}")
    print(f"  mean: {type(converted['statistics']['mean'])}")

    # Verify JSON serialization works
    import json
    try:
        json_str = json.dumps(converted, indent=2)
        print("\n✓ JSON serialization successful!")
        print(f"JSON length: {len(json_str)} bytes")
        return True
    except TypeError as e:
        print(f"\n✗ JSON serialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_cache_fix()
    sys.exit(0 if success else 1)
