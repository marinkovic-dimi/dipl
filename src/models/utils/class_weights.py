from typing import Dict


def calculate_class_weights(class_counts: Dict[int, int], strategy: str = 'balanced') -> Dict[int, float]:
    if strategy == 'balanced':
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)

        weights = {}
        for class_id, count in class_counts.items():
            weights[class_id] = total_samples / (num_classes * count)

    elif strategy == 'inverse':
        total_samples = sum(class_counts.values())
        weights = {}
        for class_id, count in class_counts.items():
            weights[class_id] = total_samples / count

    elif strategy == 'log':
        import math
        max_count = max(class_counts.values())
        weights = {}
        for class_id, count in class_counts.items():
            weights[class_id] = math.log(max_count / count) + 1

    else:
        raise ValueError(f"Unsupported weighting strategy: {strategy}")

    return weights
