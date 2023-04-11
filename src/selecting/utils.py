from typing import List

import numpy as np
from reading import Reading


def select_datasets_randomly(category: str, size: int = 5) -> List[str]:
    reading = Reading()
    all_datasets = reading.return_datasets_for_category(category=category)
    return np.random.choice(all_datasets, size=size)
