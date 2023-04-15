import logging
import os
from abc import ABC
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from reading import Reading
from tqdm import tqdm
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw
from tslearn.utils import to_time_series_dataset


class Selector(ABC):
    def select(self, category: str, size: int = 5):
        pass


class RandomSelector(Selector):
    def select(self, category: str, size: int = 5) -> List[str]:
        reading = Reading()
        all_datasets = reading.return_datasets_for_category(category=category)
        return np.random.choice(all_datasets, size=size)


class DBASelector(Selector):
    def __init__(self, saving_directory="saved"):
        self.saving_directory = os.path.join(Path(__file__).parent, saving_directory)
        self.similarity_matrices = self.__load_similarity_matrices()

    def select(self, category: str, size: int = 5) -> List[str]:
        matrix = self.similarity_matrices[category]

    @staticmethod
    def __transform_to_fourier(series: pd.Series):
        return np.abs(np.fft.fft(series))

    @staticmethod
    def __partition_by_class(X: np.array, y: np.array) -> Dict[str, pd.Series]:
        classes = np.unique(y.ravel())
        result = {}
        for class_ in classes:
            result[class_] = X[y.ravel() == class_]
        return result

    def calculate_similarity_matrix(self):
        reading = Reading()
        categories = sorted(set(reading.categories.values()))
        for category in tqdm(categories, "Similarity per category..."):
            self.calculate_similarity_matrix_per_category(category)

    def calculate_similarity_matrix_per_category(self, category):
        reading = Reading()
        whole_category_X, whole_category_y = reading.read_dataset(category=category)
        partitioned_dataset = self.__partition_by_class(
            whole_category_X, whole_category_y
        )
        similarities = self.__get_similarities_per_class(
            partitioned_dataset,
        )
        similarities_per_dataset = (
            similarities.groupby("dataset_1", "dataset_2")
            .min("similarity")
            .reset_index()
        )
        matrix: pd.DataFrame = similarities_per_dataset.pivot(
            index="dataset_1", columns="dataset_2", values="similarity"
        )
        assert not np.isnan(matrix.values).any()
        matrix.to_csv(os.path.join(self.saving_directory, f"{category}.csv"))

    def __get_similarities_per_class(self, partitioned_dataset: Dict[str, np.array]):
        similarities = []
        classes_product = [
            (c_1, c_2)
            for c_1 in partitioned_dataset
            for c_2 in partitioned_dataset
            if c_1 >= c_2
        ]
        for class_1, class_2 in tqdm(classes_product, leave=False):
            x_1 = partitioned_dataset[class_1]
            x_2 = partitioned_dataset[class_2]
            if class_1 == class_2:
                dba_similarity = self.dba_similarity(x_1, x_2)
            else:
                dba_similarity = float("inf")
            similarities.append(
                {
                    "dataset_1": class_1.split("_")[0],
                    "dataset_2": class_2.split("_")[0],
                    "similarity": dba_similarity,
                }
            )
            similarities.append(
                {
                    "dataset_1": class_2.split("_")[0],
                    "dataset_2": class_1.split("_")[0],
                    "similarity": dba_similarity,
                }
            )
        return pd.DataFrame(similarities)

    @staticmethod
    def dba_similarity(x_1, x_2) -> float:
        barycenter_1 = dtw_barycenter_averaging(to_time_series_dataset(x_1))
        barycenter_2 = dtw_barycenter_averaging(to_time_series_dataset(x_2))
        return dtw(barycenter_1, barycenter_2)

    def __load_similarity_matrices(self) -> Dict[str, pd.DataFrame]:
        matrices = {}
        files = os.listdir(self.saving_directory)
        if not files:
            logging.warning(
                "%s is empty. Please run calculate_similarity_matrix method",
                self.saving_directory,
            )
        for file in files:
            matrices[file.rstrip(".csv")] = pd.read_csv(file)
        return matrices


if __name__ == "__main__":
    DBASelector().calculate_similarity_matrix()
