import logging
import os
from abc import ABC
from pathlib import Path
from typing import Dict
from typing import List

import mlflow
import numpy as np
import pandas as pd
from reading import Reading
from tqdm import tqdm
from tslearn.barycenters import dtw_barycenter_averaging_subgradient
from tslearn.metrics import dtw


class Selector(ABC):
    def select(self, dataset: str, size: int = 5):
        raise NotImplementedError()

    def _log_datasets(self, datasets):
        mlflow.log_param("Datasets used for ensemble", ", ".join(datasets))


class RandomSelector(Selector):
    def select(self, dataset: str, size: int = 5) -> List[str]:
        reading = Reading()
        category = reading.categories[dataset]
        all_datasets = reading.return_datasets_for_category(category=category)
        result = np.random.choice(all_datasets, size=size)
        self._log_datasets(result)
        return result


class DBASelector(Selector):
    def __init__(self, saving_directory="saved"):
        self.saving_directory = os.path.join(Path(__file__).parent, saving_directory)
        self.similarity_matrices = self.__load_similarity_matrices()

    def select(self, dataset: str, size: int = 5) -> List[str]:
        reading = Reading()
        category = reading.categories[dataset]
        matrix = self.similarity_matrices[category]
        similarities_for_dataset = matrix[dataset]
        nsmallest = similarities_for_dataset.nsmallest(n=size)
        mlflow.log_metric("Mean DBA similarity", float(np.mean(nsmallest)))
        mlflow.log_param("DBA similarities", str(nsmallest.to_dict()))
        result = list(nsmallest.index)
        self._log_datasets(result)
        return result

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
            similarities.groupby(["dataset_1", "dataset_2"])
            .min("similarity")
            .reset_index()
        )
        matrix: pd.DataFrame = (
            similarities_per_dataset.pivot(
                index="dataset_1", columns="dataset_2", values="similarity"
            )
            .rename_axis(None, axis=0)
            .rename_axis(None, axis=1)
        )
        np.fill_diagonal(matrix.values, val=float("inf"))
        assert not np.isnan(matrix.values).any()
        assert np.allclose(matrix.values, matrix.values.T)
        matrix.to_csv(os.path.join(self.saving_directory, f"{category}.csv"))

    def __get_similarities_per_class(self, partitioned_dataset: Dict[str, np.array]):
        similarities = []
        barycenters = self.__get_barycenters_per_class(partitioned_dataset)
        classes_product = [
            (c_1, c_2)
            for c_1 in partitioned_dataset
            for c_2 in partitioned_dataset
            if c_1 > c_2
        ]
        for class_1, class_2 in classes_product:
            dba_similarity = dtw(barycenters[class_1], barycenters[class_2])
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
    def __get_barycenters_per_class(partitioned_dataset: Dict[str, np.array]):
        result = {}
        for class_name in tqdm(partitioned_dataset, desc="Calculating barycenters ..."):
            X = partitioned_dataset[class_name]
            X = np.random.choice(X, size=min(30, len(X)))
            X = [x[:1000] for x in X]
            result[class_name] = dtw_barycenter_averaging_subgradient(X)
        return result

    def __load_similarity_matrices(self) -> Dict[str, pd.DataFrame]:
        matrices = {}
        files = os.listdir(self.saving_directory)
        if not files:
            logging.warning(
                "%s is empty. Please run calculate_similarity_matrix method",
                self.saving_directory,
            )
        for file in files:
            matrices[file.rstrip(".csv")] = pd.read_csv(
                os.path.join(self.saving_directory, file), index_col=0
            )
        return matrices


if __name__ == "__main__":
    DBASelector().calculate_similarity_matrix()
