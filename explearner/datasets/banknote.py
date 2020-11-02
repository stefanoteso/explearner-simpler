import numpy as np
import pandas as pd
import os, requests

from itertools import product
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct

from . import TreeDataset, EqJaccardRewardMixin


class BanknoteAuth(EqJaccardRewardMixin, TreeDataset):
    def __init__(self, **kwargs):
        urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"]
        self.load_dataset('data', urls)

        # total 1372 instances, 610 1s = fake, 762 0s = real
        columns = ["variance", "skew", "curtosis", "entropy", "class"]
        dataset = pd.read_csv("data/data_banknote_authentication.txt", names=columns)

        # target values: 1 is fake, 0 is real
        y = dataset['class'].to_numpy()
        # creating the feature vector
        X = dataset.drop('class', axis=1).to_numpy()

        clf = DecisionTreeClassifier()
        clf = clf.fit(X, y)
        self.tree = clf.tree_
        # Extremely sparse explanations
        Z = clf.decision_path(X).toarray()

        # Avoid sparse features - Maybe in the future
        # decision_path = clf.decision_path(X).toarray()
        # for i in np.arange(len(decision_path.indptr) - 1):
        #     decision_path.indices[decision_path.indptr[i]:decision_path.indptr[i+1]]
        # Z = np.array(Z)

        # Kernels
        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = DotProduct(sigma_0=1, sigma_0_bounds=(1, 1))  # Explanations are sparse
        ky = DotProduct(sigma_0=1, sigma_0_bounds=(1, 1))

        # Extract all possible root-to-leaf explanations
        arms_z = self.root_to_leaf_paths(0)
        arms_y = np.array([0, 1])
        arms = list(product(arms_z, arms_y))
        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)

    @staticmethod
    def load_dataset(path, urls):
        """
        Download the content for a list of URLs and save them to a folder
        :param path: The path to the location where the data will be saved
        :param urls: The list of URLs from which the content will be downloaded and saved
        """
        if not os.path.exists(path):
            os.mkdir(path)

        for url in urls:
            data = requests.get(url).content
            filename = os.path.join(path, os.path.basename(url))
            with open(filename, "wb") as file:
                file.write(data)
