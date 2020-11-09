import numpy as np
from itertools import product
from sympy.utilities.iterables import multiset_permutations
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
from treeinterpreter import treeinterpreter as ti

from . import Dataset, EqKendallRewardMixin
from .. import KendallKernel, rank_items, kendall_tau_dist, load, dump


class BreastCancer(EqKendallRewardMixin, Dataset):
    """
    Breast cancer dataset with ranking explanations extracted from a Random Forest classifier.
    """

    def __init__(self, **kwargs):
        # Samples per class	212(M),357(B)
        dataset = load_breast_cancer()
        pca_dim = 8

        # Data
        X = dataset.data
        # 0 is "malignant", 1 is "benign"
        y = dataset.target

        # We reduce the dimensionality to be able to generate all possible rankings as explanations
        normalized_data = StandardScaler().fit_transform(X)
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(normalized_data)

        rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        rf.fit(X, y)
        _, _, contributions = ti.predict(rf, X)

        Z = np.array([rank_items(contr) if y[i] else rank_items(-contr)
                      for i, contr in enumerate(contributions[:, :, 0])])

        # Kernels
        kx = RBF(length_scale=1, length_scale_bounds=(1, 1))
        kz = KendallKernel()  # ranking kernels
        ky = DotProduct(sigma_0=1, sigma_0_bounds=(1, 1))

        # The space of explanations is all possible permutations!
        # TODO: Improve efficiency by excluding some permutation
        arms_z = np.array(list(multiset_permutations(np.arange(pca_dim))))
        arms_y = np.array([0, 1])
        arms = list(product(arms_z, arms_y))

        super().__init__(X, Z, y, kx, kz, ky, arms, **kwargs)
