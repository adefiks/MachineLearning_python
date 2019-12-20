import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.random as rnd
import os
import pandas as pd
import sklearn.linear_model
from sklearn.base import BaseEstimator, TransformerMixin


# Generowanie ładnych wykresów
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Lokacja, w której będą zapisywane rysunki
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "chapter_1"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "drawings",
                        CHAPTER_ID, fig_id + ".png")
    print("Zapisywanie rysunku", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # żadnych zmiennych *args ani **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nie robi nic innego

    def transform(self, X, y=None):
        Pokoje_na_rodzinę = X[:, rooms_ix] / X[:, household_ix]
        Populacja_na_rodzinę = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            Sypialnie_na_pokoje = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, Pokoje_na_rodzinę, Populacja_na_rodzinę,
                         Sypialnie_na_pokoje]
        else:
            return np.c_[X, Pokoje_na_rodzinę, Populacja_na_rodzinę]


# Tworzy klasę wybierającą numeryczne i kategorialne kolumny,
# gdyż moduł Scikit-Learn nie zawiera jeszcze obsługi obiektów DataFrame
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def display_scores(scores):
    print("Wyniki:", scores)
    print("Średnia:", scores.mean())
    print("Odchylenie standardowe:", scores.std())


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
