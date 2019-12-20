import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.random as rnd
import os
import pandas as pd
import sklearn.linear_model
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from categoricalEncoder import CategoricalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from functions import split_train_test, CombinedAttributesAdder, DataFrameSelector, save_fig

HOUSING_PATH = os.path.join("data", "housing")

print('Housing Info!')


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
housing.rename(columns={"longitude": "Dł. geograficzna", "latitude": "Szer. geograficzna", "housing_median_age": "Mediana wieku mieszkań",
                        "total_rooms": "Całk. liczba pokoi", "total_bedrooms": "Całk. liczba sypialni", "population": "Populacja",
                        "households": "Rodziny", "median_income": "Mediana dochodów", "median_house_value": "Mediana cen mieszkań",
                        "ocean_proximity": "Odległość do oceanu"}, inplace=True)

# Dzieli wartość przez 1,5 w celu ograniczenia liczby kategorii dochodów
housing["kat_dochodów"] = np.ceil(housing["Mediana dochodów"] / 1.5)
# Wartości przekraczające wartość 5 zostają przyporządkowane do klasy 5
housing["kat_dochodów"].where(housing["kat_dochodów"] < 5, 5.0, inplace=True)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["kat_dochodów"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("kat_dochodów", axis=1, inplace=True)

housing = strat_train_set.copy()

housing["Pokoje_na_rodzinę"] = housing["Całk. liczba pokoi"]/housing["Rodziny"]
housing["Sypialnie_na_pokoje"] = housing["Całk. liczba sypialni"] / \
    housing["Całk. liczba pokoi"]
housing["Populacja_na_rodzinę"] = housing["Populacja"]/housing["Rodziny"]

# corr_matrix = housing.corr()
# print(corr_matrix["Mediana cen mieszkań"])

# usuwa etykiety w zbiorze uczącym
housing = strat_train_set.drop("Mediana cen mieszkań", axis=1)
housing_labels = strat_train_set["Mediana cen mieszkań"].copy()

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows = sample_incomplete_rows.drop(
    "Całk. liczba sypialni", axis=1)

housing_num = housing.drop('Odległość do oceanu', axis=1)

print(housing_num.head())

imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=list(housing.index.values))

housing_cat = housing['Odległość do oceanu']

cat_encoder = CategoricalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(
    housing.columns)+["Pokoje_na_rodzinę", "Populacja_na_rodzinę"])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["Odległość do oceanu"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
