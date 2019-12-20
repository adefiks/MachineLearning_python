import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.random as rnd
import os
import pandas as pd
from scipy.stats import randint
from categoricalEncoder import CategoricalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from functions import split_train_test, CombinedAttributesAdder, DataFrameSelector, save_fig, display_scores, load_housing_data

HOUSING_PATH = os.path.join("data", "housing")

print('Housing Info!')

housing = load_housing_data(HOUSING_PATH)
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

# usuwa etykiety w zbiorze uczącym
housing = strat_train_set.drop("Mediana cen mieszkań", axis=1)
housing_labels = strat_train_set["Mediana cen mieszkań"].copy()

housing_num = housing.drop('Odległość do oceanu', axis=1)

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


forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

param_grid = [
    # wypróbowuje 12 (3×4) kombinacji hiperparametrów
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # następnie wypróbowuje 6 (2×3) kombinacji z wyłączonym parametrem bootstrap (False)
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

# przeprowadza proces uczenia wobec pięciu podzbioru, czyli łącznie (12+6)*5=90 przebiegów
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("Mediana cen mieszkań", axis=1)
y_test = strat_test_set["Mediana cen mieszkań"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(final_rmse)
