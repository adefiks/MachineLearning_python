import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.random as rnd
import os
import pandas as pd
import sklearn.linear_model

from functions import save_fig, prepare_country_stats
# from functions import prepare_country_stats

# datapath = os.path.join("data", "satysfaction". "")
datapath = os.path.join("data", "satisfaction")


def load_bli_data(housing_path=datapath):
    csv_path = os.path.join(housing_path, "BLI.csv")
    return pd.read_csv(csv_path, thousands=',')


def load_gdp_data(housing_path=datapath):
    csv_path = os.path.join(housing_path, "gdp_per_capita.csv")
    return pd.read_csv(csv_path, thousands=',')


oecd_bli = load_bli_data()
gdp_per_capita = load_gdp_data()

# Przygotowuje dane
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
Y = np.c_[country_stats["Life satisfaction"]]

country_stats.rename(
    columns={"GDP per capita": "PKB per capita"}, inplace=True)
country_stats.rename(
    columns={"Life satisfaction": "Satysfakcja z życia"}, inplace=True)

print(country_stats)

# Wizualizuje dane
country_stats.plot(kind='scatter', x="PKB per capita", y='Satysfakcja z życia')
plt.show()

# Wybiera model liniowy
model = sklearn.linear_model.LinearRegression()

# Uczy model
model.fit(X, Y)

# Oblicza prognozy dla Cypru
X_new = [[22587]]  # PKB per capita Cypru
print(model.predict(X_new))  # wyniki [[ 5.96242338]]
