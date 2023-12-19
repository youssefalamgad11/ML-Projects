# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 07:50:04 2023

@author: youss
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:/Users/youss/Downloads/cars.csv/cars.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', 100)
# Import data set

cars = pd.read_csv('C:/Users/youss/Downloads/cars.csv/cars.csv')
cars.head()
cars = cars.drop(columns=['feature_0','feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','feature_9'], axis=1)
cars.head()
cars.shape
cars.describe()
cars.columns
cars.dtypes
cars.isnull().sum()
cars = cars.dropna()
cars.isnull().sum()
# Calculate the age of the cars

cars['age'] = 2023 - cars['year_produced']
cars.head()
# All numeric(float and int) variables in dataset
cars_numeric = cars.select_dtypes(include=['float64', 'int64'])
cars_numeric.head()

# Correlation matrix
cor = cars_numeric.corr()
cor
# # Figure size
# plt.figure(figsize=(16,8))

# # Heatmap
# sns.heatmap(cor, cmap="YlGnBu", annot=True)
# plt.show()
# plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = cars.manufacturer_name.value_counts().plot(kind='bar')
plt.title('Companies Histogram')
plt1.set(xlabel = 'Car company', ylabel='Frequency of company')

plt.subplot(1,3,2)
plt1 = cars.body_type.value_counts().plot(kind='bar')
plt.title('Body Type')
plt1.set(xlabel = 'Body Type', ylabel='Frequency of Body Type')

plt.subplot(1,3,3)
plt1 = cars.engine_type.value_counts().plot(kind='bar')
plt.title('Engine Type Histogram')
plt1.set(xlabel = 'Engine Type', ylabel='Frequency of Engine type')

plt.show()

plt.figure(figsize=(30, 10))

df = pd.DataFrame(cars.groupby(['manufacturer_name'])['price_usd'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Company Name vs Average Price')
plt.show()

df = pd.DataFrame(cars.groupby(['engine_fuel'])['price_usd'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Fuel Type vs Average Price')
plt.show()

df = pd.DataFrame(cars.groupby(['body_type'])['price_usd'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Car Type vs Average Price')
plt.show()
cars['price_usd'] = cars['price_usd'].astype('float64')
temp = cars.copy()

table = temp.groupby(['manufacturer_name'])['price_usd'].mean()
temp = temp.merge(table.reset_index(), how='left', on='manufacturer_name')
bins = [0,10000,25000,50000]
cars_bins = ['Budget','Medium', 'Highend']
def onehot_encode(cars, columns, prefixes):
    cars = cars.copy()
    
    for column, prefix in zip(columns, prefixes): 
       dummies = pd.get_dummies(cars[column], prefix = prefix)
       cars = pd.concat([cars, dummies], axis=1)
       cars = cars.drop(column, axis = 1)
    return cars
onehot_columns = [
    'manufacturer_name',
    'color',
    'engine_fuel',
    'body_type',
    'state',
    'drivetrain',
    'location_region'
    ]
onehot_prefixes = [
    'm',
    'c',
    'e',
    'b',
    's',
    'd',
    'l'
]

df = onehot_encode(cars, onehot_columns, onehot_prefixes)
print("Remaining non-numeric columns:", (cars.dtypes == 'object').sum())

label_mapping = {
    'gasoline': 0,
    'diesel': 1,
    'electric':2
}
cars['engine_type'] = cars['engine_type'].replace(label_mapping)
print("Remaining non-numeric columns:", (cars.dtypes == 'object').sum())
cars = cars[['price_usd','year_produced']]
cars.hist()
plt.show()
x=cars["price_usd"]
y=cars["year_produced"]
plt.scatter(x,y)
plt.show
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

scaler = MinMaxScaler()
num_vars = ['price_usd','year_produced']
cars[num_vars] = scaler.fit_transform(cars[num_vars])
cars.head()

# Create Dependent and Independent variables
X = cars.drop(['price_usd'], axis=1)
y = cars['price_usd']

# Split Data into Train and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

# checking for shapes 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lm = LinearRegression()
lm.fit(X_train, y_train)
lm_y_pred = lm.predict(X_test)
print(lm.coef_)
print(lm.intercept_)
lm_mse = mean_squared_error(lm_y_pred, y_test)
lm_r2 = r2_score(lm_y_pred, y_test)
print(f"Mean Squared Error: {lm_mse}")
print(f"R2 Score: {lm_r2}")
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

rfr_y_pred = rfr.predict(X_test)
rfr_mse = mean_squared_error(rfr_y_pred, y_test)
rfr_r2 = r2_score(rfr_y_pred, y_test)
print(f"Mean Squared Error: {rfr_mse}")
print(f"R2 Score: {rfr_r2}")
# Cross Valiadtion
lm_cv_score = cross_val_score(lm, X_train, y_train, cv=5)
print(f"Linear Regression cross validation score: {lm_cv_score}")

# Random Forest Regression cross validation
rfr_cv_score = cross_val_score(rfr, X_train, y_train, cv=5)
print(f"Random Forest Regression cross validation: {rfr_cv_score}")

if lm_cv_score.mean() > rfr_cv_score.mean():
    mld_select = lm
    print("Linear Regression Model is selected.")
else:
    mld_select = rfr
    print("Random Forest Regression is selected.")
    
    selected_model = mld_select.fit(X_train, y_train)
print(selected_model)

final_pred = selected_model.predict(X_test)
final_pred
fnl_rfr_mse = mean_squared_error(final_pred, y_test)
fnl_rfr_r2 = r2_score(final_pred, y_test)
print(f"Mean Squared Error: {fnl_rfr_mse}")
print(f"R2 Score: {fnl_rfr_r2}")

