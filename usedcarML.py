#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler



def find_optimal_k(X_std, y, visualize=None):
    k_best = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for k in range(2, 100):
        model = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(model, X_std, y, scoring="neg_mean_squared_error", cv=kf)
        k_best.update({np.sqrt(abs(np.mean(scores))): k})

    k_min = k_best[min(list(k_best.keys()))]

    if visualize:
        plt.figure(figsize=(15, 7))
        plt.errorbar(k_best.values(), k_best.keys(), fmt='o')
        plt.plot(k_best.values(), k_best.keys())
        plt.xlabel("K Values")
        plt.ylabel("Root Mean Squared Error")
        plt.savefig(visualize)
        plt.show()

    return k_min


def knn_sklearn(X, y, X_pred):
    mm = MinMaxScaler()
    X_std = mm.fit_transform(X)
    k_min = find_optimal_k(X_std, y, visualize="Homework6.png")
    model = KNeighborsRegressor(n_neighbors=k_min)
    model.fit(X_std, y)
    y_pred = [round(i, 8) for i in list(model.predict(mm.transform(X_pred)))]
    return y_pred, k_min

# function for EUCLIDIAN_DIST
def euclidean_distance(p, q):
    d = 0
    for i in range(len(p)):
        d = d + (p[i] - q[i]) ** 2
    return np.sqrt(d)

# function for KNN
def knn(k, X, y, X_pred):
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_std = scaler.transform(X)
    X_pred_std = scaler.transform(X_pred)
    y_pred = []

    for i in range(len(X_pred)):  
        nn = []  
        nn_dist = []  
        sum_neighbors_y = 0  
        all_distance = [10] * len(X)  
        for j in range(len(X)): 
            all_distance[j] = euclidean_distance(X_pred_std[i], X_std[j])
        for kval in range(k): 
            min_distance = min(all_distance)
            min_index = all_distance.index(min_distance)
            nn.append(min_index)
            nn_dist.append(min_distance)
            all_distance[
                min_index] = 10.0 
            sum_neighbors_y = sum_neighbors_y + y[min_index]
            if kval == 18:
                y_pred.append(round(sum_neighbors_y / k,8))  

    return y_pred


df = pd.read_csv('susedcars.csv', usecols=['price', 'mileage', 'year'])
df['age'] = 2015 - df.pop('year')
X = df[['mileage', 'age']].to_numpy()
y = df['price'].to_numpy()

X_pred = [[100000, 10], [50000, 3]]
y_pred, k_min = knn_sklearn(X, y, X_pred)
y_pred2 = knn(k_min, X, y, X_pred)
print("Best Value of K = ", k_min)
print("Task1 Y Predictions = ", y_pred)
print("Task2 Y Predictions = ", y_pred2)

