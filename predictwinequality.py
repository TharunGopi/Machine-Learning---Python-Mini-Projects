import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)
  
def find_optimal_trees(X, y, visualize=None):
    count = 100
    tree = []
    i = 95
    mse_best = {}
    while i <= 495:
        tree.append(i + 5)
        i = i + 5
    oob_error = []


    n = tree
#for n in range(500):
    for i in n:

        model = RandomForestRegressor(n_estimators= i, max_depth=1000, max_features='sqrt', n_jobs=-1, oob_score=True)
        model.fit(X, y)
    #print(model.oob_score_)
        oob_score = model.oob_score_

        MSE = (1 - oob_score) * y.var()

        oob_error.append(MSE)
    #tree.append(i)
        mse_best.update({MSE: i})
        i = i + 5
    mse_min = mse_best[min(list(mse_best.keys()))]
    #print(oob_error)
    #print(mse_min)
    plt.barh(n, oob_error)
    plt.show()

    if visualize:
        plt.figure(figsize=(15, 7))
        #plt.errorbar(mse_best.values(), mse_best.keys(), fmt='o')
        plt.barh(n, oob_error)
        plt.xlabel("MSE")
        plt.ylabel("Number of Trees")
        plt.savefig(visualize)
        plt.show()

    return mse_min

values = range(500)
#n = [100, 105, 110, 115, 120]
i = 100

df = pd.read_csv('winequality-red.csv')
X = df.drop(columns=['quality'])
y = df['quality']
mse_min = find_optimal_trees(X, y,  visualize="Homework8.png")
model = RandomForestRegressor(n_estimators=mse_min, max_depth=1000, max_features='sqrt', n_jobs=-1, oob_score=True)
model.fit(X, y)
oob_score = model.oob_score_

# Convert oob_score_to MSE
MSE = (1 - oob_score) * y.var()
print("Best predicted Number of Trees(B) based on MSE is", mse_min)
print("OOB score for the model fit with identified number of trees" , model.oob_score_)
print("MSE score for the model fit with identified number of trees", MSE)
print("Feature Importance is", model.feature_importances_)

