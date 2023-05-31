import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Defining F1 score
def f1_score(y, y_pred):
    tp = ((y == 1) & (y_pred == 1)).sum()
    fn = ((y == 1) & (y_pred == 0)).sum()
    fp = ((y == 0) & (y_pred == 1)).sum()
    tn = ((y == 0) & (y_pred == 0)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('------------------------------------------')
    print('                      Actual Value')
    print('------------------------------------------')
    print('                   Positive       Negative')
    print(f'Positive          {tp:^8}         {fp:^8}')
    print(f'Negative          {fn:^8}         {tn:^8}')
    print('------------------------------------------')
    return f1

# Importing the dataset and cleaning the data
df = pd.read_csv('bank_shuffle.csv')
df['y'] = df['y'].apply(lambda x:x == 'yes')
df = pd.get_dummies(df, drop_first = True)

# Splitting the data into train and test 80 and 20 percent
df_train = df.iloc[:int(len(df) * 0.8),]
df_test = pd.concat([df,df_train]).drop_duplicates(keep=False)

# Seperating independent and dependent variables
X_train = df_train.drop('y', axis = 1)
y_train = df_train['y']
X_test = df_test.drop('y', axis = 1)
y_test = df_test['y']

# Using 5 fold cv to check the best parameters for random forest classifier
params = { 'max_features' : ['sqrt','log2'],
         'criterion':['gini','entropy'],
         'n_estimators': [50,100,150,200,250,300,350,400,450,500]}
R = GridSearchCV(RandomForestClassifier(),params,cv = 5,scoring='f1')
R.fit(X_train,y_train)
print(f'The best parameters are {R.best_estimator_}')

# Training the entire dataset with the best parameters
best = RandomForestClassifier(criterion='gini', max_features='sqrt',n_estimators=500)
best.fit(X_train,y_train)
f1_score(y_train, best.predict(X_train))

# Error rate, confusion matrix and f1 score on the test data
error_rate = sum(y_test != best.predict(X_test)) / len(y_test)
f1 = f1_score(y_test, best.predict(X_test))

print(f' The error rate is {error_rate}')
print(f' The f1 score is {f1}')