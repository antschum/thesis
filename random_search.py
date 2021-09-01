import pandas as pd
from sklearn import model_selection
import datapreprocessing as dp
import functions as f
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score, mean_squared_error, make_scorer
import sys
import multiprocessing as mp
from joblib import Parallel, delayed
import pickle
import os
import shutil
import time


# Have not use this yet. No Use. 

start = time.time()
def randomSearch_validation(X, y, estimator, param_grid, n_splits, path=None):

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    

    
    # Non_nested parameter search and scoring; holding back data. 
    clf = RandomizedSearchCV(estimator=estimator, 
                       param_grid=param_grid, 
                       scoring = {'r2': 'r2', 'neg_mean_squared_error':'neg_mean_squared_error'}, 
                       cv=inner_cv, verbose=1.1, 
                       return_train_score=True,
                       refit = 'r2',
                       n_iter=100)

    clf.fit(X, y)
    
    if path is not None:
        a_file = open(path, "wb")
        pickle.dump(clf, a_file)
        a_file.close()
    
    return clf

pred, X, velocity_genes, y = dp.get_data(louvain=True)

n_splits = 10
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y['louvain'], test_size = 0.25, shuffle=True, random_state=random_state)


# Lasso
lasso = Lasso(random_state=random_state, max_iter=1000)
path = './model_selection/lasso/'
#lasso_grid = [{'alpha':[1, 1.1, 1.5, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 150]}]
#lasso_grid = [{'alpha':[1.91203544e-01, 1.78317065e-01, 1.66299092e-01, 1.55091090e-01,

    #    1.44638469e-01, 1.34890321e-01, 1.25799164e-01, 1.17320722e-01,
    #    1.09413698e-01, 1.02039581e-01, 9.51624546e-02, 8.87488236e-02,
    #    8.27674500e-02, 7.71892008e-02, 7.19869070e-02, 6.71352304e-02,
    #    6.26105406e-02, 5.83907997e-02, 5.44554552e-02, 5.07853398e-02,
    #    4.73625779e-02, 4.41704987e-02, 4.11935549e-02, 3.84172471e-02,
    #    3.58280533e-02, 3.34133624e-02, 3.11614136e-02, 2.90612386e-02,
    #    2.71026084e-02, 2.52759833e-02, 2.35724666e-02, 2.19837613e-02,
    #    2.05021293e-02, 1.91203544e-02, 1.78317065e-02, 1.66299092e-02,
    #    1.55091090e-02, 1.44638469e-02, 1.34890321e-02, 1.25799164e-02,
    #    1.17320722e-02, 1.09413698e-02, 1.02039581e-02, 9.51624546e-03,
    #    8.87488236e-03, 8.27674500e-03, 7.71892008e-03, 7.19869070e-03,
    #    6.71352304e-03, 6.26105406e-03, 5.83907997e-03, 5.44554552e-03,
    #    5.07853398e-03, 4.73625779e-03, 4.41704987e-03, 4.11935549e-03,
    #    3.84172471e-03, 3.58280533e-03, 3.34133624e-03, 3.11614136e-03,
    #    2.90612386e-03, 2.71026084e-03, 2.52759833e-03, 2.35724666e-03,
    #    2.19837613e-03, 2.05021293e-03, 1.91203544e-03, 1.78317065e-03,
    #    1.66299092e-03, 1.55091090e-03, 1.44638469e-03, 1.34890321e-03,
    #    1.25799164e-03, 1.17320722e-03, 1.09413698e-03, 1.02039581e-03,
    #    9.51624546e-04, 8.87488236e-04, 8.27674500e-04, 7.71892008e-04,
    #    7.19869070e-04, 6.71352304e-04, 6.26105406e-04, 5.83907997e-04,
    #    5.44554552e-04, 5.07853398e-04, 4.73625779e-04, 4.41704987e-04,
    #    4.11935549e-04, 3.84172471e-04, 3.58280533e-04, 3.34133624e-04,
    #    3.11614136e-04, 2.90612386e-04, 2.71026084e-04, 2.52759833e-04,
    #    2.35724666e-04, 2.19837613e-04, 2.05021293e-04, 1.91203544e-04]}]

lasso_grid = [{'alpha':[0.191203544, 0.0951624546, 0.0473625779, 0.0235724666, 0.0117320722, 0.00583907997, 0.00290612386, 0.00144638469, 0.00071986907, 0.000358280533]}]

## Ridge
ridge = Ridge(random_state=random_state)
path = './model_selection/ridge.pkl'
# 38 seemed really good overall..
ridge_grid = [{'alpha':[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 150]}]

# clf = gridSearch_validation(X_train.loc[:, X_train.columns != 'louvain'], 
#                         y_train.loc[:, y_train.columns != 'louvain'],
#                         ridge, ridge_grid, n_splits, path)

## Random Forest
rf = RandomForestRegressor(n_jobs=-1, random_state=random_state)
rf_grid = [{'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_features': ['auto', 'sqrt', 40, 'log2'],
            'n_estimators': [10, 100]
            }]




#CHANGE
model = rf
grid = rf_grid

### comment this in!!
path ='./model_selection/R'+str(model)+'/' 
if os.path.exists(path):
    print('Path does exist under directory', os.getcwd()+path)
    shutil.rmtree(path)
os.mkdir(path)


# X_train.loc[:, (X_train.columns != 'louvain') & (X_train.columns != 'Mcm3')]
total = Parallel(n_jobs=-1)(delayed(randomSearch_validation) (X_train.loc[:, X_train.columns != 'louvain'], 
                                                            y_train.loc[:, t],
                                                            model, grid, n_splits, path+t+'.pkl') 
                                        for t in [x for x in y.columns.tolist() if x != 'louvain'])

# for t in [x for x in y.columns.tolist() if x != 'louvain'][:3]:
#     clf = gridSearch_validation(X_train.loc[:, X_train.columns != 'louvain'], 
#                             y_train.loc[:, t],
#                             model, grid, n_splits, path+t+'.pkl')
#     print('done with '+t)



end = time.time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("THis is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
