import pandas as pd
from sklearn import model_selection
import datapreprocessing as dp
import functions as f
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score, mean_squared_error, make_scorer
import sys
import multiprocessing as mp
from joblib import Parallel, delayed
import pickle
import os
import shutil
import time

start = time.time()
def gridSearch_validation(X, y, estimator, param_grid, n_splits, path=None):

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    

    
    # Non_nested parameter search and scoring; holding back data. 
    clf = GridSearchCV(estimator=estimator, 
                       param_grid=param_grid, 
                       scoring = {'r2': 'r2', 'neg_mean_squared_error':'neg_mean_squared_error'}, 
                       cv=inner_cv, verbose=1.1, 
                       return_train_score=True,
                       refit = 'r2')

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
lasso_grid = [{'alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]}]



## Ridge
ridge = Ridge(random_state=random_state)
path = './model_selection/ridge.pkl'
# 38 seemed really good overall..
ridge_grid = [{'alpha':[0.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 150]}]

# clf = gridSearch_validation(X_train.loc[:, X_train.columns != 'louvain'], 
#                         y_train.loc[:, y_train.columns != 'louvain'],
#                         ridge, ridge_grid, n_splits, path)

## Random Forest
rf = RandomForestRegressor(n_jobs=-1, random_state=random_state, max_depth=30)
rf_grid = [{'max_features':[3, 5, 12, 40, 100, 151]}]




#CHANGE
model = rf
grid = rf_grid

### comment this in!!
path ='./model_selection/'+str(model)+'/' 
if os.path.exists(path):
    print('Path does exist under directory', os.getcwd()+path)
    shutil.rmtree(path)
os.mkdir(path)


# Achtung: Ich lasse nur die ersten X trainieren. 
total = Parallel(n_jobs=-1)(delayed(gridSearch_validation) (X_train.loc[:, X_train.columns != 'louvain'], 
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
