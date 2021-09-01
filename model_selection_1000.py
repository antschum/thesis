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

os.environ['JOBLIB_TEMP_FOLDER'] = './tmp'

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

##### Models #####
random_state = 42
# Lasso
lasso = Lasso(random_state=random_state, max_iter=1000)
#lasso_grid = [{'alpha':[1, 1.1, 1.5, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 150]}]
# lasso_grid = [{'alpha':array([0.46860071, 0.43701859, 0.407565  , 0.38009648, 0.35447925,
#        0.33058854, 0.30830798, 0.28752905, 0.26815056, 0.25007811,
#        0.23322368, 0.21750518, 0.20284606, 0.18917492, 0.17642516,
#        0.16453469, 0.15344561, 0.14310389, 0.13345916, 0.12446446,
#        0.11607597, 0.10825284, 0.10095696, 0.0941528 , 0.08780722,
#        0.0818893 , 0.07637024, 0.07122314, 0.06642294, 0.06194626,
#        0.05777129, 0.05387769, 0.05024652, 0.04686007, 0.04370186,
#        0.0407565 , 0.03800965, 0.03544793, 0.03305885, 0.0308308 ,
#        0.02875291, 0.02681506, 0.02500781, 0.02332237, 0.02175052,
#        0.02028461, 0.01891749, 0.01764252, 0.01645347, 0.01534456,
#        0.01431039, 0.01334592, 0.01244645, 0.0116076 , 0.01082528,
#        0.0100957 , 0.00941528, 0.00878072, 0.00818893, 0.00763702,
#        0.00712231, 0.00664229, 0.00619463, 0.00577713, 0.00538777,
#        0.00502465, 0.00468601, 0.00437019, 0.00407565, 0.00380096,
#        0.00354479, 0.00330589, 0.00308308, 0.00287529, 0.00268151,
#        0.00250078, 0.00233224, 0.00217505, 0.00202846, 0.00189175,
#        0.00176425, 0.00164535, 0.00153446, 0.00143104, 0.00133459,
#        0.00124464, 0.00116076, 0.00108253, 0.00100957, 0.00094153,
#        0.00087807, 0.00081889, 0.0007637 , 0.00071223, 0.00066423,
#        0.00061946, 0.00057771, 0.00053878, 0.00050247, 0.0004686 ])}]

lasso_grid = [{'alpha':[0.46860071, 0.33058854, 0.23322368, 0.16453469, 0.11607597, 0.0818893, 0.05777129, 0.0407565, 0.02875291, 0.02028461, 0.01431039, 0.0100957, 0.00712231, 0.00502465, 0.00354479, 0.00250078, 0.00176425, 0.00124464, 0.00087807, 0.00061946]}]


## Ridge
ridge = Ridge(random_state=random_state)
# 38 seemed really good overall..
ridge_grid = [{'alpha':[1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]}]


## Random Forest
rf = RandomForestRegressor(n_jobs=-1, random_state=random_state)
rf_grid = [{'max_depth': [10, 30, 50, 70, 90],
            'max_features': ['auto', 'sqrt', 40, 'log2'],
            'n_estimators': [10]
            }]


linear = LinearRegression()
linear_grid = [{}]


### CHANGE ###
model = linear 
grid = linear_grid
path ='./model_selection/'+'vpred/'+str(model)+'/' 

pred, X, velocity_genes, y = dp.get_data(predictors='velocity_genes', louvain=True)




n_splits = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y['louvain'], test_size = 0.25, shuffle=True, random_state=random_state)


if os.path.exists(path):
    print('Path does exist under directory', os.getcwd()+path)
    shutil.rmtree(path)
os.makedirs(path)
os.makedirs(path+'plots/')

## HERE I CHANGED SOMETHING. adding plots directory..

### change hier to exclude target from predictors. 
# X_train.loc[:, (X_train.columns != 'louvain') & (X_train.columns != 'Mcm3')]
total = Parallel(n_jobs=-1)(delayed(gridSearch_validation) (X_train.loc[:, (X_train.columns != 'louvain') & (X_train.columns != t)], 
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
print("This is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
