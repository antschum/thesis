import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
import random 
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score, mean_squared_error, make_scorer
import sys
#import memory_profiler


def proportion(y, y_pred):
    return np.mean(np.log(abs(y_pred-y)/abs(y)))

#@profile
def generating_regressions(model,predictors, t, X, y, n_splits, path):  
        coefs = pd.DataFrame()

        # train and test set
        

        # prepare cross validation, data shuffled before split into batches
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # I could define my own scorer object; since error, want to minimize it -> the smaller the values the better. 
        # do not use jobs. this is what kills the kernel (for some reason..) !! not true. still happend, 
        s = cross_validate(model, X, y, cv=cv, scoring = {'r2': 'r2', 'neg_mean_squared_error':'neg_mean_squared_error', 'proportion': make_scorer(proportion)}, return_train_score=True, return_estimator=True)


        for x in s['estimator']:
            coefs = pd.concat([pd.DataFrame(data={'predictors': predictors, 'coefficients': x.coef_[0], 'target': t}), coefs])
        #store results for each gene at the end of the loop before going on to the next gene. 

        # removing unnecessary fit and score times
        s.pop('fit_time', None)
        s.pop('score_time',None)

        scores = pd.DataFrame.from_dict(s)
        scores['target'] = t
        
        scores.to_pickle(path+'Scores.pkl')
        coefs.to_pickle(path+'Coefs.pkl')


def reorder_like_clustermap(data, clustermap):
    
    col = data.columns.tolist()
    row = data.index.tolist()
    newcol = [col[x] for x in clustermap.dendrogram_col.reordered_ind]
    newrow = [row[x] for x in clustermap.dendrogram_row.reordered_ind]
    # this is not doing 
    reordered = data.loc[newrow,newcol]
    
    return reordered