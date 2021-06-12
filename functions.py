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


def proportion(y, y_pred):
    return abs(y_pred-y).sum()/abs(y).sum()

def generating_regressions(model, vdata, predictors, targets, n_splits):

    coefs = pd.DataFrame()
    scores = pd.DataFrame()

    #paralization
    for g in targets:

        temp = pd.DataFrame()
        # target
        t = [g]

        # train and test set
        X, y = vdata[:, predictors].layers['Ms'], vdata[:, t].layers['velocity']

        # prepare cross validation, data shuffled before split into batches
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # I could define my own scorer object; since error, want to minimize it -> the smaller the values the better. 
        s = cross_validate(model, X, y, cv=cv, scoring = {'r2': 'r2', 'neg_mean_squared_error':'neg_mean_squared_error', 'proportion': make_scorer(proportion)}, return_train_score=True, return_estimator=True, n_jobs=-1)

        for x in s['estimator']:
            temp['predictors'], temp['coefficients'], temp['target']= predictors, x.coef_[0], t*len(predictors)
            coefs = pd.concat([temp,coefs])
        
        # try to trigger garbage collection
        del temp

        #store results for each gene at the end of the loop before going on to the next gene. 

        # removing unnecessary fit and score times
        s.pop('fit_time', None)
        s.pop('score_time',None)

        df = pd.DataFrame.from_dict(s)
        df['target'] = t*n_splits
        scores = pd.concat([df, scores])

        # delete unnecessary df
        del df

    return scores, coefs


def reorder_like_clustermap(data, clustermap):
    
    col = data.columns.tolist()
    row = data.index.tolist()
    newcol = [col[x] for x in clustermap.dendrogram_col.reordered_ind]
    newrow = [row[x] for x in clustermap.dendrogram_row.reordered_ind]
    # this is not doing 
    reordered = data.loc[newrow,newcol]
    
    return reordered