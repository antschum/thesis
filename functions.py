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




def help_summary(msl, regnet_all):
    
    data = pd.DataFrame()
    summary = {}
    #percentages = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    percentages = list(np.arange(0, 1.02, 0.02))


    for i in percentages:
        n = int(len(msl)*i/2)
        smallest = msl.nsmallest(columns='mean/sd', n=n)
        largest = msl.nlargest(columns='mean/sd', n=n)
        data = pd.concat([smallest, largest], axis=0)
        # Comparing Data from RegNetWeb

        predictors = data['predictors'].drop_duplicates().tolist()

        for x in predictors:
            t = regnet_all[regnet_all['regulator_symbol']==x]['target_symbol'].tolist()
            d = data[data['predictors']==x]['target'].tolist()
            s = list(set(t) & set(d))
            summary.setdefault(str(i), {})[x]={'total': len(t), 'matchP': s}

        for x in predictors:
            t = regnet_all[regnet_all['target_symbol']==x]['regulator_symbol'].tolist()
            d = data[data['predictors']==x]['target'].tolist()
            s = list(set(t) & set(d))
            summary[str(i)][x]['total']+= len(t)
            summary[str(i)][x]['matchT']= s
            
        if str(i) not in summary:
            summary.setdefault(str(i), {})['dummy']={'total': 0, 'matchP': [], 'matchT': []}

    return summary, percentages
