import os
import shutil
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
import multiprocessing as mp
#import memory_profiler


def proportion(y, y_pred):
    return np.mean(np.log(abs(y_pred-y)/abs(y)))

def help_pivot_to_df(coefs):

    coefs.index.rename('target', inplace=True)
    coefs.columns.rename('predictor', inplace=True)
    df = coefs.stack().reset_index()
    df.columns = ['target', 'predictors', 'coefficients']

    return df

#@profile
def generating_regressions(model,predictors, targets, X, y, n_splits, path, transpose_coefs=False):  
        coefs = pd.DataFrame()


        if os.path.exists(path):
            print('Path does exist under directory', os.getcwd()+path)
            shutil.rmtree(path)
        os.mkdir(path)


        # prepare cross validation, data shuffled before split into batches
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # I could define my own scorer object; since error, want to minimize it -> the smaller the values the better. 
        # do not use jobs. this is what kills the kernel (for some reason..) !! not true. still happend, 
        s = cross_validate(model, X, y, cv=cv, scoring = {'r2': 'r2', 'neg_mean_squared_error':'neg_mean_squared_error', 'proportion': make_scorer(proportion)}, return_train_score=True, return_estimator=True)


        #store results for each gene at the end of the loop before going on to the next gene. 
        # for some reason, pls.coef_ gives back (152, 1109), thats why we need to transpose. 
        if transpose_coefs:
            for x in s['estimator']:
                coefs = pd.concat([pd.DataFrame(x.coef_.transpose(), index = targets, columns = predictors), 
                                coefs])
            
        else:
            for x in s['estimator']:
                coefs = pd.concat([pd.DataFrame(x.coef_, index = targets, columns = predictors), 
                                coefs])


        # removing unnecessary fit and score times
        s.pop('fit_time', None)
        s.pop('score_time',None)

        scores = pd.DataFrame.from_dict(s)
        
        scores.to_pickle(path+'Scores.pkl')
        coefs.to_pickle(path+'Coefs.pkl')

        return coefs, scores

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

    # Are there suitibel pandas functions that can do the same as I am here? 
    # Cut down on for loops!!
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



def shuffle_meanSD(msl):
        permut = pd.DataFrame()
        permut['predictors'], permut['target'] = msl['predictors'], msl['target']
        permut['mean/sd'] = msl['mean/sd'].sample(frac=1).reset_index(drop=True)
        return permut

def help_summary_to_count(summary, percentages):
    count = []

    for p,i in zip(percentages, list(range(0, len(percentages)))):
        count+=[sum([len(l['matchP'])+len(l['matchT']) for l in summary[str(p)].values()])]
    
    return count


def permutations(msl, regnet_all, c = 1):
    permut = shuffle_meanSD(msl)

    summary, percentages = help_summary(permut, regnet_all)
    
    count = help_summary_to_count(summary, percentages)     
    #should I take the mean of the random counts? and then see where that takes me? 
    # --> plot average and sd --> have to keep track of number of matches per percentile. 
    
    if c%50==0:
        print("currently at ",c, 'permutation')
        print(permut)
    
    #maybe saving the summary takes time too?? not sure here. going to take it out. 
    # summary was returned for all others. 
    return count

def collect_result(result):
    global total
    total.append(result)

def help_meanSD(coefs):
    msl = pd.DataFrame()
    msl['mean'] = coefs.groupby(['predictors', 'target']).coefficients.mean()
    msl['sd'] = coefs.groupby(['predictors', 'target']).coefficients.std()
    msl['mean/sd'] = msl['mean']/msl['sd']

    msl = msl.reset_index(col_fill =['predictors', 'target', 'mean', 'sd', 'mean/sd']) 
    print('This is the output of meanSD', msl)
    return msl

def help_import_database(database):
    regnet_all = pd.read_pickle(database)

    # capitalize first letter to make strings comparable
    regnet_all['regulator_symbol'] = [x.capitalize() for x in regnet_all['regulator_symbol']]
    regnet_all['target_symbol'] = [x.capitalize() for x in regnet_all['target_symbol']] 
    return regnet_all


def evaluate_permutations(coefs, database, path):
    # import datasets
    regnet_all = help_import_database(database)

    # update df for mena/sd ratio
    msl = help_meanSD(coefs)

    total = []
    
    #THIS IS STILL A WORK IN PROGRESS. 
    # permutations

    for c in list(range(0, 1000)):
        c = permutations(msl, regnet_all, c)
        total = total+c


# this is the long way. 
    #total = [permutations(msl, regnet_all, c) for c in list(range(0, 1000))]
    percentages = list(np.arange(0, 1.02, 0.02))
    # this can already be plotted. This is the mean of the matchings of all 1000 permutations. 
    permut = pd.DataFrame()   
    permut['mean'] = np.mean(total, axis=0)
    permut['std'] = np.std(total, axis=0)
    permut['percentage'] = percentages 

    permut.to_pickle(path+'permutations.pkl')
    return permut


def merge_regnet(regnet_regulators_file, regnet_targets_file, targets, file_name):
    # import datasets
    regnet_regulators = pd.read_csv(regnet_regulators_file)
    regnet_targets = pd.read_csv(regnet_targets_file)
    print(regnet_regulators)
    
    # capitalize first letter to make strings comparable
    for d in [regnet_regulators, regnet_targets]:
        d['regulator_symbol'] = [x.capitalize() for x in d['regulator_symbol']]
        d['target_symbol'] = [x.capitalize() for x in d['target_symbol']]
    

    print(regnet_regulators)
    # filter out valid target-regulator pairs
    regnet_regulators = regnet_regulators[regnet_regulators['target_symbol'].isin(targets)]
    print(regnet_regulators)
    regnet_targets = regnet_targets[regnet_targets['regulator_symbol'].isin(targets)]
    regnet_all = pd.concat([regnet_regulators, regnet_targets], axis=0)

    regnet_all.to_pickle(file_name)
    print('this', regnet_all, 'is pickled.')

    return regnet_all

