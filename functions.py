import os
import shutil
from joblib.parallel import delayed
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
from joblib import Parallel, delayed
import scipy.stats as st
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
def generating_regressions(model,predictors, targets, X, y, n_splits, path, transpose_coefs=False, pca=False):  
        coefs = pd.DataFrame()


        if os.path.exists(path):
            print('Path does exist under directory', os.getcwd()+path)
            shutil.rmtree(path)
        os.makedirs(path)


        # prepare cross validation, data shuffled before split into batches
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # I could define my own scorer object; since error, want to minimize it -> the smaller the values the better. 
        # do not use jobs. this is what kills the kernel (for some reason..) !! not true. still happend, 
        # removed proportion scorer. not used very much anyway and threw errors with the clustering (maybe because Im returning a dataframe..)
        s = cross_validate(model, X, y, cv=cv, scoring = {'r2': 'r2', 'neg_mean_squared_error':'neg_mean_squared_error'}, return_train_score=True, return_estimator=True, n_jobs=-1)


        #store results for each gene at the end of the loop before going on to the next gene. 
        # for some reason, pls.coef_ gives back (152, 1109), thats why we need to transpose. 
        if transpose_coefs:
            for x in s['estimator']:
                coefs = pd.concat([pd.DataFrame(x.coef_.transpose(), index = targets, columns = predictors), 
                                coefs])
    
        elif pca:
            for x in s['estimator']:
                # irgendwas ist komisch.. glaube es sollte der index der targets addierbar sein..
                coefs = pd.concat([pd.DataFrame(x.coef_, index = targets), 
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
        smallest = msl.nsmallest(columns='median', n=n)
        largest = msl.nlargest(columns='median', n=n)
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


# data = msl
#     data['median'] = msl['median'].abs()
#     data = data.sort_values('median')
#     summary = {}
    
#     #percentages = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
#     percentages = np.arange(0, 1.02, 0.02)
#     indices = np.rint(percentages * len(msl)).astype(int)
#     start = 0
#     last = {x: {'total': 0, 'matchP':[], 'matchT':[]} for x in msl['predictors'].drop_duplicates().tolist()}

#     # Are there suitibel pandas functions that can do the same as I am here? 
#     # Cut down on for loops!!
#     # I only need to go through everything once. 
#     for p, i in list(zip(percentages, indices)):
#         section = data.iloc[start:i]
#         start=i

#         predictors = section['predictors'].drop_duplicates().tolist()

#         for x in predictors:
#             t = regnet_all[regnet_all['regulator_symbol']==x]['target_symbol'].tolist()
#             d = section[section['predictors']==x]['target'].tolist()
#             s = list(set(t) & set(d))
#             last[x]['total']+=len(t) 
#             last[x]['matchP']+= s
#             summary.setdefault(str(p), {})[x] = last[x]

#         for x in predictors:
#             t = regnet_all[regnet_all['target_symbol']==x]['regulator_symbol'].tolist()
#             d = section[section['predictors']==x]['target'].tolist()
#             s = list(set(t) & set(d))
#             last[x]['total']+=len(t) 
#             last[x]['matchT']+= s
#             summary.setdefault(str(p), {})[x] = last[x]
            
#         if str(p) not in summary:
#             summary.setdefault(str(p), {})['dummy']={'total': 0, 'matchP': [], 'matchT': []}


#     # for i in percentages:
#     #     n = int(len(msl)*i/2)
#     #     smallest = msl.nsmallest(columns='mean/sd', n=n)
#     #     largest = msl.nlargest(columns='mean/sd', n=n)
#     #     data = pd.concat([smallest, largest], axis=0)
#     #     # Comparing Data from RegNetWeb

#     #     predictors = data['predictors'].drop_duplicates().tolist()

#     #     for x in predictors:
#     #         t = regnet_all[regnet_all['regulator_symbol']==x]['target_symbol'].tolist()
#     #         d = data[data['predictors']==x]['target'].tolist()
#     #         s = list(set(t) & set(d))
#     #         summary.setdefault(str(i), {})[x]={'total': len(t), 'matchP': s}

#     #     for x in predictors:
#     #         t = regnet_all[regnet_all['target_symbol']==x]['regulator_symbol'].tolist()
#     #         d = data[data['predictors']==x]['target'].tolist()
#     #         s = list(set(t) & set(d))
#     #         summary[str(i)][x]['total']+= len(t)
#     #         summary[str(i)][x]['matchT']= s
            
#     #     if str(i) not in summary:
#     #         summary.setdefault(str(i), {})['dummy']={'total': 0, 'matchP': [], 'matchT': []}


#     return summary, percentages
def shuffle_meanSD(msl):
        permut = pd.DataFrame()
        permut['predictors'], permut['target'] = msl['predictors'], msl['target']
        permut['median'] = msl['median'].sample(frac=1).reset_index(drop=True)
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

# GO WITH MEDIAN HERE>i
def help_median(coefs):
    msl = pd.DataFrame()
    msl['mean'] = coefs.groupby(['predictors', 'target']).coefficients.mean()
    msl['sd'] = coefs.groupby(['predictors', 'target']).coefficients.std()
    msl['median'] = coefs.groupby(['predictors', 'target']).coefficients.median() 

    msl = msl.reset_index(col_fill =['predictors', 'target', 'mean', 'sd', 'median']) 
    print('This is the output of median', msl)
    return msl

def help_import_database(database):
    regnet_all = pd.read_pickle(database)

    # capitalize first letter to make strings comparable
    regnet_all['regulator_symbol'] = [x.capitalize() for x in regnet_all['regulator_symbol']]
    regnet_all['target_symbol'] = [x.capitalize() for x in regnet_all['target_symbol']] 
    return regnet_all

def generate_count(msl, regnet_all, path):
    summary, percentages = help_summary(msl, regnet_all)
    count = help_summary_to_count(summary, percentages)

    with open(path+'count.pkl', 'wb') as f:  
         pickle.dump(count, f)
    return count


def evaluate_permutations(coefs, database_file, path):
    # import datasets
    regnet_all = help_import_database(database_file)

    # update df for mena/sd ratio
    msl = help_median(coefs)
    
    #THIS IS STILL A WORK IN PROGRESS. 
    # permutations

    #for c in list(range(0, 1000)):
        #c = permutations(msl, regnet_all, c)
        #total = total+c

    total = Parallel(n_jobs=-1)(delayed(permutations) (msl, regnet_all, i) for i in range(1000))


# this is the long way. 
    #total = [permutations(msl, regnet_all, c) for c in list(range(0, 1000))]
    percentages = list(np.arange(0, 1.02, 0.02))
    # this can already be plotted. This is the mean of the matchings of all 1000 permutations. 
    permut = pd.DataFrame()   
    permut['mean'] = np.mean(total, axis=0)
    permut['std'] = np.std(total, axis=0)
    permut['percentage'] = percentages 

    permut.to_pickle(path+'permutations.pkl')

    count = generate_count(msl, regnet_all, path)
    return permut, count


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

def predicted_counts(coefs, database_file, filepath):
    # generate predicted counts
    regnet_all = help_import_database(database_file)
    coefs = help_median(coefs)
    summary, percentages = help_summary(coefs, regnet_all)
    count = help_summary_to_count(summary, percentages)

    with open(filepath+'count.pkl', 'wb') as f:  
        pickle.dump(count, f)
    
    return count


def get_pvalue(count, permut):
    # calculating p-value without first and last group, since std is 0 here. 
    z_score = (np.array(count['matches'][1:-1])-np.array(permut['mean'][1:-1]))/np.array(permut['std'][1:-1])


    #cumulative distribution function
    # --> not sure if I should take 1- ? for righttailed test? yes. or just use sf.
    p_values = st.norm.sf(z_score)


    # significance 5%
    top = st.norm.ppf(.975)
    bottom = st.norm.ppf(.025)
    
    return (p_values, z_score)
