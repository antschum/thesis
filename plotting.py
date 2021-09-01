import seaborn as sns
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Load Data 
def compile_gridsearches_rf(filepath):

    
    best ={}
    best_idx = {}
    
    
    for filename in os.listdir(filepath):
        if os.path.isfile(os.path.join(filepath, filename)):
            with open(filepath+filename, 'rb') as file:
                gcv = pickle.load(file)
                idx = gcv.best_index_
                
                l = []
                for s in gcv.cv_results_['param_max_features'].data:
                    if s=='sqrt': l+=[12]
                    elif s=='auto': l+=[151]
                    elif s=='log2': l+=[7]
                    else: l+=[s]
                #gcv.cv_results_['best_param_features'] is a mask, not easily changed.. 
                
                target, apx = filename.split('.')
                best_idx[target] = {'features': l[idx],
                                    'depth': gcv.cv_results_['param_max_depth'][idx],
                                'mean_test_r2': gcv.cv_results_['mean_test_r2'][idx], 
                              'mean_train_r2': gcv.cv_results_['mean_train_r2'][idx],
                              'mean_test_neg_mean_squared_error': gcv.cv_results_['mean_test_neg_mean_squared_error'][idx], 
                              'mean_train_neg_mean_squared_error': gcv.cv_results_['mean_train_neg_mean_squared_error'][idx],
                              'std_test_neg_mean_squared_error': gcv.cv_results_['std_test_neg_mean_squared_error'][idx],
                              'std_train_neg_mean_squared_error': gcv.cv_results_['std_train_neg_mean_squared_error'][idx],
                              'std_test_r2': gcv.cv_results_['std_test_r2'][idx],
                              'std_train_r2':gcv.cv_results_['std_train_r2'][idx],
                              }
                
                #'best_param': gcv.cv_results_['param_alpha'],
                best[target]={'features': l,
                              'depth': gcv.cv_results_['param_max_depth'],
                            'mean_test_r2': gcv.cv_results_['mean_test_r2'], 
                              'mean_train_r2': gcv.cv_results_['mean_train_r2'],
                              'mean_test_neg_mean_squared_error': gcv.cv_results_['mean_test_neg_mean_squared_error'], 
                              'mean_train_neg_mean_squared_error': gcv.cv_results_['mean_train_neg_mean_squared_error'],
                              'std_test_neg_mean_squared_error': gcv.cv_results_['std_test_neg_mean_squared_error'],
                              'std_train_neg_mean_squared_error': gcv.cv_results_['std_train_neg_mean_squared_error'],
                              'std_test_r2': gcv.cv_results_['std_test_r2'],
                              'std_train_r2':gcv.cv_results_['std_train_r2'],
                              }
    return best, best_idx

def compile_gridsearches_lin(filepath, lin=False):
    
    best ={}
    best_idx = {}
    
    
    for filename in os.listdir(filepath):
        if os.path.isfile(os.path.join(filepath, filename)):
            with open(filepath+filename, 'rb') as file:
                gcv = pickle.load(file)
                idx = gcv.best_index_
                
                target, apx = filename.split('.')

                if lin:
                    best_idx[target] = {
                                    'mean_test_r2': gcv.cv_results_['mean_test_r2'][idx], 
                                'mean_train_r2': gcv.cv_results_['mean_train_r2'][idx],
                                'mean_test_neg_mean_squared_error': gcv.cv_results_['mean_test_neg_mean_squared_error'][idx], 
                                'mean_train_neg_mean_squared_error': gcv.cv_results_['mean_train_neg_mean_squared_error'][idx],
                                'std_test_neg_mean_squared_error': gcv.cv_results_['std_test_neg_mean_squared_error'][idx],
                                'std_train_neg_mean_squared_error': gcv.cv_results_['std_train_neg_mean_squared_error'][idx],
                                'std_test_r2': gcv.cv_results_['std_test_r2'][idx],
                                'std_train_r2':gcv.cv_results_['std_train_r2'][idx],
                                }
                    
                    #'best_param': gcv.cv_results_['param_alpha'],
                    best[target]={
                                'mean_test_r2': gcv.cv_results_['mean_test_r2'], 
                                'mean_train_r2': gcv.cv_results_['mean_train_r2'],
                                'mean_test_neg_mean_squared_error': gcv.cv_results_['mean_test_neg_mean_squared_error'], 
                                'mean_train_neg_mean_squared_error': gcv.cv_results_['mean_train_neg_mean_squared_error'],
                                'std_test_neg_mean_squared_error': gcv.cv_results_['std_test_neg_mean_squared_error'],
                                'std_train_neg_mean_squared_error': gcv.cv_results_['std_train_neg_mean_squared_error'],
                                'std_test_r2': gcv.cv_results_['std_test_r2'],
                                'std_train_r2':gcv.cv_results_['std_train_r2'],
                                } 
                
                else: 
                    best_idx[target] = {'best_param': gcv.cv_results_['param_alpha'][idx],
                                'mean_test_r2': gcv.cv_results_['mean_test_r2'][idx], 
                              'mean_train_r2': gcv.cv_results_['mean_train_r2'][idx],
                              'mean_test_neg_mean_squared_error': gcv.cv_results_['mean_test_neg_mean_squared_error'][idx], 
                              'mean_train_neg_mean_squared_error': gcv.cv_results_['mean_train_neg_mean_squared_error'][idx],
                              'std_test_neg_mean_squared_error': gcv.cv_results_['std_test_neg_mean_squared_error'][idx],
                              'std_train_neg_mean_squared_error': gcv.cv_results_['std_train_neg_mean_squared_error'][idx],
                              'std_test_r2': gcv.cv_results_['std_test_r2'][idx],
                              'std_train_r2':gcv.cv_results_['std_train_r2'][idx],
                              }
                
                    #'best_param': gcv.cv_results_['param_alpha'],
                    best[target]={'params': gcv.cv_results_['param_alpha'],
                                'mean_test_r2': gcv.cv_results_['mean_test_r2'], 
                                'mean_train_r2': gcv.cv_results_['mean_train_r2'],
                                'mean_test_neg_mean_squared_error': gcv.cv_results_['mean_test_neg_mean_squared_error'], 
                                'mean_train_neg_mean_squared_error': gcv.cv_results_['mean_train_neg_mean_squared_error'],
                                'std_test_neg_mean_squared_error': gcv.cv_results_['std_test_neg_mean_squared_error'],
                                'std_train_neg_mean_squared_error': gcv.cv_results_['std_train_neg_mean_squared_error'],
                                'std_test_r2': gcv.cv_results_['std_test_r2'],
                                'std_train_r2':gcv.cv_results_['std_train_r2'],
                                }

    return best, best_idx

def reform(dicts):
    return {(outerKey, innerKey): values for outerKey, innerDict in dicts.items() for innerKey, values in innerDict.items()}


## Plots
def parameter_plot(df, model='', parameter=''):
    fig = plt.figure()
    ax = sns.histplot(df.loc['best_param'])
    print('Average: {} ({})'.format(np.mean(df.loc['best_param']), 
                                    round(np.std(df.loc['best_param']),4)))
    ax.set_title(model+': Parameter Distribution across all targets')
    ax.set_xlabel(parameter)
    return fig


def cluster_overall(model, data, X_train, y_train, path, clustering='complete', threshold=-0.25, lin=False):
    
    index = best_overall_index(data)
    slip = data[[idx for idx in data.columns if idx[1]=='mean_test_neg_mean_squared_error']].iloc[index,:].droplevel(1)
    low = slip[slip>threshold]
    targets = low.index



    coefficients = {}  

    for idx in range(len(targets)):
        if lin: m = model
        else: m =  model.set_params(alpha=data[('Mcm3','params')][index], random_state=42)
        fitted = m.fit(X_train.loc[:, (X_train.columns != 'louvain') & (X_train.columns != targets[idx])], 
                       y_train.loc[:, targets[idx]])

        # here I should save the coefficients. --> using all training data. can validate later on. 
        # dont think I need the estimatros. well, maybe? but can also just rerun this.
        coef = fitted.coef_.tolist()
        if targets[idx] in X_train.columns: coef.insert(X_train.columns.get_loc(targets[idx]), 0)

        coefficients[targets[idx]]=coef

    coefs = pd.DataFrame(coefficients, index=X_train.loc[:,X_train.columns != 'louvain'].columns)
    cmap = sns.clustermap(coefs, method=clustering, cmap='coolwarm')   
    cmap.savefig(path+'clustermap_all_'+clustering+'.png')
    return cmap


def cluster_individual(model, data, X_train, y_train, path, clustering='complete', threshold=-0.25, lin=False):
    low = data.transpose().loc[data.transpose()['mean_test_neg_mean_squared_error']>=threshold]
    targets = low.index
    if not lin: hp = low['best_param']

    coefficients = {}  

    for idx in range(len(targets)):
        if lin: m=model
        else: m = model.set_params(alpha=hp[idx], random_state=42)
        fitted = m.fit(X_train.loc[:, (X_train.columns != 'louvain') & (X_train.columns != targets[idx])], 
                       y_train.loc[:, targets[idx]])

        # here I should save the coefficients. --> using all training data. can validate later on. 
        # dont think I need the estimatros. well, maybe? but can also just rerun this.
        coef = fitted.coef_.tolist()
        if targets[idx] in X_train.columns: coef.insert(X_train.columns.get_loc(targets[idx]), 0)

        coefficients[targets[idx]]=coef

    coefs = pd.DataFrame(coefficients, index=X_train.loc[:,X_train.columns != 'louvain'].columns)
    cmap = sns.clustermap(coefs, method=clustering, cmap='coolwarm')   
    cmap.savefig(path+'clustermap_ind_'+clustering+'.png')
    return cmap


# return index of best alpha overall.
def best_overall_index(df):
    return df[[idx for idx in df.columns if idx[1]=='mean_test_neg_mean_squared_error']].mean(axis=1).argmax()


