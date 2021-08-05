import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import functions as f


#This file will do everything for me
# INPUT: (ann)data with ms and velocity layer. , predictors, model, regnet values/databank (predictor and target must be called the same.)
# Output:trained model, permutation scores/graph

#Steps

def run_all(anndata, predictors, database_file, targets = None, model = LinearRegression(),  scale=True, filepath = './bin/'):

    # preprocessing
    #-> check if all predictors available in dataset. print the ones that are not. and throw these out. 
    # return scaled and 

    
    if targets is None:     # default targets are velocity genes
        targets = anndata.var.index[anndata.var['velocity_genes'] == True].tolist()
    
    predictors = check_variables(anndata, predictors)
    
    if scale:
        sc.pp.scale(anndata, layer = 'Ms')
        sc.pp.scale(anndata, layer = 'velocity')

    X = anndata[:, predictors].layers['Ms']
    y = anndata[:, targets].layers['velocity']

    coefs, scores = f.generating_regressions(model, predictors, targets, X, y, 10, path = filepath)

    permut = f.evaluate_permutations(coefs, database_file, path = filepath)


    return permut, scores, model


def check_variables(anndata, variables):
    data_var = set(anndata.var_names)
    var = set(variables)
    var_in_data =  data_var & var
    var_not_in_data = var - data_var

    print('The following where not found in dataset: ', var_not_in_data)

    return var_in_data
    