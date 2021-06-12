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
import functions as f


# Load dataset with velocity values
vdata = sc.read_h5ad("velocity_adata.h5ad")

# Scale Ms and velocity layer with zero mean and unit variance 
    # sc.pp.scale adds most recent mean and std as variables to var

sc.pp.scale(vdata, layer='Ms')
sc.pp.scale(vdata, layer='velocity')


velocity_genes = tuple(vdata.var.index[vdata.var['velocity_genes'] == True])

transcription8 = ('Klf2', 'Mcm3', 'Mcm5', 'Hmgb2', 'Cdk4', 'Hif1a', 'Mcm6', 'Tox')


targets = velocity_genes
model = Lasso(alpha=0.000013)

dd, d = f.generating_regressions(model, vdata, transcription8, targets, 10)

d.to_pickle('lasso0_000013Regression8.pkl')
dd.to_pickle('lasso0_000013RegressionScores8.pkl')

