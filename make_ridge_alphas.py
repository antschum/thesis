import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
import random 
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score, mean_squared_error, make_scorer
import functions as f


# Load dataset with velocity values
vdata = sc.read_h5ad("velocity_adata.h5ad")
print('data loaded')

# Scale Ms and velocity layer with zero mean and unit variance 
    # sc.pp.scale adds most recent mean and std as variables to var

sc.pp.scale(vdata, layer='Ms')
sc.pp.scale(vdata, layer='velocity')


velocity_genes = tuple(vdata.var.index[vdata.var['velocity_genes'] == True])

transcription8 = ('Klf2', 'Mcm3', 'Mcm5', 'Hmgb2', 'Cdk4', 'Hif1a', 'Mcm6', 'Tox')


targets = velocity_genes

alpha = []
alphas = 10**np.linspace(5,-4,50)*0.5

for t in velocity_genes:
    
    target= t
    ridge = Ridge()

    X, y = vdata[:, transcription8].layers['Ms'], vdata[:, target].layers['velocity']

    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

    ridgecv = RidgeCV(alphas=alphas, cv=5, normalize=True)
    ridgecv.fit(X_train, y_train)
    ridge.set_params(alpha=ridgecv.alpha_)
    #print("Alpha=", ridgecv.alpha_)
    ridge.fit(X_train, y_train)
    #print("mse = ",mean_squared_error(y_test, ridge.predict(X_test)))
    #print("best model coefficients:")
    #pd.Series(ridge.coef_, index=X.columns)
    alpha+= [ridgecv.alpha_]

    del ridge, ridgecv

pd.DataFrame(alpha).to_pickle('best_alphas_lasso.pkl')
print('data saved')

print(np.mean(pd.DataFrame(alpha)))