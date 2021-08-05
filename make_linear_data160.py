import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
import os
import shutil
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


# velocity genes
targets = vdata.var.index[vdata.var['velocity_genes'] == True].tolist()

# transcription160
# load 160 transcriptionfactors
open_file = open('data/transcriptionfactors160.pkl', "rb")
transcription160 = pickle.load(open_file)
open_file.close()
# remove factors not available in vdata.var_names
for x in  ['Junb', 'mt-Nd1', 'Fgl2', 'mt-Co1', 'mt-Nd4', 'Rraga', 'mt-Nd2']:
    if x in transcription160:
        transcription160.remove(x)

predictors = transcription160
X = vdata[:, predictors].layers['Ms']

### Model
model = LinearRegression()


path = './'+'linear160test'
if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)

# normally go through all targets individually, generate this huge folder. try doing it directly. 
#for t in targets:
   #f.generating_regressions(model=model, predictors=predictors, t=t, X=X, y=vdata[:, t].layers['velocity'], n_splits=10, path = path+'/'+t) 

f.generating_regressions(model=model, predictors=predictors, targets=targets, X=X, y=vdata[:, targets].layers['velocity'], n_splits=10, path = path+'/'+'this') 

#path = f.generating_regressions(model, vdata[:10], transcription8, targets, 10, 'lasso_000013')

print('this is the path '+path)
#d.to_pickle('lasso0_000013Regression8.pkl')
#dd.to_pickle('lasso0_000013RegressionScores8.pkl')

