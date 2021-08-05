import pickle
import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score, mean_squared_error, make_scorer
import functions as f
from sklearn.decomposition import PCA

# Load dataset with velocity values
vdata = sc.read_h5ad("velocity_adata.h5ad")

# Scale Ms and velocity layer with zero mean and unit variance 
    # sc.pp.scale adds most recent mean and std as variables to var

sc.pp.scale(vdata, layer='Ms')
sc.pp.scale(vdata, layer='velocity')


# velocity genes
velocity_genes = vdata.var.index[vdata.var['velocity_genes'] == True].tolist()

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

# PCA
n_comp = 152
pca = PCA(n_components=n_comp).fit(X)
pca_components = pca.components_

# take first componentes
pca_transform = pca.transform(X)
lin = LinearRegression()
#lin.fit(pca_transform[:,:n_comp], Y)
coefs, scores = f.generating_regressions(lin, 
                         predictors = vdata[:,transcription160].var_names, 
                         targets = vdata[:, velocity_genes].var_names, 
                         X=pca_transform, 
                         y=vdata[:, velocity_genes].layers['velocity'], 
                         n_splits = 10, 
                         path='./pca/')


# scores has linear regressions, can get lin there. 
# coefficients of genes..

#so, I need to multiply the corresponding vlaues with the corresponding regression; --> get huge loading matrix. 
# can I just expand the pca_components so that it is long enough and then do mult?

# get dataframe from loadings. 
gene_impact = pd.DataFrame(np.matmul(coefs,pca_components[:n_comp])) #, columns=vdata[:, transcription160].var_names
gene_impact.columns = vdata[:, transcription160].var_names 

# reformat pivot table
gene_df = f.help_pivot_to_df(gene_impact)

gene_df.to_pickle('pca/gene_impact.pkl')


# just downloaded the dataset, have to import is and save it somewhere. 
#f.merge_regnet('./downloads/regnet160_regulator.csv', './downloads/regnet160_target.csv', './data/regnet160_all.pkl')
permut = f.evaluate_permutations(gene_df, 'data/regnet160_all.pkl', './pca/')


