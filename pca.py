import pickle
import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
import pickle
import datapreprocessing as dp
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score, mean_squared_error, make_scorer
import functions as f
from sklearn.decomposition import PCA

pred, X, velocity_genes, Y = dp.get_data()

# PCA
n_comp = 10
path='./pca_10/'
pca = PCA(n_components=n_comp).fit(X)
pca_components = pca.components_

# take first componentes
pca_transform = pca.transform(X)
lin = LinearRegression()
#lin.fit(pca_transform[:,:n_comp], Y)
coefs, scores = f.generating_regressions(lin, 
                         predictors = pred, 
                         targets = velocity_genes, 
                         X=pca_transform, 
                         y=Y, 
                         n_splits = 10, 
                         path=path,
                         pca=True)


# scores has linear regressions, can get lin there. 
# coefficients of genes..

#so, I need to multiply the corresponding vlaues with the corresponding regression; --> get huge loading matrix. 
# can I just expand the pca_components so that it is long enough and then do mult?

# get dataframe from loadings. 


###Also, weil ich mit datapreprocessing alles lade, gibts auch kein anndata.
gene_impact = pd.DataFrame(np.matmul(coefs,pca_components[:n_comp])) #, columns=vdata[:, transcription160].var_names
### das habe ich geändert.. evtl klappt das nicht. 
gene_impact.columns = pred

# reformat pivot table
gene_df = f.help_pivot_to_df(gene_impact)

gene_df.to_pickle(path+'gene_impact.pkl')

permut = f.evaluate_permutations(gene_df, 'data/regnet160_all.pkl', path)


