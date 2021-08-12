import datapreprocessing as dp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate, KFold
import functions as f
import pandas as pd
import os
import shutil
import time
import joblib
import pickle
import scanpy as sc

start = time.time()
filepath = './rf_oob/'
database_file = 'data/regnet160_all.pkl'

model = RandomForestRegressor(n_jobs=-1)

vdata = sc.read_h5ad('velocity_adata.h5ad')
sc.pp.scale(vdata, layer='Ms')
sc.pp.scale(vdata, layer='velocity')
velocity_genes = vdata.var.index[vdata.var['velocity_genes'] == True].tolist()

open_file = open('data/transcriptionfactors160.pkl', "rb")
pred = pickle.load(open_file)
open_file.close()    

# remove factors not available in vdata.var_names -> should be filtered out by default. 
for x in  ['Junb', 'mt-Nd1', 'Fgl2', 'mt-Co1', 'mt-Nd4', 'Rraga', 'mt-Nd2']:
    if x in pred:
        pred.remove(x)


X = vdata[:, pred].layers['Ms']

def generate_reg(X, y, model, target, n_splits=10):
    rfs = []
    for x in list(range(n_splits)):
        print('model will be fit.')
        m = model.fit(X, y)
        rfs = rfs+[m]
        print('is done.')

    result = pd.DataFrame()
    result['estimator'], result['target']=rfs, [target]*n_splits
    print('ran rfs for '+target)
    return result

def help_feature_importances(estimator, predictors, target):
    return pd.DataFrame(estimator.feature_importances_.reshape(1,-1), index=target, columns=predictors)

print('prep done.')
results = pd.concat([generate_reg(X, vdata[:, v].layers['velocity'].ravel(), model, v) for v in velocity_genes[:3]])
results.to_pickle(filepath+'Scores.pkl')
print('CV generated.')
importance = pd.concat([help_feature_importances(x, pred, t) for x, t in results[['estimator', 'target']]])
importance.to_pickle(filepath+'Coefs.pkl')
print('importances saved.')

#joblib.dump(model, filepath+"/random_forest_cv_mrpl15.joblib")
end = time.time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("THis is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

