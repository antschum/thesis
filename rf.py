from joblib.parallel import Parallel, delayed
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
import sys

# Generate Regressions for rf.
# Running this for all 160tf took a little less than 2 hr

start = time.time()
filepath = './rf_lID/'
database_file = 'data/regnet160_all.pkl'

if os.path.exists(filepath):
    print('Path does exist under directory', os.getcwd()+filepath)
    shutil.rmtree(filepath)
os.mkdir(filepath)
os.mkdir(filepath+'estimators')
os.mkdir(filepath+'coefs')


model = RandomForestRegressor(n_jobs=-1, max_depth=10, max_features='sqrt')

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

# generate with this!!!
X = vdata[:, pred].to_df('Ms')
X['louvain'] = vdata[:, pred].obs['louvain']
X['sampleID'] = vdata[:, pred].obs['sampleID']

def help_feature_importances(estimator, predictors, target):
    return pd.DataFrame(estimator.feature_importances_.reshape(1,-1), index=target, columns=predictors)


def generate_reg(X, y, model, target, n_splits=10):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # removed proportion score... and n_jobs
    s = cross_validate(model, X, y.ravel(), cv=cv, scoring = {'r2': 'r2', 'neg_mean_squared_error':'neg_mean_squared_error'}, return_train_score=True, return_estimator=True, n_jobs=-1)
    s['target'] = [target]*n_splits
    print('done for:'+target)
    
    for x in list(range(n_splits)):
        joblib.dump(s['estimator'][x], filepath+'estimators/'+target+'_'+str(x)+'.joblib', compress=3)
   
    importance = pd.concat([help_feature_importances(x[0], X.columns, [x[1]]) for x in list(zip(s['estimator'],s['target']))])
    importance.to_pickle(filepath+'coefs/'+target+'.pkl')
    print('importances saved.') 
    del importance
    del s['estimator']
    return pd.DataFrame.from_dict(s)


print('prep done.')

#results = Parallel(n_jobs=10)(delayed(generate_reg) (X, vdata[:, v].layers['velocity'], model, v) for v in velocity_genes)
#results = pd.concat(results)
results = pd.concat([generate_reg(X, vdata[:, v].layers['velocity'], model, v) for v in velocity_genes])
results.to_pickle(filepath+'Scores.pkl')
print('CV generated.')



#joblib.dump(model, filepath+"/random_forest_cv_mrpl15.joblib")
end = time.time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("THis is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


