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
filepath = './rf_md30/'
database_file = 'data/regnet160_all.pkl'
max_depth = 30
max_features = 40

if os.path.exists(filepath):
    print('Path does exist under directory', os.getcwd()+filepath)
    shutil.rmtree(filepath)
os.mkdir(filepath)
os.mkdir(filepath+'estimators')
os.mkdir(filepath+'coefs')


model = RandomForestRegressor(n_jobs=-1, max_depth=max_depth, max_features=max_features)

predictors, X, velocity_genes, y = dp.get_data('tf160')

# X['louvain'] = vdata[:, pred].obs['louvain']
# X['sampleID'] = vdata[:, pred].obs['sampleID']

def help_feature_importances(estimator, predictors, target):
    return pd.DataFrame(estimator.feature_importances_.reshape(1,-1), index=target, columns=predictors)


def generate_reg(X, y, model, target, n_splits=10):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # removed proportion score... and n_jobs
    s = cross_validate(model, X, y.ravel(), cv=cv, scoring = {'r2': 'r2', 'neg_mean_squared_error':'neg_mean_squared_error'}, return_train_score=True, return_estimator=True, n_jobs=-1)
    s['target'] = [target]*n_splits
    print('done for:'+target)
    
    ### not saving estimators anymore. not relevant.
    #for x in list(range(n_splits)):
     #   joblib.dump(s['estimator'][x], filepath+'estimators/'+target+'_'+str(x)+'.joblib', compress=3)
   
    importance = pd.concat([help_feature_importances(x[0], X.columns, [x[1]]) for x in list(zip(s['estimator'],s['target']))])
    importance.to_pickle(filepath+'coefs/'+target+'.pkl')
    print('importances saved.') 
    del importance
    del s['estimator']
    return pd.DataFrame.from_dict(s)


print('prep done.')

#results = Parallel(n_jobs=10)(delayed(generate_reg) (X, vdata[:, v].layers['velocity'], model, v) for v in velocity_genes)
#results = pd.concat(results)
results = pd.concat([generate_reg(X, y, model, v) for v in velocity_genes])
results.to_pickle(filepath+'Scores.pkl')
print('CV generated.') 


coefs = pd.DataFrame()

# load coefs into one df
for (dirpath, dirnames, filenames) in os.walk(filepath+'coefs/'):
    print(filenames)
    for name in filenames:
        with open(filepath+'coefs/'+name, 'rb') as file:
             df = pickle.load(file)
             coefs = pd.concat([coefs, df])

coefs.to_pickle(filepath+'Coefs.pkl')

# create count
regnet_all = f.help_import_database(database_file)

# start = time.process_time()
# permutations rf
coefs = f.help_pivot_to_df(coefs)
# print('help_pivot_to_df', time.process_time() - start)
# start = time.process_time()

permut  = f.evaluate_permutations(coefs, database_file,  filepath)

#joblib.dump(model, filepath+"/random_forest_cv_mrpl15.joblib")
end = time.time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("THis is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


