import datapreprocessing as dp
from sklearn.cross_decomposition import PLSRegression
import functions as f
import time
import pickle
import pandas as pd

filepath = './test_work/'
database_file = 'data/regnet160_all.pkl'
#test_work = ['Klf2', 'Mcm3', 'Hif1a', 'Tcf7']
regnet_all = pd.read_pickle('data/regnet160_all.pkl')

predictors, X, velocity_genes, y = dp.get_data(predictors='test_work')

model = PLSRegression(n_components=4)


#somehow the coefficeints, the dimensinos have to be transposed, pls.coef returns different dimensions. is this only for pca?
coefs, scores = f.generating_regressions(model, predictors, velocity_genes, X, y, 10, path = filepath, transpose_coefs=True)

coefs = f.help_pivot_to_df(coefs)
print('This goes into evalualte coefs', coefs)

permut = f.evaluate_permutations(coefs, database_file, path = filepath)

#### Not tested yet!!
# generate model summary and count
start = time.process_time()
# this is the crux. 
coefs = f.help_meanSD(coefs)
print('meanSD takes', time.process_time() - start)
start = time.process_time()
summary, percentages = f.help_summary(coefs, regnet_all)
print('Help_summary', time.process_time() - start)
print(summary)

count = f.help_summary_to_count(summary, percentages)

with open('pls/count.pkl', 'wb') as f:
     pickle.dump(count, f)
