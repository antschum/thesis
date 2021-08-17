from sklearn.cross_decomposition import PLSRegression
import functions as f
import datapreprocessing as dp
import time

start = time.time()
# Update this. 
filepath = './plsCluster/'
database_file = 'data/regnet160_all.pkl'
model = PLSRegression(n_components=100)

predictors, X, velocity_genes, y = dp.get_data('tf160', louvain=True)
# run regressions for each cluster here. 

# 1. test if it works. see what it looks like. 
# 2. try it for all w/ for loop
# 3. try generalizing to available cluster numbers. (also in terms of saving it later on..) 

clustering = 'louvain'
clusters = X[clustering].cat.categories.tolist()

for c in clusters:
    num = c
    path = filepath+'cluster_'+c+'/'

    # should drop louvain column..
    X_cluster = X[X[clustering]==num].drop(columns='louvain')
    Y_cluster = y[y[clustering]==num].drop(columns='louvain')

    coefs, scores = f.generating_regressions(model, predictors, velocity_genes, 
                                            X_cluster, Y_cluster, 10, path = path, transpose_coefs=True)

    coefs = f.help_pivot_to_df(coefs)

    permut = f.evaluate_permutations(coefs, database_file, path = path)

end = time.time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("This is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))