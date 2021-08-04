import datapreprocessing as dp
from sklearn.cross_decomposition import PLSRegression
import functions as f

filepath = './pls/'
database_file = 'data/regnet160_all.pkl'

predictors, X, velocity_genes, y = dp.get_data('tf160')

model = PLSRegression(n_components=100)

#somehow the coefficeints, the dimensinos have to be transposed, pls.coef returns different dimensions. is this only for pca?
coefs, scores = f.generating_regressions(model, predictors, velocity_genes, X, y, 10, path = filepath, transpose_coefs=True)

coefs = f.help_pivot_to_df(coefs)

permut = f.evaluate_permutations(coefs, database_file, path = filepath)


