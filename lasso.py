from sklearn.linear_model import Lasso 
import functions as f
import datapreprocessing as dp
import time


# THis does not converge. super wierd. 
start = time.time()
# Update this. 
filepath = './lasso/'
database_file = 'data/regnet160_all.pkl'
model = Lasso(alpha=0.000013)

predictors, X, velocity_genes, y = dp.get_data('tf160')

coefs, scores = f.generating_regressions(model, predictors, velocity_genes, X, y, 10, path = filepath)

coefs = f.help_pivot_to_df(coefs)

permut = f.evaluate_permutations(coefs, database_file, path = filepath)

end = time.time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("This is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))