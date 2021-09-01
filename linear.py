from sklearn.linear_model import LinearRegression
import functions as f
import datapreprocessing as dp
import time

start = time.time()
# Update this. 
filepath = './linearQuick/'
database_file = 'data/regnet160_all.pkl'
model = LinearRegression(random_state=42)

predictors, X, velocity_genes, y = dp.get_data('tf160')

coefs, scores = f.generating_regressions(model, predictors, velocity_genes, X, y, 10, path = filepath)

coefs = f.help_pivot_to_df(coefs)

permut = f.evaluate_permutations(coefs, database_file, path = filepath)

end = time.time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("This is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))