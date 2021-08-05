#I want to test multiprocessing. 
import pandas as pd
import functions as f

coefs = pd.read_pickle('linearQuick/Coefs.pkl')
coefs = f.help_pivot_to_df(coefs)
permut = f.evaluate_permutations(coefs, database = 'data/regnet160_all.pkl', path = './data/')