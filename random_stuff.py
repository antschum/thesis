import functions as f
import pandas as pd
import pickle
import time
import datapreprocessing
import os

start = time.time()

coefs = pd.read_pickle('linearQuick/Coefs.pkl')

# create count
regnet_all = f.help_import_database('data/regnet160_all.pkl')

# permutations rf
coefs = f.help_pivot_to_df(coefs)

database_file = 'data/regnet160_all.pkl'
path = './linearQuick/'

msl = f.help_meanSD(coefs)
regnet = f.help_import_database(database_file)
count  = f.generate_count(msl, regnet,  path)

end = time.time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("This is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))








