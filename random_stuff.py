import functions as f
import pandas as pd
import pickle
import time
import datapreprocessing
import os

start = time.time()

# merge all rf importances to one df
path = './rf_lID/coefs'

coefs = pd.DataFrame()

for (dirpath, dirnames, filenames) in os.walk(path):
    print(filenames)
    for name in filenames:
        with open(path+'/'+name, 'rb') as file:
            df = pickle.load(file)
            coefs = pd.concat([coefs, df])

coefs.to_pickle('rf_lID/Coefs.pkl')

# create count
regnet_all = f.help_import_database('data/regnet160_all.pkl')

# permutations rf
coefs = f.help_pivot_to_df(coefs)

database_file = 'data/regnet160_all.pkl'
path = './rf_lID/'

permut  = f.evaluate_permutations(coefs, database_file,  path)


# evaluate counts
coefs = f.help_meanSD(coefs)

summary, percentages = f.help_summary(coefs, regnet_all)

count = f.help_summary_to_count(summary, percentages)

with open('rf_lID/count.pkl', 'wb') as f:  
    pickle.dump(count, f)

end = time.time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("This is the time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))








