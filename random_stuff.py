import functions as f
import pandas as pd
import pickle
import time
import datapreprocessing
import os


# merge all rf importances to one df
path = './rf/coefs'

coefs = pd.DataFrame()

for (dirpath, dirnames, filenames) in os.walk(path):
    print(filenames)
    for name in filenames:
        with open(path+'/'+name, 'rb') as file:
            df = pickle.load(file)
            coefs = pd.concat([coefs, df])

coefs.to_pickle('rf/Coefs.pkl')

# create count
regnet_all = f.help_import_database('data/regnet160_all.pkl')

# start = time.process_time()
coefs = f.help_pivot_to_df(coefs)
# print('help_pivot_to_df', time.process_time() - start)
# start = time.process_time()
coefs = f.help_meanSD(coefs)
# print('meanSD takes', time.process_time() - start)
# start = time.process_time()
summary, percentages = f.help_summary(coefs, regnet_all)
# print('Help_summary', time.process_time() - start)
# print(summary)

count = f.help_summary_to_count(summary, percentages)

with open('rf/count.pkl', 'wb') as f:  
    pickle.dump(count, f)





