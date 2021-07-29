import functions as f
import pandas as pd
import pickle
import time

# done July 22 or so. 
#f.merge_regnet('./downloads/regnet160_regulator.csv', './downloads/regnet160_target.csv', './data/regnet160_all.pkl')

pca_coefs = pd.read_pickle('pca/Coefs.pkl')
regnet_all = f.help_import_database('data/regnet160_all.pkl')

start = time.process_time()
coefs = f.help_pivot_to_df(pca_coefs)
print('help_pivot_to_df', time.process_time() - start)
start = time.process_time()
coefs = f.help_meanSD(coefs)
print('meanSD takes', time.process_time() - start)
start = time.process_time()
summary, percentages = f.help_summary(coefs, regnet_all)
print('Help_summary', time.process_time() - start)
start = time.process_time()
count = f.help_summary_to_count(summary, percentages)
print('Help_summary_to_count', time.process_time() - start)

with open('pca/count.pkl', 'wb') as f:
    pickle.dump(count, f)




 