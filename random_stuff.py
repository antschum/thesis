import functions as f
import pandas as pd
import pickle
import time
import datapreprocessing


# Linear Aug 5
lin_coefs = pd.read_pickle('linearQuick/Coefs.pkl')
regnet_all = f.help_import_database('data/regnet160_all.pkl')

# start = time.process_time()
coefs = f.help_pivot_to_df(lin_coefs)
# print('help_pivot_to_df', time.process_time() - start)
# start = time.process_time()
coefs = f.help_meanSD(coefs)
# print('meanSD takes', time.process_time() - start)
# start = time.process_time()
summary, percentages = f.help_summary(coefs, regnet_all)
# print('Help_summary', time.process_time() - start)
# print(summary)

count = f.help_summary_to_count(summary, percentages)

with open('linearQuick/count.pkl', 'wb') as f:  
    pickle.dump(count, f)





