import time
import functions as f
import pandas as pd
import pickle

pca_impact = pd.read_pickle('pca/gene_impact.pkl')
regnet_all = f.help_import_database('data/regnet160_all.pkl')

start = time.process_time()
coefs = f.help_meanSD(pca_impact)
print('meanSD takes', time.process_time() - start)
start = time.process_time()
summary, percentages = f.help_summary(coefs, regnet_all)
print('Help_summary', time.process_time() - start)
start = time.process_time()
count = f.help_summary_to_count(summary, percentages)
print('Help_summary_to_count', time.process_time() - start)

with open('pca/count_gene_impact.pkl', 'wb') as f:
 pickle.dump(count, f)