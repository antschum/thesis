import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import functions as f
import random
import pickle

# import datasets
regnet_regulators = pd.read_csv('regnetworkweb_all8tf.csv')
regnet_targets = pd.read_csv('regnet_targets.csv')
regnet_all = pd.concat([regnet_regulators, regnet_targets], axis=0)


# capitalize first letter
regnet_all['regulator_symbol'] = [x.capitalize() for x in regnet_all['regulator_symbol']]
regnet_all['target_symbol'] = [x.capitalize() for x in regnet_all['target_symbol']]


# Curate Data for the percentiles...
linearCoefs8 = pd.read_pickle('linearCoefs8.pkl')

mean_coefficients8 = linearCoefs8.pivot_table(index="predictors", columns="target", values="coefficients", aggfunc = np.mean)
std_coefficients8 = linearCoefs8.pivot_table(index="predictors", columns="target", values="coefficients", aggfunc = np.std)

msl = pd.DataFrame()
msl['mean'] = linearCoefs8.groupby(['predictors', 'target']).coefficients.mean()
msl['sd'] = linearCoefs8.groupby(['predictors', 'target']).coefficients.std()
msl['mean/sd'] = msl['mean']/msl['sd']

msl = msl.reset_index(col_fill =['predictors', 'target', 'mean', 'sd', 'mean/sd'])

def shuffle_meanSD(msl):
    permutations = pd.DataFrame()
    permutations['predictors'], permutations['target'] = msl['predictors'], msl['target']
    permutations['mean/sd'] = random.shuffle(msl['mean/sd'])
    return permutations

# permutations
c=0

total = []
while c<=1000:
    permutations = shuffle_meanSD(msl)
    summary, percentages = f.help_summary(msl, regnet_all)
    
    count = []
    
    for p,i in zip(percentages, list(range(0, len(percentages)))):
        count+=[sum([len(l['matchP'])+len(l['matchT']) for l in summary[str(p)].values()])]
            
    #should I take the mean of the random counts? and then see where that takes me? 
    # --> plot average and sd --> have to keep track of number of matches per percentile. 
    c+=1
    total+=[count]
    if c%50==0:
        print("currently at ",c)

permut = pd.DataFrame()   
permut['mean'] = np.mean(total, axis=0)
permut['std'] = np.std(total, axis=0)

permut.to_pickle('permutations.pkl')
print(permut)