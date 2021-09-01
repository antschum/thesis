import pickle
import scanpy as sc

def get_predictors(predictors):
    
    if predictors=='tf8':
        pred = ['Klf2', 'Mcm3', 'Mcm5', 'Hmgb2', 'Cdk4', 'Hif1a', 'Mcm6', 'Tox']
    
    elif predictors=='tf160':
        open_file = open('data/transcriptionfactors160.pkl', "rb")
        pr = pickle.load(open_file)
        open_file.close()    

        # remove factors not available in vdata.var_names -> should be filtered out by default. 
        for x in  ['Junb', 'mt-Nd1', 'Fgl2', 'mt-Co1', 'mt-Nd4', 'Rraga', 'mt-Nd2']:
            if x in pr:
                pr.remove(x)
        pred = []
        [pred.append(x) for x in pr if x not in pred]

    elif predictors=='test4':
        pred = ['Klf2', 'Mcm3', 'Hif1a', 'Tcf7']
    
    else:
        open_file = open(predictors, "rb")
        pred = pickle.load(open_file)
        open_file.close()   

    return pred


def get_data(predictors = 'tf160',targets='velocity_genes', datafile="velocity_adata.h5ad", louvain=False):
    # load dataset
    vdata = sc.read_h5ad(datafile)

     # velocity genes
    velocity_genes = vdata[:,vdata.var['velocity_genes']].var.index.tolist()

    if predictors=='velocity_genes':
        pred = velocity_genes
    else: 
        pred = get_predictors(predictors)

    # check if all variables are in dataset. 
    # not working right now..
    #pred = help_check_variables(vdata, pred)
    
    
    # Scale Ms and velocity layer with zero mean and unit variance 
    # sc.pp.scale adds most recent mean and std as variables to var

    sc.pp.scale(vdata, layer='Ms')
    sc.pp.scale(vdata, layer='velocity')


    X = vdata[:, pred].to_df('Ms')
    Y = vdata[:, velocity_genes].to_df('velocity')

    if louvain:
        X['louvain'] = vdata.obs['louvain']
        Y['louvain'] = vdata.obs['louvain']

    return pred, X, velocity_genes, Y

def help_check_variables(anndata, variables):
    data_var = set(anndata.var_names)
    var = set(variables)
    var_in_data =  data_var & var
    var_not_in_data = var - data_var

    print('The following where not found in dataset: ', var_not_in_data)

    return var_in_data