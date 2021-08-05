import pickle
import scanpy as sc

def get_predictors(predictors):
    
    if predictors=='tf8':
        pred = ['Klf2', 'Mcm3', 'Mcm5', 'Hmgb2', 'Cdk4', 'Hif1a', 'Mcm6', 'Tox']
    
    elif predictors=='tf160':
        open_file = open('data/transcriptionfactors160.pkl', "rb")
        pred = pickle.load(open_file)
        open_file.close()    

        # remove factors not available in vdata.var_names -> should be filtered out by default. 
        for x in  ['Junb', 'mt-Nd1', 'Fgl2', 'mt-Co1', 'mt-Nd4', 'Rraga', 'mt-Nd2']:
            if x in pred:
                pred.remove(x)

    elif predictors=='test_work':
        pred = ['Klf2', 'Mcm3', 'Hif1a', 'Tcf7']
    
    else:
        open_file = open(predictors, "rb")
        pred = pickle.load(open_file)
        open_file.close()   

    return pred


def get_data(predictors = 'tf160', datafile="velocity_adata.h5ad"):
    # load dataset
    vdata = sc.read_h5ad(datafile)
    pred = get_predictors(predictors)

    # check if all variables are in dataset. 
    # not working right now..
    #pred = help_check_variables(vdata, pred)
    
    
    # Scale Ms and velocity layer with zero mean and unit variance 
    # sc.pp.scale adds most recent mean and std as variables to var

    sc.pp.scale(vdata, layer='Ms')
    sc.pp.scale(vdata, layer='velocity')


    # velocity genes
    velocity_genes = vdata.var.index[vdata.var['velocity_genes'] == True].tolist()

    X = vdata[:, pred].layers['Ms']
    Y = vdata[:, velocity_genes].layers['velocity']

    return pred, X, velocity_genes, Y

def help_check_variables(anndata, variables):
    data_var = set(anndata.var_names)
    var = set(variables)
    var_in_data =  data_var & var
    var_not_in_data = var - data_var

    print('The following where not found in dataset: ', var_not_in_data)

    return var_in_data