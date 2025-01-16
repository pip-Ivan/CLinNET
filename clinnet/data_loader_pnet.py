import logging
import numpy as np
import pandas as pd
import random
import os
import gc

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from os.path import join

# Set a seed for reproducibility
SEED = 112
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


processed_path = "data/PNET_data/"

# use this one
cnv_filename = "P1000_data_CNA_paper.csv"
response_filename = "response_paper.csv"
gene_important_mutations_only = "P1000_final_analysis_set_cross_important_only.csv"
genes_value = "data/tissue_expression/FANTOM_tissue_specific_gene_expression.csv"

cached_data = {}
#-------------------------------------------------------------------------------------------------------------------------------------
def load_data(filename, selected_genes=None):
    filename = join(processed_path, filename)
    logging.info("loading data from %s," % filename)
    if filename in cached_data:
        logging.info("loading from memory cached_data")
        data = cached_data[filename]
    else:
        data = pd.read_csv(filename, index_col=0)
        cached_data[filename] = data
    logging.info(data.shape)

    if "response" in cached_data:
        logging.info("loading from memory cached_data")
        labels = cached_data["response"]
    else:
        labels = get_response()
        cached_data["response"] = labels

    # join with the labels
    all = data.join(labels, how="inner")
    all = all[~all["response"].isnull()]

    response = all["response"]
    samples = all.index

    
    x = all
    genes = all.columns
    

    if not selected_genes is None:
        intersect = sorted(set.intersection(set(genes), selected_genes))
        if len(intersect) < len(selected_genes):
            # raise Exception('wrong gene')
            logging.warning("some genes dont exist in the original data set")
        x = x.loc[:, intersect]
        genes = intersect
    logging.info(
        "loaded data %d samples, %d variables, %d responses "
        % (x.shape[0], x.shape[1], response.shape[0])
    )
    logging.info(len(genes))
 
    if "express_values" in cached_data:
        logging.info("loading express_values from memory cached_data")
        express_values = cached_data["express_values"]
    else:
        genes_value_df = pd.read_csv(genes_value, index_col="geneName")
        express_values = genes_value_df["prostate gland"].to_dict()
        cached_data["express_values"] = express_values

    del all
    gc.collect()
    return x, response, samples, genes, express_values
#-------------------------------------------------------------------------------------------------------------------------------------
def get_response():
    logging.info("loading response from %s" % response_filename)
    labels = pd.read_csv(join(processed_path, response_filename))
    labels = labels.set_index("id")
    return labels
#-------------------------------------------------------------------------------------------------------------------------------------
def load_data_type(data_type="gene", cnv_levels=5, cnv_filter_single_event=True, mut_binary=False, selected_genes=None):

    logging.info("loading {}".format(data_type))
    if data_type == "mut_important":
        x, response, info, genes, express_values = load_data(gene_important_mutations_only, selected_genes)
        if mut_binary:
            logging.info("mut_binary = True")
            x.values[x.values > 1.0] = 1.0
        
    if data_type == 'cnv_del':
        x, response, info, genes,express_values = load_data(cnv_filename, selected_genes)      
       
        logging.debug("Data loaded for cnv_del")
        x.values[x.values >= 0] = 0.0
        if cnv_levels == 3:
                if cnv_filter_single_event:
                    x.values[x.values == -1.0] = 0.0
                    x.values[x.values == -2.0] = 1.0
                else:
                    x.values[x.values < 0.0] = 1.0

        else:  # cnv_levels == 5
                x.values[x.values == -1.0] = 0.5
                x.values[x.values == -2.0] = 1.0
                
    if data_type == 'cnv_amp':
        x, response, info, genes,express_values = load_data(cnv_filename, selected_genes)
        
        x.values[x.values <= 0.0] = 0.0
        if cnv_levels == 3:
                if cnv_filter_single_event:
                    x.values[x.values == 1.0] = 0.0
                    x.values[x.values == 2.0] = 1.0
                else:
                    x.values[x.values > 0.0] = 1.0
        else:  # cnv_levels == 5
                x.values[x.values == 1.0] = 0.5
                x.values[x.values == 2.0] = 1.0

    return x, response, info, genes, express_values
#-------------------------------------------------------------------------------------------------------------------------------------
def combine(x_list, y_list, rows_list, cols_list, data_type_list, combine_type, use_coding_genes_only=False):
    
    cols_list_set = [set(list(c)) for c in cols_list]
    if combine_type == "intersection":
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)

    if use_coding_genes_only:
        f = "data/selected_genes/protein-coding_gene_with_coordinate_minimal.txt"
        coding_genes_df = pd.read_csv(f, sep="\t", header=None)
        coding_genes_df.columns = ["chr", "start", "end", "name"]
        coding_genes = set(coding_genes_df["name"].unique())
        cols = cols.intersection(coding_genes)

    # the unique (super) set of genes
    all_cols = sorted(cols)
    all_cols_df = pd.DataFrame(index=all_cols)

    df_list = []
    for x, y, r, c in zip(x_list, y_list, rows_list, cols_list):
        df = pd.DataFrame(x, columns=list(c), index=r)
        df = df.T.join(all_cols_df, how="right")
        df = df.T
        df = df.fillna(0)
        df_list.append(df)

    all_data = pd.concat(df_list, keys=data_type_list, join="inner", axis=1)

    del df_list
    gc.collect()

    # put genes on the first level and then the data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)

    x = all_data.values

    reordering_df = pd.DataFrame(index=all_data.index)
    y = reordering_df.join(y, how="left")

    y = y.values
    cols = all_data.columns
    rows = all_data.index
    logging.info("After combining, loaded data %d samples, %d variables, %d responses " % (x.shape[0], x.shape[1], y.shape[0]))

    del all_cols_df
    gc.collect()

    return x, y, rows, cols
#-------------------------------------------------------------------------------------------------------------------------------------
class PNETData:
    def __init__(self, data_type="mut", cnv_levels=5, cnv_filter_single_event=True, mut_binary=False, selected_genes=None, combine_type="intersection",
                 use_coding_genes_only=False, balanced_data=False, shuffle=False, selected_samples=None, stratify=True, multipeled=True,
                 n_split=5):
        self.data_type = data_type
        self.cnv_levels = cnv_levels
        self.cnv_filter_single_event = cnv_filter_single_event
        self.mut_binary = mut_binary
        self.selected_genes = selected_genes
        self.combine_type = combine_type
        self.use_coding_genes_only = use_coding_genes_only
        self.balanced_data = balanced_data
        self.shuffle = shuffle
        self.selected_samples = selected_samples
        self.stratify = stratify
        self.multipeled = multipeled
        self.n_splits = n_split
        self.cv_dir = f"data/PNET_data/{n_split}fold_cross_validation"
        os.makedirs(self.cv_dir, exist_ok=True)
        self.cv_df_path = os.path.join(self.cv_dir, 'cross_validation.csv')
        if os.path.exists(self.cv_df_path):
            self.cv_df = pd.read_csv(self.cv_df_path)
        else:
            self.load_data()
            self.cv_df = self.cross_val(self.n_splits)
    #=================================================================================================================================
    def load_data(self):
        if not self.selected_genes is None:
            if type(self.selected_genes) != list:
                df = pd.read_csv(self.selected_genes, header=0)
                self.selected_genes = list(df["genes"])

        if type(self.data_type) == list:
            x_list = []
            y_list = []
            rows_list = []
            cols_list = []

            for t in self.data_type:
                x, y, rows, cols,express_values = load_data_type(t, self.cnv_levels, self.cnv_filter_single_event, self.mut_binary, self.selected_genes)
                x_list.append(x), y_list.append(y), rows_list.append(rows), cols_list.append(cols)
            x, y, rows, cols = combine(x_list, y_list, rows_list, cols_list, self.data_type, self.combine_type, self.use_coding_genes_only)
            x = pd.DataFrame(x, columns=cols)

        else:
            x, y, rows, cols, express_values = load_data_type(self.data_type, self.cnv_levels, self.cnv_filter_single_event, self.mut_binary, self.selected_genes)

        if type(x) == pd.DataFrame:
            x = x.values

        if self.balanced_data:
            pos_ind = np.where(y == 1.0)[0]
            neg_ind = np.where(y == 0.0)[0]
            n_pos = pos_ind.shape[0]
            n_neg = neg_ind.shape[0]
            n = min(n_pos, n_neg)
            pos_ind = np.random.choice(pos_ind, size=n, replace=False)
            neg_ind = np.random.choice(neg_ind, size=n, replace=False)
            ind = np.sort(np.concatenate([pos_ind, neg_ind]))

            y = y[ind]
            x = x[ind,]
            rows = rows[ind]

        if self.shuffle:
            n = x.shape[0]
            ind = np.arange(n)
            np.random.shuffle(ind)
            x = x[ind, :]
            y = y[ind, :]
            rows = rows[ind]

        if self.selected_samples is not None:
            selected_samples_file = join(processed_path, self.selected_samples)
            df = pd.read_csv(selected_samples_file, header=0)
            selected_samples_list = list(df["Tumor_Sample_Barcode"])

            x = pd.DataFrame(x, columns=cols, index=rows)
            y = pd.DataFrame(y, index=rows, columns=["response"])

            x = x.loc[selected_samples_list, :]
            y = y.loc[selected_samples_list, :]
            rows = x.index
            cols = x.columns
            y = y["response"].values
            x = x.values

        self.x = x
        self.y = y
        self.info = rows
        self.columns = cols
        if self.multipeled:
            logging.info("**********data multipled by express_P.values************")
            self.preprocess_genes_multi(express_values)

        self.express_values = express_values
        logging.info(f"First 10 columns inside class of express value:{list(self.express_values.items())[:10]}")

        del x_list, y_list, rows_list, cols_list
        gc.collect()
        self.kf = self.cross_val(self.n_splits)

        classes = np.unique(self.y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.y[:,0])
        self.weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    #=================================================================================================================================
    def preprocess_genes_multi(self, express_values):
    # Ensure self.x is a float32 numpy array
     self.x = self.x.astype('float32')

     for gene, multiplier in express_values.items():
        # Skip the multiplication if multiplier is 0
        if multiplier == 0:
            continue

        for idx, column_tuple in enumerate(self.columns):
            if gene == column_tuple[0]:
                # Perform multiplication only if necessary
                self.x[:, idx] *= multiplier
    #=================================================================================================================================
    def get_train_test(self):
        if self.stratify:
            return train_test_split(self.x, self.y, test_size=self.test_size, stratify=self.y, random_state=SEED)
        else:
            return train_test_split(self.x, self.y, test_size=self.test_size, random_state=SEED)
    #=================================================================================================================================
    def get_train_validate_test(self):
        if self.stratify:
            x_train, x_temp, y_train, y_temp = train_test_split(self.x, self.y, test_size=self.test_size * 2, stratify=self.y, random_state=SEED)
        else:
            x_train, x_temp, y_train, y_temp = train_test_split(
                self.x, self.y, test_size=self.test_size * 2, random_state=SEED)

        if self.stratify:
            x_val, x_test, y_val, y_test = train_test_split(
                x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)
        else:
            x_val, x_test, y_val, y_test = train_test_split(
                x_temp, y_temp, test_size=0.5, random_state=SEED)

        return x_train, x_val, x_test, y_train, y_val, y_test
    #=================================================================================================================================
    def cross_val(self, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        df = pd.DataFrame(np.full((self.x.shape[0], n_splits), 'index'), columns=[f'kf{i}' for i in range(1,n_splits+1)])
        for idx, (train_val_index, test_index) in enumerate(skf.split(self.x, self.y)):
            train_index, valid_index = train_test_split(train_val_index, stratify=self.y[train_val_index], test_size=1/(n_splits-1), random_state=SEED)
            df.loc[test_index,f'kf{idx+1}'] = 'Test'
            df.loc[train_index,f'kf{idx+1}'] = 'Train'
            df.loc[valid_index,f'kf{idx+1}'] = 'Valid'
        df.to_csv(self.cv_df_path,index=False)
        return df 
    #=================================================================================================================================
    def get_kf(self, kf=1):
        kf_path = os.path.join(self.cv_dir, f'fold_{kf}.npz')
        if os.path.exists(kf_path):
            x_train, y_train, x_valid, y_valid, x_test, y_test, genes, gene_status, class_weight = np.load(kf_path, allow_pickle=True).values()
            return x_train, y_train, x_valid, y_valid, x_test, y_test, genes, gene_status, class_weight.item()
        if not hasattr(self, 'x'):
            self.load_data()        
        x_train = self.x[self.kf[f'kf{kf}'] == 'Train',:]
        x_test = self.x[self.kf[f'kf{kf}'] == 'Test',:]
        x_valid = self.x[self.kf[f'kf{kf}'] == 'Valid',:]
        y_train = self.y[self.kf[f'kf{kf}'] == 'Train',:]
        y_test = self.y[self.kf[f'kf{kf}'] == 'Test',:]
        y_valid = self.y[self.kf[f'kf{kf}'] == 'Valid',:]
        np.savez_compressed(file=kf_path, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test, 
                    genes=self.columns.levels[0], gene_status=self.columns.levels[1], class_weight=self.weight_dict)
        return x_train, y_train, x_valid, y_valid, x_test, y_test, self.columns.levels[0], self.columns.levels[1], self.weight_dict