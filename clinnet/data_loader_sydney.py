import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


#-------------------------------------------------------------------------------------------------------------------------------------
def MakeBinaryFeatures(x):
    x1 = (x==1).astype('int8')
    x2 = (x==-1).astype('int8')
    res = np.concatenate([x1,x2], axis=1)
    order = [[i, j] for i,j in zip(range(x.shape[1]), range(x.shape[1],x.shape[1]*2))]
    return res[:, np.array(order).reshape(-1)]
#-------------------------------------------------------------------------------------------------------------------------------------
def balance_dataset(X, Y, method='oversample'):
    classes, counts = np.unique(Y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    minority_count = np.min(counts)
    majority_count = np.max(counts)
    if method == 'undersample':
        majority_indices = np.where(Y == majority_class)[0]
        sampled_indices = np.random.choice(majority_indices, minority_count, replace=False)
        balanced_indices = np.concatenate([sampled_indices, np.where(Y == minority_class)[0]])
    elif method == 'oversample':
        minority_indices = np.where(Y == minority_class)[0]
        oversampled_indices = np.random.choice(minority_indices, majority_count - minority_count, replace=True)
        balanced_indices = np.concatenate([minority_indices, oversampled_indices])
    else:
        raise ValueError("Invalid method. Use 'oversample' or 'undersample'.")
    np.random.seed(42)
    np.random.shuffle(balanced_indices)
    X_balanced = X[balanced_indices]
    Y_balanced = Y[balanced_indices]
    return X_balanced, Y_balanced
#-------------------------------------------------------------------------------------------------------------------------------------
def save_crossvalidation_df(n_splits, X, Y, saving_path):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    df = pd.DataFrame(np.full((X.shape[0], n_splits), 'index'), columns=[f'kf{i}' for i in range(1, n_splits+1)])
    for idx, (train_val_index, test_index) in enumerate(skf.split(X, Y)):
        train_index, valid_index = train_test_split(train_val_index, stratify=Y[train_val_index], test_size=1/(n_splits-1), random_state=42)
        df.loc[test_index,f'kf{idx+1}'] = 'Test'
        df.loc[train_index,f'kf{idx+1}'] = 'Train'
        df.loc[valid_index,f'kf{idx+1}'] = 'Valid'
    df.to_csv(saving_path, index=False)
    return df
#-------------------------------------------------------------------------------------------------------------------------------------
class SydneyData:
    def __init__(self, data_dir, coding_gene=True, unique_feature=False, balance=None, split_cnvs=True, n_split=20, unique_sample=False):
        self.data_dir = data_dir
        self.coding_gene = coding_gene
        self.unique_feature = unique_feature
        self.balance = balance
        self.split_cnvs = split_cnvs
        if split_cnvs: self.gene_status=['CNV_AMP', 'CNV_DEL']
        else: self.gene_status=['CNV']
        self.n_split = n_split
        self.unique_sample = unique_sample
        self.cv_dir = f"{data_dir}/{n_split}fold{'_'+balance if balance else ''}{'_coding' if coding_gene else ''}_uniquegenename{'_splitcsv' if split_cnvs else ''}{'_uniquesample' if unique_sample else ''}"
        self.cv_df_path = os.path.join(self.cv_dir, 'cross_validation.csv')
        os.makedirs(self.cv_dir, exist_ok=True)
        if os.path.exists(self.cv_df_path):
            self.cv_df = pd.read_csv(self.cv_df_path)
        else:
            self.load_data()
            self.cv_df = save_crossvalidation_df(self.n_split, self.X_processed, self.Y_processed, self.cv_df_path)
    #=================================================================================================================================
    def load_data(self):
        # data
        data = np.load(os.path.join(self.data_dir, 'Sydney_Data_c.npz'))
        self.X = data['X']
        self.Y = data['Y']
        self.Sample_ID = data['Sample_ID']
        self.Gene_Annot = data['Gene_Annot']
        self.Cohort = data['Cohort']
        #self.Uniq_columns = data['Uniq_columns']
        #print(self.Cohort)
        
        # coding gene data (18334 genes which are also unique)
        if self.coding_gene:
            self.X_processed, self.genes_processed = self.get_coding_gene(self.X, self.Gene_Annot[:, 1])
        # unique features (37965 genes)
        #elif self.unique_feature:
        #    self.X_processed, self.genes_processed = self.X[:, self.Uniq_columns], self.Gene_Annot[self.Uniq_columns,1]
        # all data (82520 genes)
        else:
            self.X_processed, self.genes_processed = self.X, self.Gene_Annot[:, 1]
        
        # get unique genes name
        self.genes_processed, index = np.unique(self.genes_processed, return_index=True)
        self.X_processed = self.X_processed[:,index]

        # Balance the data 
        if self.balance:
            self.X_processed, self.Y_processed = balance_dataset(self.X_processed, self.Y, method=self.balance)
        else: 
            self.Y_processed = self.Y

        # split cnv to amp and del
        if self.split_cnvs:
            self.X_processed = MakeBinaryFeatures(self.X_processed)
        classes = np.unique(self.Y_processed)
        print(f"Shape of Y_processed: {self.Y_processed.shape}")  # Debugging: Print the shape of Y_processed
    
        # Handle dimensionality of Y_processed
        if self.Y_processed.ndim == 1:  # If Y_processed is 1D, pass it directly
           class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.Y_processed)
        else:  # If Y_processed is 2D, access the first column
           class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.Y_processed[:, 0])
        self.weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    #=================================================================================================================================
    def get_coding_gene(self, X, genes):
        coding_gene = pd.read_csv('data/selected_genes/protein-coding_gene_with_coordinate_minimal.txt', sep='\t', header=None)
        coding_gene = coding_gene[3]
        self.coding_gene_mask = pd.Series(genes).isin(coding_gene)
        X = X[:, self.coding_gene_mask]
        genes = genes[self.coding_gene_mask]
        return X, genes
    #=================================================================================================================================
    def get_kf(self, kf=1):
        kf_path = os.path.join(self.cv_dir, f'fold_{kf}.npz')
        if os.path.exists(kf_path):
            return np.load(kf_path, allow_pickle=True).values()
        if not hasattr(self, 'X_processed'):
            self.load_data()
            
             # Ensure Y_processed is at least 2D for consistent indexing
        if self.Y_processed.ndim == 1:
           self.Y_processed = self.Y_processed.reshape(-1, 1)
        x_train = self.X_processed[self.cv_df[f'kf{kf}'] == 'Train', :]
        x_test = self.X_processed[self.cv_df[f'kf{kf}'] == 'Test', :]
        x_valid = self.X_processed[self.cv_df[f'kf{kf}'] == 'Valid', :]
        y_train = self.Y_processed[self.cv_df[f'kf{kf}'] == 'Train', :]
        y_test = self.Y_processed[self.cv_df[f'kf{kf}'] == 'Test', :]
        y_valid = self.Y_processed[self.cv_df[f'kf{kf}'] == 'Valid', :]
        np.savez_compressed(file=kf_path, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test, 
                            genes=self.genes_processed, gene_status=self.gene_status, class_weight=self.weight_dict)
        return x_train, y_train, x_valid, y_valid, x_test, y_test, self.genes_processed, self.gene_status, self.weight_dict