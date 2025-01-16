# General 
import os
import json
import pickle # import dill as pickle
import gzip
import logging
import datetime
import random

# Data Analysis
import numpy as np
import pandas as pd
import networkx as nx

# Deep Learning
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras
from keras.models import load_model as keras_load_model
from keras.layers import Dense, Dropout, PReLU, RNN
from tensorflow.keras.activations import gelu

from keras.regularizers import l2,L1L2 
from keras.utils import plot_model as keras_plot_model
from keras.initializers import HeUniform
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


from sklearn.metrics import log_loss

# CLinNet 
from clinnet.network_go import load_maps
from clinnet.layers_custom import f1, Diagonal, SparseTF, GeneSelection, LeCunUniform
from clinnet.model_utils import plot_history, plot_confusion_matrix, plot_ROC, get_metrics, get_th

# set seed 
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
#-------------------------------------------------------------------------------------------------------------------------------------
# 1. get mask
def get_adj_mat_path():
    go_adj_mat_dir = 'clinnet/Gene_Ontology/adjacency_matrix'
    reac_adj_mat_dir = 'clinnet/Reactome/adjacency_matrix'

    adj_matrix_path = {'GO': os.path.join(go_adj_mat_dir, 'maps_hid_num_n_hid_total.pkl'),
                    'GO_CC': os.path.join(go_adj_mat_dir,'maps_hid_num_n_hid_cellular_component.pkl'),
                    'GO_BP': os.path.join(go_adj_mat_dir, 'maps_hid_num_n_hid_biological_process.pkl'),
                    'GO_MF': os.path.join(go_adj_mat_dir, 'maps_hid_num_n_hid_molecular_function.pkl'),
                    'Reactome': os.path.join(reac_adj_mat_dir, 'maps_hid_num_n_hid.pkl')}
    return (go_adj_mat_dir, reac_adj_mat_dir), adj_matrix_path
#-------------------------------------------------------------------------------------------------------------------------------------
# 2. get mask 
def get_mask(genes, tissue_name='all', network_database=None, ex_source=None):
    genes = pd.Series(genes)
    logging.debug(f"genes shape: {genes.shape}")
    (go_adj_mat_dir, reac_adj_mat_dir), adj_matrix_path = get_adj_mat_path()

    # Determine layer_1_gene based on network_database
    if network_database is None:
        layer_1_gene = genes
    else:
        if network_database == "Reactome":
            print("reactome detected")
            map_1 = load_maps(os.path.join(reac_adj_mat_dir, "maps_hid_num_2.h5"))["/map_1"]
        elif network_database.split("_")[0] == "GO":
            print("go detected")
            map_1 = load_maps(os.path.join(go_adj_mat_dir, "maps_hid_num_2_total.h5"))["/map_1"]
        else:
            # If network_database is not recognized, include all genes
            layer_1_gene = genes
        layer_1_gene = map_1.index if 'map_1' in locals() else genes

    # Determine ex_gene0 based on tissue_name and ex_source
    if tissue_name == 'all':
        ex_gene0 = genes  # Include all genes if 'all' is specified as the tissue name
    else:
        if ex_source == 'fantom':
            ex0 = pd.read_csv("data/tissue_expression/FANTOM_tissue_specific_gene_expression.csv")
            ex_gene0 = ex0.loc[ex0[tissue_name] > 0, "geneName"]
        elif ex_source == 'gtex':
            ex0 = pd.read_csv('data/tissue_expression/GTEx_tissue_specific_gene_expression.csv')
            ex0[tissue_name] = pd.to_numeric(ex0[tissue_name], errors='coerce')
            ex_gene0 = ex0.loc[ex0[tissue_name] > 0, "geneName"]
        else:
            # If ex_source is not recognized, include all genes
            ex_gene0 = genes

    # Create masks based on the filtered genes
    mask_ex = genes.isin(ex_gene0)
    mask_1 = genes.isin(layer_1_gene)

    # Combine masks (if both conditions should apply)
    combined_mask = mask_ex & mask_1

    logging.info(f"There are {combined_mask.sum()} genes after filtering.")

    # If no filtering was applied, set mask to all True
    if tissue_name == 'all' and network_database is None:
        combined_mask = pd.Series([True] * len(genes), index=genes.index)
        logging.info("No genes were filtered; returning mask of all ones.")
    
    # Convert combined_mask to the same format as mask_ex
    combined_mask = np.array(combined_mask, dtype="bool")
    return combined_mask
#-------------------------------------------------------------------------------------------------------------------------------------
# 3. preparing a saving directory base on time
def saving_dir(path):
    result_dir = 'Result'
    timeStamp = '{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())
    if path is None:
        saving_directory = f'result_{timeStamp}'
    else:
        saving_directory = f'{path}_{timeStamp}'
    saving_directory = os.path.join(result_dir, saving_directory)    
    os.makedirs(saving_directory, exist_ok=True)
    logging.info(f'Result directory: {saving_directory}')
    return saving_directory
#-------------------------------------------------------------------------------------------------------------------------------------
# 4. load model
def load_clinnet(saving_dir):
    with gzip.open(os.path.join(saving_dir, 'saved_model/clinnet_attrs.pkl.gz'), 'rb') as file:
        model_dict = pickle.load(file)

    model = CLinNET(None, None, None, None)
    weights = keras_load_model(os.path.join(saving_dir, 'saved_model/weights'), custom_objects={'GeneSelection':GeneSelection, 'Diagonal': Diagonal})
    setattr(model, 'model', weights)
    for key, value in model_dict.items():
        setattr(model, key, value)
    return model
#-------------------------------------------------------------------------------------------------------------------------------------
# 5. plot model architecture
def plot_model(model, saving_dir):
    plot_filename = os.path.join(saving_dir, 'model_architecture.png')
    keras_plot_model(model, to_file=plot_filename, expand_nested=True, show_layer_activations=True, show_shapes=True, show_layer_names=True)
    logging.info(f'Model architecture plot saved: {plot_filename}')
#-------------------------------------------------------------------------------------------------------------------------------------
# 6. check if networks are available
def get_network_map_path(n_hids):
    _, adj_matrix_path = get_adj_mat_path()
    nets = {}
    for network, n_hid in n_hids.items():
        net_maps_path = adj_matrix_path[network].replace('n_hid', str(n_hid))
        if os.path.exists(net_maps_path):
            nets[network] = (n_hid, net_maps_path)
    if len(nets) == 0:
        raise ValueError("Please check provided network and hidden layers. There is no suitable network.")
    else:
        logging.info(f'There is {len(nets)} network for provided number of hidden layer.')
    return nets
#-------------------------------------------------------------------------------------------------------------------------------------
# 7. get final probability from layers probability
def converge_layers_prob(layers_prob_test, layers_prob_valid, y_valid=None, method='average'):
    if method=='average':
        y_prob_test = np.average(layers_prob_test, axis=1)
        y_prob_valid = np.average(layers_prob_valid, axis=1)

    elif method=='average-bayes':

        layer_confidence = [1 / log_loss(y_valid, layers_prob_valid[:, idx]) for idx in range(layers_prob_test.shape[1])]
        y_prob_test = np.average(layers_prob_test, axis=1, weights=layer_confidence)
        y_prob_valid = np.average(y_prob_valid, axis=1, weights=layer_confidence)

    elif method=='last-layer':
        y_prob_test = layers_prob_test[:,-1]
        y_prob_valid = layers_prob_valid[:,-1]

    elif method=='first-layer':
        y_prob_test = layers_prob_test[:,0]
        y_prob_valid = layers_prob_valid[:,0]
        
    elif type(method) == int:
        y_prob_test = layers_prob_test[:,method]
        y_prob_valid = layers_prob_valid[:,method]   

    return y_prob_test, y_prob_valid
def get_callbacks(patience_es=30, patience_rlr=15, factor=0.2, min_lr=1e-6, monitor='val_loss'):

      early_stopping = EarlyStopping(
         monitor=monitor,
         patience=patience_es,
         restore_best_weights=True)
      reduce_lr = ReduceLROnPlateau(
         monitor=monitor,
         factor=factor,
         patience=patience_rlr,
         min_lr=min_lr)
      return [early_stopping, reduce_lr]
#-------------------------------------------------------------------------------------------------------------------------------------
# 8. CLinNET class
class CLinNET:
    def __init__(self, genes, gene_status, tissue, saving_dir, n_hids={'GO':5,'Reactome':5}, ex_source='fantom',
                 w_regs={'Diag':0,'GO':[0]*10,'Reactome':[0]*10}, w_regs_outcome={'GS':0,'Diag':0,'GO':[0]*10,'Reactome':[0]*10},
                 learning_rate=.0001, drop_rate=[.5, .1, .1, .1, .1, .1], verbose=True):
        if np.all(genes != None):
            # arguments
            self.genes = genes
            self.gene_status = gene_status
            self.tissue = tissue
            self.w_regs = w_regs
            self.n_hids = n_hids
            self.ex_source = ex_source
            self.w_regs_outcome = w_regs_outcome
            self.learning_rate = learning_rate
            self.drop_rate = drop_rate
            self.verbose = verbose
            self.seed = 42

            # attriburtes
            self.nets = get_network_map_path(n_hids)
            self.saving_dir = os.path.join('result', saving_dir, tissue)
            os.makedirs(self.saving_dir, exist_ok=True)
            self.features = np.array([f'{g}_{s}' for g in genes for s in gene_status])
            self.n_features = self.features.shape[0]
            self.n_genes = len(genes)
            self.n_status = len(gene_status)
            self.graph = nx.DiGraph()
            self.model = self.build_model(loss_weights=[ 1,1, 2, 4, 8, 16, 32, 64,  4, 8, 16, 32, 64])
    #=================================================================================================================================
    def build_model(self, loss_weights=None):
        self.loss_weights = loss_weights
        ## Input
        Inp = keras.Input(shape=(self.n_features,), dtype='float32', name=f'Input')
        Inp_Dense = Dense(1, activation='sigmoid', name=f'Input_Dense', kernel_regularizer=l2(self.w_regs_outcome['GS']), kernel_initializer=HeUniform(seed=self.seed))(Inp)

        ## Gene Selection
        self.mask = get_mask(genes=self.genes, tissue_name=self.tissue, ex_source=self.ex_source)
        GS = GeneSelection(self.mask, self.n_status, name=f'GS')(Inp)
        GS_Drop = Dropout(rate=self.drop_rate[0], name=f'GS_Drop')(GS)
        GS_Denes = Dense(1, activation='sigmoid', name=f'GS_Dense', kernel_regularizer=l2(self.w_regs_outcome['GS']),kernel_initializer=HeUniform(seed=self.seed))(GS_Drop)
        features_mask = np.repeat(self.mask, self.n_status)
        self.graph.add_node(f'GS', feature_id=self.features[features_mask], mask=features_mask)
        

        ## Diagonal layer
        Diag = Diagonal(self.mask.sum(), input_shape=(self.mask.sum() * self.n_status,), activation=PReLU(), name=f'Diag', kernel_initializer=HeUniform(seed=self.seed), 
                        bias_initializer=LeCunUniform(seed=self.seed), W_regularizer=L1L2(self.w_regs['Diag']))(GS_Drop)
        Diag_Drop = Dropout(rate=self.drop_rate[1], name=f'Diag_Drop')(Diag)
        Diag_Dense = Dense(1, activation='sigmoid', name=f'Diag_Dense', kernel_regularizer=L1L2(self.w_regs_outcome['Diag']), kernel_initializer=HeUniform(seed=self.seed))(Diag_Drop)
        self.graph.add_node(f'Diag', feature_id=self.genes[self.mask])
        cm = np.repeat(np.eye(self.mask.sum()), self.n_status, axis=0).reshape(self.mask.sum()*self.n_status,-1).astype(bool)
        cm = pd.DataFrame(data=cm, index=self.features[features_mask], columns=self.genes[self.mask])
        self.graph.add_edge(f'GS', f'Diag', cm=cm)

        ## outputs
        self.outcomes = [ Inp_Dense ,GS_Denes, Diag_Dense]
        self.graph.add_node(f'root', feature_id=np.array(['root']))

        ## Sparse layers
        for network, (n_hid, maps_path) in self.nets.items():
            Hidd = Diag_Drop
            pre_layer_name = 'Diag'
            maps = load_maps(maps_path)
            non_zero_rows = self.genes[self.mask]
            for n_layer in range(1, n_hid+1):
                mapp = maps[f'map_{n_layer}']
                df = pd.DataFrame(0, index=non_zero_rows, columns=mapp.columns, dtype="int8")
                temp_df = mapp[mapp.index.isin(non_zero_rows)].astype(np.int8)
                df.loc[temp_df.index, temp_df.columns] = temp_df
                mapp = df.loc[:,df.sum() != 0]
                non_zero_rows=mapp.columns
                _, n_pathways = mapp.shape
                post_layer_name = f'Hidd{n_layer}_{network}'
                Hidd = SparseTF(n_pathways, mapp, activation="tanh", name=post_layer_name, W_regularizer=l2(self.w_regs[network][n_layer-1]), 
                                kernel_initializer=HeUniform(seed=self.seed), bias_initializer=LeCunUniform(seed=self.seed))(Hidd)
                Hidd_Dense = Dense(1, activation='sigmoid', name=f'Hidd{n_layer}_Dense_{network}', kernel_regularizer=l2(self.w_regs_outcome[network][n_layer-1]),
                                    kernel_initializer=HeUniform(seed=self.seed))(Hidd)
                self.outcomes.append(Hidd_Dense)
                self.graph.add_node(post_layer_name, feature_id=mapp.columns)
                self.graph.add_edge(pre_layer_name, post_layer_name, cm=mapp)
                pre_layer_name = post_layer_name
            
            self.graph.add_edge(post_layer_name, 'root', cm=maps[f'map_{n_layer+1}'])
    
        self.n_output = len(self.outcomes)
        ## build model
        model = keras.Model(inputs=[Inp], outputs=self.outcomes, name=f'model')

        ## compile the model 
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        metrics = [f1, 'accuracy', 'AUC', 'Recall', 'Precision']
        model.compile(optimizer=optimizer, loss=['binary_crossentropy']*self.n_output, metrics=[metrics]*self.n_output, loss_weights=loss_weights)

        ## plot the model
        plot_model(model, self.saving_dir)
        return model
    
    #=================================================================================================================================
    def load_weights(self, weights_path):
        if weights_path==None:
            self.model.load_weights(os.path.join(self.saving_dir, 'weights.h5'))
        else:
            self.model.load_weights(weights_path)
    #=================================================================================================================================

    
    def train(self, x_train, y_train, x_valid, y_valid, *args, class_weight=None, **kwargs):
        # convert class_weight to sample_weight
        if class_weight is not None and len(np.unique(class_weight)) > 1:       
            sample_weight = class_weight[y_train.flatten()]
        else: sample_weight=None
        default_callbacks = get_callbacks()
        callbacks = kwargs.get('callbacks', [])
       
        callbacks.extend(default_callbacks)
        # fit the model
        history = self.model.fit(x_train, [y_train]*self.n_output, validation_data=(x_valid, [y_valid]*self.n_output), *args, sample_weight=sample_weight,
                                 callbacks=callbacks, 
                                 **kwargs
                                 )
        self.history = history.history
        # save the history file
        history_filename = os.path.join(self.saving_dir, f'history.csv')
        df = pd.DataFrame(self.history)
        df.to_csv(history_filename, index=False)
        # plot the history 
        plot_history(df, self.saving_dir, multiple_output=True)
        # save the model
        self.batch_size = kwargs.get('batch_size', None) 
        self.save()
    #=================================================================================================================================
    def save(self):
        # make model saving directory
        model_saved_dir = os.path.join(self.saving_dir, 'saved_model')
        os.makedirs(model_saved_dir, exist_ok=True)

        # save arguments
        prams = {'tissue':self.tissue,
                'saving_dir':self.saving_dir,
                'n_hids': self.n_hids,
                'ex_source': self.ex_source,
                'w_regs': self.w_regs,
                'w_regs_outcome': self.w_regs_outcome,
                'learning_rate': self.learning_rate,
                'drop_rate': self.drop_rate}
        params_path = os.path.join(self.saving_dir, 'params.json')
        with open(params_path, "w") as json_file:
            json.dump(prams, json_file, indent=4)

        # save model attributes 
        model_dict = self.__dict__.copy()
        del model_dict['model'], model_dict['outcomes']
        with gzip.open(os.path.join(model_saved_dir, 'clinnet_attrs.pkl.gz'), 'wb') as file:
            pickle.dump(model_dict, file)

        # save model weights 
        saved_model_path = os.path.join(model_saved_dir, 'weights')
        os.makedirs(saved_model_path, exist_ok=True)
        self.model.save(saved_model_path)
    #=================================================================================================================================
    def save_predictions(self, **kwargs):
        arr_dict = {}
        for k, v in kwargs.items():
            if k.startswith('x_'):
                arr_dict[k.replace('x','y_prob_layers')] = np.array(self.model.predict(v, batch_size=self.batch_size)).squeeze().T
            elif k.startswith('y_'):
                arr_dict[k] = v 
        np.savez_compressed(os.path.join(self.saving_dir, 'probabilities.npz'), **arr_dict)
    #=================================================================================================================================
    def predict(self, x):
        # get the y_prob 
        y_prob = self.model.predict(x, self.batch_size)
        return y_prob
    #=================================================================================================================================
    def evaluate(self, x_valid, y_valid, x_test, y_test, converge_method):

        # get test and valid datasets y_probs for each layer 
        self.y_prob_layers_valid = np.array(self.model.predict(x_valid, self.batch_size)).squeeze().T
        self.y_prob_layers_test = np.array(self.model.predict(x_test, self.batch_size)).squeeze().T

        # converge layers y_prob
        self.y_prob_test, self.y_prob_valid = converge_layers_prob(self.y_prob_layers_test, self.y_prob_layers_valid, y_valid, method=converge_method)
        
        # get the best (f1 maximizing) threshold
        self.th = get_th(y_valid, self.y_prob_valid)

        # get test set prediction
        self.y_pred_test = (self.y_prob_test > self.th).astype(int)

        # plot confusion matrix and ROC curve
        plot_confusion_matrix(y_true=y_test, y_pred=self.y_pred_test, save_dir=self.saving_dir)
        plot_ROC(y_test, self.y_prob_test, self.saving_dir)
        
        # get metrics 
        self.metrics_dict = get_metrics(y_test, self.y_prob_test, save_dir=self.saving_dir, th=self.th)

        return self.metrics_dict
    #=================================================================================================================================

