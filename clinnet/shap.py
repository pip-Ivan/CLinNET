import os
import time 
import logging 
import numpy as np
import pandas as pd 
import keras 
import tensorflow as tf

import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#-------------------------------------------------------------------------------------------------------------------------------------
# 1. calculate executaion time decorator
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Function {func.__name__} took {end-start:.4f} seconds to complete.")
        return result
    return wrapper
#-------------------------------------------------------------------------------------------------------------------------------------
# 2. get reactome pathway:name dictionary 
def reactome_pathway_name():
    pathways = pd.read_csv('clinnet/Reactome/cache/pathways.csv')
    pathways = dict(zip(pathways.reactome_id, pathways.pathway_name))
    return pathways
#-------------------------------------------------------------------------------------------------------------------------------------
# 3. get go pathway:name dictionary 
def go_pathway_name():
    pathways = pd.read_csv('clinnet/Gene_Ontology/cache/go_node.csv')
    pathways = dict(zip(pathways.GO_ID, pathways.GO_name))
    return pathways
#-------------------------------------------------------------------------------------------------------------------------------------
# 4. get go pathway:name dictionary 
def is_startswith(string: str, chars: list) -> bool:
    res = False 
    for c in chars:
        res = res or string.startswith(c)
    return res
#-------------------------------------------------------------------------------------------------------------------------------------
# 5. replace multiple text
def multi_replace(string: str, from_list: list, to_list: list):
    for f, t in zip(from_list,to_list):
        if string.__contains__(f):
            string = string.replace(f, t)
            break
    return string
#-------------------------------------------------------------------------------------------------------------------------------------
# 6. get dataframe of feature importance
def save_feature_importance(shap, layer, saving_directory):
    sv = shap.sv[layer]
    sv_norm = shap.sv_norm[layer]
    node = shap.graph.nodes[layer]
    feature_ids = node['feature_id']
    feature_names = node['feature_name']
    in_degree = node.get('in_degree', [0]*len(node['feature_name']))
    out_degree = node.get('out_degree', [0]*len(node['feature_name']))
    imp = np.abs(sv).mean(0)
    imp_norm = np.abs(sv_norm).mean(0)
    rank_index = node['rank_index']

    feature_importance = pd.DataFrame(list(zip(feature_ids, feature_names, imp, imp_norm, rank_index, in_degree, out_degree)),
                                      columns=['feature_id', 'feature_name','importance', 'importance_norm', 'rank_index', 'in_degree','out_degree'])
    # feature_importance.importance /= feature_importance.importance.sum()
    # feature_importance.importance_norm /= feature_importance.importance_norm.sum()
    data_filename = os.path.join(saving_directory, f"shap_data_{layer}.csv")
    feature_importance.to_csv(data_filename, index=False)
    logging.info(f"SHAP data saved for layer {layer}: {data_filename}")
#-------------------------------------------------------------------------------------------------------------------------------------
# 7. get name of next layers $NOT USED$
def get_next_layer(model, layer):
    selected_layer = model.model.get_layer(layer)
    layer_name = [l.name for l in model.model.layers]
    next_layers = [] 
    for out_node in selected_layer.outbound_nodes:
        name = out_node.outbound_layer.name 
        if name in layer_name:
            next_layers.append(name)
    return next_layers
#-------------------------------------------------------------------------------------------------------------------------------------
# 8. get name of previous layers $NOT USED$
def get_previous_layer(model, layer):
    selected_layer = model.model.get_layer(layer)
    previous_layers = [] 
    in_node = selected_layer.inbound_nodes[0]
    in_layer = in_node.inbound_layers
    if type(in_layer) == list:
        if len(in_layer)>0:
            for layer in in_layer:
                previous_layers.append(layer.name)
    else:
        previous_layers.append(in_layer.name)
    return previous_layers
#-------------------------------------------------------------------------------------------------------------------------------------
# 8. get samples from data for shap background and test set
#def get_sample(X, Y, n_sample=None, random_state=111):
 #   if n_sample >= X.shape[0]:
 #       return X
 #   else:
 #       _, _, _, y = train_test_split(X, Y, test_size=n_sample/len(X), stratify= Y, random_state=random_state)
 #       return y
    
def get_sample(X, Y, n_sample=None, random_state=111):
    _, x, _, y = train_test_split(X, Y, test_size=n_sample/len(X), stratify=Y, random_state=random_state)
    return x, y


#-------------------------------------------------------------------------------------------------------------------------------------
# 9. get each layer output to calculate the shap values
def get_layer_output(model, layer_shap, x_train, x_test, y_train, y_test, train_n_sample, test_n_sample):
    train_, _ = get_sample(x_train, y_train, n_sample=train_n_sample if train_n_sample is not None else 1000)
    test_, _ = get_sample(x_test, y_test, n_sample=test_n_sample if test_n_sample is not None else 1000)

    
    new_inputs_train = {}
    new_inputs_test = {}
    for l in layer_shap:
        desired_layer = model.get_layer(l)
        new_model = keras.Model(inputs=model.input, outputs=desired_layer.output)
        
        
        new_inputs_train[l] = new_model.predict(train_, verbose=0)
        new_inputs_test[l] = new_model.predict(test_, verbose=0)
    logging.info("Output of hidden layers calculated.")
    return new_inputs_train, new_inputs_test
#-------------------------------------------------------------------------------------------------------------------------------------
# 10. shap value class
class SHAP:
    def __init__(self, model, norm_method='degree', train_n_sample=None, test_n_sample=None, saving_dir=None):
        self.n_status = model.gene_status
        self.n_hids = model.n_hids
        self.graph = model.graph.copy()
        self.train_n_sample = train_n_sample
        self.test_n_sample = test_n_sample
        self.norm_method = norm_method
        self.model_dir = model.saving_dir
        self.interpret_dir = os.path.join(model.saving_dir, 'interpretability')
        self.shap_dir = os.path.join(self.interpret_dir, 'SHAP') if saving_dir is None else saving_dir
        os.makedirs(self.shap_dir, exist_ok=True)
        self.layer_name = [l.name for l in model.model.layers]
        self.layer_shap = [l for l in self.layer_name if is_startswith(l,['Hidd', 'GS', 'Diag']) and not l.__contains__('_D')]
        self.get_feature_name()
        self.get_layer_info()
        self.get_degree()
    #=================================================================================================================================
    def get_layer_info(self):
        for n in self.graph.nodes:
            if n.startswith('GS'):
                self.graph.nodes[n]['network'] = 'common'
                self.graph.nodes[n]['layer_id'] = 0
            elif n.startswith('Diag'):
                self.graph.nodes[n]['network'] = 'common'
                self.graph.nodes[n]['layer_id'] = 1
            elif n!='root':
                temp = n.split('-')[0].split('_')
                maxi = int(temp[0].replace('Hidd',''))+1
                self.graph.nodes[n]['layer_id'] = maxi
                self.graph.nodes[n]['network'] = temp[1]
        
        self.graph.nodes['root']['network'] = 'common'
        self.graph.nodes['root']['layer_id'] = maxi+1
    #=================================================================================================================================
    def get_feature_name(self):
        id_name = {}
        for net, ـ in self.n_hids.items():
            if net == 'Reactome':
                id_name.update(reactome_pathway_name())
            else:
                id_name.update(go_pathway_name())
        
        id_name['root'] = 'Output'
        for n in self.graph.nodes:
            self.graph.nodes[n]['feature_name'] = [i if is_startswith(n,['GS','Diag']) else id_name[i] for i in self.graph.nodes[n]['feature_id']]            
        logging.info("Features name prepared.")
    #=================================================================================================================================
    @timeit
    def get_degree(self):
        for pre_layer, post_layer in self.graph.edges:
            cm = self.graph.edges[pre_layer, post_layer]['cm']
            pre_node = self.graph.nodes[pre_layer]
            post_node = self.graph.nodes[post_layer]
            pre_node['out_degree'] = pre_node.get('out_degree', 0) + cm.sum(axis=1)
            post_node['in_degree'] = post_node.get('in_degree', 0) + cm.sum(axis=0)
    #=================================================================================================================================
    @timeit
    def get_layer_shap(self, model, x_train, x_test, y_train, y_test):
        new_inputs_train, new_inputs_test = get_layer_output(model, self.layer_shap, x_train, x_test, y_train, y_test, self.train_n_sample, self.test_n_sample)
        sv = {}
        for input_layer in self.layer_shap:
            new_inp = model.get_layer(input_layer).output
            output_layer = input_layer.split('_')
            output_layer.insert(1, 'Dense')
            output_layer = '_'.join(output_layer)
            new_out = model.get_layer(output_layer)(new_inp)
            new_model = keras.Model(inputs=new_inp, outputs=new_out)
            explainer = shap.DeepExplainer(new_model, new_inputs_train[input_layer])
            test_shap_values = explainer.shap_values(new_inputs_test[input_layer])
            sv[input_layer] = test_shap_values.squeeze()
        sv['root'] = np.ones((sv[input_layer].shape[0], 1)) 
        self.sv = sv
        self.save_shap_plot(new_inputs_test)
        self.normalization(method=self.norm_method)
        self.get_rank_index()
        
    #=================================================================================================================================
    def normalization(self, method: str = "degree"): # add subgraph method & during training normalization method
        self.sv_norm = {}
        for layer in self.sv.keys():
            node = self.graph.nodes[layer]
            if method=='degree':
                degree = np.array(node.get('in_degree', 0)) + np.array(node.get('out_degree', 0))
            elif method in ['in_degree', 'out_degree']:
                degree = np.array(node.get(method, 0))

            if 0 in degree: 
                print(f'Layer {layer} has 0 in degrees. Check it.')
                degree[degree==0] = 1
            
            sv_norm = self.sv[layer] / degree[None, :]
            self.sv_norm[layer] = sv_norm
    #=================================================================================================================================
    def get_rank_index(self):
        for layer in self.sv.keys():
            sorted_indices = np.abs(self.sv_norm[layer]).mean(axis=0).argsort()[::-1]
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(0, len(sorted_indices))
            self.graph.nodes[layer]['rank_index'] = ranks
        self.graph.nodes['root']['rank_index'] = np.array([0])
    #=================================================================================================================================
    def save_shap_plot(self, new_inputs_test):
        for layer in self.layer_shap:
            # Save SHAP plot
            plt.figure(figsize=(16, 16))
            shap.summary_plot(self.sv[layer], new_inputs_test[layer], feature_names=self.graph.nodes[layer]['feature_name'], plot_type='dot', plot_size=(20,20),show=False, max_display=10)
            plt.gca().title.set_size(24)  
            plt.gca().xaxis.label.set_size(26) 
            plt.gca().yaxis.label.set_size(26)  
            plt.xticks(fontsize=18)  
            plt.yticks(fontsize=18) 
            plot_filename = os.path.join(self.shap_dir, f'{layer}.png')
            logging.info(f"SHAP plot saved for layer {layer}: {plot_filename}")
            plt.savefig(plot_filename, bbox_inches='tight',dpi=600)
            plt.close()

    
    #=================================================================================================================================    
    def save_shap_csv(self):
        for layer in self.layer_shap:
            # Save SHAP values as DataFrame
            save_feature_importance(self, layer, self.shap_dir)
        np.savez_compressed(f"{self.shap_dir}/shap_values_normalized.npz", sv=self.sv, sv_norm=self.sv_norm)