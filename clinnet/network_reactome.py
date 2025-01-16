import os
import re
import itertools
import pickle
import logging
import numpy as np
import networkx as nx
import pandas as pd
from os.path import join, exists



Databae_dir = 'data/Network/Reactome2023/'
cache_dir = 'clinnet/Reactome/cache'
adj_matrix_dir = 'clinnet/Reactome/adjacency_matrix'

# https://reactome.org/download/current/ReactomePathwaysRelation.txt
relations_file_name = 'ReactomePathwaysRelation.txt'
# https://reactome.org/download/current/ReactomePathways.txt
pathway_names = 'ReactomePathways.txt'
# https://reactome.org/download/current/ReactomePathways.gmt.zip
pathway_genes = 'ReactomePathways.gmt'


#-------------------------------------------------------------------------------------------------------------------------------------
def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G
#-------------------------------------------------------------------------------------------------------------------------------------
def complete_network(G, n_leveles=4):
    '''
    Suppose you have a graph with 11 layers(based of distance from root) and you just wants 6 layer 
    of network. So, you will lost all 7 to 11 layers information(nodes and edges). On the other hand,
    some terminal nodes(connected to genes) are located in layers < 6. Means some terminal nodes have 
    Distance from root of less than 6. Complete_network() with coopration of add_edges() make some fake 
    node(by adding copy{number} to terminal node) to strictly make a 6 layer network.
    '''
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)
    return sub_graph
#-------------------------------------------------------------------------------------------------------------------------------------
def get_neural_network(net, n_hid):
    nodes_level = {}
    for i in range(13):
        temp = get_nodes_at_level(net, i)
        for j in temp:
            nodes_level[f'{j}'] = i

    net = nx.ego_graph(net, 'root', radius=n_hid)
    layers = get_layers_from_net(net, n_hid+1)
    rm_edges = []
    for idx, l in enumerate(layers):
        for k, v in l.items():
            temp = nodes_level[k]
            for i in v:
                if temp >= nodes_level[i]:
                    rm_edges.append((k,i))
    net.remove_edges_from(rm_edges)
    return net
#-------------------------------------------------------------------------------------------------------------------------------------
def get_nodes_at_level(net, distance):
    nodes = set(nx.ego_graph(net, 'root', radius=distance))
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))
    return list(nodes)
#-------------------------------------------------------------------------------------------------------------------------------------
def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub('_copy.*', '', n)
            next = net.successors(n)
            dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next]
        layers.append(dict)
    return layers
#-------------------------------------------------------------------------------------------------------------------------------------
def get_map_from_layer(layer_dict):
    pathways = sorted(layer_dict.keys())
    genes = list(itertools.chain.from_iterable(list(layer_dict.values())))
    genes = sorted(np.unique(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in list(layer_dict.items()):
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1
    df = pd.DataFrame(mat, index=pathways, columns=genes)
    return df.T
#-------------------------------------------------------------------------------------------------------------------------------------
def load_maps(path):
    # with pd.HDFStore(path, 'r') as store:
    #     maps = {key: store[key] for key in store.keys()}
    
    with open(path, 'rb') as file:
        maps = pickle.load(file)
    return maps
#-------------------------------------------------------------------------------------------------------------------------------------
def maps_info(input_genes, maps_file, just_involved_gene=False):
    logging.info(maps_file.split('/')[-1])
    data_gene = set(input_genes)
    maps = load_maps(maps_file)
    map_gene = set(maps['map_1'].index[maps['map_1'].sum(axis=1)>0])
    logging.info(f'Involved genes: {len(data_gene.intersection(map_gene))}/{len(data_gene)}')
    total_neuron = {neuron for map in maps.values() for neuron in map.columns}
    logging.info(f'total neuron in network: {len(total_neuron)}')
    if not just_involved_gene:
        for k, m in maps.items():
            logging.info(f'{k}: {m.shape}')
            logging.info('   connection:')
            logging.info(f'\t{sum(m.sum(axis=1)>0)}/{m.shape[0]}')
            logging.info(f'\t{sum(m.sum(axis=0)>0)}/{m.shape[1]}\n')
#-------------------------------------------------------------------------------------------------------------------------------------
def gmt_load_data(filename, genes_col=1, pathway_col=0):
    data_dict_list = []
    with open(filename) as gmt:
        data_list = gmt.readlines()
        for row in data_list:
            genes = row.strip().split('\t')
            genes = [re.sub('_copy.*', '', g) for g in genes]
            genes = [re.sub('\\n.*', '', g) for g in genes]
            for gene in genes[genes_col:]:
                pathway = genes[pathway_col]
                dict = {'group': pathway, 'gene': gene}
                data_dict_list.append(dict)
    df = pd.DataFrame(data_dict_list)
    return df
#-------------------------------------------------------------------------------------------------------------------------------------
class Reactome_Data():
    def __init__(self):
        self.pathway_names = self.load_names()
        self.hierarchy = self.load_hierarchy()
        self.pathway_genes = self.load_genes()
    #=================================================================================================================================
    def load_names(self):
        filename = join(Databae_dir, pathway_names)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['reactome_id', 'pathway_name', 'species']
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.exists(os.path.join(cache_dir, 'pathways.csv')):
            pathways = df[df.species == 'Homo sapiens'].reset_index(drop=True)
            pathways.drop('species', inplace=True,axis=1)
            pathways.to_csv(os.path.join(cache_dir, 'pathways.csv'), index=0)
        return df
    #=================================================================================================================================
    def load_genes(self):
        filename = join(Databae_dir, pathway_genes)
        df = gmt_load_data(filename, pathway_col=1, genes_col=3)
        return df
    #=================================================================================================================================
    def load_hierarchy(self):
        filename = join(Databae_dir, relations_file_name)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['child', 'parent']
        return df
#-------------------------------------------------------------------------------------------------------------------------------------
class Reactome():
    def __init__(self):
        self.reactome = Reactome_Data()
        self.netx = self.get_reactome_networkx()
    #=================================================================================================================================
    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.reactome.hierarchy
        # filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph())
        net.name = 'reactome'
        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)
        return net
    #=================================================================================================================================
    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = get_neural_network(self.netx, n_levels)
            net = complete_network(net, n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = complete_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]

        # get the last layer (genes level)
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]  # set of terminal pathways
        temp=[i.split('_')[0] for i in terminal_nodes]
        logging.info(f'Terminal nodes: {len(terminal_nodes)}\n')
        # we need to find genes belonging to these pathways
        genes_df = self.reactome.pathway_genes
        genes_df = genes_df[genes_df.group.isin(temp)]
        genes_df = genes_df[~genes_df.duplicated()]
        genes_dict = genes_df.groupby('group')['gene'].apply(list).to_dict()
        layers.append(genes_dict)
        return layers
    #=================================================================================================================================
    def create_network_adj(self, n_hid, add_unk_genes=False, direction='root_to_leaf'):

        if not os.path.exists(adj_matrix_dir):
            os.makedirs(adj_matrix_dir)

        file_path = os.path.join(adj_matrix_dir, f'maps_hid_num_{n_hid}.pkl')
        if exists(file_path):
            logging.info(f"File '{file_path}' already exists!")
            logging.info("Load data using the load_maps function.")
            return 
        
        reactome_layers = self.get_layers(n_hid, direction)
        maps = {}
        maps1 = {}
        logging.info(f'{"#"*20}REACTOME GRAPH WITH {n_hid} HIDDEN LAYERS{"#"*20}')
        # making first map
        pathways = sorted(reactome_layers[-1].keys())
        genes = sorted(set([gene for _, genes in reactome_layers[-1].items() for gene in genes]))
        n_pathways = len(pathways)
        n_genes = len(genes)
        print('n_genes', n_genes)
        print('n_pathways', n_pathways)
        mat = np.zeros((n_pathways, n_genes))
        for p, gs in list(reactome_layers[-1].items()):
            g_inds = [genes.index(g) for g in gs]
            p_ind = pathways.index(p)
            mat[p_ind, g_inds] = 1
        mapp = pd.DataFrame(mat, index=pathways, columns=genes).T
        if add_unk_genes:
            mapp['UNK'] = 0
            ind = mapp.sum(axis=1) == 0
            mapp.loc[ind, 'UNK'] = 1
        mapp = mapp.fillna(0)
        maps[f'map_{1}'] = mapp.astype('bool')
        for i, layer in enumerate(reactome_layers[-2::-1]):
            mapp = get_map_from_layer(layer)
            mapp = mapp.loc[mapp.index.isin(pathways), :]
            pathways = mapp.columns
            if add_unk_genes:
                mapp['UNK'] = 0
                ind = mapp.sum(axis=1) == 0
                mapp.loc[ind, 'UNK'] = 1

            mapp = mapp.fillna(0)
            maps[f'map_{i+2}'] = mapp.astype('bool')

        m = maps['map_1']
        r_mask = m.sum(axis=1)!=0
        m = m.loc[r_mask,:]
        c_mask = m.sum(axis=0)!=0
        maps1['map_1'] = m.loc[:, c_mask]
        for i in range(1,len(maps)):
            m = maps[f'map_{i+1}']
            r_mask = c_mask.copy()
            m = m.loc[r_mask,:]
            c_mask = m.sum(axis=0)!=0
            maps1[f'map_{i+1}'] = m.loc[:, c_mask]

        for k, m in maps1.items():
            logging.info(f'{k}: {m.shape}')
            logging.info('   connection:')
            logging.info(f'\t{sum(m.sum(axis=1)>0)}/{m.shape[0]}')
            logging.info(f'\t{sum(m.sum(axis=0)>0)}/{m.shape[1]}\n')
        # with pd.HDFStore(file_path, 'w') as store:
        #     for k, m in maps1.items():
        #         store[k] = m
        with open(file_path, 'wb') as file:
            pickle.dump(maps1, file)
            # with pd.HDFStore(file_paths[key], 'w') as store:
            #     for k, m in maps1.items():
            #         store[k] = m


