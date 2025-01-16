import os
import re
import json
import pickle
import itertools
import logging
import obonet
import pandas as pd 
import networkx as nx
import numpy as np


gaf_path = 'data/Network/GO/goa_human.gaf' # gene to go_id
obo_path = 'data/Network/GO/go.obo' # go_id to go_id
cache_dir = 'clinnet/Gene_Ontology/cache'
adj_matrix_dir = 'clinnet/Gene_Ontology/adjacency_matrix'

#-------------------------------------------------------------------------------------------------------------------------------------
def changing_info(func):
    def wrapper(*args, **kwargs):
        if 'net' in kwargs:
            net = kwargs['net']
        else:
            net = args[0] if args else None
        logging.info(f'Net before {func.__name__}:')
        logging.info(f'    nodes:{len(net.nodes)}')
        logging.info(f'    edges:{len(net.edges)}\n')
        result = func(*args, **kwargs)
        logging.info(f'Net after {func.__name__}:')
        logging.info(f'    nodes:{len(result.nodes)}')
        logging.info(f'    edges:{len(result.edges)}\n')
        return result
    return wrapper
#-------------------------------------------------------------------------------------------------------------------------------------
def extract_gaf(gaf_path):
    columns=['DB','DB_Object_ID','DB_Object_Symbol','Qualifier','GO_ID',
         'DB_Reference','Evidence_Code','With_or_From','Aspect',
         'DB_Object_Name','DB_Object_Synonym','DB_Object_Type',
         'Taxon','Date','Assigned_By','Annotation_Extension']
    gene_annotations = []
    with open(gaf_path, 'r') as file:
        for line in file:
            # Skip lines starting with '!'
            if line.startswith('!'):
                continue
            gene_annotations.append(line.split('\t')[:-1])
    logging.debug(f'GO annotation files(goa_human.gaf) extracted.')
    return pd.DataFrame(gene_annotations, columns=columns)
#-------------------------------------------------------------------------------------------------------------------------------------
def get_nodes_at_level(net, distance):
    nodes = set(nx.ego_graph(net, 'root', radius=distance))
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))
    logging.debug(f'List of nodes at distance of {distance} from root created.')
    return list(nodes)
#-------------------------------------------------------------------------------------------------------------------------------------
@changing_info
def complete_network(G, n_hid):
    """
    Suppose you have a graph with 11 layers(based of distance from root) and you just wants 6
    layer of network. So, you will lost all 7 to 11 layers information(nodes and edges). On the
    other hand, some terminal nodes(connected to genes) are located in layers < 6. Means some 
    terminal nodes have Distance from root of less than 6. Complete_network() with coopration of 
    add_edges() make some fake node(by adding copy{number} to terminal node) to strictly make a 6 
    layer network.
    """

    net = nx.ego_graph(G, 'root', radius=n_hid)
    terminal_nodes = [n for n, d in net.out_degree() if d == 0]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(net, source='root', target=node))
        if distance <= n_hid:
            diff = n_hid - distance + 1
            net = add_edges(net, node, diff)
    return net
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
def load_maps(path):
    # with pd.HDFStore(path, 'r') as store:
    #     maps = {key: store[key] for key in store.keys()}
    
    with open(path, 'rb') as file:
        maps = pickle.load(file)
    return maps
#-------------------------------------------------------------------------------------------------------------------------------------
def cytoscape(graph, file_name='graph.json'):
    cytoscape_json = nx.cytoscape_data(graph)
    # Save the JSON to a file
    with open(file_name, 'w') as json_file:
        json.dump(cytoscape_json, json_file)
#-------------------------------------------------------------------------------------------------------------------------------------
class GO():
    def __init__(self, obo_path=obo_path, gaf_path=gaf_path):
        self.graph = obonet.read_obo(obo_path)
        logging.debug(f'GO graph(go.obo) readed.')
        self.goa = extract_gaf(gaf_path)
        self.graph_total = self.graph.reverse().copy()
        self.graph_total.name = 'graph_total'
        roots = [n for n, d in self.graph_total.in_degree if d == 0]
        edges = [('root', n) for n in roots]
        self.graph_total.add_edges_from(edges)
        logging.info('Total graph:')
        logging.info(f'    nodes:{len(self.graph_total.nodes)}')
        logging.info(f'    edges:{len(self.graph_total.edges)}')
        self.layers_list = self.get_layers_list()
        self.node_edge_info()
        self.get_class_graph()
    #=================================================================================================================================
    def get_layers_list(self):
        file_name = os.path.join(cache_dir, 'layers_list.txt')

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if os.path.exists(file_name):
            logging.debug(f"File '{file_name}' already exists!")
            layers_list = []
            with open(file_name, 'r') as file:
                for line in file:
                    names = line.strip().split(',')
                    layers_list.append(names)
            return layers_list
        else:
            logging.debug(f"File '{file_name}' is not exists.")
            layers_list = []
            for i in range(13):
                layers_list.append(get_nodes_at_level(self.graph_total, i))
            with open(file_name, 'w') as file:
                for l_list in layers_list:
                    file.write(','.join(l_list) + '\n')
            logging.debug(f"File '{file_name}' is created.")
            return layers_list
    #=================================================================================================================================
    def node_edge_info(self):

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if os.path.exists(os.path.join(cache_dir,'go_term_edges.csv')):
            self.go_term_edges = pd.read_csv(os.path.join(cache_dir, 'go_term_edges.csv'))
            self.go_gene_edges = pd.read_csv(os.path.join(cache_dir, 'go_gene_edges.csv'))
            self.go_node = pd.read_csv(os.path.join(cache_dir, 'go_node.csv'))
            return 
        
        ## GO term edges
        child = []
        parent = []
        typee = []
        for c, p, t in list(self.graph.edges):
            child.append(c)
            parent.append(p)
            typee.append(t)
        self.go_term_edges = pd.DataFrame(data={'Child':child, 'Parent':parent, 'Type':typee})
        
        ## GO gene edges
        col = {'DB_Object_Symbol':'Child', 'GO_ID':'Parent',
               'Qualifier':'Qualifier', 'Evidence_Code':'Evidence',
               'DB_Object_Name':'gene_name', 'DB_Object_Synonym':'gene_synonym'}
        self.go_gene_edges = self.goa.loc[:,col.keys()]
        self.go_gene_edges.rename(columns=col, inplace=True)
        self.go_gene_edges = self.go_gene_edges[self.go_gene_edges.Child != '']

        ## Node informarion
        GO_ID = []
        GO_name = []
        GO_class = []
        Distance_from_root = []
        for layer_number, layer in enumerate(self.layers_list[1:]):
            for node in layer:
                GO_ID.append(node)
                node_info = self.graph_total.nodes[node]
                GO_name.append(node_info['name'])
                GO_class.append(node_info['namespace'])
                Distance_from_root.append(layer_number)
        self.go_node = pd.DataFrame(data={'GO_ID':GO_ID, 'GO_name':GO_name, 'GO_class':GO_class,
                                          'Distance_from_root':Distance_from_root})
        
        self.go_term_edges.to_csv(os.path.join(cache_dir, 'go_term_edges.csv'), index=0)
        self.go_gene_edges.to_csv(os.path.join(cache_dir, 'go_gene_edges.csv'),index=0)
        self.go_node.to_csv(os.path.join(cache_dir, 'go_node.csv'),index=0)
        logging.debug(f'go_term_edges.csv, go_gene_edges.csv, and go_node.csv is created.')
    #=================================================================================================================================
    def get_class_graph(self):
        for c in self.go_node.GO_class.unique():
            nodes = self.go_node[self.go_node.GO_class == c].GO_ID.tolist()
            nodes.append('root')
            net = self.graph_total.subgraph(nodes).copy()
            net.name = f'graph_{c}'
            setattr(self, net.name, net)
            logging.info(f'{c} graph:')
            logging.info(f'    nodes:{len(getattr(self, f"graph_{c}").nodes)}')
            logging.info(f'    edges:{len(getattr(self, f"graph_{c}").edges)}')
    #=================================================================================================================================
    # @changing_info
    def rm_retrograde_edges(self, net, name):
        if name == 'total':
            df = self.go_node
        else:
            df = self.go_node[self.go_node.GO_class == name]
        node_level = dict(zip(df.GO_ID, df.Distance_from_root))
        df = self.go_term_edges.copy()
        df = df[df.Child.isin(node_level.keys())]
        df = df[df.Parent.isin(node_level.keys())]
        df['Child_level'] = [node_level[i] for i in df.Child]
        df['Parent_level'] = [node_level[i] for i in df.Parent]
        rm_edges_df = df.loc[df.Child_level <= df.Parent_level,['Child', 'Parent']]
        net.remove_edges_from(list(zip(rm_edges_df.Parent, rm_edges_df.Child)))
        return net 
    #=================================================================================================================================
    def get_layers(self, net, n_hid):
        layers_dict = get_layers_from_net(net, n_hid)
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]
        temp = [i.split('_')[0] for i in terminal_nodes]
        genes_df = self.go_gene_edges.loc[:,['Child', 'Parent']]
        genes_df = genes_df[genes_df.Parent.isin(temp)]
        genes_df = genes_df[~genes_df.duplicated()]
        genes_dict = genes_df.groupby('Parent')['Child'].apply(list).to_dict()
        layers_dict.append(genes_dict)
        logging.debug(f'Layers dict is created.')
        return layers_dict
    #=================================================================================================================================
    def create_network_adj(self, n_hid, class_wise=False):

        if not os.path.exists(adj_matrix_dir):
            os.makedirs(adj_matrix_dir)
        nets = {}
        file_paths = {}
        if class_wise:
            for c in self.go_node.GO_class.unique():
                file_paths[c] = os.path.join(adj_matrix_dir, f'maps_hid_num_{n_hid}_{c}.pkl')
                nets[c] = getattr(self, f'graph_{c}').copy()
        else:
            nets['total'] = self.graph_total.copy()
            file_paths['total'] = os.path.join(adj_matrix_dir, f'maps_hid_num_{n_hid}_total.pkl')
        
        for p in file_paths.values():
            finish=False
            if os.path.exists(p):
                logging.info(f"File '{p}' already exists!")
                logging.info("Load data using the load_maps function.")
                finish=True
                continue
        if finish:
            return
        
        for key, net in nets.items():
            maps = {}
            maps1 = {}
            logging.info(f'{"#"*20}{key.upper()} GRAPH WITH {n_hid} HIDDEN LAYERS{"#"*20}')
            if key == 'total': n_level = n_hid
            else: n_level = n_hid+1
            net = self.rm_retrograde_edges(net=net, name=key)
            net = complete_network(net, n_level)
            logging.debug(f'Complete network is created.')
            layers_dict = self.get_layers(net, n_level)
            # 1 all genes in row of first connection map
            # filtering_index = sorted(self.goa.DB_Object_Symbol.unique())
            # 2 actual connected genes in row of first connecttion map
            filtering_index = sorted(set([i for _, v in layers_dict[-1].items() for i in v]))
            for i, layer in enumerate(layers_dict[::-1]):
                mapp = get_map_from_layer(layer)
                filter_df = pd.DataFrame(index=filtering_index)
                filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
                filtered_map = filtered_map.fillna(0)
                filtering_index = filtered_map.columns
                maps[f'map_{i+1}'] = filtered_map.astype('bool')########### for dropping rows and columns with all zero values
            
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

            with open(file_paths[key], 'wb') as file:
                pickle.dump(maps1, file)
            # with pd.HDFStore(file_paths[key], 'w') as store:
            #     for k, m in maps1.items():
            #         store[k] = m
        return maps1




