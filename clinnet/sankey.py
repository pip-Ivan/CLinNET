import os 
import logging 
import re
import matplotlib
import json
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from nltk.stem.porter import PorterStemmer
#from IPython.display import display
import plotly.io as pio

# Set renderer for Jupyter Notebook
pio.renderers.default = 'notebook'


#-------------------------------------------------------------------------------------------------------------------------------------
# 1 get edge info from shap graph 
def get_edge_df(graph, sv_norm, gene_status, top_n=10):
    feature_dict = {'source_id':[], 'target_id':[],
                    'source_name':[], 'target_name':[],
                    'source_layer_index':[], 'target_layer_index':[],
                    'source_rank':[], 'target_rank':[],
                    'source_sv_norm':[], 'target_sv_norm':[],
                    'source_layer_name':[], 'target_layer_name':[],
                    'source_layer_id':[], 'target_layer_id':[],
                    'source_network':[], 'target_network':[]}
    for pre_layer, post_layer in graph.edges:

        pre_node  = graph.nodes[pre_layer]
        post_node = graph.nodes[post_layer]

        # just to reduce over calculation
        pre_node_sv_norm  = np.abs(sv_norm[pre_layer]).mean(0)
        post_node_sv_norm = np.abs(sv_norm[post_layer]).mean(0)
        
        cm = graph.edges[(pre_layer, post_layer)]['cm']

        pre_node_top_idx  = np.where(pre_node['rank_index']<top_n)[0]
        post_node_top_idx = np.where(post_node['rank_index']<top_n)[0]

        post_node_res_idx = np.where(cm.iloc[pre_node_top_idx, :].any(axis=0).to_numpy())[0]
        pre_node_res_idx  = np.where(cm.iloc[:, post_node_top_idx].any(axis=1).to_numpy())[0]

        pre_node_idx  = np.unique(np.concatenate((pre_node_top_idx, pre_node_res_idx), axis=0))
        post_node_idx = np.unique(np.concatenate((post_node_top_idx, post_node_res_idx), axis=0))

        new_cm = cm.iloc[pre_node_idx, post_node_idx]
        for pre_i, post_i in zip(*np.nonzero(new_cm.values)):
            feature_dict['source_id'].append(pre_node['feature_id'][pre_node_idx[pre_i]])
            feature_dict['target_id'].append(post_node['feature_id'][post_node_idx[post_i]])
            
            feature_dict['source_name'].append(pre_node['feature_name'][pre_node_idx[pre_i]])
            feature_dict['target_name'].append(post_node['feature_name'][post_node_idx[post_i]])

            feature_dict['source_layer_index'].append(pre_node_idx[pre_i])
            feature_dict['target_layer_index'].append(post_node_idx[post_i])     
            
            feature_dict['source_rank'].append(pre_node['rank_index'][pre_node_idx[pre_i]])
            feature_dict['target_rank'].append(post_node['rank_index'][post_node_idx[post_i]])
            
            feature_dict['source_sv_norm'].append(pre_node_sv_norm[pre_node_idx[pre_i]])
            feature_dict['target_sv_norm'].append(post_node_sv_norm[post_node_idx[post_i]])
            
            feature_dict['source_layer_name'].append(pre_layer)            
            feature_dict['target_layer_name'].append(post_layer)

            feature_dict['source_network'].append(pre_node['network']) 
            feature_dict['target_network'].append(post_node['network'])

            feature_dict['source_layer_id'].append(pre_node['layer_id'])
            feature_dict['target_layer_id'].append(post_node['layer_id'])
    
    edge_df = pd.DataFrame(data=feature_dict)
    for gs in gene_status:
        edge_df['source_id']   = edge_df['source_id'].apply(lambda x: gs if x.endswith(gs) else x)
        edge_df['source_name'] = edge_df['source_name'].apply(lambda x: gs if x.endswith(gs) else x)

    return edge_df
#-------------------------------------------------------------------------------------------------------------------------------------
# 2 get node info from edge info
def get_node_df(edge_df):
    # get dataframe of source and target
    source_edge_df = edge_df.iloc[:,::2]
    target_edge_df = edge_df.iloc[:,1::2]
    # rename columns for concatenation
    source_edge_df.columns = [re.sub(r'^[^_]*_', '', col) for col in source_edge_df.columns]
    target_edge_df.columns = [re.sub(r'^[^_]*_', '', col) for col in target_edge_df.columns]
    node_df = pd.concat([source_edge_df, target_edge_df])
    node_df = node_df.drop_duplicates(ignore_index=True)

    # aggregate values for gs layer
    node_df = node_df.groupby(by=['id','name','layer_name', 'layer_id','network'], as_index=False).agg({'rank':'first', 'layer_index':'first', 'sv_norm':'sum'})
    # add rank to GS layer
    node_df.loc[node_df['layer_id'] == 0, 'rank'] = np.argsort(np.argsort(-node_df.loc[node_df['layer_id']==0, 'sv_norm'].to_numpy()))
    # normalization sv_norm for each layer
    layer_value_sum = dict(node_df.groupby('layer_name',as_index=False).agg({'sv_norm':'sum'}).values)
    node_df['value_norm'] = node_df.apply(lambda x: x['sv_norm']/layer_value_sum[x['layer_name']], axis=1)
    # convert non-ranked term to residual
    cols = ['id','name','rank']
    node_df.loc[:,cols] = node_df.apply(lambda x: [f"Residual_{x['layer_name']}", 'Residual',10] if x[cols[2]]>9 else list(x[cols]), axis=1).to_list()
    node_df = node_df.drop(['layer_index'], axis=1)
    # sum values for Residual
    node_df = node_df.groupby(by=['id','name','layer_name','layer_id','network'], as_index=False).agg({'rank':'min','sv_norm':'sum', 'value_norm':'sum'})
    # sort data before adding a graph index
    node_df = node_df.sort_values(by=['layer_id','rank']).reset_index(drop=True)
    # adding graph index for refering in sankey diagram
    node_df['graph_index'] = node_df.index
    # value norm for Reactome and GO would be divided by 2
    node_df['value_norm'] = node_df.apply(lambda x: x['value_norm'] if x['network'] == 'Common' else x['value_norm']/2, axis=1)
    return node_df
#-------------------------------------------------------------------------------------------------------------------------------------
# 3 abbreviate the long pathway name
def abbreviate(bio_process, gen_abbs, spe_abbs=None, max_len=40):
    if len(bio_process)<30:
        return bio_process
    
    if spe_abbs:
        if spe_abbs.get(bio_process):
            return spe_abbs[bio_process]
    
    def shorting(word, max_len):
        if len(word)<max_len:
            return word
        word = ' '.join(word.split()[:-1])
        return shorting(f'{word}.', max_len)

    stemmer = PorterStemmer()
    words = bio_process.split()
    abbreviated_words = [gen_abbs.get(stemmer.stem(word), word) for word in words]
    abbreviated_words = " ".join(abbreviated_words)
    return shorting(abbreviated_words, max_len=max_len) # [:40]  # Ensure the result is at most 40 characters
#-------------------------------------------------------------------------------------------------------------------------------------
# 4 adding residual, normalizing, dropping unnecessary columns 
def processing_edge_df(edge_df, node_df):
    # edge_df correct GS layer features' Rank 
    GS_rank_dict = dict(node_df.loc[node_df['layer_id'] == 0, ['id', 'rank']].values)
    edge_df['source_rank'] = edge_df.apply(lambda x: GS_rank_dict[x['source_id']] if x['source_layer_id']==0 else x['source_rank'], axis=1)
    # edge_df add residual and correct the rank
    cols = ['source_id','source_name','source_rank']
    edge_df.loc[:,cols] = edge_df.apply(lambda x: [f"Residual_{x['source_layer_name']}", 'Residual',10] if x[cols[2]]>9 else list(x[cols]), axis=1).to_list()
    cols = ['target_id','target_name','target_rank']
    edge_df.loc[:,cols] = edge_df.apply(lambda x: [f"Residual_{x['target_layer_name']}", 'Residual',10] if x[cols[2]]>9 else list(x[cols]), axis=1).to_list()
    # edge_df drop unnecessary columns
    edge_df = edge_df.drop(['source_layer_index','target_layer_index'], axis=1)
    edge_df = edge_df.drop_duplicates().reset_index(drop=True)
    # add value to edge_df but before it correct the norme_value by number of connection of prelayer node
    edge_df = pd.merge(left=edge_df, right=node_df.loc[:,['id','layer_name','value_norm','graph_index']], left_on=['source_id','source_layer_name'], right_on=['id', 'layer_name'])
    edge_df = edge_df.rename(columns={'graph_index': 'graph_index_source', 'value_norm':'source_value'})
    edge_df = edge_df.drop(columns=['id', 'layer_name'])
    edge_df = pd.merge(left=edge_df, right=node_df.loc[:,['id','layer_name','value_norm','graph_index']], left_on=['target_id','target_layer_name'], right_on=['id', 'layer_name'])
    edge_df = edge_df.rename(columns={'graph_index': 'graph_index_target', 'value_norm':'target_value'})
    edge_df = edge_df.drop(columns=['id', 'layer_name'])
    # calculating edge_value and add to edge_df
    edge_df['edge_value'] = 0.0
    for i in node_df.graph_index:
        temp_df = edge_df[edge_df['graph_index_source'] == i]
        edge_df.loc[temp_df.index,'edge_value'] = temp_df['source_value']*temp_df['target_value']/temp_df['target_value'].sum()    
    return edge_df
#-------------------------------------------------------------------------------------------------------------------------------------
# 5  adding abbreviation, adding name, 
def processing_node_df(edge_df, node_df, gen_abbs, spe_abbs):
    # add value form edge_df to node_df
    source_edge_df = edge_df.loc[:,['graph_index_source', 'source_layer_name', 'edge_value']]
    target_edge_df = edge_df.loc[:,['graph_index_target', 'target_layer_name', 'edge_value']]

    source_edge_df.columns = ['graph_index', 'layer_name', 'value']
    target_edge_df.columns = ['graph_index', 'layer_name', 'value']

    source_edge_df = source_edge_df.groupby(by=['graph_index', 'layer_name'], as_index=False).value.sum()
    target_edge_df = target_edge_df.groupby(by=['graph_index', 'layer_name'], as_index=False).value.sum()
    temp_df = pd.concat([source_edge_df, target_edge_df])
    temp_df = temp_df.groupby(by=['graph_index', 'layer_name'], as_index=False).value.max()
    node_df['edge_value'] = temp_df['value']
    name_dict = {'mut_important':'Mutation',
                'cnv_del':'Deletion',
                'cnv_amp':'Amplification',
                'cnv':'CNV'}
    node_df['name'] = node_df['name'].replace(name_dict)
    node_df['name_abb'] = node_df.apply(lambda x: abbreviate(x['name'], gen_abbs, spe_abbs, max_len=40) if x['layer_id']>1 else x['name'],axis=1)
    return node_df
#-------------------------------------------------------------------------------------------------------------------------------------
# 6 location of each node in plot 
def get_x_y(node_df):
    # x
    x = node_df.layer_id/(node_df.layer_id.nunique()-1)
    x = [.001 if v==0 else .999 if v==1 else v for v in x]
    node_df['x'] = x
    # y
    node_df['y'] = 0.0
    for i in node_df['layer_id'].unique():
        df = node_df.loc[node_df['layer_id'] == i,['id', 'network','rank','edge_value']]
        df = df.sort_values(by=['network', 'rank'])
        df['n_value'] =  (df.edge_value.to_numpy() / df.edge_value.sum())
        df['n_value/2'] =  df['n_value']/2
        df['value_cumsum'] = df.n_value.cumsum()
        df['y'] = df['value_cumsum'] - df['n_value/2']
        node_df.loc[df.index,'y'] = df['y']
        
    node_df.loc[node_df['id'] == 'root', 'y'] = 0.5
    return node_df
#-------------------------------------------------------------------------------------------------------------------------------------
# 7 color of each node 
def get_node_color(node_df, GS_color, cmap='OrRd'):
    vmin = 0
    vmax = 100
    GS_color = [f'rgba({c[0]},{c[1]},{c[2]},{c[3]})' for c in GS_color]
    node_cmap = cmap
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    cmap = plt.cm.ScalarMappable(norm = norm, cmap = node_cmap)
    color = [cmap.to_rgba(number, alpha=0.75) for number in range(100,0,-node_df['rank'].max())]
    node_df['color'] = node_df.apply(lambda x: (0.9254, 0.9254, 0.9254, .9254) if x['name'] == 'Residual' else color[x['rank']] ,axis=1) # (0.9254, 0.9254, 0.9254, .75)
    node_df['color'] = node_df.apply(lambda x: f"rgba({x['color'][0]*255},{x['color'][1]*255},{x['color'][2]*255},{x['color'][3]})", axis=1)
    df = node_df[node_df['layer_id']==0]
    df.loc[:,'color'] = GS_color
    node_df.loc[df.index, 'color'] = df['color']
    return node_df
#-------------------------------------------------------------------------------------------------------------------------------------
# 8 color of edges 
def get_edge_color(edge_df, node_df):
    color_map = dict(zip(node_df['id'], node_df['color']))    
    edge_df['color'] = edge_df.apply(lambda x: color_map[x['source_id']].replace('.75)','.20)'), axis=1)
    return edge_df
#-------------------------------------------------------------------------------------------------------------------------------------
# 9 prepare node_df in sankey format
def prepare_nodes(node_df, use_abb=False):
    x = node_df['x']
    y = node_df['y']
    if use_abb: labels = node_df['name_abb'] # add abb_name
    else: labels = node_df['id'] # add abb_name
    colors = node_df['color']
    nodes = dict(pad=0, thickness=10, line=dict(color="white", width=0),
                 label=list(labels), color=list(colors), x=list(x), y=list(y))
    return nodes
#-------------------------------------------------------------------------------------------------------------------------------------
# 10 prepare edge_df in sankey format
def prepare_links(edge_df, value_column = 'normـvalue'): 
    sources = edge_df['graph_index_source']
    targets = edge_df['graph_index_target']
    values = edge_df[value_column]
    color_edge = edge_df['color']
    links = dict(source=list(sources), target=list(targets),
                 value=list(values), color=list(color_edge))
    return links
#-------------------------------------------------------------------------------------------------------------------------------------
# 11 plot sankey and save in html
def plot_sankey(nodes, links, saving_name,display=True):
    
    scale = 1
    width = 2100*scale
    height = 0.621 * width
    data_trace = dict(type='sankey', arrangement='fixed',
                      domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]),
                      orientation="h",
                      valueformat=".0f",
                      node=dict(pad=1*scale, thickness=20*scale,
                                line=dict(color="white", width=1*scale),
                                label=nodes['label'],
                                x=nodes['x'],
                                y=nodes['y'],
                                color=nodes['color']),
                      link=dict(source=links['source'],
                                target=links['target'],
                                value=links['value'],
                                color=links['color']))

    layout = dict(height=height,
                  width=width,
                  margin=go.layout.Margin(l=0,  # left margin
                                          r=0,  # right margin
                                          b=0,  # bottom margin
                                          t=0,  # top margin
                                          ),
                  font=dict(size=18*scale, family='Times New Roman',color='black'))
    
    fig = dict(data=[data_trace], layout=layout)
    fig = go.Figure(fig)

    # Display the figure in the Jupyter notebook
    if display:
        # Display the figure
        fig.show()
    
    # fig.write_image somtimes take a long time. I don't know why. 
    # fig.write_image(f'{saving_name}.png')
    fig.write_html(f'{saving_name}.html')
    
#-------------------------------------------------------------------------------------------------------------------------------------
# 12 get color for gene_status
def get_GS_color(n_status):
    if n_status == 2:
        colors = [[255, 152, 101, 0.75],
                  [ 80, 104, 164, 0.75]]
    elif n_status == 3:
        colors = [[255, 152, 101, 0.75],
                  [156, 192, 208, 0.75],
                  [ 80, 104, 164, 0.75]]
    else:
        colors = [[202, 231, 196, 1],
                  [246, 247, 194, 1],
                  [252, 225, 177, 1],
                  [249, 185, 160, 1],
                  [251, 189, 204, 1],
                  [239, 195, 219, 1],
                  [211, 186, 219, 1],
                  [194, 183, 219, 1],
                  [189, 195, 227, 1],
                  [189, 195, 227, 1],
                  [189, 195, 227, 1]]
        idx = np.random.choice(list(range(len(colors))), n_status, replace=False)
        colors[idx]
    return colors
#-------------------------------------------------------------------------------------------------------------------------------------
# 13 Sankey 
class Sankey:
    def __init__(self, graph, interpret_dir, sv_norm, gene_status, top_n=10, saving_dir=None):
        self.top_n = top_n
        self.graph = graph
        self.interpret_dir = interpret_dir
        self.sankey_dir = os.path.join(interpret_dir, 'sankey') if saving_dir is None else saving_dir
        os.makedirs(self.sankey_dir, exist_ok=True)
        self.sv_norm = sv_norm
        self.gene_status = gene_status
        GS_colors = get_GS_color(len(gene_status))
        self.gen_abbs, self.spe_abbs = self.read_abbs()
        self.edge_df = get_edge_df(graph, sv_norm, gene_status, top_n=10)
        self.node_df = get_node_df(self.edge_df)
        self.edge_df = processing_edge_df(self.edge_df, self.node_df)
        self.node_df = processing_node_df(self.edge_df, self.node_df, self.gen_abbs, self.spe_abbs)
        self.node_df = get_x_y(self.node_df)
        self.node_df = get_node_color(self.node_df, GS_color=GS_colors)
        self.edge_df = get_edge_color(self.edge_df, self.node_df)
        self.saving_node_edge_df()
    #=================================================================================================================================
    def prepare_graph(self, shap, tissue_name): # don't need so far
        G = deepcopy(shap.graph)
        layer_to_remove = []
        for n in G.nodes:
            if not n.endswith(tissue_name.replace(' ','_')) and n != 'root':
                layer_to_remove.append(n)
        G.remove_nodes_from(layer_to_remove)
        return G
    #=================================================================================================================================
    def read_abbs(self): # convert it to function
        with open('clinnet/abbreviations.txt', 'r') as file:
            file_contents = file.read()
            gen_abbs = json.loads(file_contents)
        spe_abb_file = os.path.join(self.sankey_dir, f'abbreviations.txt')
        if os.path.exists(spe_abb_file):
            with open(spe_abb_file, 'r') as file:
                file_contents = file.read()
                spe_abbs = json.loads(file_contents)
        else:
            spe_abbs = None
        return gen_abbs, spe_abbs
    #=================================================================================================================================
    def saving_node_edge_df(self):
        self.node_df.to_csv(os.path.join(self.sankey_dir, f'node_df.csv'), index=False)
        self.edge_df.to_csv(os.path.join(self.sankey_dir, f'edge_df.csv'), index=False)
    #=================================================================================================================================
    def read_node_edge_df(self, node_df_path, edge_df_path):
        self.node_df = pd.read_csv(node_df_path)
        self.edge_df = pd.read_csv(edge_df_path)
    #=================================================================================================================================
    def get_node_name(self, gt=30):
        return [n for n in self.node_df['name'].to_list() if len(n)>gt]
    #=================================================================================================================================
    def set_node_abb(self, abb_list):
        self.node_df['name_abb'] = abb_list
    #=================================================================================================================================
    def get_abbs(self, gt=30):
        abbs = self.node_df.apply(lambda x: abbreviate(x['name'], self.abbs, max_len=200), axis=1)
        name = self.node_df['name'].tolist()
        return [(n,a) for n, a in zip(name, abbs) if len(n)>gt]
    #=================================================================================================================================
    def plot_sankey(self, use_abb=False,display=True):
        sankey_path = os.path.join(self.sankey_dir, f'Sankey')
        nodes = prepare_nodes(self.node_df, use_abb=use_abb)
        links = prepare_links(self.edge_df, value_column='edge_value')
        plot_sankey(nodes, links, saving_name=sankey_path,display=display)
        logging.info(f"Sankey diagram saved: {sankey_path}")
    #=================================================================================================================================