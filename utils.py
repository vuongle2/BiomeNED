#import torch
import pickle
import csv
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt
import csv

def write_matrix_to_csv(a, fn):
    with open(fn, 'w', newline='') as csvfile:
        thiswriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for ra in a:
            thiswriter.writerow(ra)
def consistency_index(w1, w2, threshold = 0.0):
    """
    w1 and w2 are two matrix or two list of a matrix
    :param w1:
    :param w2:
    :return: iou
    """
    if isinstance(w1, list) and isinstance(w2,list):
        if (len(w1)) == 0 and (len(w2)) == 0:
            return 1
        fw1 = []
        fw2 = []
        for ew1, ew2 in zip(w1,w2):
           fw1.append(ew1.flatten())
           fw2.append(ew2.flatten())

        w1 = np.hstack(fw1)
        w2 = np.hstack(fw2)
    #convert to binary
    bw1 = np.absolute(w1) >threshold
    bw2 = np.absolute(w2) > threshold
    i = np.multiply(bw1,bw2) #intersection
    o = np.where(bw1+bw2) #union
    return np.count_nonzero(i)/float(np.count_nonzero(o)) #iou

def get_subgraph(nodes, weights, kept_final_node = None):
    # Get the subgraph of some final node, prune out the parts that are not interested,
    # including middle nodes that are not connected to either end
    if kept_final_node is None:
        kept_final_node = list(range(len(nodes[-1])))

    subnodes = [[] for _ in range(len(nodes))]
    subweights = [[] for _ in range(len(weights))]
    subnodes[-1] = [nodes[-1][node_i] for node_i in kept_final_node] # argsort is index of best nodes, not names

    kept_dest_nodes = kept_final_node
    #bottom up prune: anything does not connect to needed final
    for lay_i in range(len(nodes) - 2, -1, -1):  # each layer before the last
        layer_weight = weights[lay_i]
        kept_source_nodes = []

        for node_i_s in range(len(nodes[lay_i])):  # each source node
            if np.count_nonzero(layer_weight[node_i_s, kept_dest_nodes]) >0:  # keep the node
                    kept_source_nodes.append(node_i_s)
        kept_source_nodes = sorted(list(set(kept_source_nodes)))  # unique, sorted
        assert len(kept_source_nodes) != 0, "no node left"
        #transfer name
        subnodes[lay_i] = [nodes[lay_i][node_i] for node_i in kept_source_nodes]
        subweights[lay_i] = layer_weight[np.array(kept_source_nodes)[:, None], np.array(kept_dest_nodes)]
        kept_dest_nodes = kept_source_nodes
    """ the other direction should be done through flipping graph
    subsubnodes = [[] for _ in range(len(subnodes))]
    subsubweights = [[] for _ in range(len(subweights))]

    subsubnodes[0] = subnodes[0] #top layer stays
    kept_source_nodes = list(range(len(subnodes[0])))
    #top down: some may still connect the final but loss connection to beginning, therefore unwanted
    for lay_i in range(1, len(nodes)):  # each layer after the first
        layer_weight = subweights[lay_i]
        kept_dest_nodes = []
        for node_i_d in range(len(subnodes[lay_i])):  # each dest node
            for node_i_s in range(len(subnodes[lay_i -1])):  # each source node
                if node_i_s in kept_source_nodes and layer_weight[node_i_s, node_i_d] != 0:  # keep the node
                    kept_dest_nodes.append(node_i_d)
        kept_dest_nodes = sorted(list(set(kept_dest_nodes)))  # unique, sorted
        assert (len(kept_dest_nodes) != 0, "no node left")
        #transfer name
        subsubnodes[lay_i] = [subnodes[lay_i][node_i] for node_i in kept_dest_nodes]
        subsubweights[lay_i] = layer_weight[np.array(kept_source_nodes)[:, None], np.array(kept_dest_nodes)]
        kept_source_nodes = kept_dest_nodes
    """
    return (subnodes, subweights)

def prune_subgraph(nodes, weights, kept_final_node):
    bot_up_nodes, bot_up_weights = get_subgraph(nodes, weights, kept_final_node)
    bot_up_flip_nodes, bot_up_flip_weights = flip_graph(bot_up_nodes, bot_up_weights)
    top_down_flip_nodes, top_down_flip_weights = get_subgraph(bot_up_flip_nodes, bot_up_flip_weights)
    top_down_nodes, top_down_weights = flip_graph(top_down_flip_nodes, top_down_flip_weights)
    return (top_down_nodes, top_down_weights)

def flip_graph(nodes, weights):
    flip_nodes = [nodes[layer] for layer in range(len(nodes)-1, -1, -1)]
    flip_weights = [weights[layer].T for layer in range(len(weights) - 1, -1, -1)]
    return (flip_nodes, flip_weights)


def draw_weight_graph(node_names, link_weights, file_name, Name1=None, Name2=None, x1_DA=None, x2_DA=None, z_DA=None):
    """
    Draw a graph of weights of a feed forward model
    :param node_names: list of layers x nodes of each layer
    :param link_weights: list (layers-1) of 2d numpy array for weights of each links
    :param file_name: write to
    :return:
    """
    node_colors = ["#DDDDDD", "#FDE725", "#21908C"]

    def get_node_color(node_id, DA):
        if DA is not None and float(DA[int(node_id)]["FDR"]) < 0.05:
            if float(DA[int(node_id)]["Control"]) > float(DA[int(node_id)]["IBD"]):
                node_color = node_colors[1]
            else:
                node_color = node_colors[2]
        else:
            node_color = node_colors[0]
        return node_color
    u = Digraph(file_name.split('/')[-1], graph_attr={'nodesep': '0.5','ranksep': '5', 'rankdir':'LR'})

    for l, layer in enumerate(node_names):
        for n, node_id in enumerate(layer):
            node_color = node_colors[0] #default
            if l == 0:
                node_name = Name1[int(node_id)] if Name1 is not None else str(int(node_id)+1)
                if x2_DA is not None:# first layer
                    node_color = get_node_color(node_id, x1_DA)
            elif l == len(node_names)-1:
                node_name = Name2[int(node_id)] if Name2 is not None else str(int(node_id)+1)
                if x2_DA is not None: #last layer
                    node_color = get_node_color(node_id, x2_DA)
            else: #middle
                if z_DA is not None:
                    node_color = get_node_color(node_id, z_DA)
                node_name = str(int(node_id)+1) # +1 so that z indexes start with 1
            u.node("%d_%d"%(l,n), label=str(node_name), fillcolor=node_color, style="filled")

    for l, layer_weights in enumerate(link_weights):
        maxw = np.max(np.abs(layer_weights))
        minw = np.min(np.abs(layer_weights))
        for b, bn in enumerate(node_names[l]):
            for e, en in enumerate(node_names[l+1]):
                w = layer_weights[b,e]
                if w!=0.0:
                    intensity = int((abs(w)-minw)/maxw *50)# range from 1 -100
                    u.edge("%d_%d"%(l,b), "%d_%d"%(l+1,e), color="Gray%d"%(intensity), arrowhead=None, arrowtail=None)

    u.render(filename=file_name,format='png')

def read_csv_table_raw(filename):
    """
    Assume the csv has the first row to be col names, first col are row name, rest are numbers
    :param filename:
    :return: (col_name, data)
    """
    with open(filename,'r') as f:
        spamreader = csv.reader(f, delimiter=',', quotechar='"')
        data_list = []
        for r, row in enumerate(spamreader):
            if r ==0:
                col_name = row # first cell is empty
            else:
                d = {}
                for i, item in enumerate(row):
                    d[col_name[i]] = item
                data_list.append(d)
        output = (col_name, data_list)
    return output

def read_csv_table(filename, output_dict=False):
    """
    Assume the csv has the first row to be col names, first col are row name, rest are numbers
    :param filename:
    :return:
    if not output_dict:
    (row_name, col_name, data as np array)
    if output_dict:
    [{"row_name":row_name, col_name: col_value}]
    """
    with open(filename,'r') as f:
        spamreader = csv.reader(f, delimiter=',', quotechar='"')
        data_list = []
        out_dict = {}
        row_name = []
        for r, row in enumerate(spamreader):
            if r ==0:
                col_name = row[1:] # first cell is empty
            else:
                row_name.append(row[0])
                if output_dict:
                    one_out_dict = {}
                    for i in range(len(row) - 1):
                        one_out_dict[col_name[i]] = row[i + 1]
                    out_dict[row[0]] = one_out_dict
                else:
                    data_list.append(np.array(row[1:]))

    if output_dict:
        output = out_dict
    else:
        data = np.vstack(data_list)
        output = (row_name, col_name, data)
    return output


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot
"""
def load_data(data_file):
    with open(data_file,"rb") as df:
        content = pickle.load(df)

    row_name_met, col_name_met, met, met_W, met_H, row_name_mic, col_name_mic, mic, diagnosis = content
    pathway_df = data['pathway_df']
    pathway_df['PAM50'] = pathway_df['PAM50'].apply(lambda x: c2i[x])
    train_idxs = data['train_idxs']
    test_idxs = data['test_idxs']

    cols = pathway_df.columns[:-2]

    X_train = pathway_df.loc[train_idxs][cols].values
    y_train = pathway_df.loc[train_idxs]['PAM50'].values
    X_test = pathway_df.loc[test_idxs][cols].values
    y_test = pathway_df.loc[test_idxs]['PAM50'].values

    # normalize per person
    means = X_train.mean(1)
    X_train = X_train/means[:, None]
    means = X_test.mean(1)
    X_test = X_test/means[:, None]

    # scale
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train = (X_train-means) / stds
    X_test = (X_test-means) / stds

    return (X_train, y_train), (X_test, y_test)

"""