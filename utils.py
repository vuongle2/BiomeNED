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
    w1 and w2 are two matrix
    :param w1:
    :param w2:
    :return: iou
    """
    #convert to binary
    bw1 = np.find(w1.abs() >threshold)
    bw2 = np.find(w2.abs() > threshold)
    i = np.multiply(bw1,bw2) #intersection
    o = np.find(bw1+bw2) #union
    return np.count_nonzero(i)/float(np.count_nonzero(o)) #iou

def get_subgraph(nodes, weights, kept_final_node):
    # Get the subgraph of topk
    subnodes = [[] for _ in range(len(nodes))]
    subweights = [[] for _ in range(len(weights))]
    subnodes[-1] = kept_final_node  # argsort is index of best nodes, not names
    for lay_i in range(len(nodes) - 2, -1, -1):  # each layer before the last
        layer_weight = weights[lay_i]
        kept_source_nodes = []
        for node_i_s in range(len(nodes[lay_i])):  # each source node
            for node_i_d in range(len(nodes[lay_i + 1])):  # each destination node
                if node_i_d in subnodes[lay_i + 1] and layer_weight[node_i_s, node_i_d] != 0:  # keep the node
                    kept_source_nodes.append(node_i_s)
        kept_source_nodes = sorted(list(set(kept_source_nodes)))  # unique, sorted
        assert (len(kept_source_nodes) != 0, "no node left")
        subnodes[lay_i] = [nodes[lay_i][node_i] for node_i in kept_source_nodes]
        subweights[lay_i] = layer_weight[np.array(subnodes[lay_i])[:, None], np.array(subnodes[lay_i + 1])]
    return (subnodes, subweights)

def draw_weight_graph(node_names, link_weights, file_name):
    """
    Draw a graph of weights of a feed forward model
    :param node_names: list of layers x nodes of each layer
    :param link_weights: list (layers-1) of 2d numpy array for weights of each links
    :param file_name: write to
    :return:
    """
    u = Digraph(file_name.split('/')[-1], graph_attr={'nodesep': '1','ranksep': '5'})
    for l, layer in enumerate(node_names):
        for n, node_name in enumerate(layer):
            u.node("%d_%d"%(l,n), label=str(node_name), color='lightblue2')

    for l, layer_weights in enumerate(link_weights):
        maxw = np.max(np.abs(layer_weights))
        minw = np.min(np.abs(layer_weights))
        for b, bn in enumerate(node_names[l]):
            for e, en in enumerate(node_names[l+1]):
                w = layer_weights[b,e]
                if w!=0.0:
                    intensity = int((abs(w)-minw)/maxw *100)# range from 1 -100
                    u.edge("%d_%d"%(l,b), "%d_%d"%(l+1,e), color="Gray%d"%(intensity), arrowhead=None, arrowtail=None)

    u.render(filename=file_name,format='png')


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