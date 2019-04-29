#import torch
import pickle
import csv
import numpy as np


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