import os
import time
import argparse
import csv
import numpy as np
import pickle
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from utils import read_csv_table



def main(args):
    #data_type = "clr"# log(x) - mean(log(x))
    data_type = "0-1" #x/sum(x)

    met_dr = "PCA"
    #met_dr = "NMF"

    metabolite_file = "/data/BioHealth/IBD/ibd-x1-met-zComp-{}.csv".format(data_type)
    bacteria_file   = "/data/BioHealth/IBD/ibd-x2-mic-zComp-{}.csv".format(data_type)
    bacteria_taxa_file = "/data/BioHealth/IBD/mic-taxa-itis-v2.csv"
    label_file      = "/data/BioHealth/IBD/ibd-y.csv"
    output_file ="/data/BioHealth/IBD/ibd_{}.pkl".format(data_type)

    #Process metabolite: NMF
    subject_ids, met_ids, met_fea = read_csv_table(metabolite_file)
    met_fea = met_fea.astype(np.float)
    #undo preprocessing:
    #x1 = np.exp(x1) # from log(x)- mean(log(x) to x/(x1*x2*...)^(1/n), centered (product) at 1

    if data_type == "0-1":
        met_fea = np.log(met_fea)

    if met_dr == "NMF":
        #NMF - TODO: better way to make data positive
        met_fea = met_fea - met_fea.min(axis=0)
        nmf = NMF(n_components=20, init='random', random_state=1)
        met_W = nmf.fit_transform(met_fea)
        met_H = nmf.components_
    elif met_dr == "PCA":
        #PCA
        pca = PCA(copy=True, iterated_power='auto', n_components=20, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
        met_W = pca.fit_transform(met_fea)
        met_H = pca.components_

    # Process bacteria
    row_name_bac, bac_ids, bac_fea = read_csv_table(bacteria_file)
    assert (row_name_bac == subject_ids)
    bac_fea = bac_fea.astype(np.float)
    if data_type == "0-1":
        bac_fea = np.log(bac_fea)
    bac_taxa = read_csv_table(bacteria_taxa_file,output_dict=True)
    #Get a list of unique genuses
    genuses = set()
    for bac_id, this_bac_taxa in bac_taxa.items():
        genuses.add(this_bac_taxa["genus"])
    genuses = list(genuses)
    # assign a list of col ind to each genus ind
    gen_to_bac = [[] for i in range(len(genuses))]

    for bac_col, bac_id in enumerate(bac_ids):
        gen_to_bac[genuses.index(bac_taxa[bac_id]["genus"])].append(bac_col)

    genus_fea = np.zeros((len(subject_ids),len(genuses)))
    for i in range(len(subject_ids)):
        for g in range(len(genuses)):
            if data_type =="clr":
                genus_fea[i,g] = np.mean(bac_fea[i, gen_to_bac[g]])
            elif data_type =="0-1":
                genus_fea[i, g] = np.sum(bac_fea[i, gen_to_bac[g]])

    # Process y
    row_name_lab, col_name_lab, labels = read_csv_table(label_file)
    assert (row_name_bac == subject_ids)
    diagnosis = labels[:,1]
    diag_encode = {
        "Control":0,
        "CD":1,
        "UC":2
    }

    train_idx = []
    val_idx = []
    for i, row_name in enumerate(row_name_lab):
        if "Validation" in row_name_lab[i]:
            val_idx.append(i)
        else:
            train_idx.append(i)
    diagnosis = [diag_encode[d] for d in diagnosis]
    diagnosis = np.array(diagnosis)
    # TODO:use tree structure or binning to preprocess

    out_data = {
            "subject_ids":subject_ids,
            "met_fea":met_fea,
            "met_W":met_W,
            "met_H":met_H,
            "bac_ids":bac_ids,
            "bac_fea":bac_fea,
            "genuses": genuses,
            "genus_fea":genus_fea,
            "train_idx":train_idx,
            "val_idx":val_idx,
            "diagnosis":diagnosis}
    with open(output_file,'wb') as of:
        pickle.dump(out_data, of, protocol=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
