import os
import sys
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
#from pyrcca import rcca
import logging
from sklearn import metrics
#import tensorboard_logger as tl

from sklearn.cross_decomposition import CCA
from sklearn import svm
DATA_ROOT = "/data/BioHealth/IBD"

parser = argparse.ArgumentParser()
FORMAT = '[%(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def main(args):
    data_path = os.path.join(DATA_ROOT, "ibd_{}.pkl".format(args.data_type))
    with open(data_path,"rb") as df:
        content = pickle.load(df)

    model_alias = 'ShallowBiome_%s+%s_%s+fea1_%s+fea2_%s+%s' % (
        args.model,
        args.dataset_name, args.data_type,
        args.fea1,args.fea2,
        args.extra)
    print(model_alias)

    X1 = content[args.fea1] # n x m1
    X2 = content[args.fea2] # n x m2
    Y = content["diagnosis"]

    #Suppress to two classes
    for i in range(len(Y)):
        if Y[i] !=0:
            Y[i] =1

    y_train_keys = [-1,1]#+list(set(Y)) #-1 means all data
    y_ind  = {}
    y_ind[-1] = range(len(Y))
    for y_val in set(Y):
        y_ind[y_val] = [i for i in range(len(Y)) if Y[i] == y_val]

    #Split train and val
    val_ind = content["val_idx"]
    train_ind = content["train_idx"]
    X1_train = X1[train_ind]
    X1_val = X1[val_ind]
    X2_train = X2[train_ind]
    X2_val = X2[val_ind]
    y_train = Y[train_ind]
    y_val = Y[val_ind]

    #X1 = np.random.rand(X1.shape[0], X1.shape[1])
    #X2 = np.random.rand(X2.shape[0], X2.shape[1])


    CENTER_AND_SCALE = 0 # normally not because sklearn does it inside cca
    NUM_PLOT_ROW = 2
    NUM_PLOT_COL = 3

    fig, axes = plt.subplots(nrows=2, ncols=NUM_PLOT_COL, figsize=(np.sqrt(2)*12, 1*12))
    fig.suptitle("CCA Latent space analysis - %s"%model_alias, fontsize=16)
    """
    axes[0][0].set_title("Canonical corellations")
    axes[0][1].set_title("Prediction accuracy")
    row_names = ['Y= {}'.format(y) for y in y_keys]
    axes[1][0].set_title("All")
    axes[1][0].set_title("All")
    """

    # SVM
    clf = svm.SVC(gamma='auto')
    clf.fit(X1_train, y_train)
    y_val_hat_X1 = clf.predict(X1_val)
    acc_X1 = metrics.f1_score(y_val, y_val_hat_X1)

    clf.fit(X2_train, y_train)
    y_val_hat_X2 = clf.predict(X2_val)
    acc_X2 = metrics.f1_score(y_val, y_val_hat_X2)

    X1_X2_train = np.hstack([X1_train, X2_train])
    X1_X2_val = np.hstack([X1_val, X2_val])

    clf.fit(X1_X2_train, y_train)
    y_val_hat_X1_X2 = clf.predict(X1_X2_val)
    acc_X1_X2 = metrics.f1_score(y_val, y_val_hat_X1_X2)

    print("SVM f1, raw:", acc_X1, acc_X2, acc_X1_X2)
    #Train the CCA using some set of data: all, or by y class?
    for count_train_y, each_train_y in enumerate(y_train_keys): # for each class
        #Train
        x1_train = X1[list(set(y_ind[each_train_y]) & set(train_ind)),:]
        x2_train = X2[list(set(y_ind[each_train_y]) & set(train_ind)),:]
        mu1 = x1_train.mean(axis=0)
        mu2 = x2_train.mean(axis=0)
        std1 = x1_train.std(axis=0, ddof=1)
        std1[std1 == 0.0] = 1.0
        std2 = x2_train.std(axis=0, ddof=1)
        std2[std2 == 0.0] = 1.0
        if (CENTER_AND_SCALE):
            #Center and Scale
            x1_train = x1_train - mu1
            x2_train = x2_train - mu2
            x1_train /= std1
            x2_train /= std2

        x1_val = X1[val_ind, :]
        x2_val = X2[val_ind, :]
        if (CENTER_AND_SCALE):
            x1_val = (x1_val - mu1) / std1
            x2_val = (x2_val - mu2) / std2

        max_n_comp = min(x1_train.shape[1], x2_train.shape[1])
        n_comp = int(max_n_comp)

        #CCA
        cca12 = CCA(n_components=n_comp)
        cca12.fit(x1_train,x2_train)
        x1_c, x2_c = cca12.transform(x1_train, x2_train)
        train_corrs1 = [np.corrcoef(x1_c[:, i], x2_c[:, i])[0, 1] for i in range(x1_c.shape[1])]

        # Test
        x1_c_val = {}
        x2_c_val = {}
        val_corrs ={}
        for count_test_y, each_test_y in enumerate(list(set(Y))):
            x1_val = X1[list(set(y_ind[each_test_y]) & set(val_ind)), :]
            x2_val = X2[list(set(y_ind[each_test_y]) & set(val_ind)), :]
            if (CENTER_AND_SCALE):
                x1_val = (x1_val - mu1) / std1
                x2_val = (x2_val - mu2) / std2

            # Test correlation: Correlation between one signal and the predicted one
            x1_c_val[each_test_y], x2_c_val[each_test_y] = cca12.transform(x1_val, x2_val)
            val_corrs[each_test_y] = [np.corrcoef(x1_c_val[each_test_y][:,i], x2_c_val[each_test_y][:,i])[0, 1] for i in range(x2_c_val[each_test_y].shape[1])]

        X1_c, X2_c = cca12.transform(X1_train, X2_train)
        X1_c_val,X2_c_val = cca12.transform(X1_val, X2_val)

        clf.fit(X1_c, y_train)
        y_val_hat_X1_c = clf.predict(X1_c_val)
        acc_X1_c = metrics.f1_score(y_val, y_val_hat_X1_c)

        clf.fit(X2_c, y_train)
        y_val_hat_X2_c = clf.predict(X2_c_val)
        acc_X2_c = metrics.f1_score(y_val, y_val_hat_X2_c)

        X1_X2_c = np.hstack([X1_c,X2_c])
        X1_X2_c_val = np.hstack([X1_c_val, X2_c_val])
        clf.fit(X1_X2_c, y_train)
        y_val_hat_X1_X2_c = clf.predict(X1_X2_c_val)
        acc_X1_X2_c = metrics.f1_score(y_val, y_val_hat_X1_X2_c)

        print("svm f1, CCA trained on Y={}".format(each_train_y), acc_X1_c, acc_X2_c, acc_X1_X2_c)

        markers = {0:'.',1:'+'}
        colors = {0: 'g', 1: 'r'}
        #plot col1: correlation of tranformed
        plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL,1+count_train_y*NUM_PLOT_COL)
        plt.plot(np.arange(n_comp) + 1, train_corrs1, marker ='o', c='b')
        for count_test_y, each_test_y in enumerate(list(set(Y))):
            plt.plot(np.arange(n_comp) + 1, val_corrs[each_test_y], marker = markers[each_test_y], c=colors[each_test_y])
        plt.xlim(0.5, 0.5 + n_comp)
        plt.xticks(np.arange(n_comp) + 1)
        plt.xlabel('Canonical component')
        plt.ylabel('Canonical correlation')
        plt.legend(['train']+['Y={}'.format(each_test_y) for each_test_y in list(set(Y))])
        plt.title('Train on Y={}'.format(each_train_y))
        #print (train_corrs1)



        # Plot col 2: separability of latents
        min_x = min(x1_c_val[each_test_y][:,0].min(), x2_c_val[each_test_y][:,0].min())
        max_x = max(x1_c_val[each_test_y][:, 0].max(), x2_c_val[each_test_y][:, 0].max())
        min_y = min(x1_c_val[each_test_y][:, 1].min(), x2_c_val[each_test_y][:, 1].min())
        max_y = max(x1_c_val[each_test_y][:, 1].max(), x2_c_val[each_test_y][:, 1].max())
        #tranform of x2 to cca space
        plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, 1 + count_train_y * NUM_PLOT_COL + 1)
        for count_test_y, each_test_y in enumerate(list(set(Y))):
            plt.scatter(x1_c_val[each_test_y][:,0], x1_c_val[each_test_y][:,1], marker = markers[each_test_y], c=colors[each_test_y])
            plt.xlabel('CCA comp1')
            plt.ylabel('CCA comp2')
            plt.title('Latent space of X1')
            plt.xlim(min_x, max_x)
            plt.ylim(min_y, max_y)
        #tranform of x2 to cca space
        plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, 1 + count_train_y * NUM_PLOT_COL + 2)
        for count_test_y, each_test_y in enumerate(list(set(Y))):
            plt.scatter(x2_c_val[each_test_y][:,0], x2_c_val[each_test_y][:,1], marker = markers[each_test_y], c=colors[each_test_y])
            plt.xlabel('CCA comp1')
            plt.ylabel('CCA comp2')
            plt.title('Latent space of X2')
            plt.xlim(min_x, max_x)
            plt.ylim(min_y, max_y)
    plt.show()


    #
    pass
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='CCA')
    parser.add_argument("--fea1", type=str, default='met_W_pca')
    parser.add_argument("--fea2", type=str, default='genus_fea')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: log (x/sum(x)))")
    parser.add_argument("--extra", type=str, default='')

    args = parser.parse_args()

    main(args)
