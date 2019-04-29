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

DATA_ROOT = "/data/BioHealth/IBD"

parser = argparse.ArgumentParser()
FORMAT = '[%(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)
def cv_divising(num_folds, lbls):
    ind_pos = []
    ind_neg = []
    for i in range(len(lbls)):
        if Y_full[i] > 0:
            ind_pos.append(i)
        else:
            ind_neg.append(i)
    np.random.seed(RANSOM_SEED)
    np.random.shuffle(ind_pos)
    np.random.shuffle(ind_neg)

    Y_pred_full = [None for i in range(len(lbls))]
    for test_part in range(num_folds):
        test_parts = [test_part]
        train_parts = list(set(range(num_folds)) - set(test_parts))
        X_test_pos = get_parts([feats[i] for i in ind_pos], test_parts, num_folds)
        X_test_neg = get_parts([feats[i] for i in ind_neg], test_parts, num_folds)
        X_train_pos = get_parts([feats[i] for i in ind_pos], train_parts, num_folds)
        X_train_neg = get_parts([feats[i] for i in ind_neg], train_parts, num_folds)
        ind_test_pos = get_parts(ind_pos, test_parts, num_folds)
        ind_test_neg = get_parts(ind_neg, test_parts, num_folds)

        X_test = np.vstack(X_test_pos + X_test_neg)
        X_train = np.vstack(X_train_pos + X_train_neg)
        Y_train = np.concatenate((np.ones((len(X_train_pos), 1)), np.zeros((len(X_train_neg), 1))), axis=0)
        Y_test = np.concatenate((np.ones((len(X_test_pos), 1)), np.zeros((len(X_test_neg), 1))), axis=0)
        ind_test = ind_test_pos + ind_test_neg


def main(args):
    data_type = args.data_type
    data_path = os.path.join(DATA_ROOT, "ibd_{}.pkl".format(data_type))
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

    y_keys = [-1, -1]#+list(set(Y)) #-1 means all data
    test_y_keys = [-1]+list(set(Y)) #-1 means all data
    y_ind  = {}
    y_ind[-1] = range(len(Y))
    for y_val in set(Y):
        y_ind[y_val] = [i for i in range(len(Y)) if Y[i] == y_val]

    #TODO: Cross validation
    if not args.cross_val_folds:
        #Split train and val
        val_ind = content["val_idx"]
        train_ind = content["train_idx"]
    else:

        X1_train = X1[train_ind]
        X1_val = X1[val_ind]
        X2_train = X2[train_ind]
        X2_val = X2[val_ind]
        y_train = X1[train_ind]
        y_val = X1[val_ind]

        #X1 = np.random.rand(X1.shape[0], X1.shape[1])
        #X2 = np.random.rand(X2.shape[0], X2.shape[1])


    CENTER_AND_SCALE = 0 # normally not because sklearn does it inside cca
    NUM_PLOT_COL = 4

    fig, axes = plt.subplots(nrows=len(y_keys), ncols=NUM_PLOT_COL, figsize=(np.sqrt(2)*12, 1*12))
    fig.suptitle("CCA Component correlation analysis - %s"%model_alias, fontsize=16)
    axes[0][0].set_title("Canonical corellations")
    axes[0][1].set_title("Prediction accuracy")
    row_names = ['Y= {}'.format(y) for y in y_keys]
    axes[1][0].set_title("All")
    axes[1][0].set_title("All")

    #Train the CCA using some set of data: all, or by y class?
    for fold_count, x1_train, x2_train, y_train, x1_val, x2_val in enumerate(y_keys): # for each class

        #Train
        x1_train = X1[list(set(y_ind[each_y]) & set(train_ind)),:]
        x2_train = X2[list(set(y_ind[each_y]) & set(train_ind)),:]
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

        x1_val = X1[list(set(y_ind[each_y]) & set(val_ind)), :]
        x2_val = X2[list(set(y_ind[each_y]) & set(val_ind)), :]
        if (CENTER_AND_SCALE):
            x1_val = (x1_val - mu1) / std1
            x2_val = (x2_val - mu2) / std2

        y = Y[list(set(y_ind[each_y]) & set(train_ind))]

        max_n_comp = min(x1_train.shape[1], x2_train.shape[1])
        n_comp = max_n_comp/2

        #X1-->X2
        cca12 = CCA(n_components=n_comp)
        cca12.fit(x1_train,x2_train)
        x1_c, x2_c = cca12.transform(x1_train, x2_train)
        train_corrs1 = [np.corrcoef(x1_c[:, i], x2_c[:, i])[0, 1] for i in range(x1_c.shape[1])]

        markers = {0:'.',1:'+'}
        colors = {0: 'g', 1: 'r'}
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

        # X2-->X1
        cca21 = CCA(n_components=n_comp)
        cca21.fit(x2_train, x1_train)
        x2_c, x1_c = cca21.transform(x2_train, x1_train)
        train_corrs2 = [np.corrcoef(x2_c[:, i], x1_c[:, i])[0, 1] for i in range(x2_c.shape[1])]

        plt.subplot(len(y_keys), NUM_PLOT_COL,1+count_y*NUM_PLOT_COL)
        plt.plot(np.arange(n_comp) + 1, train_corrs1, marker ='o', c='b')
        for count_test_y, each_test_y in enumerate(list(set(Y))):
            plt.plot(np.arange(n_comp) + 1, val_corrs[each_test_y], marker=markers[each_test_y], c=colors[each_test_y])

        plt.xlim(0.5, 0.5 + n_comp)
        plt.ylim(-1.0, 1.0)
        plt.xticks(np.arange(n_comp) + 1)
        plt.xlabel('Canonical component')
        plt.ylabel('Canonical correlation')
        plt.legend(['train', 'val'])
        plt.title('Train/Val CCs Y= {}'.format(each_y))
        print (train_corrs1, train_corrs2)

        # Test
        for count_test_y, each_test_y in enumerate(test_y_keys):
            x1_val = X1[list(set(y_ind[each_test_y]) & set(val_ind)), :]
            x2_val = X2[list(set(y_ind[each_test_y]) & set(val_ind)), :]
            if (CENTER_AND_SCALE):
                x1_val = (x1_val - mu1) / std1
                x2_val = (x2_val - mu2) / std2

            # Test correlation: Correlation between one signal and the predicted one
            x2_val_hat = cca12.predict(x1_val)
            test_corrs1 = np.nan_to_num([np.corrcoef(x2_val_hat[:, i], x2_val[:, i])[0, 1] for i in range(x2_val.shape[1])])
            argsort1 = np.argsort(-test_corrs1) # descending
            sort1 = test_corrs1[argsort1]
            avg_acc1 = np.sqrt(metrics.mean_squared_error(x2_val, x2_val_hat))

            # X2-->X1
            x1_val_hat = cca21.predict(x2_val)
            test_corrs2 = np.nan_to_num([np.corrcoef(x1_val_hat[:, i], x1_val[:, i])[0, 1] for i in range(x1_val.shape[1])])
            argsort2 = np.argsort(-test_corrs2) #- for descending
            sort2 = test_corrs1[argsort2]
            avg_acc2 = np.sqrt(metrics.mean_squared_error(x1_val, x1_val_hat))

            plt.subplot(len(y_keys), NUM_PLOT_COL,1+count_y*NUM_PLOT_COL+1+count_test_y)
            nTicks = max(len(test_corrs1), len(test_corrs2))
            plt.plot(np.arange(len(test_corrs1)) + 1, test_corrs1, '+', color='r')
            plt.plot(np.arange(len(test_corrs2)) + 1, test_corrs2, '.', color='g')


            plt.xlim(0.5, 0.5 + nTicks + 3)
            plt.ylim(-1.0, 1.0)
            #plt.xticks(np.arange(nTicks) + 1)
            plt.xlabel('Dataset dimension')
            plt.ylabel('Prediction correlation')
            plt.title('Validation accuracy Y= {}'.format(each_test_y))
            plt.legend(['x2 from x1: %3.2f, best %d:%3.2f'%(avg_acc1, argsort1[0],sort1[0]), 'x1 from x2: %3.2f, best %d:%3.2f'%(avg_acc2, argsort2[0],sort2[0])])
            print('''The prediction accuracy x1-->x2:%s''' % test_corrs1)
            print('''The prediction accuracy x2-->x1:%s''' % test_corrs2)
    plt.show()


    #
    pass
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='CCA')
    parser.add_argument("--fea1", type=str, default='met_W_nmf')
    parser.add_argument("--fea2", type=str, default='genus_fea')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: log (x/sum(x)))")
    parser.add_argument("--cross_val_folds", type=int, default=3)
    parser.add_argument("--extra", type=str, default='')

    args = parser.parse_args()

    main(args)
