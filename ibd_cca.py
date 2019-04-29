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

RANDOM_SEED =1
DATA_ROOT = "/data/BioHealth/IBD"

parser = argparse.ArgumentParser()
FORMAT = '[%(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_parts(x_full, parts, num_parts):
    this_part = []
    part_length = len(x_full) / num_parts
    for part in parts:
        part_start = part_length * part
        part_end = part_length * (part + 1) if part < num_parts - 1 else len(x_full)
        this_part += x_full[part_start:part_end]
    return this_part

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
    # X1 = np.random.rand(X1.shape[0], X1.shape[1])
    # X2 = np.random.rand(X2.shape[0], X2.shape[1])
    Y = content["diagnosis"]

    #Suppress to two classes
    for i in range(len(Y)):
        if Y[i] !=0:
            Y[i] =1

    test_y_keys = [-1]+list(set(Y)) #-1 means all data
    y_ind  = {}
    y_ind[-1] = range(len(Y))
    for y_val in set(Y):
        y_ind[y_val] = [i for i in range(len(Y)) if Y[i] == y_val]

    #TODO: Cross validation
    num_folds =args.cross_val_folds
    if not num_folds:
        #Split train and val
        val_ind = content["val_idx"]
        train_ind = content["train_idx"]
    else:

        ind_pos = []
        ind_neg = []
        for i, y in enumerate(Y):
            if y > 0:
                ind_pos.append(i)
            else:
                ind_neg.append(i)
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(ind_pos)
        np.random.shuffle(ind_neg)


        CENTER_AND_SCALE = 0 # normally not because sklearn does it inside cca
        NUM_PLOT_COL = 4 #cca fit + y_test types
        NUM_PLOT_ROW = 2 #x1 to x2 and x2 to x1

        fig, axes = plt.subplots(nrows=NUM_PLOT_ROW, ncols=NUM_PLOT_COL, figsize=(np.sqrt(2)*12, 1*12))
        fig.suptitle("CCA Component correlation analysis - %s"%model_alias, fontsize=16)
        axes[0][0].set_title("Canonical corellations")
        axes[0][1].set_title("Prediction accuracy")
        row_names = ["X2->X1", "X1->X2"]
        axes[1][0].set_title("All")
        axes[1][0].set_title("All")

    #Cross validation, we will see if the reconstruction ability is consistent in the data
    for test_part in range(num_folds):
        test_parts = [test_part]
        train_parts = list(set(range(num_folds)) - set(test_parts))
        
        x1_train_pos = get_parts([X1[i] for i in ind_pos], train_parts, num_folds)
        x1_train_neg = get_parts([X1[i] for i in ind_neg], train_parts, num_folds)
        x1_train_all = np.vstack(x1_train_pos + x1_train_neg)
        
        x2_train_pos = get_parts([X2[i] for i in ind_pos], train_parts, num_folds)
        x2_train_neg = get_parts([X2[i] for i in ind_neg], train_parts, num_folds)
        x2_train_all = np.vstack(x2_train_pos + x2_train_neg)
        
        x1_val_pos = get_parts([X1[i] for i in ind_pos], test_parts, num_folds)
        x1_val_neg = get_parts([X1[i] for i in ind_neg], test_parts, num_folds)
        x1_val_all = x1_val_pos + x1_val_neg
        x1_vals = [np.vstack(x1_val_all), np.vstack(x1_val_neg), np.vstack(x1_val_pos)]
        
        x2_val_pos = get_parts([X2[i] for i in ind_pos], test_parts, num_folds)
        x2_val_neg = get_parts([X2[i] for i in ind_neg], test_parts, num_folds)
        x2_val_all = x2_val_pos + x2_val_neg
        x2_vals = [np.vstack(x2_val_all), np.vstack(x2_val_neg), np.vstack(x2_val_pos)]

        ind_test_pos = get_parts(ind_pos, test_parts, num_folds)
        ind_test_neg = get_parts(ind_neg, test_parts, num_folds)
        ind_test = ind_test_pos + ind_test_neg
        
        Y_train = np.concatenate((np.ones((len(ind_pos), 1)), np.zeros((len(ind_neg), 1))), axis=0)
        Y_test = np.concatenate((np.ones((len(ind_test_pos), 1)), np.zeros((len(ind_test_neg), 1))), axis=0)
        
        #Train CCA on all or part?
        x1_train = x1_train_all
        x2_train = x2_train_all
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

        max_n_comp = min(x1_train.shape[1], x2_train.shape[1])
        n_comp = max_n_comp/2


        #Train X1-->X2
        cca12 = CCA(n_components=n_comp)
        cca12.fit(x1_train,x2_train)
        x1_c, x2_c = cca12.transform(x1_train, x2_train)
        train_corrs1 = [np.corrcoef(x1_c[:, i], x2_c[:, i])[0, 1] for i in range(x1_c.shape[1])]

        #Train X2-->X1
        cca21 = CCA(n_components=n_comp)
        cca21.fit(x2_train, x1_train)
        x2_c, x1_c = cca21.transform(x2_train, x1_train)
        train_corrs2 = [np.corrcoef(x2_c[:, i], x1_c[:, i])[0, 1] for i in range(x2_c.shape[1])]

        #Plot col 1: overall fitness
        markers = {0:'.',1:'+', 2:'o'}
        colors = {0: 'g', 1: 'r', 2:'b'}
        x1_c_val = {}
        x2_c_val = {}
        val_corrs ={}

        for val_ind, (x1_val, x2_val) in enumerate(zip(x1_vals, x2_vals)):
            if (CENTER_AND_SCALE):
                x1_val = (x1_val - mu1) / std1
                x2_val = (x2_val - mu2) / std2
            # Test correlation: Correlation between one signal and the predicted one
            x1_c_val[val_ind], x2_c_val[val_ind] = cca12.transform(x1_val, x2_val)
            val_corrs[val_ind] = [np.corrcoef(x1_c_val[val_ind][:,i], x2_c_val[val_ind][:,i])[0, 1] for i in range(x2_c_val[val_ind].shape[1])]


        plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL,1)
        plt.plot(np.arange(n_comp) + 1, train_corrs1, marker ='x', c=colors[test_part])
        for val_ind, val_corr in val_corrs.iteritems():
            plt.plot(np.arange(n_comp) + 1, val_corr, marker=markers[val_ind], c=colors[test_part])

        plt.xlim(0.5, 0.5 + n_comp)
        plt.ylim(-1.0, 1.0)
        plt.xticks(np.arange(n_comp) + 1)
        plt.xlabel('Canonical component')
        plt.ylabel('Canonical correlation')
        plt.legend(['train', 'val'])
        plt.title('Train/Val X2->X1')



        # Plot col 2..n: VALIDATION DETAILS
        for val_ind, (x1_val, x2_val) in enumerate(zip(x1_vals, x2_vals)):
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

            plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL,2+val_ind)
            nTicks = len(test_corrs1)
            plt.plot(np.arange(len(test_corrs1)) + 1, test_corrs1,  marker=markers[val_ind], c=colors[test_part])

            plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, NUM_PLOT_COL + 2 + val_ind)
            nTicks = len(test_corrs2)
            plt.plot(np.arange(len(test_corrs2)) + 1, test_corrs2,  marker=markers[val_ind], c=colors[test_part])


            plt.xlim(0.5, 0.5 + nTicks + 3)
            plt.ylim(-1.0, 1.0)
            #plt.xticks(np.arange(nTicks) + 1)
            plt.xlabel('Dataset dimension')
            plt.ylabel('Prediction correlation')
            plt.title('Validation accuracy Y= {}'.format(val_ind))
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
