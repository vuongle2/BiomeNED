import os
import sys
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
#from pyrcca import rcca
#import logging
from sklearn import metrics
#import tensorboard_logger as tl

from biome_ae import BiomeAE, BiomeCCA, BiomeAESnip

RANDOM_SEED =1
DATA_ROOT = "/data/BioHealth/IBD"
vis_dir = os.path.join(DATA_ROOT, 'vis')
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
parser = argparse.ArgumentParser()
FORMAT = '[%(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'
#logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
#logger = logging.getLogger(__name__)


def get_parts(x_full, parts, num_parts):
    this_part = []
    part_length = int(len(x_full) / num_parts)
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

    model_alias = '%s+sparse_%s+%s_%s+fea1_%s+fea2_%s+%s' % (
        args.model,
        args.sparse,
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
        num_folds =1

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
    #Cross validation, we will see if the reconstruction ability is consistent in the data
    CV_results = [{} for i in range(num_folds)]
    for fold in range(num_folds):
        test_parts = [fold]
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

        vals_meanings = ["all"]#, "negatives", "positives"]

        ind_test_pos = get_parts(ind_pos, test_parts, num_folds)
        ind_test_neg = get_parts(ind_neg, test_parts, num_folds)

        ind_train_pos = get_parts(ind_pos, train_parts, num_folds)
        ind_train_neg = get_parts(ind_neg, train_parts, num_folds)

        
        y_train = np.concatenate((np.ones((len(ind_train_pos), 1)), np.zeros((len(ind_train_neg), 1))), axis=0)
        y_test = np.concatenate((np.ones((len(ind_test_pos), 1)), np.zeros((len(ind_test_neg), 1))), axis=0)
        y_vals = [y_test, np.ones((len(ind_test_pos), 1)), np.zeros((len(ind_test_neg), 1))]

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

        #Train X1-->X2
        if args.model == "BiomeCCA":
            model12 = BiomeCCA(args)
            model21 = BiomeCCA(args)
        elif args.model in ["BiomeAE","BiomeAEL0"]:
            model12 = BiomeAE(args)
            model21 = BiomeAE(args)
        elif args.model == "BiomeAESnip":
            model12 = BiomeAESnip(args)
            model21 = BiomeAESnip(args)
        else:
            assert("Unknown model %s"%args.model)
        model12.fit(x1_train,x2_train,y_train, x1_vals[0], x2_vals[0], y_test, args)
        #Train X2-->X1
        model21.fit(x2_train, x1_train, y_train, x2_vals[0], x1_vals[0], y_test, args)

        if args.model == "CCA":
            x1_c, x2_c = model12.transform(x1_train, x2_train)
            latent_corrs_train_cv = [np.corrcoef(x1_c[:, i], x2_c[:, i])[0, 1] for i in range(x1_c.shape[1])]
        else:
            latent_corrs_train_cv = [0 for i in range(x1_train.shape[1])]


        #Test
        test_corrs1_cv = [None for _ in range(len(vals_meanings))]
        test_corrs2_cv = [None for _ in range(len(vals_meanings))]
        latent_corrs_val_cv = [None for _ in range(len(vals_meanings))]
        avg_acc1_cv= [None for _ in range(len(vals_meanings))]
        avg_acc2_cv = [None for _ in range(len(vals_meanings))]
        for val_ind, (x1_val, x2_val, y_val, meaning) in enumerate(zip(x1_vals, x2_vals, y_vals, vals_meanings)):
            if (CENTER_AND_SCALE):
                x1_val = (x1_val - mu1) / std1
                x2_val = (x2_val - mu2) / std2

            # Test correlation: Correlation between org signal and the predicted one
            x2_val_hat = model12.predict(x1_val, x2_val, y_val, args)
            test_corrs1_cv[val_ind] = np.nan_to_num([np.corrcoef(x2_val_hat[:, i], x2_val[:, i])[0, 1] for i in range(x2_val.shape[1])])
            avg_acc1_cv[val_ind] = np.sqrt(metrics.mean_squared_error(x2_val, x2_val_hat))

            # X2-->X1
            x1_val_hat = model21.predict(x2_val, x1_val, y_val, args)
            test_corrs2_cv[val_ind] = np.nan_to_num([np.corrcoef(x1_val_hat[:, i], x1_val[:, i])[0, 1] for i in range(x1_val.shape[1])])
            avg_acc2_cv[val_ind] = np.sqrt(metrics.mean_squared_error(x1_val, x1_val_hat))

            if args.model == "CCA":
                # Test correlation: Correlation between two transformed vectors into latent space
                x1_c_val, x2_c_val = model12.transform(x1_val, x2_val)
                latent_corrs_val_cv[val_ind] = [np.corrcoef(x1_c_val[:, i], x2_c_val[:, i])[0, 1] for i in  range(x1_c_val.shape[1])]
            else:
                #TODO: write latent get for pytorcgh
                latent_corrs_val_cv = [0 for i in  range(x1_val.shape[1])]
            l0_weight_12 = model12.param_l0()
            l0_weight_21 = model21.param_l0()

        CV_results[fold] = {'test_corrs1':test_corrs1_cv,
                            'test_corrs2': test_corrs2_cv,
                            'avg_acc1':avg_acc1_cv,
                            'avg_acc2': avg_acc2_cv,
                            'latent_corrs_train': latent_corrs_train_cv,
                            'latent_corrs': latent_corrs_val_cv,
                            'trans12':model12.get_transformation(),
                            'trans21': model21.get_transformation()}

    #ANALYSIS and PLOTS

    #Join the folds
    latent_corrs_train_allfolds = [CV_results[fold]['latent_corrs_train'] for fold in range(num_folds)]
    latent_corrs_train = np.array(latent_corrs_train_allfolds).mean(axis=0)
    latent_corrs = [None for _ in range(len(vals_meanings))]
    for v in range(len(vals_meanings)):
        latent_corrs_allfolds = [CV_results[fold]['latent_corrs'][v] for fold in range(num_folds)]
        latent_corrs[v] = np.array(latent_corrs_allfolds).mean(axis=0)


    markers = {0: '.', 1: '+', 2: 'o'}
    colors = {0: 'g', 1: 'r', 2: 'b', -1:'black'}
    if args.model =="CCA":
        # Plot col 1: overall fitness
        plt.figure()
        plt.plot(np.arange(args.latent_size) + 1, latent_corrs_train, marker='.', c='black', label="train")
        for v, meaning in enumerate(vals_meanings):
            plt.plot(np.arange(args.latent_size) + 1, latent_corrs[v], marker='.', c=colors[v], label="val on %s"%(meaning))
        plt.xticks(np.arange(args.latent_size) + 1)
        plt.xlabel('Canonical component')
        plt.ylabel('Canonical correlation')
        plt.legend()
        plt.title('Latent corr of CCA fitting')


    NUM_PLOT_COL = len(vals_meanings) #cca fit + y_test types
    NUM_PLOT_ROW = 2 #x1 to x2 and x2 to x1
    if args.visualize =="subplot":
        fig, axes = plt.subplots(nrows=NUM_PLOT_ROW, ncols=NUM_PLOT_COL, figsize=(np.sqrt(2)*12, 1*12))
        fig.suptitle("%s"%model_alias, fontsize=16)
        #axes[0][0].set_title("Canonical corellations")
        #axes[0][1].set_title("Prediction accuracy")
        row_names = ["X2->X1", "X1->X2"]
        #axes[1][0].set_title("All")
        #axes[1][0].set_title("All")

    if args.visualize == "subplot":
        plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, 1)
    else:
        plt.figure()

    # Plot 2..n: VALIDATION DETAILS
    for v, meaning in enumerate(vals_meanings): # for each choice of val selection, we take the mean of the folds
        test_corrs1_allfolds = [CV_results[fold]['test_corrs1'][v] for fold in range(num_folds)]
        test_corrs2_allfolds = [CV_results[fold]['test_corrs2'][v] for fold in range(num_folds)]
        test_corrs1 = np.array(test_corrs1_allfolds).mean(axis=0)
        test_corrs2 = np.array(test_corrs2_allfolds).mean(axis=0)

        meancc1 = test_corrs1.mean()
        argsort1 = np.argsort(-test_corrs1)  # descending
        sort1 = test_corrs1[argsort1]
        meantopk1 = sort1[:args.topk].mean()

        meancc2 = test_corrs2.mean()
        argsort2 = np.argsort(-test_corrs2)  # - for descending
        sort2 = test_corrs2[argsort2]
        meantopk2 = sort2[:args.topk].mean()

        avg_acc1_allfolds = [CV_results[fold]['avg_acc1'][v] for fold in range(num_folds)]
        avg_acc1 = np.array(avg_acc1_allfolds).mean(axis=0)
        avg_acc2_allfolds = [CV_results[fold]['avg_acc2'][v] for fold in range(num_folds)]
        avg_acc2 = np.array(avg_acc2_allfolds).mean(axis=0)


        #first row 1->2
        if args.visualize == "subplot":
            plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, 1 + v)
        else:
            plt.figure()

        for fold in range(num_folds):
            plt.plot(np.arange(len(test_corrs1_allfolds[fold])) + 1, test_corrs1_allfolds[fold], marker='x', c=colors[fold], label="fold %s"%(fold))
        plt.plot(np.arange(len(test_corrs1)) + 1, test_corrs1, marker=markers[val_ind],
                 c=colors[-1], label="mean cv")
        plt.title('Y %s, x1->x2:%3.2f, best %d:%3.2f' % (meaning, avg_acc1, argsort1[0], sort1[0]))
        plt.legend()
        plt.ylim(-1.0, 1.0)


        #2nd row 2->1
        if args.visualize == "subplot":
            plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, 1 +NUM_PLOT_COL+ v)
        else:
            plt.figure()

        for fold in range(num_folds):
            plt.plot(np.arange(len(test_corrs2_allfolds[fold])) + 1, test_corrs2_allfolds[fold], marker='x', c=colors[fold], label="fold %s"%(fold))
        plt.plot(np.arange(len(test_corrs2)) + 1, test_corrs2, marker=markers[val_ind],
                 c=colors[-1], label="mean cv")
        plt.title('Y %s, x2->x1:%3.2f, best %d:%3.2f' % (meaning, avg_acc2, argsort2[0], sort2[0]))
        plt.legend()
        plt.ylim(-1.0, 1.0)
    # plt.xticks(np.arange(nTicks) + 1)
    print(model_alias)
    print("x1-->x2, rmse %4.2f, meancc:%4.2f, cc best %s:%4.2f, top %d:%4.2f, #link %s"%(
        avg_acc1,
        meancc1,
        argsort1[0], sort1[0],
        args.topk, meantopk1,
        l0_weight_12,
    ))
    print("x2-->x1, rmse %4.2f, meancc:%4.2f, cc best %s:%4.2f, top %d:%4.2f, #link %s"%(
        avg_acc2,
        meancc2,
        argsort2[0], sort2[0],
        args.topk, meantopk2,
        l0_weight_21,
    ))

    fname = os.path.join(vis_dir, model_alias+'.png')
    if fname is None:
        plt.show(block=False)
    else:
        plt.savefig(fname)


    #
    pass
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='BiomeAE')
    parser.add_argument("--fea1", type=str, default='met_W_pca')
    parser.add_argument("--fea2", type=str, default='genus_fea')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: log (x/sum(x)))")
    parser.add_argument("--cross_val_folds", type=int, default=3)
    parser.add_argument("--topk", type=int, default=10)

    parser.add_argument("--sparse", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--activation", type=str, default='ReLU')
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=1, type=bool)
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    parser.add_argument('--gpu_num', default="0", type=str)
    parser.add_argument("--extra", type=str, default='')

    parser.add_argument("--visualize", type=str, default='subplot')

    args = parser.parse_args()

    main(args)
