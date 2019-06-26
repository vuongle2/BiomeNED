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

from biome_ae import BiomeAE, BiomeCCA, BiomeAESnip, BiomeLasso, BiomeMultiTaskLasso, BiomeLinear, BiomeOneLayerSnip, BiomeLassoAESnip

from utils import draw_weight_graph, write_matrix_to_csv, get_subgraph, consistency_index,prune_subgraph

RANDOM_SEED =1
DATA_ROOT = "/data/BioHealth/IBD"
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

def init_model(args):
    # Train X1-->X2
    if args.model == "BiomeCCA":
        translator = BiomeCCA(args)
    elif args.model == "BiomeLinear":
        translator = BiomeLinear(args)
    elif args.model == "BiomeLasso":
        translator = BiomeLasso(args)
    elif args.model == "BiomeMultiTaskLasso":
        translator = BiomeMultiTaskLasso(args)
    elif args.model in ["BiomeOneLayerSnip"]:
        translator = BiomeOneLayerSnip(args)
    elif args.model in ["BiomeAE", "BiomeAEL0"]:
        translator = BiomeAE(args)
    elif args.model == "BiomeAESnip":
        translator = BiomeAESnip(args)
    elif args.model == "BiomeLassoAESnip":
        translator = BiomeLassoAESnip(args)
    else:
        assert 0, "Unknown model %s" % args.model
    return translator
def main(args):
    data_type = args.data_type
    data_path = os.path.join(DATA_ROOT, "ibd_{}.pkl".format(data_type))
    with open(data_path,"rb") as df:
        content = pickle.load(df)

    extra = ""
    if args.nonneg_weight:
        extra+="+nonneg_weight"
    if args.normalize_input:
        extra += "+normalize_input"
    if args.model in ["BiomeAE", "BiomeAESnip"]: #deep learning model, add deep learning params in alias
        extra += "+bs_%d+ac_%s+lr_%s" %(
            args.batch_size,
            args.activation,
            args.learning_rate
        )

    model_alias = 'Translator+%s_%s+%s-%s+cv_%d+%s+sparse_%s+ls_%d+%s' % (
        args.dataset_name, args.data_type,
        args.fea1, args.fea2,
        args.cross_val_folds,
        args.model,
        args.sparse, args.latent_size,
        extra + args.extra)

    args.model_alias = model_alias
    print(model_alias)

    vis_dir = os.path.join(DATA_ROOT, 'vis', model_alias)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    args.vis_dir = vis_dir

    X1 = content[args.fea1] # n x m1
    X2 = content[args.fea2] # n x m2
    Name1 = content[args.fea1.replace("fea","ids")]
    Name2 = content[args.fea2.replace("fea","ids")]

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

    if args.model in ["BiomeAEL0"] or args.normalize_input:
        CENTER_AND_SCALE = 1 # so that the lL0 weight makes unique sense
    else:
        CENTER_AND_SCALE = 0  # normally not because sklearn does it inside cca


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
        if (CENTER_AND_SCALE):
            mu1 = (x1_train.max(axis=0) + x1_train.min(axis=0))/2.0
            x1_train = x1_train - mu1
            std1 = (x1_train.max(axis=0) - x1_train.min(axis=0))/2.0
            x1_train /= std1
            x1_vals = [(x1_val - mu1) / std1 for x1_val in x1_vals]

            mu2 = (x2_train.max(axis=0) + x2_train.min(axis=0))/2.0
            x2_train = x2_train - mu2
            std2 = (x2_train.max(axis=0) - x2_train.min(axis=0))/2.0
            x2_train /= std2
            x2_vals = [(x2_val - mu2) / std2 for x2_val in x2_vals]
        #Train
        args.contr = fold
        translator = init_model(args)
        translator.fit(x1_train,x2_train,y_train, x1_vals[0], x2_vals[0], y_test, args)


        if args.model == "CCA":
            x1_c, x2_c = translator.transform(x1_train, x2_train)
            latent_corrs_train_cv = [np.corrcoef(x1_c[:, i], x2_c[:, i])[0, 1] for i in range(x1_c.shape[1])]
        else:
            latent_corrs_train_cv = [0 for i in range(x1_train.shape[1])]


        #Test
        test_corrs1_cv = [None for _ in range(len(vals_meanings))]
        test_corrs2_cv = [None for _ in range(len(vals_meanings))]
        latent_corrs_val_cv = [None for _ in range(len(vals_meanings))]
        avg_acc1_cv= [None for _ in range(len(vals_meanings))]
        for val_ind, (x1_val, x2_val, y_val, meaning) in enumerate(zip(x1_vals, x2_vals, y_vals, vals_meanings)):
            # Test correlation: Correlation between org signal and the predicted one
            x2_val_hat = translator.predict(x1_val, x2_val, y_val, args)
            test_corrs1_cv[val_ind] = np.nan_to_num([np.corrcoef(x2_val_hat[:, i], x2_val[:, i])[0, 1] for i in range(x2_val.shape[1])])
            avg_acc1_cv[val_ind] = np.sqrt(metrics.mean_squared_error(x2_val, x2_val_hat))

            if args.model == "CCA":
                # Test correlation: Correlation between two transformed vectors into latent space
                x1_c_val, x2_c_val = translator.transform(x1_val, x2_val)
                latent_corrs_val_cv[val_ind] = [np.corrcoef(x1_c_val[:, i], x2_c_val[:, i])[0, 1] for i in  range(x1_c_val.shape[1])]
            else:
                #TODO: write latent get for pytorcgh
                latent_corrs_val_cv = [0 for i in  range(x1_val.shape[1])]
            l0_weight_12 = translator.param_l0()

            nodes12, weights12 = translator.get_graph()
            argsort1 = np.argsort(-test_corrs1_cv[val_ind])
            if args.draw_graph and args.sparse is not None and args.fea1 !="met_fea":
                prunedn12, prunedw12 = prune_subgraph(nodes12, weights12, list(argsort1[:args.topk].T))
                graph_fname12 = os.path.join(vis_dir, 'graph_translator_fold%d' % fold)
                draw_weight_graph(prunedn12, prunedw12, graph_fname12)

        CV_results[fold] = {'test_corrs1':test_corrs1_cv,
                            'avg_acc1':avg_acc1_cv,
                            'latent_corrs_train': latent_corrs_train_cv,
                            'latent_corrs': latent_corrs_val_cv,
                            'trans12':translator.get_transformation(),
                            'nodes12':nodes12,
                            'weights12':weights12,
                            }

    #Join the folds
    latent_corrs_train_allfolds = [CV_results[fold]['latent_corrs_train'] for fold in range(num_folds)]
    latent_corrs_train = np.array(latent_corrs_train_allfolds).mean(axis=0)
    latent_corrs = [None for _ in range(len(vals_meanings))]
    for v in range(len(vals_meanings)):
        latent_corrs_allfolds = [CV_results[fold]['latent_corrs'][v] for fold in range(num_folds)]
        latent_corrs[v] = np.array(latent_corrs_allfolds).mean(axis=0)

    #Plots
    markers = {0: '.', 1: '+', 2: 'o'}
    colors = {0: 'g', 1: 'r', 2: 'b', 3:'m',4:'y', -1:'black'}
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
    NUM_PLOT_ROW = 1
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
        test_corrs1 = np.array(test_corrs1_allfolds).mean(axis=0)

        meancc1 = test_corrs1.mean()
        argsort1 = np.argsort(-test_corrs1)  # descending
        sort1 = test_corrs1[argsort1]
        meantopk1 = sort1[:args.topk].mean()

        avg_acc1_allfolds = [CV_results[fold]['avg_acc1'][v] for fold in range(num_folds)]
        avg_acc1 = np.array(avg_acc1_allfolds).mean(axis=0)

        #first row 1->2
        if args.visualize == "subplot":
            plt.subplot(NUM_PLOT_ROW, NUM_PLOT_COL, 1 + v)
        else:
            plt.figure()

        for fold in range(num_folds):
            plt.plot(np.arange(len(test_corrs1_allfolds[fold])) + 1, test_corrs1_allfolds[fold], marker='x', c=colors[fold], label="fold %s"%(fold))
        plt.plot(np.arange(len(test_corrs1)) + 1, test_corrs1, marker=markers[val_ind],
                 c=colors[-1], label="mean cv")
        plt.title('Y %s, x1->x2 top %d:%3.2f, best %d:%3.2f' % (meaning, args.topk, meantopk1, argsort1[0], sort1[0]))
        plt.legend()
        plt.ylim(-1.0, 1.0)

        fname = os.path.join(vis_dir, 'cv_val%s_corcoeff.png'%(meaning))
        if fname is None:
            plt.show(block=False)
        else:
            plt.savefig(fname)
            plt.close()

        # --join the graph
        # ----- basically nodes should be the same
        for fold in range(num_folds):
            assert (CV_results[fold]['nodes12'] == CV_results[0]['nodes12'])
        join_nodes = CV_results[0]['nodes12']

        # -----we will just join the weights
        join_weights = []
        for lay_i in range(len(CV_results[0]['weights12'])):  # each layer
            all_weights = [CV_results[fold]['weights12'][lay_i] for fold in range(num_folds)]
            jw = np.zeros(all_weights[0].shape)
            for i in range(all_weights[0].shape[0]):  # each input node
                for j in range(all_weights[0].shape[1]):  # each output node
                    ws = np.array([all_weights[fold][i][j] for fold in range(num_folds)])
                    # vote on existent
                    if np.count_nonzero(ws) < num_folds:
                        jw[i, j] = 0
                    else:
                        jw[i, j] = ws.mean()
            join_weights.append(jw)

        #Measure the stability using the mean of pair-wise consistency index of non negative weights
        if 1:
            ci = []
            for fold1 in range(0, num_folds-1):
                for fold2 in range(fold1+1, num_folds):
                    ci.append(consistency_index(CV_results[fold1]['weights12'], CV_results[fold2]['weights12']))
            mean_ci = np.array(ci).mean()

        if args.draw_graph:
            # Now prune more for interpretation
            LINK_THRESHOLD = 0.1
            # --prune all the weights that are less important than 0.1
            for lay_i in range(len(join_weights)):  # each layer
                lay_w = join_weights[lay_i]
                lay_w[lay_w < LINK_THRESHOLD] = 0.0
            # --prune all the nodes that are not connected to the destination
            subnodes, subweights = prune_subgraph(join_nodes, join_weights, list(argsort1[:args.topk].T))
            graph_fname = os.path.join(vis_dir, 'joint_graph_top_%d_thresh%s_pruned' % (args.topk,LINK_THRESHOLD))
            draw_weight_graph(subnodes, subweights, graph_fname, Name1, Name2)

        #FINAL MODEL
        # Train a single model on all data
        if (CENTER_AND_SCALE):
            # Center and Scale
            mu1 = (X1.max(axis=0) + X1.min(axis=0))/2.0
            X1 = X1 - mu1
            std1 = (X1.max(axis=0) - X1.min(axis=0))/2.0
            X1 /= std1

            mu2 = (X2.max(axis=0) + X2.min(axis=0)) / 2.0
            X2 = X2 - mu2
            std2 = (X2.max(axis=0) - X2.min(axis=0)) / 2.0
            X2 /= std2
        translator.fit(X1, X2, Y, X1, X2, Y, args)

        nodes, weights = translator.get_graph()
        if args.model in ["BiomeAE", "BiomeAESnip","CCA"]:
            z = translator.transform(X1, X2, Y, args)
            write_matrix_to_csv(z, os.path.join(vis_dir, 'latent_z.txt'))
            write_matrix_to_csv(weights[0], os.path.join(vis_dir, 'x1_to_z_weight.txt'))
            write_matrix_to_csv(weights[1], os.path.join(vis_dir, 'z_to_x2_weight.txt'))

        X2_hat = translator.predict(X1, X2, Y, args)
        test_corrs = np.nan_to_num(
            [np.corrcoef(X2_hat[:, i], X2[:, i])[0, 1] for i in range(X2.shape[1])])
        argsort = np.argsort(-test_corrs)  # descending
        sort = test_corrs[argsort]
        avg_acc = np.sqrt(metrics.mean_squared_error(X2, X2_hat))
        meancc = test_corrs.mean()
        meantopk = sort[:args.topk].mean()
        l0_weight = translator.param_l0()

        if args.draw_graph:
            # --prune all the weights that are less important than 0.1
            for lay_i in range(len(weights)):  # each layer
                lay_w = join_weights[lay_i]
                lay_w[lay_w < LINK_THRESHOLD] = 0.0
            subnodes, subweights = prune_subgraph(nodes, weights, list(argsort[:args.topk].T))
            graph_fname = os.path.join(vis_dir, 'graph_top%d' % args.topk)
            draw_weight_graph(subnodes, subweights, graph_fname, Name1, Name2)

        print(model_alias, meaning)
        print("CV %s-->%s, ci %4.2f rmse %4.2f, meancc:%4.2f, cc best %s/%4.2f, top %d:%4.2f, #link %s \n top10:%s"%(
            args.fea1,
            args.fea2,
            mean_ci,
            avg_acc1,
            meancc1,
            argsort1[0], sort1[0],
            args.topk, meantopk1,
            l0_weight_12,
            ", ".join(["%d:%4.2f"%(k,a) for k,a in zip(argsort1[:args.topk],sort1[:args.topk])])
        ))
        print("All %s-->%s, rmse %4.2f, meancc:%4.2f, cc best %s/%4.2f, top %d:%4.2f, #link %s \n top10:%s"%(
            args.fea1,
            args.fea2,
            avg_acc,
            meancc,
            argsort[0], sort[0],
            args.topk, meantopk,
            l0_weight,
            ", ".join(["%d:%4.2f"%(k,a) for k,a in zip(argsort[:args.topk],sort[:args.topk])])
        ))
    #
    pass
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='BiomeAE')
    parser.add_argument("--fea1", type=str, default='bac_group_fea')
    parser.add_argument("--fea2", type=str, default='met_fea')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: x/sum(x)")
    parser.add_argument("--cross_val_folds", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)

    parser.add_argument("--sparse", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="tanh_None")
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=1, type=bool)
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')
    parser.add_argument("--nonneg_weight", action='store_true')
    parser.add_argument("--normalize_input", action='store_true')

    parser.add_argument('--gpu_num', default="0", type=str)
    parser.add_argument("--extra", type=str, default='')

    parser.add_argument("--visualize", type=str, default='subplot')
    parser.add_argument("--draw_graph", action='store_true')
    args = parser.parse_args()

    main(args)
