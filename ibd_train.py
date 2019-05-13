import os
import sys
import time
import torch
import pickle
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import logging
import tensorboard_logger as tl

from models import  AE, VAE

DATA_ROOT = "/data/BioHealth/IBD"
output_dir = os.path.join(DATA_ROOT, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
parser = argparse.ArgumentParser()
FORMAT = '[%(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def load_data(data_file):
    with open(data_file,"rb") as df:
        content = pickle.load(df)

    X1 = content[args.fea1] # n x m1
    X2 = content[args.fea2] # n x m2
    Y = content["diagnosis"]

    #Suppress to two classes
    for i in range(len(Y)):
        if Y[i] !=0:
            Y[i] =1

    y_keys = [-1]+list(set(Y)) #-1 means all data
    y_ind  = {}
    y_ind[-1] = range(len(Y))
    for y_val in set(Y):
        y_ind[y_val] = [i for i in range(len(Y)) if Y[i] == y_val]

    #TODO: Cross validation
    #Split train and val
    val_ind = content["val_idx"]
    train_ind = content["train_idx"]
    X1_train = X1[train_ind]
    X1_val = X1[val_ind]
    X2_train = X2[train_ind]
    X2_val = X2[val_ind]
    y_train = Y[train_ind]
    y_val = Y[val_ind]

    return (X1_train, X2_train, y_train),(X1_val, X2_val, y_val)

def get_dataloader(X1, X2, y, batch_size, shuffle=True):
    X1_tensor = torch.FloatTensor(X1)
    X2_tensor = torch.FloatTensor(X2)
    y_tensor = torch.LongTensor(y)
    ds = TensorDataset(X1_tensor, X2_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

def main(args):

    model_alias = 'DeepBiome_%s+%s_%s+fea1_%s+fea2_%s+bs_%s+%s' % (
        args.model,
        args.dataset_name, args.data_type,
        args.fea1,args.fea2,
        args.batch_size,
        args.extra)

    tl.configure("runs/ds.{}".format(model_alias))
    tl.log_value(model_alias, 0)

    """ no stat file needed for now
    stat_alias = 'obj_DataStat+%s_%s' % (args.dataset_name, args.dataset_subset)
    stat_path = os.path.join(
        output_dir, '%s.pkl' % (stat_alias)
    )
    with open(stat_path,'rb') as sf:
        data_stats = pickle.load(sf)
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    data_path = os.path.join(DATA_ROOT, "ibd_{}.pkl".format(args.data_type))


    logger.info("Initializing train dataset")

    # load data
    print('==> loading data'); print()
    (X1_train, X2_train,  y_train), (X1_val, X2_val, y_val) = load_data(data_path)
    train_loader = get_dataloader (X1_train, X2_train, y_train, args.batch_size)
    test_loader = get_dataloader(X1_val, X2_val, y_val, args.batch_size)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    def loss_fn(model, recon_x, x, mean, log_var):
        if model =="AE":
            mseloss = torch.nn.MSELoss()
            return torch.sqrt(mseloss(recon_x, x))
        elif model =="VAE":
            BCE = torch.nn.functional.binary_cross_entropy(
                recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return (BCE + KLD) / x.size(0)

    if args.model == "AE":
        predictor = AE(
            encoder_layer_sizes=[X1_train.shape[1]],
            latent_size=args.latent_size,
            decoder_layer_sizes=[X2_train.shape[1]],
            activation=args.activation,
            batch_norm= args.batch_norm,
            dropout=args.dropout,
            conditional=args.conditional,
            num_labels=10 if args.conditional else 0).to(device)
    else:
        predictor = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            decoder_layer_sizes=args.decoder_layer_sizes,
            activation=args.activation,
            batch_norm=args.batch_norm,
            dropout=args.dropout,
            conditional=args.conditional,
            num_labels=10 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x1, x2, y) in enumerate(train_loader):

            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            if args.conditional:
                x2_hat, z, mean, log_var = predictor(x1, y)
            else:
                x2_hat, z, mean, log_var = predictor(x1)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(args.model, x2_hat, x2, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(train_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(train_loader)-1, loss.item()))
                if args.model =="VAE":
                    if args.conditional:
                        c = torch.arange(0, 10).long().unsqueeze(1)
                        x = predictor.inference(n=c.size(0), c=c)
                    else:
                        x = predictor.inference(n=10)
                """
                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(10):
                    plt.subplot(5, 2, p+1)
                    if args.conditional:
                        plt.text(
                            0, 0, "c={:d}".format(c[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].view(28, 28).cpu().data.numpy())
                    plt.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')
                """
        #Batch test
        x1, x2, y = torch.FloatTensor(X1_val).to(device), torch.FloatTensor(X2_val).to(device), torch.FloatTensor(y_val).to(device)
        if args.conditional:
            x2_hat, z, mean, log_var = predictor(x1, y)
        else:
            x2_hat, z, mean, log_var = predictor(x1)
        val_loss = loss_fn(args.model, x2_hat, x2, mean, log_var)
        print("val_loss: {:9.4f}", val_loss.item())
        """
        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        g = sns.lmplot(
            x='x1', y='x2', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)
        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)

        """
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='AE')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: log (x/sum(x)))")
    parser.add_argument("--fea1", type=str, default='met_W_pca')
    parser.add_argument("--fea2", type=str, default='genus_fea')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
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

    args = parser.parse_args()

    main(args)
