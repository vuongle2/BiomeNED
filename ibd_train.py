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

from models import CCA, AE, VAE

DATA_ROOT = "/data/BioHealth/IBD"

parser = argparse.ArgumentParser()
FORMAT = '[%(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def load_data(data_file):

    c2i = {'LumA': 1, 'LumB': 0}

    data = pd.read_pickle(data_file)
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

def get_dataloader(X, y, batch_size, shuffle=True):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

def main(args):

    model_alias = 'IBD+%s+%s_%s+%s+%s' % (
        args.model,
        args.dataset_name, args.dataset_subset,
        args.batch_size,
        args.extra)

    tl.configure("runs/ds.{}".format(model_alias))
    tl.log_value(model_alias, 0)

    stat_alias = 'obj_DataStat+%s_%s' % (args.dataset_name, args.dataset_subset)
    stat_path = os.path.join(
        args.output_dir, '%s.pkl' % (stat_alias)
    )
    with open(stat_path,'rb') as sf:
        data_stats = pickle.load(sf)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    data_path = os.path.join(DATA_ROOT, "{}-{}-zComp-clr.csv".format(args.dataset_name, args.dataset_subset))

    logger.info("Initializing train dataset")

    # load data
    print('==> loading data'); print()
    (X_train, y_train), (X_test, y_test) = load_data(data_path)
    train_loader = get_dataloader(X_train, y_train, args.batch_size)
    test_loader = get_dataloader(X_test, y_test, args.batch_size)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    dataset = MNIST(
        root='data', train=True, transform=transforms.ToTensor(),
        download=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if args.conditional:
                    c = torch.arange(0, 10).long().unsqueeze(1)
                    x = vae.inference(n=c.size(0), c=c)
                else:
                    x = vae.inference(n=10)

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

        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        g = sns.lmplot(
            x='x', y='y', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)
        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='AE')
    parser.add_argument("--dataset", type=str, default='ibd')
    parser.add_argument("--data_subset", type=str, default='x2-mic')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)
