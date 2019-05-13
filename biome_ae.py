import os
import copy
import time
import torch
import torch.nn as nn
import pickle
import argparse
import types
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import logging
import tensorboard_logger as tl

from sklearn.cross_decomposition import CCA
from models import  AE, VAE
from snip.snip import snip_forward_linear, snip_forward_conv2d

DATA_ROOT = "/data/BioHealth/IBD"
output_dir = os.path.join(DATA_ROOT, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
parser = argparse.ArgumentParser()
FORMAT = '[%(asctime)s: %(filename)s: %(lineno)4d]: %(message)s'
#logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
#logger = logging.getLogger(__name__)

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


class BiomeCCA():
    def __init__(self, args):
        self.CCA = CCA(n_components=args.latent_size)
    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        return self.CCA.fit(X1_train, X2_train)
    def transform(self,x1_train, x2_train):
        return self.CCA.transform(x1_train, x2_train)
    def predict(self,X1_val, X2_val, y_val, args):
        return self.CCA.predict(X1_val)
    def get_transformation(self):
        return self.CCA.coef_
    def param_l0(self):
        return {"Encoder":self.CCA.coef_.shape[0]*self.CCA.n_components,
                "Decoder":self.CCA.n_components*self.CCA.coef_.shape[1]}
        return self.CCA.coef_.shape[0]

class BiomeAE():
    def __init__(self, args):
        if args.model in ["BiomeAEL0"]:
            self.mlp_type = "L0"
        else:
            self.mlp_type = None

        self.model_alias = 'DeepBiome_%s+%s_%s+fea1_%s+fea2_%s+sparse_%s+bs_%s+%s' % (
            args.model,
            args.dataset_name, args.data_type,
            args.fea1,args.fea2,
            args.sparse,
            args.batch_size,
            args.extra)
        self.model= args.model

        #tl.configure("runs/ds.{}".format(model_alias))
        #tl.log_value(model_alias, 0)

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

        self.predictor = None

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #logger.info("Initializing train dataset")


    def transform(self,x1_train, x2_train):
        #TODO: tranform of a pytorhc net???
        assert("Not implemented")

    def get_transformation(self):
        return None

    def loss_fn(self, recon_x, x, mean, log_var):
        if self.model in ["BiomeAE","BiomeAESnip", "BiomeAEL0"]:
            mseloss = torch.nn.MSELoss()
            return torch.sqrt(mseloss(recon_x, x))
        elif self.model =="BiomeVAE":
            BCE = torch.nn.functional.binary_cross_entropy(
                recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return (BCE + KLD) / x.size(0)

    def param_l0(self):
        return self.predictor.param_l0()

    def init_fit(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val, args, ):
        self.train_loader = get_dataloader (X1_train, X2_train, y_train, args.batch_size)
        self.test_loader = get_dataloader(X1_val, X2_val, y_val, args.batch_size)

        if args.model in ["BiomeAE","BiomeAESnip", "BiomeAEL0"]:
            self.predictor = AE(
                encoder_layer_sizes=[X1_train.shape[1]],
                latent_size=args.latent_size,
                decoder_layer_sizes=[X2_train.shape[1]],
                activation=args.activation,
                batch_norm= args.batch_norm,
                dropout=args.dropout,
                mlp_type=self.mlp_type,
                conditional=args.conditional,
                num_labels=10 if args.conditional else 0).to(self.device)
        elif args.model == "BiomeVAE":
            self.predictor = VAE(
                encoder_layer_sizes=args.encoder_layer_sizes,
                latent_size=args.latent_size,
                decoder_layer_sizes=args.decoder_layer_sizes,
                activation=args.activation,
                batch_norm=args.batch_norm,
                dropout=args.dropout,
                conditional=args.conditional,
                num_labels=10 if args.conditional else 0).to(self.device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=args.learning_rate)

    def train(self, args):
        ts = time.time()
        logs = defaultdict(list)
        for epoch in range(args.epochs):

            tracker_epoch = defaultdict(lambda: defaultdict(dict))

            for iteration, (x1, x2, y) in enumerate(self.train_loader):

                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                if args.conditional:
                    x2_hat, z, mean, log_var = self.predictor(x1, y)
                else:
                    x2_hat, z, mean, log_var = self.predictor(x1)

                for i, yi in enumerate(y):
                    id = len(tracker_epoch)
                    tracker_epoch[id]['x'] = z[i, 0].item()
                    tracker_epoch[id]['y'] = z[i, 1].item()
                    tracker_epoch[id]['label'] = yi.item()

                loss = self.loss_fn(x2_hat, x2, mean, log_var)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logs['loss'].append(loss.item())

                if iteration % args.print_every == 0 or iteration == len(self.train_loader)-1:
                    print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        epoch, args.epochs, iteration, len(self.train_loader)-1, loss.item()))
                    if args.model =="VAE":
                        if args.conditional:
                            c = torch.arange(0, 10).long().unsqueeze(1)
                            x = self.predictor.inference(n=c.size(0), c=c)
                        else:
                            x = self.predictor.inference(n=10)

    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        self.init_fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val, args)
        self.train(args)

    def predict(self,X1_val, X2_val, y_val, args):
        #Batch test
        x1, x2, y = torch.FloatTensor(X1_val).to(self.device), torch.FloatTensor(X2_val).to(self.device), torch.FloatTensor(y_val).to(self.device)
        if args.conditional:
            x2_hat, z, mean, log_var = self.predictor(x1, y)
        else:
            x2_hat, z, mean, log_var = self.predictor(x1)
        val_loss = self.loss_fn( x2_hat, x2, mean, log_var)
        print("val_loss: {:9.4f}", val_loss.item())
        return x2_hat.detach().numpy()
        """
        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        g = sns.lmplot(
            x='x1', y='x2', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)
        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)

        """

class BiomeAESnip(BiomeAE):
    def __init__(self, args):
        super(BiomeAESnip, self).__init__(args)
        self.keep_ratio = float(args.sparse)

    def fit(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val, args, ):
        self.init_fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val, args)

        #SNIP IT!
        # Grab a single batch from the training dataset
        x1, x2, y = next(iter(self.train_loader))
        x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
        # Let's create a fresh copy of the network so that we're not worried about
        # affecting the actual training-phase
        net = copy.deepcopy(self.predictor)
        # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
        # instead of the weights
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False

            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

        # Compute gradients (but don't apply them)
        net.zero_grad()
        if args.conditional:
            x2_hat, z, mean, log_var = net(x1, y)
        else:
            x2_hat, z, mean, log_var = net(x1)

        loss = self.loss_fn(x2_hat, x2, mean, log_var)
        loss.backward()

        grads_abs = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight_mask.grad))

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        num_params_to_keep = int(len(all_scores) * self.keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        keep_masks = []
        for g in grads_abs:
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

        self.apply_prune_mask(keep_masks)

        #train after pruning
        self.train(args)

    def apply_prune_mask(self, keep_masks):

        # Before I can zip() layers and pruning masks I need to make sure they match
        # one-to-one by removing all the irrelevant modules:
        prunable_layers = filter(
            lambda layer: isinstance(layer, torch.nn.Conv2d) or isinstance(
                layer, torch.nn.Linear), self.predictor.modules())

        for layer, keep_mask in zip(prunable_layers, keep_masks):
            assert (layer.weight.shape == keep_mask.shape)

            def hook_factory(keep_mask):
                """
                The hook function can't be defined directly here because of Python's
                late binding which would result in all hooks getting the very last
                mask! Getting it through another function forces early binding.
                """

                def hook(grads):
                    return grads * keep_mask

                return hook

            # mask[i] == 0 --> Prune parameter
            # mask[i] == 1 --> Keep parameter

            # Step 1: Set the masked weights to zero (NB the biases are ignored)
            # Step 2: Make sure their gradients remain zero
            layer.weight.data[keep_mask == 0.] = 0.
            layer.weight.register_hook(hook_factory(keep_mask))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='AE')
    parser.add_argument("--dataset_name", type=str, default='ibd')
    parser.add_argument("--data_type", type=str, default='clr', help="clr: log(x) - mean(log(x)), 0-1: log (x/sum(x)))")
    parser.add_argument("--fea1", type=str, default='met_W_pca')
    parser.add_argument("--fea2", type=str, default='genus_fea')

    parser.add_argument("--sparse", type=str, default="l0")
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

