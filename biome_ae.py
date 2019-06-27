import os
import copy
import time
import torch
import torch.nn as nn
import pickle
import argparse
import types
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

from sklearn.cross_decomposition import CCA
from sklearn import linear_model
from models import  AE, VAE, MLPerceptron, Decoder
from snip.snip import snip_forward_linear, snip_forward_conv2d

def get_dataloader(X1, X2, y, batch_size, shuffle=True):
    X1_tensor = torch.FloatTensor(X1)
    X2_tensor = torch.FloatTensor(X2)
    y_tensor = torch.LongTensor(y)
    ds = TensorDataset(X1_tensor, X2_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

class BiomeLasso():
    def __init__(self, args):
        self.linears = []
    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        alpha =  1- float(args.sparse) if args.sparse is not None else 1.0
        for i, x2 in enumerate(X2_train.T):
            l =linear_model.Lasso(alpha =alpha, positive=args.nonneg_weight)
            l.fit(X1_train, x2)
            self.linears.append(l)

    def transform(self,x1_train, x2_train, y_train, args):
        #Because this is linear model, transform does nothing
        return (x1_train, x2_train)
    def predict(self,X1_val, X2_val, y_val, args):
        X2_hat = []
        for i, x2 in enumerate(X2_val.T):
             X2_hat.append(self.linears[i].predict(X1_val))
        return np.array(X2_hat).T

    def get_transformation(self):
        return np.array([self.linears[i].coef_ for i in range(len(self.linears))])

    def param_l0(self):
        non_zero_params = (np.array([self.linears[i].coef_ for i in range(len(self.linears))]) !=0).sum()
        return {"Transform":non_zero_params}
    def get_graph(self):
        """
        return nodes and weights
        :return:
        """
        nodes = []
        weights = []
        out_size = len(self.linears)
        in_size = len(self.linears[0].coef_)
        nodes.append(list(range(in_size)))
        nodes.append(list(range(out_size)))
        weights.append(self.get_transformation().T)

        return (nodes, weights)

class BiomeLinear():
    def __init__(self, args):
        self.linear = linear_model.LinearRegression()
    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        self.linear.fit(X1_train, X2_train)
    def transform(self,x1_train, x2_train):
        #Because this is linear model, transform does nothing
        return (x1_train, x2_train)
    def predict(self,X1_val, X2_val, y_val, args):
        return self.linear.predict(X1_val)
    def get_transformation(self):
        return self.linear.coef_

    def param_l0(self):
        non_zero_params = (self.linear.coef_ !=0).sum()
        return {"Transform":non_zero_params}
    def get_graph(self):
        """
        return nodes and weights
        :return:
        """
        T = self.get_transformation().T
        nodes = [list(range(T.shape[0])),list(range(T.shape[1]))]
        weights = [T]

        return (nodes, weights)


class BiomeMultiTaskLasso():
    def __init__(self, args):
        alpha = 1 - float(args.sparse) if args.sparse is not None else 1.0
        self.linear = linear_model.MultiTaskLasso(alpha=alpha)
    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        self.linear.fit(X1_train, X2_train)

class BiomeLassoAESnip():
    """
    This is the hydrid between lasso and AE. Only components can't be explained by lasso will go through AE
    """
    def __init__(self, args):
        assert args.normalize_input,"BiomeLassoAESnip does not work with normalize_input option"
        self.args_lasso = copy.deepcopy(args)
        self.args_lasso.model = "BiomeLasso"
        self.args_lasso.normalize_input = False

        self.args_ae = copy.deepcopy(args)
        self.args_ae.model = "BiomeAESnip"

        self.lasso = BiomeLasso(self.args_lasso)
        self.aesnip = BiomeAESnip(self.args_ae)
        self.switch_threshold = 0.5
    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        #Fit a lasso
        self.lasso.fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val, self.args_lasso)
        X2_hat  = self.lasso.predict(X1_val, X2_val, y_val, self.args_lasso)
        # Select the component that are good to use
        test_corr = np.nan_to_num([np.corrcoef(X2_hat[:, i], X2_val[:, i])[0, 1] for i in range(X2_val.shape[1])])
        argsort = np.argsort(-test_corr)  # descending
        sort1 = test_corr[argsort]
        passed_lasso = sort1 > self.switch_threshold
        self.passed_lasso = np.where(passed_lasso)

        #Get the residue
        residue = ~passed_lasso
        X2_train_res = X2_train[:,residue]
        X2_val_res = X2_val[:,residue]

        #Train a ae snip
        # Center and Scale to use tanh
        mu1 = X1_train.mean(axis=0)
        mu2 = X2_train_res.mean(axis=0)
        std1 = X1_train.std(axis=0, ddof=1)
        std1[std1 == 0.0] = 1.0
        std2 = X2_train_res.std(axis=0, ddof=1)
        std2[std2 == 0.0] = 1.0

        X1_train = (X1_train - mu1)/std1
        X1_val = (X1_val - mu1) / std1

        X2_train_res = (X2_train_res - mu2)/std2
        X2_val_res = (X2_val_res - mu2)

        self.aesnip.fit(X1_train, X2_train_res, y_train, X1_val, X2_val_res, y_val, self.args_ae)


    def transform(self,x1_train, x2_train):
        #Because this is linear model, transform does nothing
        return (x1_train, x2_train)
    def predict(self,X1_val, X2_val, y_val, args):
        X2_hat = []
        for i, x2 in enumerate(X2_val.T):
             X2_hat.append(self.linears[i].predict(X1_val))
        return np.array(X2_hat).T

    def get_transformation(self):
        return np.array([self.linears[i].coef_ for i in range(len(self.linears))])

    def param_l0(self):
        non_zero_params = (np.array([self.linears[i].coef_ for i in range(len(self.linears))]) !=0).sum()
        return {"Transform":non_zero_params}
    def get_graph(self, topk = None):
        """
        return nodes and weights
        :return:
        """
        nodes = []
        weights = []
        out_size = len(self.linears)
        in_size = len(self.linears[0].coef_)
        nodes.append(list(range(in_size)))
        nodes.append(list(range(out_size)))
        weights.append(self.get_transformation().T)

        return (nodes, weights)


class BiomeCCA():
    def __init__(self, args):
        self.CCA = CCA(n_components=args.latent_size)
    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        if args.latent_size > min(X1_train.shape[1], X2_train.shape[1]):
            print("Warning: auto reduce latent size")
            self.CCA = CCA(n_components=min(X1_train.shape[1], X2_train.shape[1]))
        return self.CCA.fit(X1_train, X2_train)
    def transform(self,x1_train, x2_train, y, args):
        return self.CCA.transform(x1_train, x2_train)
    def predict(self,X1_val, X2_val, y_val, args):
        return self.CCA.predict(X1_val)
    def get_transformation(self):
        return self.CCA.coef_
    def param_l0(self):
        return {"Encoder":self.CCA.coef_.shape[0]*self.CCA.n_components,
                "Decoder":self.CCA.n_components*self.CCA.coef_.shape[1]}
    def get_graph(self):
        return ([],[])
class BiomeOneLayerManual():
    def __init__(self, args):
        self.sparse = float(args.sparse) if args.sparse is not None else 1.0
        self.mlp_type = None
        self.model_alias = args.model_alias
        self.model= args.model

        #tl.configure("runs/ds.{}".format(model_alias))
        #tl.log_value(model_alias, 0)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
        self.predictor = None
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def transform(self,x1_train, x2_train):
        #TODO: tranform of a pytorhc net???
        assert("Not implemented")

    def get_transformation(self):
        return None

    def loss_fn(self, recon_x, x, mean, log_var):
        if self.model in ["BiomeOneLayer"]:
            mseloss = torch.nn.MSELoss()
            #l1 reg?
            l1_crit = nn.L1Loss(size_average=False)
            reg_loss = 0
            for layer in self.predictor.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    target = torch.zeros(layer.weight.shape).to(self.device)
                    reg_loss += l1_crit(layer.weight, target)

            return torch.sqrt(mseloss(recon_x, x))+(1-self.sparse)*reg_loss

        if self.model in ["BiomeAEL0"]:
            mseloss = torch.nn.MSELoss()
            return torch.sqrt(mseloss(recon_x, x))+self.predictor.regularization()

    def param_l0(self):
        return self.predictor.param_l0()

    def init_fit(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val, args, ):
        self.train_loader = get_dataloader (X1_train, X2_train, y_train, args.batch_size)
        self.test_loader = get_dataloader(X1_val, X2_val, y_val, args.batch_size)

        self.predictor = MLPerceptron(
                layer_sizes=[X1_train.shape[1],X2_train.shape[1]],
                activation=None,
                batch_norm= args.batch_norm,
                dropout=args.dropout,
                mlp_type=self.mlp_type,).to(self.device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=args.learning_rate)

    def train(self, args):
        ts = time.time()
        logs = defaultdict(list)
        for epoch in range(args.epochs):

            tracker_epoch = defaultdict(lambda: defaultdict(dict))

            for iteration, (x1, x2, y) in enumerate(self.train_loader):

                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                x2_hat = self.predictor(x1)

                loss = self.loss_fn(x2_hat, x2, None, None)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logs['loss'].append(loss.item())

                if iteration % args.print_every == 0 or iteration == len(self.train_loader)-1:
                    print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        epoch, args.epochs, iteration, len(self.train_loader)-1, loss.item()))

    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        self.init_fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val, args)
        self.train(args)

    def predict(self,X1_val, X2_val, y_val, args):
        #Batch test
        x1, x2, y = torch.FloatTensor(X1_val).to(self.device), torch.FloatTensor(X2_val).to(self.device), torch.FloatTensor(y_val).to(self.device)
        x2_hat = self.predictor(x1)
        val_loss = self.loss_fn( x2_hat, x2, None, None)
        print("val_loss: {:9.4f}", val_loss.item())
        return x2_hat.detach().cpu().numpy()

class BiomeOneLayer():
    def __init__(self, args):
        self.mlp_type = None
        self.model_alias = args.model_alias
        self.model= args.model
        self.snap_loc = os.path.join(args.vis_dir, "snap.pt")

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
        self.predictor = None

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_transformation(self):
        return None

    def loss_fn(self, recon_x, x):
        mseloss = torch.nn.MSELoss()
        return torch.sqrt(mseloss(recon_x, x))

    def param_l0(self):
        return self.predictor.param_l0()

    def init_fit(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val, args, ):
        self.train_loader = get_dataloader (X1_train, X2_train, y_train, args.batch_size)
        self.test_loader = get_dataloader(X1_val, X2_val, y_val, args.batch_size)

        self.predictor = Decoder(
            latent_size=X1_train.shape[1],
            layer_sizes=[X2_train.shape[1]],
            activation=args.activation,
            batch_norm= args.batch_norm,
            dropout=args.dropout,
            mlp_type=self.mlp_type,
            conditional=args.conditional,
            num_labels=10 if args.conditional else 0).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)

    def train(self, args):
        if args.contr:
            print("Loading from ", self.snap_loc)
            loaded_model_para = torch.load(self.snap_loc)
            self.predictor.load_state_dict(loaded_model_para)

        t = 0
        logs = defaultdict(list)
        iterations_per_epoch = len(self.train_loader.dataset) / args.batch_size
        num_iterations = int(iterations_per_epoch * args.epochs)
        for epoch in range(args.epochs):

            tracker_epoch = defaultdict(lambda: defaultdict(dict))

            for iteration, (x1, x2, y) in enumerate(self.train_loader):
                t+=1

                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                if args.conditional:
                    x2_hat = self.predictor(x1, y)
                else:
                    x2_hat = self.predictor(x1, None)

                loss = self.loss_fn(x2_hat, x2)

                self.optimizer.zero_grad()
                loss.backward()
                if (t + 1) % int(num_iterations / 10) == 0:
                    self.scheduler.step()
                self.optimizer.step()

                #enforce non-negative
                if args.nonneg_weight:
                    for layer in self.predictor.modules():
                        if isinstance(layer, nn.Linear):
                            layer.weight.data.clamp_(0.0)


                logs['loss'].append(loss.item())

                if iteration % args.print_every == 0 or iteration == len(self.train_loader)-1:
                    print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        epoch, args.epochs, iteration, len(self.train_loader)-1, loss.item()))

        if not args.contr:
            print("Saving to ", self.snap_loc)
            torch.save(self.predictor.state_dict(), self.snap_loc)

    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        self.init_fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val, args)
        self.train(args)

    def get_graph(self):
        """
        return nodes and weights
        :return:
        """
        nodes = []
        weights = []
        for l, layer in enumerate(self.predictor.modules()):
            if isinstance(layer, nn.Linear):
                lin_layer =layer
                nodes.append(["%s"%(x) for x in list(range(lin_layer.in_features))])
                weights.append(lin_layer.weight.detach().cpu().numpy().T)
        nodes.append(["%s"%(x) for x in list(range(lin_layer.out_features))]) #last linear layer

        return (nodes, weights)

    def predict(self,X1_val, X2_val, y_val, args):
        #Batch test
        x1, x2, y = torch.FloatTensor(X1_val).to(self.device), torch.FloatTensor(X2_val).to(self.device), torch.FloatTensor(y_val).to(self.device)
        if args.conditional:
            x2_hat = self.predictor(x1, y)
        else:
            x2_hat = self.predictor(x1)
        val_loss = self.loss_fn( x2_hat, x2)
        print("val_loss: {:9.4f}", val_loss.item())
        return x2_hat.detach().cpu().numpy()

    def transform(self,X1_val, X2_val, y_val, args):
        x1, x2, y = torch.FloatTensor(X1_val).to(self.device), torch.FloatTensor(X2_val).to(
            self.device), torch.FloatTensor(y_val).to(self.device)
        if args.conditional:
            x2_hat = self.predictor(x1, y)
        else:
            x2_hat = self.predictor(x1)

        return x2_hat.detach().cpu().numpy()

    def get_influence_matrix(self):
        return self.predictor.get_influence_matrix()


class BiomeOneLayerSnip(BiomeOneLayer):
    def __init__(self, args):
        super(BiomeOneLayerSnip, self).__init__(args)
        self.keep_ratio = float(args.sparse)

    def fit(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val, args, ):
        self.init_fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val, args)

        #SNIP IT!
        # Use all data to estimate the gradient
        X1_tensor = torch.FloatTensor(X1_train).to(self.device)
        X2_tensor = torch.FloatTensor(X2_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)

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
            x2_hat = net(X1_tensor, y_tensor)
        else:
            x2_hat = net(X1_tensor, None)

        loss = self.loss_fn(x2_hat, X2_tensor)
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

class BiomeAE():
    def __init__(self, args):
        if args.model in ["BiomeAEL0"]:
            self.mlp_type = "L0"
        else:
            self.mlp_type = None

        self.model_alias = args.model_alias
        self.model= args.model
        self.snap_loc = os.path.join(args.vis_dir, "snap.pt")

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
        self.predictor = None

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def get_transformation(self):
        return None

    def loss_fn(self, recon_x, x, mean, log_var):
        if self.model in ["BiomeAE","BiomeAESnip"]:
            mseloss = torch.nn.MSELoss()
            return torch.sqrt(mseloss(recon_x, x))
        if self.model in ["BiomeAEL0"]:
            mseloss = torch.nn.MSELoss()
            return torch.sqrt(mseloss(recon_x, x))+self.predictor.regularization()
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
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)

    def train(self, args):
        if args.contr:
            print("Loading from ", self.snap_loc)
            loaded_model_para = torch.load(self.snap_loc)
            self.predictor.load_state_dict(loaded_model_para)

        t = 0
        logs = defaultdict(list)
        iterations_per_epoch = len(self.train_loader.dataset) / args.batch_size
        num_iterations = int(iterations_per_epoch * args.epochs)
        for epoch in range(args.epochs):

            tracker_epoch = defaultdict(lambda: defaultdict(dict))

            for iteration, (x1, x2, y) in enumerate(self.train_loader):
                t+=1

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
                if (t + 1) % int(num_iterations / 10) == 0:
                    self.scheduler.step()
                self.optimizer.step()

                #enforce non-negative
                if args.nonneg_weight:
                    for layer in self.predictor.modules():
                        if isinstance(layer, nn.Linear):
                            layer.weight.data.clamp_(0.0)


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

        if not args.contr:
            print("Saving to ", self.snap_loc)
            torch.save(self.predictor.state_dict(), self.snap_loc)

    def fit(self,X1_train, X2_train, y_train, X1_val, X2_val, y_val, args,):
        self.init_fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val, args)
        self.train(args)

    def get_graph(self):
        """
        return nodes and weights
        :return:
        """
        nodes = []
        weights = []
        for l, layer in enumerate(self.predictor.modules()):
            if isinstance(layer, nn.Linear):
                lin_layer =layer
                nodes.append(["%s"%(x) for x in list(range(lin_layer.in_features))])
                weights.append(lin_layer.weight.detach().cpu().numpy().T)
        nodes.append(["%s"%(x) for x in list(range(lin_layer.out_features))]) #last linear layer

        return (nodes, weights)

    def predict(self,X1_val, X2_val, y_val, args):
        #Batch test
        x1, x2, y = torch.FloatTensor(X1_val).to(self.device), torch.FloatTensor(X2_val).to(self.device), torch.FloatTensor(y_val).to(self.device)
        if args.conditional:
            x2_hat, z, mean, log_var = self.predictor(x1, y)
        else:
            x2_hat, z, mean, log_var = self.predictor(x1)
        val_loss = self.loss_fn( x2_hat, x2, mean, log_var)
        print("val_loss: {:9.4f}", val_loss.item())
        return x2_hat.detach().cpu().numpy()

    def transform(self,X1_val, X2_val, y_val, args):
        x1, x2, y = torch.FloatTensor(X1_val).to(self.device), torch.FloatTensor(X2_val).to(
            self.device), torch.FloatTensor(y_val).to(self.device)
        if args.conditional:
            x2_hat, z, mean, log_var = self.predictor(x1, y)
        else:
            x2_hat, z, mean, log_var = self.predictor(x1)

        return z.detach().cpu().numpy()

    def get_influence_matrix(self):
        return self.predictor.get_influence_matrix()

class BiomeAESnip(BiomeAE):
    def __init__(self, args):
        super(BiomeAESnip, self).__init__(args)
        self.keep_ratio = float(args.sparse)

    def fit(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val, args, ):
        self.init_fit(X1_train, X2_train, y_train, X1_val, X2_val, y_val, args)

        #SNIP IT!
        # Use all data to estimate the gradient
        X1_tensor = torch.FloatTensor(X1_train).to(self.device)
        X2_tensor = torch.FloatTensor(X2_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)

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
            x2_hat, z, mean, log_var = net(X1_tensor, y_tensor)
        else:
            x2_hat, z, mean, log_var = net(X1_tensor)

        loss = self.loss_fn(x2_hat, X2_tensor, mean, log_var)
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



