import torch
import torch.nn as nn

from L0_regularization.models import L0MLP
from utils import idx2onehot

def make_mlp(type, dim_list, activation=None, batch_norm=True, dropout=0, N=200):
    layers = []
    if type is None:
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)
    elif type in ["L0"]:
        l0mlp = L0MLP(dim_list[0], dim_list[-1], layer_dims=dim_list[1:-1],N=N)
        return l0mlp
    else:
        assert (0, "unknown mlp type=%s"%type)


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).cuda()
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, z, means, log_var

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).cuda()

        recon_x = self.decoder(z, c)

        return recon_x

class AE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size,decoder_layer_sizes, activation, batch_norm, dropout,
                 mlp_type = None, conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, activation, batch_norm, dropout, conditional, num_labels, mlp_type=mlp_type)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, activation, batch_norm, dropout, conditional, num_labels, mlp_type=mlp_type)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        batch_size = x.size(0)

        z = self.encoder(x, c)

        recon_x = self.decoder(z, c)

        return recon_x,  z, None, None

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).cuda()

        recon_x = self.decoder(z, c)

        return recon_x
    def param_l0(self):
        return {"Encoder":self.encoder.param_l0(),
                "Decoder":self.decoder.param_l0(),}
class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, activation, batch_norm, dropout,  conditional, num_labels, mlp_type=None):

        super().__init__()
        self.mlp_type = mlp_type
        self.latent_size = latent_size
        self.layer_sizes = layer_sizes
        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = make_mlp(
                mlp_type,
                layer_sizes+[latent_size],
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )
    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        z = self.MLP(x)

        return z

    def param_l0(self):
        if self.mlp_type == "L0":
            return self.MLP.get_exp_flops_l0()[1]
        else:
            #Count all non-zeros param
            nonzero_count = 0
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    nonzero_count +=(layer.weight!=0.0).sum().item()
            return nonzero_count

class Decoder(nn.Module):

    def __init__(self, layer_sizes,latent_size, activation ,batch_norm, dropout, conditional, num_labels, mlp_type=None):

        super().__init__()
        self.mlp_type = mlp_type
        self.latent_size = latent_size
        self.layer_sizes = layer_sizes
        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        self.MLP = make_mlp(
                mlp_type,
                [latent_size]+layer_sizes,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10).cuda()
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x

    def param_l0(self):
        if self.mlp_type == "L0":
            return self.MLP.get_exp_flops_l0()[1]
        else:
            #Count all non-zeros param
            nonzero_count = 0
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    nonzero_count +=(layer.weight!=0.0).sum().item()
            return nonzero_count
