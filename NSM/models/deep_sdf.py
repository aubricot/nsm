import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import warnings

PROGRESSIVE_PARAMS = {
    "n_layers": 3,
    "layers": {
        5: {  # Or -3, -2, -1....
            "start_epoch": 200,
            "warmup_epochs": 200,
        },
        6: {  # Or -3, -2, -1....
            "start_epoch": 600,
            "warmup_epochs": 200,
        },
        7: {  # Or -3, -2, -1....
            "start_epoch": 1010,
            "warmup_epochs": 200,
        },
    },
}


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        n_objects=1,
        dropout=None,
        dropout_prob=0.2,
        norm_layers=(),  # DEPRECATED
        latent_in=(),
        weight_norm=True,
        # batch_norm=False,
        xyz_in_all=None,
        activation="relu",  # "relu" or "sin"
        final_activation="tanh",  # "sin", "linear"
        concat_latent_input=False,
        progressive_add_depth=False,
        progressive_depth_params=PROGRESSIVE_PARAMS,
        latent_noise_sigma=None,
        layer_split=None,
        **kwargs,
    ):
        """
        latent_size (int): size of the latent input vector to the decoder network
        dims (list of ints): list containing the size of each layer in MLP.
        n_objects (int): number of objects to predict
        dropout (list of ints): where to apply dropout to the encoder
        dropout_prob (float) : probability with which dropout is applied
        norm_layers (list of ints): where to apply weightnorm/batchnorm to the decoder
        latent_in (list of ints): where to repeat the latent vector in the decoder
        weight_norm (bool): whether to apply weight normalization
        xyz_in_all (bool): for deepSDF decoder, include XYZ at each layer
        use_tanh (bool): for deepSDF decoder, tanh final layer to [0, 1]
        """
        super(Decoder, self).__init__()

        if "latent_dropout" in kwargs:
            warnings.warn("latent_dropout is deprecated. Use dropout instead.", DeprecationWarning)

        # DEPRECATED
        self.norm_layers = norm_layers

        def make_sequence():
            return []

        self._activation_ = activation
        self._final_activation_ = final_activation
        self.concat_latent_input = concat_latent_input
        self.dims = [latent_size + 3] + dims
        self.latent_in = latent_in
        self.progressive_add_depth = progressive_add_depth
        self.progressive_depth_params = progressive_depth_params
        self.latent_noise_sigma = latent_noise_sigma
        self.layer_split = layer_split
        self.n_objects = n_objects

        # layers:
        # 0: input
        # 1 to N-hidden: NN
        # -1: output
        if (n_objects == 1) or ((n_objects > 1) and (self.layer_split is not None)):
            self.dims = self.dims + [1]
        else:
            self.dims = self.dims + [n_objects]

        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()

        # Add the rest of the layers
        for layer in range(len(self.dims) - 1):
            # get layer input and output dimensions
            in_dim, out_dim = self.get_layer_dims(layer)
            if self.layer_split is not None and layer >= self.layer_split:
                lin_layer = nn.ModuleList()
                for _ in range(self.n_objects):
                    lin_layer.append(
                        self.lin_layer_(
                            in_dim=in_dim, out_dim=out_dim, layer=layer, weight_norm=weight_norm
                        )
                    )
            else:
                lin_layer = self.lin_layer_(
                    in_dim=in_dim, out_dim=out_dim, layer=layer, weight_norm=weight_norm
                )
            self.layers.append(lin_layer)

        self.activation = get_activation(self._activation_)
        self.final_activation = get_activation(self._final_activation_)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.epoch = None

    def lin_layer_(self, in_dim, out_dim, layer, weight_norm):
        lin_layer = nn.Linear(in_dim, out_dim)
        # initialize the weights - particularly for the sine activation
        init_weights(module=lin_layer, activation=self._activation_, first_layer=layer == 0)
        # add weight norm if specified
        # if weight_norm is True and layer in self.norm_layers:
        if weight_norm is True:
            lin_layer = nn.utils.weight_norm(lin_layer)
        elif self.norm_layers is not None and layer in self.norm_layers:
            self.bn.append(nn.LayerNorm(out_dim))
        return lin_layer

    def get_layer_dims(self, layer):
        if self.concat_latent_input == False:
            in_dim = self.dims[layer]
            if layer + 1 in self.latent_in:
                out_dim = self.dims[layer + 1] - self.dims[0]
            else:
                out_dim = self.dims[layer + 1]
        elif self.concat_latent_input == True:
            out_dim = self.dims[layer + 1]
            if layer in self.latent_in:
                in_dim = self.dims[layer] + self.dims[0]
            else:
                in_dim = self.dims[layer]
        else:
            in_dim = self.dims[layer]
            out_dim = self.dims[layer + 1]

        return in_dim, out_dim

    def forward_branch_(self, x, input_, layer, layer_idx):
        if layer_idx in self.latent_in:
            xi = torch.cat([x, input_], 1)
        else:
            xi = x

        if (self.progressive_add_depth is True) and (
            layer_idx in self.progressive_depth_params["layers"]
        ):
            if self.epoch >= self.progressive_depth_params["layers"][layer_idx]["start_epoch"]:
                x = self.progressive_layer(xi, layer, layer_idx)
            else:
                return
        else:
            x = layer(xi)

        # only apply normalization/ regular activation to
        # hidden layers (not output)
        if layer_idx < len(self.layers) - 1:

            if len(self.bn) > 0 and layer_idx in self.norm_layers:
                x = self.bn[layer_idx](x)
            x = self.activation(x)

            if (
                self.dropout is not None and layer_idx in self.dropout
            ):  # and (self._activation_ != "sin")
                x = F.dropout(x, p=self.dropout_prob, training=self.training)

        return x

    # input: N x (L+3)
    def forward(self, input_, epoch=None):
        # Assign the epoch in case needed (for progressive depth)
        if epoch is not None:
            self.epoch = epoch

        xyz = input_[:, -3:]
        x = input_

        for layer_idx, layer in enumerate(self.layers):  # range(0, self.num_layers - 1):
            if self.layer_split is not None and layer_idx >= self.layer_split:

                if layer_idx == self.layer_split:
                    x = [
                        x,
                    ] * self.n_objects

                for i in range(self.n_objects):
                    x_ = self.forward_branch_(
                        x=x[i], input_=input_, layer=layer[i], layer_idx=layer_idx
                    )
                    if x_ is None:
                        continue
                    else:
                        x[i] = x_
            else:
                x = self.forward_branch_(x=x, input_=input_, layer=layer, layer_idx=layer_idx)

            if x is None:
                continue

        if self.layer_split is not None:
            x = torch.cat(x, dim=1)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def progressive_layer(self, xi, layer, layer_idx):
        # use this as a way to store the progress of the network so far.
        # this way if we try to use the partly tuned model for inference, it will be able to
        # use the weights that have been tuned so far.

        # progresive tuning of latter layers is from Curriculum DeepSDF
        # code was adapted from:
        # https://github.com/haidongz-usc/Curriculum-DeepSDF
        start = self.progressive_depth_params["layers"][layer_idx]["start_epoch"]
        warmup = self.progressive_depth_params["layers"][layer_idx]["warmup_epochs"]
        end = start + warmup
        if self.epoch < start:
            raise exception("Epoch is before start of progressive depth")
        elif start < self.epoch < end:
            # during warmup... linearly phase this block in
            # https://github.com/haidongz-usc/Curriculum-DeepSDF/blob/ca216dda8edc6435139a6f657c45800791be94a7/networks/deep_sdf_decoder_train.py#L113
            new_weight = (self.epoch - start) / warmup
            new_weight = new_weight**2
            base_weight = 1 - new_weight
            # base_weight = ((end-self.epoch)/warmup)

            x_base = xi * base_weight
            x_new = layer(xi) * new_weight
            x = x_base + x_new
        else:
            # after start + warmup epochs just apply this block as a normal layer
            x = layer(xi)

        return x


def init_weights(module, activation, first_layer=False):
    """
    Initializes the weights of a linear layer based on the activation function.
    """
    if isinstance(module, nn.Linear):
        num_input = module.weight.size(-1)
        if activation == "sin":
            with torch.no_grad():
                if first_layer is True:
                    b = 1 / num_input
                elif first_layer is False:
                    b = np.sqrt(6 / num_input) / 30

                torch.nn.init.uniform_(module.weight, -b, b)


def weight_norm_all(module):
    """
    Applies weight normalization to all linear layers in a PyTorch module.
    """

    def apply_weight_norm(module):
        if isinstance(module, nn.Linear):
            nn.utils.weight_norm(module)

    module.apply(apply_weight_norm)


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "swish":
        return nn.SiLU()
    elif activation == "sin":
        return Sine()
    elif activation == "linear":
        return None
    else:
        raise ValueError(f"Unknown activation function: {activation}")
