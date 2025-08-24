from torch import nn
import torch

from .triplanar import TriplanarDecoder
from .deep_sdf import Decoder

default_triplanar_params = {
    "latent_dim": 256,
    "n_objects": 2,
    "conv_hidden_dims": [512, 512, 512, 512, 512],
    "conv_deep_image_size": 2,
    "conv_norm": True,
    "conv_norm_type": "layer",
    "conv_start_with_mlp": True,
    "sdf_latent_size": 128,
    "sdf_hidden_dims": [512, 512, 512],
    "sdf_weight_norm": True,
    "sdf_final_activation": "tanh",
    "sdf_activation": "relu",
}

default_mlp_params = {
    "latent_size": 256,
    "dims": (512, 512, 512, 512, 512, 512, 512, 512),
    "n_objects": 2,
    "dropout": None,
    "dropout_prob": 0.0,
    "norm_layers": (0, 1, 2, 3, 4, 5, 6, 7),  # DEPRECATED
    "latent_in": (),
    "weight_norm": True,
    "xyz_in_all": None,
    "activation": "relu",  # "relu" or "sin"
    "final_activation": "tanh",  # "sin", "linear"
    "concat_latent_input": True,
}


class TwoStageDecoder(nn.Module):
    """
    Create a two stage model that takes in a latent vector and 3d coordinates and
    outputs the SDF for each point.

    It takes 1/2 of the latent vector and passes it through a triplanar decoder
    It takes the other 1/2 of the latent vector and passes it through an MLP

    These outputs both predict the SDF for each points, the outputs are then summed
    and returned as the final SDF prediction.
    """

    def __init__(
        self,
        latent_size=512,
        n_objects=2,
        triplanar_params: dict = default_triplanar_params,
        mlp_params: dict = default_mlp_params,
    ):
        super(TwoStageDecoder, self).__init__()

        self.latent_size = latent_size
        self.model_latent_size = latent_size // 2
        assert latent_size % 2 == 0, "latent_size must be even"

        self.n_objects = n_objects

        triplanar_params["latent_dim"] = self.model_latent_size
        triplanar_params["n_objects"] = self.n_objects
        mlp_params["latent_size"] = self.model_latent_size
        mlp_params["n_objects"] = self.n_objects

        self.triplanar_params = triplanar_params
        self.mlp_params = mlp_params

        self.triplanar = TriplanarDecoder(**triplanar_params)
        self.mlp = Decoder(**mlp_params)

    def forward(self, input, epoch=None):
        # Split the latent vector in half
        latent_triplanar = input[:, : self.model_latent_size]
        latent_mlp = input[:, self.model_latent_size : self.model_latent_size * 2]

        # get the xyz coordinates
        xyz = input[:, -3:]

        # Pass the latent vector  & xyz through the triplanar decoder
        sdf_triplanar = self.triplanar(torch.cat([latent_triplanar, xyz], dim=-1))
        # Pass the other half of the latent vector & xyz through the MLP
        sdf_mlp = self.mlp(torch.cat([latent_mlp, xyz], dim=-1))

        # Sum the outputs
        sdf = sdf_triplanar + sdf_mlp

        return sdf
