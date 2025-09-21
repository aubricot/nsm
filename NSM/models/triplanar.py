import torch
from torch import nn
from .deep_sdf import Decoder
from torch.nn.functional import grid_sample
import time
import logging


"""
We will create a triplanar neural implicit representation model. 
First, we will create a VAE that takes a latent vector, reshapes it into 
a CX2x2 tensor, and then uses a 2D CNN to output a C2xHxH tensor that is a
set of 2D planar feature maps. We will use the first 1/3 of the channels
as features for the xy plane, the second 1/3 for the xz plane, and the last
1/3 for the yz plane. 

Then, we will train an MLP as a SDF decoder. Instead of only taking the xyz 
position of each point and a fixed latent code, we will sample the latent code
from the planar feature mapes outputted from the VAE. This will be done using 
summation of the latent codes from each plane using bilinear interpolation. This 
way, we get a specific latent code for each point in space.
"""


class VAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        out_features=128 * 3,
        hidden_dims=[512, 512, 512, 512, 512],
        deep_image_size=2,
        norm=True,
        norm_type="batch",
        activation="leakyrelu",
        start_with_mlp=True,
    ):
        super(VAEDecoder, self).__init__()

        # self.fc = nn.Linear(latent_dim, hidden_dims[0] * deep_image_size**2)

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.deep_image_size = deep_image_size
        self.out_features = out_features
        self.norm = norm
        self.norm_type = norm_type
        self.start_with_mlp = start_with_mlp

        if activation == "leakyrelu":
            activation_fn = nn.LeakyReLU
        elif activation == "relu":
            activation_fn = nn.ReLU

        assert (
            latent_dim % deep_image_size**2 == 0
        ), "latent_dim must be divisible by deep_image_size**2"

        self.layers = nn.ModuleList()

        if self.start_with_mlp is True:
            self.fc = nn.Linear(latent_dim, hidden_dims[0] * deep_image_size**2)
            in_channels = hidden_dims[0]
        else:
            in_channels = latent_dim // deep_image_size**2

        # decoder
        for i in range(len(hidden_dims)):

            out_channels = hidden_dims[i]

            conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            )
            self.layers.append(conv)
            # norm = nn.LayerNorm([out_channels, deep_image_size**(i+2), deep_image_size**(i+2)])
            if self.norm is True:
                if self.norm_type == "batch":
                    norm = nn.BatchNorm2d(out_channels)
                elif self.norm_type == "layer":
                    norm = nn.LayerNorm(
                        [out_channels, deep_image_size ** (i + 2), deep_image_size ** (i + 2)]
                    )
                else:
                    raise ValueError("norm_type must be 'batch' or 'layer'")
                self.layers.append(norm)

            activation = activation_fn()

            # set in_channels for next loop.
            in_channels = out_channels

        # finaly layer
        final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], out_channels=self.out_features, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.layers.append(final_layer)

        self.decoder = nn.Sequential(*self.layers)

    def forward(self, x):
        # reshape x into a 2D tensor

        if self.start_with_mlp is True:
            x = self.fc(x)
            x = x.view(-1, self.hidden_dims[0], self.deep_image_size, self.deep_image_size)

        if len(x.shape) in (1, 2):
            x = x.view(
                -1,
                self.latent_dim // self.deep_image_size**2,
                self.deep_image_size,
                self.deep_image_size,
            )
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 4:
            pass
        else:
            raise ValueError("x must be a 1D, 2D, 3D, or 4D tensor")

        return self.decoder(x)


class UniqueConsecutive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=0, return_inverse=True):
        unique, indices = torch.unique_consecutive(input, dim=dim, return_inverse=return_inverse)
        ctx.save_for_backward(indices)
        return unique, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices=None):
        (indices,) = ctx.saved_tensors
        # Count the occurrences of each unique row
        counts = torch.bincount(indices)
        # Expand grad_output according to counts
        expanded_grad = grad_output.repeat_interleave(counts, dim=0)

        return expanded_grad, None, None


unique_consecutive = UniqueConsecutive.apply


class FastUnique(torch.autograd.Function):
    """
    Fast autograd function that mimics unique_consecutive behavior for single latent.
    
    This provides the same gradient expansion as unique_consecutive but with minimal
    forward computation cost - just unsqueeze(0) instead of expensive unique operation.
    """
    @staticmethod
    def forward(ctx, latent_input, num_points):
        ctx.num_points = num_points
        return latent_input.unsqueeze(0)  # (1, D)
    
    @staticmethod 
    def backward(ctx, grad_output):
        # Expand gradient to match original input size (like unique_consecutive does)
        # grad_output: (1, D) -> expanded_grad: (num_points, D)
        expanded_grad = grad_output.repeat(ctx.num_points, 1)
        return expanded_grad, None


class TriplanarDecoder(nn.Module):
    """
    Triplanar neural implicit representation decoder.
    
    Combines a VAE decoder (latent -> 2D feature maps) with an SDF decoder (MLP).
    Uses triplanar interpolation to sample features from xy, xz, and yz planes.
    
    Performance Notes:
    - The main bottleneck is triplanar interpolation (~90% of forward pass time)
    - VAE computation is only ~5-10% of total time
    - Previously attempted feature caching optimization, but it provided minimal 
      speedup (~1.01-1.09x) due to low hit rates and wrong bottleneck targeting
    - Current optimization: FastUnique bypass for single-latent inference scenarios
    """
    def __init__(
        self,
        latent_dim,
        n_objects=1,
        conv_hidden_dims=[512, 512, 512, 512, 512],
        conv_deep_image_size=2,
        conv_norm=True,
        conv_norm_type="batch",
        conv_start_with_mlp=True,
        sdf_latent_size=128,
        sdf_hidden_dims=[512, 512, 512],
        sdf_weight_norm=True,
        sdf_final_activation="tanh",
        sdf_activation="relu",
        sdf_dropout_prob=0.0,
        sum_sdf_features=True,
        conv_pred_sdf=False,
        padding=0.1,
        **kwargs,
    ):
        super(TriplanarDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.n_objects = n_objects
        self.conv_hidden_dims = conv_hidden_dims
        self.conv_deep_image_size = conv_deep_image_size
        self.sdf_latent_size = sdf_latent_size
        self.sdf_hidden_dims = sdf_hidden_dims
        self.sdf_weight_norm = sdf_weight_norm
        self.sdf_final_activation = sdf_final_activation
        self.sdf_activation = sdf_activation
        self.sdf_dropout_prob = sdf_dropout_prob
        self.sum_sdf_features = sum_sdf_features
        self.padding = padding
        self.conv_pred_sdf = conv_pred_sdf

        if self.sum_sdf_features is False:
            assert (
                self.sdf_latent_size % 3 == 0
            ), "sdf_latent_size must be divisible by 3 if sum_sdf_features is True"
            vae_out_features = self.sdf_latent_size
        elif self.sum_sdf_features is True:
            vae_out_features = self.sdf_latent_size * 3

        if self.conv_pred_sdf is True:
            vae_out_features += 3

        self.vae_decoder = VAEDecoder(
            latent_dim=latent_dim,
            out_features=vae_out_features,
            hidden_dims=conv_hidden_dims,
            deep_image_size=conv_deep_image_size,
            norm=conv_norm,
            norm_type=conv_norm_type,
            start_with_mlp=conv_start_with_mlp,
        )

        self.sdf_decoder = Decoder(
            latent_size=self.sdf_latent_size,
            dims=self.sdf_hidden_dims,
            n_objects=self.n_objects,
            dropout=None if self.sdf_dropout_prob == 0 else list(range(len(self.sdf_hidden_dims))),
            dropout_prob=self.sdf_dropout_prob,
            weight_norm=self.sdf_weight_norm,
            activation=self.sdf_activation,  # "relu" or "sin"
            final_activation=self.sdf_final_activation,  # "sin", "linear"
            layer_split=None,
        )



    def forward_with_plane_features(self, plane_features, query):
        """
        Sample features from triplanar representation.

        Args:
            plane_features: (3 * sdf_latent_size, H, W) - triplanar feature maps
            query: (N, 3) - query points
            
        Returns:
            plane_feats: (N, sdf_latent_size) - sampled features
        """
        latent_size = (
            self.sdf_latent_size + self.conv_pred_sdf
        )  # one sdf prediction per plane (if conv_pred_sdf is True)

        feat_xz = plane_features[:latent_size, ...]
        feat_yz = plane_features[latent_size : latent_size * 2, ...]
        feat_xy = plane_features[latent_size * 2 :, ...]

        # Sample from each plane
        plane_feats_list = [
            self.sample_plane_features(query, feat_xz, "xz"),
            self.sample_plane_features(query, feat_yz, "yz"),
            self.sample_plane_features(query, feat_xy, "xy")
        ]

        # Combine features
        if self.sum_sdf_features:
            plane_feats = sum(plane_feats_list)
        else:
            plane_feats = torch.cat(plane_feats_list, dim=1)

        return plane_feats

    def sample_plane_features(self, query, plane_feature, plane):
        """
        args:
            query: (N, 3)
            plane_feature: (sdf_latent_size, H, W)
            plane: 'xz', 'yz', 'xy'

        return:
            sampled_feats: (N, sdf_latent_size)
        """
        # normalize coords to [-1, 1] & return
        grid = self.normalize_coordinates(query.clone(), plane=plane)

        sampled_feats = (
            grid_sample(
                input=plane_feature.unsqueeze(0),
                grid=grid,
                padding_mode="border",
                align_corners=True,
                mode="bilinear",
            )
            .squeeze(-1)
            .squeeze(0)
        )

        return sampled_feats.T

    def normalize_coordinates(self, query, plane, padding=0.1):
        if plane == "xy":
            xy = query[:, [0, 1]]
        elif plane == "xz":
            xy = query[:, [0, 2]]
        elif plane == "yz":
            xy = query[:, [1, 2]]
        else:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")

        xy_new = xy / (1 + self.padding + 10e-6)
        if xy_new.min() < -1:
            xy_new[xy_new < -1] = -1
        if xy_new.max() > 1:
            xy_new[xy_new > 1] = 1

        return xy_new[None, :, None, :]

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
        """
        Forward pass through the triplanar decoder.
        
        Args:
            x: Input tensor with latent codes and xyz coordinates (legacy interface)
            latent: Single latent vector (D,) or (1,D) - for fast inference
            xyz: Query points (N,3) - for fast inference  
            epoch: Current training epoch (for logging)
            verbose: Whether to print debug information
        
        Note: 
            - Use either x OR (latent + xyz), not both
            - Using (latent + xyz) is much faster for inference with single latent
        """
        
        # Handle different input modes
        if (latent is not None and xyz is not None):
            # Fast inference mode: separate latent and xyz
            if x is not None:
                raise ValueError("Cannot specify both x and (latent, xyz). Use one interface or the other.")
            
            # Ensure latent is 1D for consistency
            if latent.dim() == 2:
                latent = latent.squeeze(0)
            if latent.dim() != 1:
                raise ValueError(f"latent must be 1D or (1,D), got shape {latent.shape}")
            if xyz.dim() != 2 or xyz.shape[1] != 3:
                raise ValueError(f"xyz must be (N,3), got shape {xyz.shape}")
            
            # Fast path: use custom autograd function that properly handles gradient expansion
            unique_latent = FastUnique.apply(latent, xyz.shape[0])
            unique_indices = torch.zeros(xyz.shape[0], dtype=torch.long, device=xyz.device)
            num_unique = 1
            
        elif x is not None:
            # Legacy mode: concatenated input
            if latent is not None or xyz is not None:
                raise ValueError("Cannot specify both x and (latent, xyz). Use one interface or the other.")
                
            if verbose:
                print("Triplanar.forward()")
                print("Epoch:", epoch)
                print(f"Device: {x.device}")
                print(f"x shape: {x.shape}, dtype: {x.dtype}")
                if x.device.type == "cuda":
                    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

            # Input parsing
            xyz = x[:, -3:]
            latent = x[:, :-3:]
            
            # Unique latent computation for legacy mode  
            unique_latent, unique_indices = unique_consecutive(latent, 0, True)
            num_unique = unique_latent.shape[0]
                
        else:
            raise ValueError("Must specify either x OR (latent, xyz)")

        # Feature computation
        per_unique_feats = self.vae_decoder(unique_latent)  # (U,C,H,W)
        
        pts_latents = []
        for idx in range(num_unique):
            feats = per_unique_feats[idx]
            pts = xyz[unique_indices == idx, :]
            pts_latents.append(self.forward_with_plane_features(feats, pts))
        plane_feats_for_points = torch.cat(pts_latents, dim=0)

        if self.conv_pred_sdf:
            low_freq_sdf = plane_feats_for_points[:, :1]
            plane_feats_for_points = plane_feats_for_points[:, 1:]

        # Final SDF computation
        sdf_features = torch.cat([plane_feats_for_points, xyz], dim=1)
        sdf = self.sdf_decoder(sdf_features)

        if self.conv_pred_sdf:
            sdf = sdf + low_freq_sdf

        return sdf


