import torch
from torch import nn
from .deep_sdf import Decoder
from torch.nn.functional import grid_sample
from contextlib import contextmanager
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

        # Sticky plane features cache for L-BFGS optimization
        self._sticky_latent = None      # 1D copy for exact comparison
        self._sticky_feats = None       # (C,H,W) on current device with gradients
        self._sticky_hit_count = 0      # Count cache hits
        self._sticky_miss_count = 0     # Count cache misses
        self._sticky_vae_time = 0.0     # Time spent in VAE decoder
        self._sticky_cache_time = 0.0   # Time spent checking/using cache

    def clear_sticky(self):
        """Clear the sticky plane features cache."""
        self._sticky_latent = None
        self._sticky_feats = None
    
    def get_sticky_stats(self):
        """Get sticky cache statistics."""
        total_calls = self._sticky_hit_count + self._sticky_miss_count
        hit_rate = self._sticky_hit_count / max(total_calls, 1) * 100
        return {
            'hits': self._sticky_hit_count,
            'misses': self._sticky_miss_count,
            'hit_rate': hit_rate,
            'vae_time': self._sticky_vae_time,
            'cache_time': self._sticky_cache_time,
            'total_calls': total_calls
        }
    
    def reset_sticky_stats(self):
        """Reset sticky cache statistics."""
        self._sticky_hit_count = 0
        self._sticky_miss_count = 0
        self._sticky_vae_time = 0.0
        self._sticky_cache_time = 0.0

    def _same_latent_as_sticky(self, z: torch.Tensor) -> bool:
        """Check if the given latent matches the cached sticky latent."""
        if self._sticky_latent is None:
            return False
        # Require exact match; we don't want to reuse across *any* change.
        return torch.equal(z.detach(), self._sticky_latent)

    def prime(self, latent: torch.Tensor, device: torch.device = None):
        """
        Precompute & store plane features for a single latent.
        Keep the graph intact for gradient flow.
        
        Args:
            latent: (D,) or (1,D) latent vector
            device: target device for plane features (defaults to model device)
        """
        start_time = time.time()
        z = latent.squeeze(0) if latent.dim() == 2 else latent
        
        vae_start = time.time()
        feats = self.vae_decoder(z.unsqueeze(0)).squeeze(0)    # NO no_grad, NO detach
        vae_time = time.time() - vae_start
        
        self._sticky_latent = z.detach().clone()               # value copy for equality check only
        self._sticky_feats = feats.to(device or next(self.parameters()).device)
        self._sticky_vae_time += vae_time
        
        total_time = time.time() - start_time
        logging.debug(f"STICKY: Primed cache with latent (VAE: {vae_time:.4f}s, Total: {total_time:.4f}s)")

    def forward_with_plane_features(self, plane_features, query):
        """

        args:
            plane_features: (N, 3 * sdf_latent_size, H, W)
            query: (N, 3)
        """
        interpolation_start = time.time()

        latent_size = (
            self.sdf_latent_size + self.conv_pred_sdf
        )  # one sdf prediction per plane (if conv_pred_sdf is True)

        feat_xz = plane_features[:latent_size, ...]
        feat_yz = plane_features[latent_size : latent_size * 2, ...]
        feat_xy = plane_features[latent_size * 2 :, ...]

        sample_start = time.time()
        plane_feats_list = []
        plane_feats_list.append(self.sample_plane_features(query, feat_xz, "xz"))
        plane_feats_list.append(self.sample_plane_features(query, feat_yz, "yz"))
        plane_feats_list.append(self.sample_plane_features(query, feat_xy, "xy"))
        sample_time = time.time() - sample_start

        combine_start = time.time()
        if self.sum_sdf_features is True:
            plane_feats = 0
            # sum plane features for each point
            for plane_feat in plane_feats_list:
                plane_feats += plane_feat

        elif self.sum_sdf_features is False:
            plane_feats = torch.cat(plane_feats_list, dim=1)
        
        combine_time = time.time() - combine_start
        total_interpolation_time = time.time() - interpolation_start
        
        logging.debug(f"TRIPLANAR: Interpolation for {query.shape[0]} points - "
                     f"Sampling: {sample_time:.4f}s, Combine: {combine_time:.4f}s, "
                     f"Total: {total_interpolation_time:.4f}s")

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

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False, sticky=False):
        """
        Forward pass through the triplanar decoder.
        
        Args:
            x: Input tensor with latent codes and xyz coordinates (legacy interface)
            latent: Single latent vector (D,) or (1,D) - for inference mode
            xyz: Query points (N,3) - for inference mode  
            epoch: Current training epoch (for logging)
            verbose: Whether to print debug information
            sticky: If True, cache plane features for repeated inference with same latent.
        
        Note: 
            - Use either x OR (latent + xyz), not both
            - Using (latent + xyz) is much faster for inference with single latent
        """
        forward_start = time.time()
        
        # Handle different input modes
        if (latent is not None and xyz is not None):
            # Fast inference mode: separate latent and xyz
            if x is not None:
                raise ValueError("Cannot specify both x and (latent, xyz). Use one interface or the other.")
            
            parse_start = time.time()
            # Ensure latent is 1D for consistency
            if latent.dim() == 2:
                latent = latent.squeeze(0)
            if latent.dim() != 1:
                raise ValueError(f"latent must be 1D or (1,D), got shape {latent.shape}")
            if xyz.dim() != 2 or xyz.shape[1] != 3:
                raise ValueError(f"xyz must be (N,3), got shape {xyz.shape}")
                
            parse_time = time.time() - parse_start
            
            # Fast path: use custom autograd function that properly handles gradient expansion
            unique_start = time.time()
            # Apply FastUnique function that preserves correct gradient flow
            unique_latent = FastUnique.apply(latent, xyz.shape[0])
            unique_indices = torch.zeros(xyz.shape[0], dtype=torch.long, device=xyz.device)
            num_unique = 1
            unique_time = time.time() - unique_start
            
            logging.debug(f"INFERENCE_MODE: Fast path with single latent {latent.shape} for {xyz.shape[0]} points")
            logging.debug(f"GRADIENT_CHECK: latent.requires_grad={latent.requires_grad}, unique_latent.requires_grad={unique_latent.requires_grad}")
            
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

            # 1. Input parsing
            parse_start = time.time()
            xyz = x[:, -3:]
            latent = x[:, :-3:]
            parse_time = time.time() - parse_start
            
            # 2. Unique latent computation for legacy mode
            unique_start = time.time()
            
            # TEMPORARILY DISABLE optimization to test if it's breaking gradients
            # Always use the original unique_consecutive method for now
            unique_latent, unique_indices = unique_consecutive(latent, 0, True)
            num_unique = unique_latent.shape[0]
            logging.debug(f"UNIQUE_OPT: Using original unique_consecutive, found {num_unique} unique latents")
                
            unique_time = time.time() - unique_start
            
        else:
            raise ValueError("Must specify either x OR (latent, xyz)")

        # 3. Feature computation (sticky or normal path)
        feature_start = time.time()
        if sticky:
            cache_start = time.time()
            
            # enforce one-latent-per-call in sticky mode
            if num_unique != 1:
                raise ValueError("sticky=True requires a single unique latent per call.")
            
            # Add safety check to prevent sticky usage during training
            if self.training:
                raise RuntimeError("sticky=True is intended for LBFGS optimization only, not training.")

            z = unique_latent[0]

            # if latent changed: drop old sticky & compute new
            if not self._same_latent_as_sticky(z):
                logging.debug(f"STICKY: Cache MISS - recomputing plane features for {xyz.shape[0]} points")
                self.clear_sticky()
                
                # Recompute with graph intact
                vae_start = time.time()
                feats = self.vae_decoder(z.unsqueeze(0)).squeeze(0)  # no detach
                vae_time = time.time() - vae_start
                
                self._sticky_latent = z.detach().clone()
                self._sticky_feats = feats
                self._sticky_miss_count += 1
                self._sticky_vae_time += vae_time
                
                logging.debug(f"STICKY: VAE computation took {vae_time:.4f}s for features shape {feats.shape}")
            else:
                logging.debug(f"STICKY: Cache HIT - reusing plane features for {xyz.shape[0]} points")
                self._sticky_hit_count += 1

            # reuse sticky features for all points in this batch  
            interp_start = time.time()
            plane_feats_for_points = self.forward_with_plane_features(self._sticky_feats, xyz)
            interp_time = time.time() - interp_start
            
            cache_time = time.time() - cache_start
            self._sticky_cache_time += cache_time
            
            # Log performance stats every 10 calls
            total_calls = self._sticky_hit_count + self._sticky_miss_count
            if total_calls % 10 == 0:
                stats = self.get_sticky_stats()
                logging.debug(f"STICKY: Stats after {total_calls} calls - Hit rate: {stats['hit_rate']:.1f}%, "
                           f"VAE time: {stats['vae_time']:.3f}s, Cache time: {stats['cache_time']:.3f}s")

        else:
            # original multi-latent path (no sticky behavior)
            vae_start = time.time()
            per_unique_feats = self.vae_decoder(unique_latent)  # (U,C,H,W)
            vae_time = time.time() - vae_start
            
            interp_start = time.time()
            pts_latents = []
            for idx in range(num_unique):
                feats = per_unique_feats[idx]
                pts = xyz[unique_indices == idx, :]
                pts_latents.append(self.forward_with_plane_features(feats, pts))
            plane_feats_for_points = torch.cat(pts_latents, dim=0)
            interp_time = time.time() - interp_start
            
        feature_time = time.time() - feature_start

        if self.conv_pred_sdf:
            low_freq_sdf = plane_feats_for_points[:, :1]
            plane_feats_for_points = plane_feats_for_points[:, 1:]

        sdf_prep_start = time.time()
        sdf_features = torch.cat([plane_feats_for_points, xyz], dim=1)
        sdf_prep_time = time.time() - sdf_prep_start
        
        sdf_mlp_start = time.time()
        sdf = self.sdf_decoder(sdf_features)
        sdf_mlp_time = time.time() - sdf_mlp_start

        if self.conv_pred_sdf:
            sdf = sdf + low_freq_sdf

        logging.debug(f"SDF_MLP: Input prep: {sdf_prep_time:.4f}s, MLP forward: {sdf_mlp_time:.4f}s "
                     f"for {xyz.shape[0]} points, input_dim: {sdf_features.shape[1]}")

        # Overall timing breakdown
        total_forward_time = time.time() - forward_start
        logging.debug(f"FORWARD_BREAKDOWN: Total: {total_forward_time:.4f}s, "
                     f"Parse: {parse_time:.4f}s ({parse_time/total_forward_time*100:.1f}%), "
                     f"Unique: {unique_time:.4f}s ({unique_time/total_forward_time*100:.1f}%), "
                     f"Features: {feature_time:.4f}s ({feature_time/total_forward_time*100:.1f}%), "
                     f"SDF_prep: {sdf_prep_time:.4f}s ({sdf_prep_time/total_forward_time*100:.1f}%), "
                     f"SDF_MLP: {sdf_mlp_time:.4f}s ({sdf_mlp_time/total_forward_time*100:.1f}%)")
        
        unaccounted_forward = total_forward_time - (parse_time + unique_time + feature_time + sdf_prep_time + sdf_mlp_time)
        logging.debug(f"FORWARD_BREAKDOWN: Unaccounted: {unaccounted_forward:.4f}s ({unaccounted_forward/total_forward_time*100:.1f}%)")

        return sdf


@contextmanager
def lbfgs_sticky_step(decoder, z):
    """
    Context manager for safe L-BFGS optimization with sticky features.
    
    Args:
        decoder: TriplanarDecoder instance
        z: Latent tensor to optimize
    """
    logging.debug(f"STICKY: Entering L-BFGS sticky context for latent shape {z.shape}")
    start_time = time.time()
    
    decoder.clear_sticky()
    decoder.reset_sticky_stats()  # Reset stats for this L-BFGS step
    decoder.prime(z)  # keeps graph
    
    try:
        yield
    finally:
        step_time = time.time() - start_time
        stats = decoder.get_sticky_stats()
        
        logging.debug(f"STICKY: L-BFGS step completed in {step_time:.4f}s")
        logging.debug(f"STICKY: Final stats - Hits: {stats['hits']}, Misses: {stats['misses']}, "
                   f"Hit rate: {stats['hit_rate']:.1f}%, VAE time: {stats['vae_time']:.4f}s")
        
        if stats['hits'] > 0:
            speedup = stats['vae_time'] / (stats['vae_time'] / max(stats['misses'], 1)) * stats['total_calls']
            theoretical_speedup = 1.0 + (stats['hit_rate'] / 100.0) * (stats['vae_time'] / stats['cache_time'])
            logging.debug(f"STICKY: Estimated speedup from caching: {theoretical_speedup:.2f}x")
        
        decoder.clear_sticky()
