"""
Classes and helper functions shape completion, uncertainty estimation, and uncertainty evaluation.
"""

import numpy as np
from pymeshfix import MeshFix
import pyvista as pv
import torch
from torch.distributions import MultivariateNormal

from pymskt.mesh.meshTools import get_pt_cloud_distances, pcu_sdf

from NSM.datasets import SDFSamples
from NSM.mesh import create_mesh, decode_sdf
from NSM.helper_funcs import NumpyTransform, convert_ply_to_vtk
from NSM.optimization import get_top_k_pcs, optimize_latent_partial
from NSM.reconstruct.utils import compute_assd, compute_chamfer

# ======== HELPER FUNCTIONS ========
def load_mesh(mesh_path):
    """Load mesh as vtk, converting from ply and saving as vtk if needed."""
    # Convert ply to vtk if needed
    if '.ply' in mesh_path.lower():
        ply_fname = mesh_path
        mesh, mesh_path = convert_ply_to_vtk(ply_fname, save=True)
    else:
        mesh = pv.read(mesh_path)
    
    return mesh, mesh_path

def compute_optimal_latent(mesh_path, model, model_config, latent_codes, l2_loss=False, n_samples=240, device='cuda'):
    """Optimize and return latent code for mesh."""

    ## ---- Generate point samples from mesh ---- 
    print("\n----Preprocessing mesh and generating point samples----\n")

    # Setup a dataset with just this mesh (to use same preprocessing used during model training)
    sdf_dataset = SDFSamples(
        list_mesh_paths=[mesh_path],
        multiprocessing=False,
        subsample=model_config["samples_per_object_per_batch"],
        print_filename=True,
        n_pts=model_config["n_pts_per_object"],
        p_near_surface=model_config['percent_near_surface'],
        p_further_from_surface=model_config['percent_further_from_surface'],
        sigma_near=model_config['sigma_near'],
        sigma_far=model_config['sigma_far'],
        rand_function=model_config['random_function'], 
        center_pts=model_config['center_pts'],
        norm_pts=model_config['normalize_pts'],
        scale_method=model_config['scale_method'],
        scale_jointly=model_config['scale_jointly'],
        reference_mesh=None,
        verbose=model_config['verbose'],
        save_cache=model_config['cache'],
        equal_pos_neg=model_config['equal_pos_neg'],
        fix_mesh=model_config['fix_mesh'])

    # Get the point/SDF data
    sdf_sample = sdf_dataset[0]  # returns a dict
    sample_dict, _ = sdf_sample
    points = sample_dict['xyz'].to(device) # shape: [N, 3]
    sdf_vals = sample_dict['gt_sdf'].to(device)  # shape: [N, 1]

    # Use a random subset of the points for optimization
    indices = torch.randperm(points.size(0))[:n_samples] # Generate n_samples random indices
    # Downsample the points and SDF values
    points = points[indices].squeeze() # (N,3)
    sdf_vals = sdf_vals[indices].reshape(-1,1) # (N,1)

    ## ---- Optimize latent code for mesh_path ---- 
    print("\n----Optimizing latent code----\n")

    # Compute latent code statistics
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    latent_std = latent_codes.std().mean()

    # Optimize latents
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.95)
    # Phase 1 - Coarse Optimization - get a global shape in the right area of latent space (close to target specimen (far enough from mean); but not so far from mean that it is noisy or unrealistic)
    latent_partial, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, model_config['latent_size'], 
                                                mean_latent=mean_latent, latent_init=latent_codes, top_k=top_k_reg, 
                                                iters=7000, lr=1e-4, lambda_reg=1e-3, clamp_val=2.0, 
                                                latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.8, 
                                                batch_inference_size=32768, multi_stage=False, device=device,
                                                l2_loss=l2_loss)
    # Phase 2 - Refinement - emphasis on local SDF samples and surface consistency to refine target specimen shape
    latent_partial, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, model_config['latent_size'],
                                                latent_init=latent_partial, top_k=top_k_reg, 
                                                iters=10000, lr=1e-5, lambda_reg=1e-3, clamp_val=2.0, 
                                                latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.8, 
                                                batch_inference_size=32768, multi_stage=True, device=device,
                                                l2_loss=l2_loss) # True because second stage using already initialized latent
    print("\nTranslated novel mesh into latent space!\n")
    
    return latent_partial

def reconstruct_mesh(latent_code, model, device='cuda', verbose=True):
    """Reconstruct mesh from latent code and repair/clean the result."""
    ## ---- Reconstruct mesh from latent code ----
    if verbose:
        print("\n----Reconstructing mesh from latent code----\n")
    
    # Mesh reconstruction settings
    recon_grid_origin = 1.0
    n_pts_per_axis = 256
    voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
    voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
    offset = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    icp_transform = NumpyTransform(np.eye(4))
    objects = 1

    # Mesh reconstruction
    recon_mesh_raw = create_mesh(
        decoder=model, latent_vector=latent_code,
        n_pts_per_axis=n_pts_per_axis,
        voxel_origin=voxel_origin, voxel_size=voxel_size,
        path_original_mesh=None,
        offset=offset, scale=scale, icp_transform=icp_transform,
        objects=objects, verbose=False, device=device,
        path_save=None,
        smooth=1.0, scale_to_original_mesh=False)
    
    ## ---- Postprocess reconstructed mesh ----
    if verbose:
        print("\n----Cleaning and repairing mesh if necessary----\n")

    # Ensure it's PyVista PolyData
    if isinstance(recon_mesh_raw, list):
        recon_mesh_raw = recon_mesh_raw[0]
    if not isinstance(recon_mesh_raw, pv.PolyData):
        recon_mesh = recon_mesh_raw.extract_surface() # Extract surface geometry of the mesh as PolyData. PolyData is a dataset consisting of surface geometry (e.g. vertices, lines, and polygons).
    else:
        recon_mesh = recon_mesh_raw

    # Repair, clean, and triangulate
    recon_mesh = recon_mesh.clean() # This merges duplicate points, removes unused points, and/or removes degenerate cells.
    recon_mesh = recon_mesh.triangulate() # Return an all triangle mesh. More complex polygons will be broken down into triangles.
    if not recon_mesh.is_all_triangles:
        mfix = MeshFix(recon_mesh)
        mfix.repair()
        recon_mesh = mfix.mesh

    # Remove region label given when converting SDF to mesh (via mesh_pv.connectivity(largest=True))
    if 'RegionId' in recon_mesh.point_data:
        recon_mesh.point_data.remove('RegionId')
    if 'RegionId' in recon_mesh.cell_data:
        recon_mesh.cell_data.remove('RegionId')

    return recon_mesh
    
def unit_scale_mesh(mesh):
    """Return a copy of the input mesh scaled to unit radius."""
    scale = np.max(np.linalg.norm(mesh.points, axis=-1), axis=-1)
    unit_mesh = mesh.scale(1.0/scale)
    return unit_mesh

def decimate_mesh(mesh, n_triangles=5000, verbose=True):
    """Reduce triangle (and vertex) count using PyVista's decimate method."""
    if n_triangles is not None and n_triangles > 0:
        if verbose:
            print(f"\n----Decimating mesh to target triangle count----")
        # Decimate mesh to reduce vertex count
        reduction_factor = max(0.0, 1.0 - n_triangles/mesh.n_faces_strict)
        mesh_dec = mesh.decimate(reduction_factor)
        # mesh_dec = mesh.decimate_pro(reduction_factor)
    else:
        # Use mesh as is
        mesh_dec = mesh.copy()
    if verbose:
        print("Number of points remaining:", mesh_dec.n_points)
        print("Number of triangles remaining:", mesh_dec.n_faces_strict)
    return mesh_dec

# ======== LAPLACE APPROXIMATOR ========
class LaplaceApproximator:
    """Laplace approximation of posterior distribution of optimal latent code."""
    
    def __init__(self, decoder, latent_code, mesh_path, model_config, device, verbose=False):
        self.decoder = decoder
        self.latent_code = latent_code
        self.jacobian = None
        self.mesh_path = mesh_path
        self.model_config = model_config
        self.device = device
        self.verbose = verbose
    
    def covariance(self, data_std=0.01, latent_std=0.01, data_weight=1.0, latent_weight=1.0, n_samples=1000, recompute_jacobian=False):
        """Compute covariance matrix."""
        # Compute Hessian of loss function at the distribution mean (precision matrix) using Gauss-Newton approximation, then invert it
        # - data_std: Expected data noise
        # - latent_std: Expected latent noise
        # - data_weight: Weight coefficient for data term
        # - latent_weight: Weight coefficient for latent term
        if self.verbose:
            print("\n----Computing covariance matrix----")
        if self.jacobian is None or recompute_jacobian:
            self.jacobian = self._residuals_jacobian(n_samples)
        with torch.no_grad():
            if data_weight != 0:
                recon_term = (1.0 / (data_std**2)) * self.jacobian.T @ self.jacobian
            else:
                recon_term = 0
            if latent_weight != 0:
                reg_term = (1.0 / (latent_std**2)) * torch.eye(self.jacobian.shape[1], device=self.device)
            else:
                reg_term = 0
            hessian = data_weight * recon_term + latent_weight * reg_term
            lower = torch.linalg.cholesky(hessian)
            covariance = torch.cholesky_inverse(lower)
            return covariance, hessian
        
    def _residuals_jacobian(self, n_samples=5000):
        """Compute Jacobian of SDF residuals wrt latent elements, at narrowband samples."""
        
        # Generate samples in the mesh surface's narrowband
        narband_pts, narband_sdfs = self._sample_mesh_narrowband(n_samples)
        
        if self.verbose:
            print("\nComputing Jacobian of residual at each point")
        
        # Prepare tensors for autograd
        self.decoder.eval()
        latent_code = self.latent_code.to(self.device).requires_grad_()
        narband_pts, narband_sdfs = narband_pts.to(self.device), narband_sdfs.to(self.device)
        
        # Compute Jacobian of residuals wrt latent elements
        jacobian = torch.empty((narband_pts.shape[0], latent_code.shape[1]), device=self.device)
        residual_sum = 0
        for i in range(narband_pts.shape[0]):
            if self.verbose:
                print(f"Processing point {i+1}/{len(narband_pts)}", end='\r')
            pred = self.decoder(torch.cat([latent_code, narband_pts[None,i]], dim=1))
            residual = pred - narband_sdfs[i] # Technically unnecessary because dresidual/dz ==  dpreds/dz
            residual_sum += abs(residual)
            residual.backward(inputs=latent_code)
            jacobian[i] = latent_code.grad
            self.decoder.zero_grad() # Technically unnecessary because only inputs=latent_code has gradients accumulated
            latent_code.grad.zero_() 

        return jacobian

    def _sample_mesh_narrowband(self, n_samples=5000):
        """Generate (point, sdf) samples in the mesh surface's narrowband."""
        # Future: 
        # - Choose downsampling method: Random subsample, mesh decimation, downsample_partial_pointcloud, _uniform_surface_sample,  ...
        # - Choose narrowband SDF sampling method: sample_near_surface, pcu.signed_distance_to_mesh, mesh_to_sdf.sample_sdf_near_surface, ...
         
        if self.verbose:
            print(f"\nSampling SDF in mesh surface's narrowband")

        # Setup a dataset with just this mesh (to use same preprocessing used during model training)
        sdf_dataset = SDFSamples(
            list_mesh_paths=[self.mesh_path],
            multiprocessing=False,
            subsample=self.model_config["samples_per_object_per_batch"],
            print_filename=True,
            n_pts=self.model_config["n_pts_per_object"],
            p_near_surface=self.model_config['percent_near_surface'],
            p_further_from_surface=self.model_config['percent_further_from_surface'],
            sigma_near=self.model_config['sigma_near'],
            sigma_far=self.model_config['sigma_far'],
            rand_function=self.model_config['random_function'], 
            center_pts=self.model_config['center_pts'],
            norm_pts=self.model_config['normalize_pts'],
            scale_method=self.model_config['scale_method'],
            scale_jointly=self.model_config['scale_jointly'],
            reference_mesh=None,
            verbose=self.model_config['verbose'],
            save_cache=self.model_config['cache'],
            equal_pos_neg=self.model_config['equal_pos_neg'],
            fix_mesh=self.model_config['fix_mesh'])

        # Get the point/SDF data
        sdf_sample = sdf_dataset[0]  # returns a dict
        sample_dict, _ = sdf_sample
        points = sample_dict['xyz'].to(self.device) # shape: [N, 3]
        sdf_vals = sample_dict['gt_sdf'].to(self.device)  # shape: [N, 1]

        # Use a random subset of the points for optimization
        indices = torch.randperm(points.size(0))[:n_samples] # Generate n_samples random indices
        # Downsample the points and SDF values
        points = points[indices]
        points = points.squeeze()
        sdf_vals = sdf_vals[indices]
        sdf_vals = sdf_vals.reshape(-1, 1)

        return points, sdf_vals


# ======== UNCERTAINTY ESTIMATORS ========
class UncertaintyEstimator:
    """Uncertainty estimation base class. Should not be instantiated directly."""
    
    def __init__(self, decoder, latent_code, device, verbose):
        self.decoder = decoder
        self.latent_code = latent_code
        self.device = device
        self.verbose = verbose

class AnalyticalUncertainty(UncertaintyEstimator):
    """SDF and positional uncertainty estimation by propagation of latent covariance via local linearization analysis."""

    def __init__(self, decoder, latent_code, device, verbose=False):
        super().__init__(decoder, latent_code, device, verbose)
        self.sdf_grad_latent = None
        self.sdf_grad_pos = None

    def sdf_uncertainty(self, covariance, points=None, recompute_gradients=False):
        """Compute SDF uncertainty via local linearization analysis."""
        if self.verbose:
            print("\n----Computing SDF uncertainty (via local linearization analysis)----")
        sdf_variance = self._sdf_variance(covariance, points, recompute_gradients)
        with torch.no_grad():
            sdf_std = torch.sqrt(sdf_variance)
            return sdf_std
    
    def pos_uncertainty(self, covariance, sdf_uncertainty=None, points=None, recompute_gradients=False):
        """Compute positional uncertainty along the surface normal."""
        if sdf_uncertainty is None:
            sdf_variance = self._sdf_variance(covariance, points, recompute_gradients)
        else:
            sdf_variance = sdf_uncertainty**2
        if self.verbose:
            print("\n----Computing positional uncertainty along the normal (via local linearization analysis)----")
        with torch.no_grad():
            pos_variance = sdf_variance / torch.sum(torch.square(self.sdf_grad_pos), dim=1)
            pos_std = torch.sqrt(pos_variance)
            return pos_std

    def _sdf_variance(self, covariance, points=None, recompute_gradients=False):
        """Compute SDF variance via local linearization analysis."""
        if self.sdf_grad_latent is None or recompute_gradients:
            self.sdf_grad_latent, self.sdf_grad_pos = self._sdf_gradients_at_points(points)
        with torch.no_grad():
            sdf_variance = torch.einsum('ij,jk,ik->i', self.sdf_grad_latent, covariance, self.sdf_grad_latent)
            return sdf_variance
        
    def _sdf_gradients_at_points(self, points):
        """Compute and store gradients of predicted SDF wrt latent elements and position."""
        if self.verbose:
            print(f"\nComputing gradients of predicted SDF wrt latent elements and position")

        if points is None:
            raise ValueError("Must specify points to compute SDF gradients!")

        # Prepare tensors for autograd
        if not torch.is_tensor(self.latent_code):
            latent_code = torch.tensor(self.latent_code).to(self.device).requires_grad_()
        else:
            latent_code = self.latent_code.detach().clone().to(self.device).requires_grad_()
        if not torch.is_tensor(points):
            points = torch.tensor(points).to(self.device).requires_grad_()
        else:
            points = points.detach().clone().to(self.device).requires_grad_()

        # Compute SDF gradients
        # - Local gradient of SDF wrt optimal latent, or ds/dz at z* at each point
        # - Local gradient of SDF wrt surface position, or ds/dx at z* at each point
        sdf_grad_latent = torch.empty((points.shape[0], latent_code.shape[1]), device=self.device) # (N,K)
        sdf_grad_pos = torch.empty((points.shape[0], points.shape[1]), device=self.device) # (N,3)
        for i in range(points.shape[0]):
            if self.verbose:
                print(f"Processing point {i+1}/{len(points)}", end='\r')
            query_pt = points[None,i]
            pred = decode_sdf(self.decoder, latent_code, query_pt)
            pred.backward(inputs=[latent_code, query_pt])
            sdf_grad_latent[i] = latent_code.grad
            sdf_grad_pos[i] = query_pt.grad
            self.decoder.zero_grad() # Technically unnecessary because only inputs=latent_code has gradients accumulated
            latent_code.grad.zero_()
            query_pt.grad.zero_() # Technically unnecessary since query_pt is recreated each loop

        return sdf_grad_latent, sdf_grad_pos

class MonteCarloUncertainty(UncertaintyEstimator):
    """SDF uncertainty estimation by propagation of latent covariance via Monte Carlo sampling."""

    def __init__(self, decoder, latent_code, device, verbose=False):
        super().__init__(decoder, latent_code, device, verbose)
        self.latent_samples = None

    def sdf_uncertainty(self, covariance, points, n_samples=2000):
        """Compute SDF uncertainty via Monte Carlo sampling. Latent samples are stored in self.latent_samples."""
        if self.verbose:
            print("\n----Computing SDF uncertainty (via Monte Carlo sampling)----")
        self.latent_samples = self._draw_samples(covariance, n_samples)
        with torch.no_grad():
            self.decoder.eval()
            if not torch.is_tensor(points):
                points = torch.tensor(points)
            points = points.to(self.device)
            # Predict SDF at each point, for each latent code
            preds_all = torch.empty((len(self.latent_samples), len(points)), device=self.device)
            for i, lc in enumerate(self.latent_samples):
                if self.verbose:
                    print(f"Processing latent sample {i+1}/{len(self.latent_samples)}", end='\r')
                preds_all[i] = decode_sdf(self.decoder, lc.to(self.device), points).squeeze(1)
            # Return std at each point
            sdf_std = torch.std(preds_all, dim=0)
            return sdf_std

    def _draw_samples(self, covariance, n_samples=2000):
        """Take samples in latent space around self.latent_code according to a Gaussian distribution."""
        distribution = MultivariateNormal(self.latent_code[0], covariance_matrix=covariance)
        latent_samples = distribution.sample((n_samples,))
        return latent_samples

# ======== UNCERTAINTY EVALUATION ========
def gaussian_pdf(x, mu, std):
    a = 1.0 / (std * np.sqrt(2.0*np.pi))
    g = a * np.exp(-0.5 * ((x - mu) / std)**2)
    return g

def eval_nll(preds, stds, points, target_mesh): 
    """Negative log likelihood of ground truth SDFs, assuming Gaussian distribution of SDF value at each point."""
    gt_sdfs = pcu_sdf(points, target_mesh) # target_mesh scale must be normalized!
    probs = gaussian_pdf(gt_sdfs, preds, stds)
    nll = -np.log(np.mean(probs)+1e-9)
    return nll

def eval_errorcurve(preds, stds, points, target_mesh, error_metric='pos_nn', n_samples=None):
    """Remove points in decreasing order of uncertainty and examine effect on average error."""
    if error_metric not in ('pos_nn', 'pos_cd', 'pos_assd', 'sdf_l1'):
        raise ValueError(f"error_metric must be one of ('pos_nn', 'pos_cd', 'pos_assd', 'sdf_l1').")
    if error_metric == 'pos_nn': # Distance nearest neighbor in target mesh
        _, errors = get_pt_cloud_distances(points, target_mesh.points)
    elif error_metric == 'sdf_l1': # Absolute difference of SDF
        gt_sdfs = pcu_sdf(points, target_mesh) # vtk_sdf is just NN dist
        errors = abs(gt_sdfs - preds)
    N = len(points)
    sorted_idxs = np.argsort(-stds)
    avg_errors = np.empty(N)
    for i in range(N):
        print(f"Points removed: {i+1}/{N}", end='\r'if i<N-1 else '\n')
        mask = sorted_idxs[i:]
        if error_metric in ('pos_nn', 'sdf_l1'):
            avg_errors[i] = np.mean(errors[mask])
        elif error_metric == 'pos_cd': # Symmetric chamfer distance
            avg_errors[i] = compute_chamfer(points[i:], target_mesh.points, n_samples)
        elif error_metric == 'pos_assd': # Average symmetric surface distance
            avg_errors[i] = compute_assd(points[i:], target_mesh.points, n_samples)
    num_removed = np.arange(N)
    return avg_errors, num_removed

def eval_tv(variances=None, covariance_mat=None):
    """Total variance (TV). Implemented here as normalized trace."""
    if variances is not None:
        return np.mean(variances)
    elif covariance_mat is not None:
        return np.trace(covariance_mat) / covariance_mat.shape[0]
    else:
        raise ValueError("Must provide list of variances or a covariance matrix.")