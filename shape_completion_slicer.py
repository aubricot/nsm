# Shape completion for partial vertebrae

import os
import torch
import numpy as np
import pymskt.mesh.meshes as meshes
import sys
import random
import slicer
import tiny3d as o3d


# Pyvista to Open3D    
def pv_to_tiny3d(mesh_pv):
    pts = np.asarray(mesh_pv.points)
    faces = np.asarray(mesh_pv.faces)
    tris = faces.reshape(-1,4)[:,1:4]
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(pts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tris)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def encoder_latent(encoder_ckpt, points_full, sdf_full, device, surf_thresh=0.01):
    """Fast mode: predict the full-shape latent from partial near-surface points
    in one forward pass."""
    from encoder.pointnet_encoder import PointNetEncoder

    if not os.path.isabs(encoder_ckpt):
        encoder_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), encoder_ckpt)
    ckpt = torch.load(encoder_ckpt, map_location=device)
    enc = PointNetEncoder(latent_size=ckpt["latent_size"]).to(device)
    enc.load_state_dict(ckpt["model"])
    enc.eval()
    n_points = ckpt["n_points"]

    abs_sdf = sdf_full.abs().squeeze()
    surf = points_full[abs_sdf < surf_thresh]
    if surf.shape[0] < 32:
        # Not enough near-surface samples: fall back to the k smallest |sdf|.
        k = min(n_points, points_full.shape[0])
        idx = torch.topk(abs_sdf, k, largest=False).indices
        surf = points_full[idx]

    n = surf.shape[0]
    sel = np.random.choice(n, n_points, replace=n < n_points)
    surf = surf[sel]
    with torch.no_grad():
        z = enc(surf.unsqueeze(0).to(device))
    return z


def main(config_path=None, model_path=None, latent_codes_path=None, input_mesh_path=None, output_folder_path=None,
         estimate_uncertainty=False, propagation_mode='analytical', data_std=2e-5, latent_prior_std=5e-4,
         data_weight=1.0, latent_weight=1.0, mc_samples=2000, n_triangles=5000,
         n_samples=240, phase1_iters=3000, phase1_lr=1e-4, phase1_lambda_reg=1e-3,
         phase2_iters=8000, phase2_lr=1e-5, phase2_lambda_reg=1e-5, n_pts_per_axis=256,
         fast_mode=False, encoder_ckpt=None, refine_iters=0, refine_lr=None, refine_lambda_reg=None):
    USE_TINY3D = True  # Set this flag based on Slicer environment

    # Dynamically import open3d/tiny3d depending on USE_TINY3D
    if USE_TINY3D:
        try:
            import tiny3d as o3d
            print("Using tiny3d as o3d")
        except ImportError:
            slicer.util.pip_install('tiny3d')
            import tiny3d as o3d  # Try again after installing
    else:
        try:
            import open3d as o3d
            print("Using open3d")
        except ImportError:
            slicer.util.pip_install('open3d')
            import open3d as o3d  # Try again after installing
    from NSM.datasets import SDFSamples
    from NSM.mesh import create_mesh
    from NSM.helper_funcs import (
        NumpyTransform, 
        load_config, 
        load_model_and_latents, 
        fixed_point_coords, 
        safe_load_mesh_scalars,
        convert_ply_to_vtk
    )
    from NSM.optimization import (
        get_top_k_pcs,
        optimize_latent_partial,
        normalize_mesh,
        get_norm_params
    )
    # Monkey Patch into pymskt.mesh.meshes.Mesh
    meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
    meshes.Mesh.point_coords = property(fixed_point_coords)

    # Define training directory
    TRAIN_DIR = "run_v57" # TO DO: Choose training directory containing model ckpt and latent codes
    if model_path is None: os.chdir(TRAIN_DIR)
    CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
    LC_PATH = 'latent_codes' + '/' + CKPT + '.pth' if latent_codes_path is None else latent_codes_path
    MODEL_PATH = 'model' + '/' + CKPT + '.pth' if model_path is None else model_path

    # Load model config
    config = load_config(config_path='model_params_config.json' if config_path is None else config_path)
    #device = config.get("device", "cuda:0")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select matching paths of partial meshes for shape completion
    if input_mesh_path is None:
        mesh_dir = "fossils/models_smooth_hollow/aligned"
        mesh_list = os.listdir(mesh_dir)
        mesh_list = [os.path.join(mesh_dir, f) for f in random.sample(mesh_list, 5)]
    else:
        mesh_list = [input_mesh_path]
    #mesh_list = [os.path.join(mesh_dir, f) for f in mesh_list]

    # Load model and latent codes
    model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    latent_std = latent_codes.std().mean()
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)

    # Loop through meshes
    summary_log = []
    for i, vert_fname in enumerate(mesh_list):    
        print(f"\033[32m\n=== Processing {os.path.basename(vert_fname)} ===\033[0m")
        print(f"\033[32m\n=== Mesh {i+1} / {len(mesh_list)} ===\033[0m")
        # Make a new dir to save predictions
        if output_folder_path is not None:
            outfpath = output_folder_path
        else:
            outfpath = 'shape_completion/predictions/' + os.path.splitext(os.path.basename(vert_fname))[0] # TO DO: Adjust to desired outpath
        print("Making a new directory to save model predictions and outputs at: ", outfpath)
        os.makedirs(outfpath, exist_ok=True)

        # Convert plys to vtks
        if '.ply' in vert_fname:
            ply_fname = vert_fname
            mesh, vert_fname = convert_ply_to_vtk(ply_fname, save=True)

        # Setup your dataset with just one mesh
        sdf_dataset = SDFSamples(
            list_mesh_paths=[vert_fname],
            multiprocessing=False,
            subsample=config["samples_per_object_per_batch"],
            print_filename=True,
            n_pts=config["n_pts_per_object"],
            p_near_surface=config['percent_near_surface'],
            p_further_from_surface=config['percent_further_from_surface'],
            sigma_near=config['sigma_near'],
            sigma_far=config['sigma_far'],
            rand_function=config['random_function'], 
            center_pts=config['center_pts'],
            norm_pts=config['normalize_pts'],
            scale_method=config['scale_method'],
            scale_jointly=config['scale_jointly'],
            reference_mesh=None,
            verbose=config['verbose'],
            save_cache=config['cache'],
            equal_pos_neg=config['equal_pos_neg'],
            fix_mesh=config['fix_mesh'])

        # Get the point/SDF data
        print("\n-----Setting up dataset-----\n")
        sdf_sample = sdf_dataset[0]  # returns a dict
        sample_dict, _ = sdf_sample
        points_full = sample_dict['xyz'].to(device).squeeze()      # shape: [N, 3]
        sdf_full = sample_dict['gt_sdf'].to(device).reshape(-1, 1)  # shape: [N, 1]

        # Use a subset of the points for optimization/refinement
        indices = torch.randperm(points_full.size(0))[:n_samples] # Generate n_samples random indices
        points = points_full[indices].squeeze()
        sdf_vals = sdf_full[indices].reshape(-1, 1)

        if fast_mode:
            # ---- Fast mode: one-shot encoder instead of full latent optimization ----
            print("\n-----Fast mode: encoding latent (single forward pass)----\n")
            latent_partial = encoder_latent(encoder_ckpt, points_full, sdf_full, device)
            if refine_iters > 0:
                lr = refine_lr if refine_lr is not None else phase2_lr
                lam = refine_lambda_reg if refine_lambda_reg is not None else phase2_lambda_reg
                print(f"Refining encoder latent for {refine_iters} iters (lr={lr}, lambda={lam})...")
                latent_partial, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], latent_init=latent_partial, top_k=top_k_reg,
                                                            iters=refine_iters, lr=lr, lambda_reg=lam, clamp_val=None, latent_std=latent_std, scheduler_step=800, 
                                                            scheduler_gamma=0.7, batch_inference_size=32768, multi_stage=True, device=device)
            print("\nEncoded novel mesh into latent space!\n")
        else:
            # Optimize latents
            print("\n-----Optimizing latents----\n")
            # Phase 1 - Coarse Optimization - get a global shape in the right area of latent space (close to target specimen (far enough from mean); but not so far from mean that it is noisy or unrealistic)
            latent_partial, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], mean_latent=mean_latent, latent_init=latent_codes, top_k=top_k_reg,
                                                             iters=phase1_iters, lr=phase1_lr, lambda_reg=phase1_lambda_reg, clamp_val=1.0, latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.9,
                                                             batch_inference_size=32768, multi_stage=False, device=device)
            # Phase 2 - Refinement - emphasis on local SDF samples and surface consistency to refine target specimen shape
            latent_partial, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], latent_init=latent_partial, top_k=top_k_reg,
                                                                 iters=phase2_iters, lr=phase2_lr, lambda_reg=phase2_lambda_reg, clamp_val=None, latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.7,
                                                                 batch_inference_size=32768, multi_stage=True, device=device) # True because second stage using already initialized latent
            print("\nTranslated novel mesh into latent space!\n")
        
        # Reconstruction parameters
        recon_grid_origin = 1.0
        voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
        voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
        offset = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        icp_transform = NumpyTransform(np.eye(4))
        objects = 1

        # Reconstruct the novel mesh
        with torch.no_grad():
            mesh_out = create_mesh(decoder=model, latent_vector=latent_partial, n_pts_per_axis=n_pts_per_axis,
                                    voxel_origin=voxel_origin, voxel_size=voxel_size, path_original_mesh=vert_fname,
                                    offset=offset, scale=scale, icp_transform=icp_transform, objects=objects,
                                    verbose=True, device=device, scale_to_original_mesh=False) #, smooth=1.0)
            
        # Normalize and scale output using model config training params
        center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
        mesh_pv = normalize_mesh(mesh_out, vert_fname, config, center, max_radius)
        mesh_pv = mesh_pv.clean()
        mesh_pv = mesh_pv.triangulate()

        if estimate_uncertainty:
            import sys
            import types
            # Mock wandb to prevent ImportError when importing NSM.reconstruct.main
            if 'wandb' not in sys.modules:
                try:
                    import wandb
                except ImportError:
                    mock_wandb = types.ModuleType('wandb')
                    mock_wandb.Object3D = lambda *args, **kwargs: None
                    sys.modules['wandb'] = mock_wandb

            import NSM.uncertainty as uncert
            
            print("\n-----Estimating Uncertainty-----\n")
            # 1. Initialize Laplace approximator
            laplace = uncert.LaplaceApproximator(model, latent_partial, vert_fname, config, device, verbose=True)
            
            # 2. Compute approximate covariance of latent posterior
            print("Computing covariance matrix using Laplace approximation...")
            covariance, hessian = laplace.covariance(
                data_std=data_std, 
                latent_std=latent_prior_std, 
                data_weight=data_weight, 
                latent_weight=latent_weight, 
                n_samples=2000, 
                recompute_jacobian=False
            )
            
            # 3. Initialize propagator
            if propagation_mode == 'analytical':
                propagator = uncert.AnalyticalUncertainty(model, latent_partial, device, verbose=True)
            else:
                propagator = uncert.MonteCarloUncertainty(model, latent_partial, device, verbose=True)
                
            # 4. Decimate the completed mesh to speed up uncertainty computation
            print(f"Decimating reconstructed mesh to {n_triangles} triangles...")
            recon_mesh_dec = uncert.decimate_mesh(mesh_pv, n_triangles=n_triangles, verbose=True)

            # Clean/triangulate BEFORE computing uncertainty
            recon_mesh_dec = recon_mesh_dec.clean()
            recon_mesh_dec = recon_mesh_dec.triangulate()
            
            # 5. Propagate covariance to get SDF uncertainty at vertices
            print(f"Propagating covariance using {propagation_mode} mode...")
            if propagation_mode == 'analytical':
                surface_sdf_std = propagator.sdf_uncertainty(covariance, recon_mesh_dec.points, recompute_gradients=False)
            else:
                surface_sdf_std = propagator.sdf_uncertainty(covariance, recon_mesh_dec.points, n_samples=mc_samples)
                
            # 6. Set SdfUncertainty scalar field
            # Raw SDF uncertainty
            unc_raw = surface_sdf_std.detach().cpu().numpy().astype(np.float32)

            # Micro-unit version for nicer Slicer visualization
            unc_um = unc_raw * 1e6

            recon_mesh_dec.point_data['SdfUncertainty'] = unc_raw
            recon_mesh_dec.point_data['SdfUncertainty_um'] = unc_um
            
            # 7. Save the decimated mesh with uncertainty
            output_path = outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_shape_completion_unc.vtk"
            recon_mesh_dec.save(output_path)
            print(f"Completed mesh with uncertainty saved to: {output_path}")
        else:
            # Save mesh
            output_path = outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_shape_completion.vtk"
            # Set color: RGB in range 0–255 or 0–1
            color = np.array([112, 215, 222], dtype=np.uint8)  
            # Broadcast color to all points
            rgb = np.tile(color, (mesh_pv.n_points, 1))
            mesh_pv.point_data.clear()
            mesh_pv.point_data['Colors'] = rgb
            mesh_pv.save(output_path)
            print(f"Completed mesh from partial pointcloud saved to: {output_path}")

        with open(os.path.join(outfpath, os.path.splitext(os.path.basename(vert_fname))[0] + ".done"), "w") as f:
            f.write(output_path)

if __name__ == "__main__":

    repoRoot = os.path.dirname(os.path.abspath(__file__))
    if repoRoot not in sys.path:
        sys.path.insert(0, repoRoot)  # so `import encoder...` (fast mode) resolves
    modulePath = os.path.join(repoRoot, 'NSM')
    print("Module path: ", modulePath)
    if modulePath not in sys.path:
        sys.path.insert(0, modulePath)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--latent_codes', type=str, default=None)
    parser.add_argument('--input_mesh', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--estimate_uncertainty', action='store_true')
    parser.add_argument('--propagation_mode', type=str, default='analytical')
    parser.add_argument('--data_std', type=float, default=2e-5)
    parser.add_argument('--latent_prior_std', type=float, default=5e-4)
    parser.add_argument('--data_weight', type=float, default=1.0)
    parser.add_argument('--latent_weight', type=float, default=1.0)
    parser.add_argument('--mc_samples', type=int, default=2000)
    parser.add_argument('--n_triangles', type=int, default=5000)
    # Optimization settings
    parser.add_argument('--n_samples', type=int, default=240)
    parser.add_argument('--phase1_iters', type=int, default=3000)
    parser.add_argument('--phase1_lr', type=float, default=1e-4)
    parser.add_argument('--phase1_lambda_reg', type=float, default=1e-3)
    parser.add_argument('--phase2_iters', type=int, default=8000)
    parser.add_argument('--phase2_lr', type=float, default=1e-5)
    parser.add_argument('--phase2_lambda_reg', type=float, default=1e-5)
    parser.add_argument('--n_pts_per_axis', type=int, default=256)
    # Fast mode (one-shot encoder)
    parser.add_argument('--fast_mode', action='store_true')
    parser.add_argument('--encoder_ckpt', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'encoder', 'checkpoints', 'encoder.pt'))
    parser.add_argument('--refine_iters', type=int, default=0)
    parser.add_argument('--refine_lr', type=float, default=None)
    parser.add_argument('--refine_lambda_reg', type=float, default=None)

    # Check if we are using positional args or argparse format
    if len(sys.argv) == 6 and not sys.argv[1].startswith('-'):
        # Fallback to positional arguments
        main(
            config_path=sys.argv[1],
            model_path=sys.argv[2],
            latent_codes_path=sys.argv[3],
            input_mesh_path=sys.argv[4],
            output_folder_path=sys.argv[5]
        )
    else:
        args = parser.parse_args()
        main(
            config_path=args.config,
            model_path=args.model,
            latent_codes_path=args.latent_codes,
            input_mesh_path=args.input_mesh,
            output_folder_path=args.output_folder,
            estimate_uncertainty=args.estimate_uncertainty,
            propagation_mode=args.propagation_mode,
            data_std=args.data_std,
            latent_prior_std=args.latent_prior_std,
            data_weight=args.data_weight,
            latent_weight=args.latent_weight,
            mc_samples=args.mc_samples,
            n_triangles=args.n_triangles,
            n_samples=args.n_samples,
            phase1_iters=args.phase1_iters,
            phase1_lr=args.phase1_lr,
            phase1_lambda_reg=args.phase1_lambda_reg,
            phase2_iters=args.phase2_iters,
            phase2_lr=args.phase2_lr,
            phase2_lambda_reg=args.phase2_lambda_reg,
            n_pts_per_axis=args.n_pts_per_axis,
            fast_mode=args.fast_mode,
            encoder_ckpt=args.encoder_ckpt,
            refine_iters=args.refine_iters,
            refine_lr=args.refine_lr,
            refine_lambda_reg=args.refine_lambda_reg,
        )
