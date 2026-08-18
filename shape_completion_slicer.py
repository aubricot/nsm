# Shape completion for partial vertebrae in 3D Slicer Fossil NSM Module

import argparse
import os
import torch
import numpy as np
import pymskt.mesh.meshes as meshes
import sys
import random
import slicer
import types

USE_TINY3D = True  # Set this flag based on Slicer environment
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
from NSM.mesh import create_mesh
from NSM.helper_funcs import (NumpyTransform, load_config, load_model_and_latents, 
                                  fixed_point_coords, safe_load_mesh_scalars, convert_ply_to_vtk)
from NSM.optimization import (get_top_k_pcs, optimize_latent_partial, normalize_mesh, get_norm_params, 
                              encode_latent_pointnet, encode_latent, build_sdf_dataset, reconstruct_mesh_from_latent)

# Mock wandb to prevent ImportError when importing NSM.reconstruct.main
if 'wandb' not in sys.modules:
    try:
        import wandb
    except ImportError:
        mock_wandb = types.ModuleType('wandb')
        mock_wandb.Object3D = lambda *args, **kwargs: None
        sys.modules['wandb'] = mock_wandb

import NSM.uncertainty as uncert

# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

def main(config_path=None, model_path=None, latent_codes_path=None, input_mesh_path=None, output_folder_path=None,
         estimate_uncertainty=False, propagation_mode='analytical', data_std=2e-5, latent_prior_std=5e-4,
         data_weight=1.0, latent_weight=1.0, mc_samples=2000, n_triangles=5000,
         n_samples=240, phase1_iters=3000, phase1_lr=1e-4, phase1_lambda_reg=1e-3,
         phase2_iters=8000, phase2_lr=1e-5, phase2_lambda_reg=1e-5, n_pts_per_axis=256,
         fast_mode=False, encoder_ckpt=None, refine_iters=0, refine_lr=None, refine_lambda_reg=None):

    # Define training directory
    TRAIN_DIR = "run_v57" # TO DO: Choose training directory containing model ckpt and latent codes
    if model_path is None: os.chdir(TRAIN_DIR)
    CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
    LC_PATH = 'latent_codes' + '/' + CKPT + '.pth' if latent_codes_path is None else latent_codes_path
    MODEL_PATH = 'model' + '/' + CKPT + '.pth' if model_path is None else model_path

    # Load model config
    config = load_config(config_path='model_params_config.json' if config_path is None else config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select matching paths of partial meshes for shape completion
    if input_mesh_path is None:
        mesh_dir = "fossils/models_smooth_hollow/aligned"
        mesh_list = os.listdir(mesh_dir)
        mesh_list = [os.path.join(mesh_dir, f) for f in random.sample(mesh_list, 5)]
    else:
        mesh_list = [input_mesh_path]

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

        # Latent encoding
        if fast_mode:
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples=None) # Use all points instead of downsampling by n_samples
            # Fast mode: one-shot encoder instead of full latent optimization 
            print("\n-----Fast mode: encoding latent (single forward pass)----\n")
            latent_opt = encode_latent_pointnet(encoder_ckpt, points, sdf_vals, device)
            if refine_iters > 0:
                lr = refine_lr if refine_lr is not None else phase2_lr
                lam = refine_lambda_reg if refine_lambda_reg is not None else phase2_lambda_reg
                print(f"Refining encoder latent for {refine_iters} iters (lr={lr}, lambda={lam})...")
                latent_opt, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], latent_init=latent_opt, top_k=top_k_reg,
                                                            iters=refine_iters, lr=lr, lambda_reg=lam, clamp_val=None, latent_std=latent_std, scheduler_step=800, 
                                                            scheduler_gamma=0.7, batch_inference_size=32768, multi_stage=True, device=device)
            print("\nEncoded novel mesh into latent space!\n")
        else:
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples)
            # Encode latent via 2 stage optimization (auto-decoder framework)
            latent_opt = encode_latent(decoder=model, points=points.squeeze(), sdf_vals=sdf_vals, latent_dim=latent_codes.shape[1], 
                                        mean_latent=mean_latent, latent_codes=latent_codes, top_k_reg=top_k_reg, latent_std=latent_std,
                                        iters1=phase1_iters, iters2=phase2_iters, lr1=phase1_lr, lr2=phase2_lr, lambda_reg1=phase1_lambda_reg, 
                                        lambda_reg2=phase2_lambda_reg, clamp_val1=1.0, clamp_val2=None, scheduler_step1=800, scheduler_step2=800, 
                                        scheduler_gamma1=0.9, scheduler_gamma2=0.7, batch_inference_size=32768) 
        
        # Reconstruction mesh from latent
        mesh_out = reconstruct_mesh_from_latent(vert_fname, model, latent_opt, config)
            
        # Normalize and scale output using model config training params
        center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
        mesh_pv = normalize_mesh(mesh_out, vert_fname, config, center, max_radius)
        mesh_pv = mesh_pv.clean().triangulate().extract_surface(algorithm=None)
        for arr in ['RegionId', 'vtkOriginalCellIds']:
            if arr in mesh_pv.cell_data.keys():
                mesh_pv.cell_data.remove(arr)

        # Estimate uncertainty
        if estimate_uncertainty:
            print("\n-----Estimating Uncertainty-----\n")
            # 1. Initialize Laplace approximator
            laplace = uncert.LaplaceApproximator(model, latent_opt, vert_fname, config, device, verbose=True)
            
            # 2. Compute approximate covariance of latent posterior
            print("Computing covariance matrix using Laplace approximation...")
            covariance, hessian = laplace.covariance(data_std=data_std, latent_std=latent_prior_std, data_weight=data_weight, 
                                                     latent_weight=latent_weight, n_samples=2000, recompute_jacobian=False)
            
            # 3. Initialize propagator
            if propagation_mode == 'analytical':
                propagator = uncert.AnalyticalUncertainty(model, latent_opt, device, verbose=True)
            else:
                propagator = uncert.MonteCarloUncertainty(model, latent_opt, device, verbose=True)
                
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
        main(config_path=sys.argv[1], model_path=sys.argv[2], latent_codes_path=sys.argv[3], input_mesh_path=sys.argv[4], output_folder_path=sys.argv[5])
    
    else:
        args = parser.parse_args()
        main(config_path=args.config,
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
                refine_lambda_reg=args.refine_lambda_reg)
