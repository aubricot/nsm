import sys, os, json, torch
import numpy as np
import pyvista as pv
import pymskt.mesh.meshes as meshes
import pymskt.mesh.meshTools as meshTools


def main(config_path, model_path, latent_codes_path, input_mesh_path, output_folder_path):
    from NSM.datasets import SDFSamples
    from NSM.mesh import create_mesh
    from NSM.helper_funcs import (
        NumpyTransform, 
        load_config, 
        load_model_and_latents,
        convert_ply_to_vtk, 
        fixed_point_coords, 
        safe_load_mesh_scalars,
    )
    from NSM.optimization import (
        get_top_k_pcs, 
        sample_near_surface,
        downsample_partial_pointcloud, 
        optimize_latent_partial,
        normalize_mesh, 
        get_norm_params
    )

    print("Loading config...")
    print(torch.cuda.is_available())
    config = load_config(config_path=config_path)
    device = config.get("device", "cuda:0")

    print("Loading model and latent codes...")
    model, latent_ckpt, latent_codes = load_model_and_latents(model_path, latent_codes_path, config, device)

    print("Running model inference... (this may take a few minutes)")
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    latent_std = latent_codes.std().mean()
    pca_model, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.95)

    # Monkey Patch into pymskt.mesh.meshes.Mesh
    meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
    meshes.Mesh.point_coords = property(fixed_point_coords)

    _original_signed_distance_to_mesh = meshTools.pcu.signed_distance_to_mesh
    def _signed_distance_to_mesh_patch(pts, points, faces):
        pts = np.asarray(pts, dtype=np.float64)     # force double precision
        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)   # ensure integer type for faces
        return _original_signed_distance_to_mesh(pts, points, faces)
    meshTools.pcu.signed_distance_to_mesh = _signed_distance_to_mesh_patch

    vert_fname = input_mesh_path   
    print(f"\033[32m\n=== Processing {os.path.basename(vert_fname)} ===\033[0m")
    # Make a new dir to save predictions
    outfpath = output_folder_path + '/' + os.path.splitext(os.path.basename(vert_fname))[0] # TO DO: Adjust to desired outpath
    print(f"Making a new directory to save model predictions and outputs at: {outfpath}")
    os.makedirs(outfpath, exist_ok=True)

    # Convert plys to vtks
    if '.ply' in vert_fname:
        ply_fname = vert_fname
        mesh, vert_fname = convert_ply_to_vtk(ply_fname, output_file=outfpath + '/partial.vtk', save=True)

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
    points = sample_dict['xyz'].to(device) # shape: [N, 3]
    sdf_vals = sample_dict['gt_sdf']  # shape: [N, 1]

    # Use a subset of the points for optizimation/reconstruction
    n_samples = 240 # TO DO: Define how many points to sample
    indices = torch.randperm(points.size(0))[:n_samples] # Generate n_samples random indices
    # Downsample the points and SDF values
    points = points[indices]
    points = points.squeeze()
    sdf_vals = sdf_vals[indices]
    sdf_vals = sdf_vals.reshape(-1, 1)

    # Optimize latents
    print("\n-----Optimizing latents----\n")
    # Phase 1 - Coarse Optimization - get a global shape in the right area of latent space (close to target specimen (far enough from mean); but not so far from mean that it is noisy or unrealistic)
    latent_partial, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], mean_latent=mean_latent, latent_init=latent_codes, top_k=top_k_reg, 
                                                       iters=3000, lr=1e-4, lambda_reg=1e-3, clamp_val=1.0, latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.9, 
                                                       batch_inference_size=32768, multi_stage=False, device=device)
    # Phase 2 - Refinement - emphasis on local SDF samples and surface consistency to refine target specimen shape
    latent_partial, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], latent_init=latent_partial, top_k=top_k_reg, 
                                                        iters=8000, lr=1e-5, lambda_reg=1e-5, clamp_val=None, latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.7, 
                                                        batch_inference_size=32768, multi_stage=True, device=device) # True because second stage using already initialized latent
    print("\nTranslated novel mesh into latent space!\n")
    
    # Reconstruction parameters
    recon_grid_origin = 1.0
    n_pts_per_axis = 256 # TO DO: Adjust resolution
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

    # Save mesh
    mesh_pv = mesh_pv.clean()
    mesh_pv = mesh_pv.triangulate()
    output_path = outfpath + '/prediction.vtk'
    # Set color: RGB in range 0–255 or 0–1
    color = np.array([112, 215, 222], dtype=np.uint8)  
    # Broadcast color to all points
    rgb = np.tile(color, (mesh_pv.n_points, 1))
    mesh_pv.point_data.clear()
    mesh_pv.point_data['Colors'] = rgb
    mesh_pv.save(output_path)
    print(f"Completed mesh from partial pointcloud saved to: {output_path}")

    with open(os.path.join(outfpath, ".done"), "w") as f:
        f.write(output_path)

if __name__ == "__main__":

    modulePath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if modulePath not in sys.path:
        sys.path.insert(0, modulePath)

    main(
        config_path=sys.argv[1],
        model_path=sys.argv[2],
        latent_codes_path=sys.argv[3],
        input_mesh_path=sys.argv[4],
        output_folder_path=sys.argv[5]
    )