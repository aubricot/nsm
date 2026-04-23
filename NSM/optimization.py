# Utility functions for fine-tuning optimization of novel meshes in trained models

from sklearn.decomposition import PCA
import torch.nn.functional as F
import numpy as np
import torch
from NSM.helper_funcs import get_sdfs  
import pyvista as pv
import open3d as o3d
import json

# Initialize latent near PCA offset mean
def pca_initialize_latent(mean_latent, latent_codes, top_k=10):
    # Convert to numpy
    latent_np = latent_codes.detach().cpu().numpy()
    mean_np = mean_latent.detach().cpu().numpy().squeeze()
    pca = PCA(n_components=latent_np.shape[1])
    pca.fit(latent_np)
    # Sample along top-K PCs
    top_components = pca.components_[:top_k]  # (K, D)
    top_eigenvalues = pca.explained_variance_[:top_k]
    scale = 0.01  # tune this
    coeffs = np.random.randn(top_k) * np.sqrt(top_eigenvalues) * scale
    pca_offset = np.dot(coeffs, top_components)  # D
    init_latent = mean_np + pca_offset
    return torch.tensor(init_latent, dtype=torch.float32, device=latent_codes.device).unsqueeze(0)

# Get top k PCA's based on defined explained variance threshold
def get_top_k_pcs(latent_codes, threshold=0.90):
    latent_np = latent_codes.cpu().numpy()
    pca = PCA()
    pca.fit(latent_np)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = np.searchsorted(cum_var, threshold)
    print(f"Selected top {k+1} PCs to explain {threshold*100:.1f}% of variance")
    return pca, k + 1

# Find the top 5 most similar meshes from training data to novel/input mesh - uses L2 (euclidian) distance in latent space
def find_similar(latent_novel, latent_codes, top_k=5, n_std=2):
    dists = torch.norm(latent_codes - latent_novel, dim=1)
    mean_dist = dists.mean()
    std_dist = dists.std()
    threshold = mean_dist + n_std * std_dist
    within = dists <= threshold
    sorted_idx = torch.argsort(dists[within])[:top_k]
    similar_ids = torch.nonzero(within).squeeze()[sorted_idx]
    print(f"similar_ids shape: {similar_ids.shape}")
    print(f"similar_ids: {similar_ids}")
    return similar_ids.tolist(), dists[similar_ids].tolist()

# Find the top 5 most similar meshes from training data to novel/input mesh - uses cosine distance in latent space
def find_similar_cos(latent_novel, latent_codes, top_k=5, n_std=2, device='cuda'):
    # Compute cosine similarity between each latent code and the novel latent code
    cosine_similarities = F.cosine_similarity(latent_codes.to(device), latent_novel.to(device), dim=1)
    cosine_distances = 1 - cosine_similarities
    mean_dist = cosine_distances.mean()
    std_dist = cosine_distances.std()
    # Apply threshold (mean + n_std * std)
    threshold = mean_dist + n_std * std_dist
    within = cosine_distances <= threshold
    # Sort distances within the threshold and get top_k
    within_indices = torch.nonzero(within, as_tuple=False).squeeze()
    if within_indices.numel() == 0:
        print("No similar items within the threshold.")
        return [], []
    # If only one index remains, ensure it's a 1D tensor
    if within_indices.ndim == 0:
        within_indices = within_indices.unsqueeze(0)
    sorted_indices = torch.argsort(cosine_distances[within_indices])[:top_k]
    similar_ids = within_indices[sorted_indices]  # 1D: shape [top_k]
    print(f"similar_ids shape: {similar_ids.shape}")
    print(f"similar_ids: {similar_ids}")
    return similar_ids.tolist(), cosine_distances[similar_ids].tolist()

# Optimize latent vector for inference (since DeepSDF has no encoder, this is how you run novel data through for inference)
def optimize_latent(decoder, points, sdf_vals, latent_size, top_k, mean_latent, latent_codes, iters=1000, lr=1e-3, device='cuda'):
    init_latent_torch = pca_initialize_latent(mean_latent, latent_codes, top_k) # initialize near mean using PCAs for regularization
    latent = init_latent_torch.clone().detach().requires_grad_()
    optimizer = torch.optim.Adam([latent], lr=lr)
    sdf_vals = sdf_vals.to(device)
    decoder = decoder.to(device)
    points = points.to(device)
    for i in range(iters):
        optimizer.zero_grad()
        pred_sdf = get_sdfs(decoder, points, latent)
        loss = F.l1_loss(pred_sdf.squeeze(), sdf_vals)
        loss.backward()
        optimizer.step()
        if i % 200 == 0 or i == iters - 1:
            print(f"[{i}/{iters}] Loss: {loss.item():.6f}")
    return latent.detach().to(device)

# Sample points near and far from surface to get range of SDF values for smooth interpolation
def sample_near_surface(mesh, surface_pts, eps=0.005, fraction_nonzero=0.3, 
                        fraction_far=0.05, far_eps=0.05, device='cuda'):
    n_pts = surface_pts.shape[0]
    # Slightly perturbed points (near-surface)
    n_nonzero = int(n_pts * fraction_nonzero)
    idx_near = torch.randperm(n_pts)[:n_nonzero]
    base_near = surface_pts[idx_near]
    dirs_near = torch.randn_like(base_near)
    dirs_near = dirs_near / torch.norm(dirs_near, dim=1, keepdim=True)
    pts_out_near = base_near + eps * dirs_near
    pts_in_near  = base_near - eps * dirs_near
    sdf_out_near = eps  * torch.ones((n_nonzero, 1), device=surface_pts.device)
    sdf_in_near  = -eps * torch.ones((n_nonzero, 1), device=surface_pts.device)
    pts_nonzero = torch.cat([pts_out_near, pts_in_near], dim=0)
    sdf_nonzero = torch.cat([sdf_out_near, sdf_in_near], dim=0)
    # Farther-away points for regularization
    n_far = int(n_pts * fraction_far)
    idx_far = torch.randperm(n_pts)[:n_far]
    base_far = surface_pts[idx_far]
    dirs_far = torch.randn_like(base_far)
    dirs_far = dirs_far / torch.norm(dirs_far, dim=1, keepdim=True)
    pts_out_far = base_far + far_eps * dirs_far
    pts_in_far  = base_far - far_eps * dirs_far
    sdf_out_far = far_eps  * torch.ones((n_far, 1), device=surface_pts.device)
    sdf_in_far  = -far_eps * torch.ones((n_far, 1), device=surface_pts.device)
    pts_far = torch.cat([pts_out_far, pts_in_far], dim=0)
    sdf_far = torch.cat([sdf_out_far, sdf_in_far], dim=0)
    # Keep remaining points exactly on the surface (SDF=0)
    n_zero = int(n_pts * (1 - fraction_nonzero - fraction_far))
    mask = torch.ones(n_pts, dtype=torch.bool)
    mask[idx_near] = False
    mask[idx_far] = False
    pts_zero = surface_pts[mask]
    sdf_zero = torch.zeros((pts_zero.shape[0], 1), device=surface_pts.device)
    # Combine everything
    pts = torch.cat([pts_zero, pts_nonzero, pts_far], dim=0)
    sdf = torch.cat([sdf_zero, sdf_nonzero, sdf_far], dim=0)
    print(f"Sampled {fraction_nonzero*100}% of points (n={n_nonzero}) near the surface with a non-zero SDF (±eps = {eps})\n \
          {fraction_far*100}% of points (n={n_far}) sampled far from the surface (±far_eps={far_eps})\n \
          {(1 - fraction_nonzero - fraction_far)*100:.1f}% of points (n={n_zero}) on the surface with SDF=0.")
    return pts, sdf

# Downsample pointcloud 
def downsample_partial_pointcloud(mesh_path, n_points=5000, voxel_fraction=0.01, method='poisson'):
    # Read mesh with PyVista
    mesh_pv = pv.read(mesh_path)
    if not mesh_pv.is_all_triangles:
        mesh_pv = mesh_pv.triangulate()
    # Smooth mesh to denoise
    mesh_pv = mesh_pv.smooth(n_iter=50, relaxation_factor=0.01)
    # Convert to Open3D mesh
    vertices = np.asarray(mesh_pv.points)
    faces = np.asarray(mesh_pv.faces.reshape(-1, 4)[:, 1:])  # PyVista stores faces as [n, i, j, k]
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces))
    mesh_o3d.compute_vertex_normals()
    # Uniform or Poisson disk sampling
    if method == 'poisson':
        pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=n_points)
    else:
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_points)
    # Voxel downsample
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(np.array(bbox.get_extent()))
    voxel_size = max(diag * voxel_fraction, 1e-5)  # ensure nonzero voxel size
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.asarray(pcd_down.points)
    print(f"Downsampled from {len(mesh_pv.points)} → to {len(pts)} points (voxel={voxel_size:.4f})")
    return pts

# Optimize latent from partial pointcloud (model has no encoder, so need to optimize before feeding in new data)
def optimize_latent_partial(decoder, partial_pts, sdfs, latent_dim, mean_latent=None, latent_init=None, iters=2000, 
                            lr=1e-4, lambda_reg=1e-4, clamp_val=None, latent_std=None, scheduler_step=1000, scheduler_gamma=0.5, 
                            top_k=200, batch_inference_size=32768, verbose=True, device='cuda', multi_stage=False, l2_loss=False):
    decoder = decoder.to(device)
    decoder.eval()
    if isinstance(partial_pts, np.ndarray):
        partial_pts = torch.tensor(partial_pts, dtype=torch.float32)
    partial_pts = partial_pts.clone().detach().to(device)
    target = torch.tensor(sdfs, dtype=torch.float32).clone().detach().to(device)
    # If multi-stage optimization, intialize from previous latent, not mean
    if multi_stage:
        print("\nPhase 2\n")
        mean_latent = latent_init.clone().detach()
        latent = latent_init.clone().detach().to(device).requires_grad_(True)
    # If single-stage, initialize from pca based mean of latent codes
    else:
        print("\nPhase 1\n")
        mean_latent = mean_latent.clone().detach().to(device)
        latent = pca_initialize_latent(mean_latent, latent_init, top_k).clone().detach().to(device).requires_grad_(True)
    
    # Define clamp
    if clamp_val is not None:
        latent_std = torch.tensor(float(latent_std), dtype=torch.float32)
        clamp_val = (torch.tensor(float(clamp_val), dtype=torch.float32) * latent_std).to(device)
    
    # Run optimization
    optimizer = torch.optim.Adam([latent], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    loss_log = []
    for step in range(iters):
        optimizer.zero_grad()
        # Evaluate predicted SDFs in mini-batches to save memory
        preds = get_sdfs(decoder, partial_pts, latent, batch_size=batch_inference_size, device=device)  # (N,1)
        # surface loss (absolute SDF near 0)
        if l2_loss:
            sdf_loss = F.mse_loss(preds.to(device), target)
        else:
            sdf_loss = F.l1_loss(preds.to(device), target)
        # latent prior: encourage closeness to mean_latent
        reg_loss = F.mse_loss(latent, mean_latent.to(device))
        loss = sdf_loss + lambda_reg * reg_loss
        loss.backward()
        # gradient clipping and step
        torch.nn.utils.clip_grad_norm_([latent], 1.0)
        optimizer.step()
        scheduler.step()
        if clamp_val is not None:
            with torch.no_grad():
                offset = latent - mean_latent.to(latent.device)   # shape (1, D)
                offset_clamped = torch.clamp(offset, -clamp_val, clamp_val)
                latent.copy_(mean_latent.to(latent.device) + offset_clamped)
        loss_log.append(float(loss.item()))
        if verbose and (step % 100 == 0 or step == iters-1):
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            print(f"[{step:4d}/{iters}] loss={loss.item():.6e} sdf={sdf_loss.item():.6e} reg={reg_loss.item():.6e} lr={lr_now:.2e}")
    return latent.detach(), loss_log

# Load in manually placed landmark points from 3D Slicer markups file
def load_slicer_mrkup_pts(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    markups = data["markups"][0]              # first markup node
    points = markups["controlPoints"]        # list of control point dicts
    # Extract positions
    pts = np.array([p["position"] for p in points], dtype=np.float32)
    return pts

# Load in a bounding box ROI from 3D slicer
def load_slicer_roi_bbox(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    roi = data["markups"][0]
    center = np.array(roi["center"])
    size = np.array(roi["size"])
    orientation = np.array(roi["orientation"]).reshape(3, 3)
    # Compute half-axes in world coordinates
    half_size = size / 2.0
    axes = orientation * half_size[np.newaxis, :]
    local_corners = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [ 1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [-1,  1,  1],
        [ 1,  1,  1]]) * half_size
    world_corners = (orientation @ local_corners.T).T + center
    # Create PyVista box mesh
    bbox_params = (half_size, orientation, center)
    return bbox_params

# Sample points on surface within input bounding box dimensions
def sample_points_in_bbox(mesh_path, bbox_params, n_points=500, method='poisson'):
    # Read mesh with PyVista
    mesh_pv = pv.read(mesh_path)
    if not mesh_pv.is_all_triangles:
        mesh_pv = mesh_pv.triangulate()
    # Smooth mesh to denoise
    mesh_pv = mesh_pv.smooth(n_iter=50, relaxation_factor=0.01)
    # Convert to Open3D mesh
    vertices = np.asarray(mesh_pv.points)
    faces = np.asarray(mesh_pv.faces.reshape(-1, 4)[:, 1:])  # PyVista stores faces as [n, i, j, k]
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces))
    mesh_o3d.compute_vertex_normals()    # Convert to Open3D for Poisson disk sampling
    oversample_factor = 5
    n_sample = n_points * oversample_factor
    if method == 'poisson':
        pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=n_sample)
    else:
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_sample)
    pts = np.asarray(pcd.points)
    # Transform points to ROI's local coordinate system
    half_size, orientation, center = bbox_params
    rel_pts = pts - center
    local_pts = rel_pts @ orientation.T  # rotate into ROI frame
    # Create mask for points inside the box extents
    in_x = np.abs(local_pts[:, 0]) <= half_size[0]
    in_y = np.abs(local_pts[:, 1]) <= half_size[1]
    in_z = np.abs(local_pts[:, 2]) <= half_size[2]
    mask = in_x & in_y & in_z
    pts_inside = pts[mask]
    if len(pts_inside) < n_points:
        print(f"Warning: only {len(pts_inside)} surface points inside ROI (requested {n_points})")
        n_points_final = len(pts_inside)
    else:
        n_points_final = n_points
        idx = np.random.choice(len(pts_inside), n_points_final, replace=False)
        pts_inside = pts_inside[idx]
    return pts_inside

# Extract normalization parameters
def get_norm_params(sdf_dataset, sample_dict, vert_fname):
    if hasattr(sdf_dataset, 'center') and sdf_dataset.center is not None:
        # If scale_jointly=True
        center = sdf_dataset.center
        max_radius = sdf_dataset.max_radius
        print(f"Using joint normalization: center={center}, max_radius={max_radius}")
    else:
        # Check if center/max_radius are in sample_dict
        if 'center_0' in sample_dict:
            center = sample_dict['center_0'].cpu().numpy()
            max_radius = sample_dict['max_radius_0'].cpu().numpy()
            print(f"Using individual normalization: center={center}, max_radius={max_radius}")
        else:
            # Compute manually from original mesh
            orig_mesh = pv.read(vert_fname)
            center = orig_mesh.points.mean(axis=0)
            max_radius = np.linalg.norm(orig_mesh.points - center, axis=1).max()
            print(f"Computed normalization: center={center}, max_radius={max_radius}")
    return center, max_radius

# Normalize mesh
def normalize_mesh(mesh_out, vert_fname, config, center, max_radius):
    # Manually normalize
    mesh_pv = pv.wrap(mesh_out)
    if config['normalize_pts'] == True:
        print("Normalizing output mesh to match training transforms...")
        mesh_pv.points = mesh_pv.points * max_radius + center
    print(f"Output mesh bounds: {mesh_pv.bounds}")
    # Compare to input mesh
    input_mesh = pv.read(vert_fname)
    print(f"Input mesh bounds: {input_mesh.bounds}")
    # Ensure it's PyVista PolyData
    if isinstance(mesh_out, list):
        mesh_out = mesh_out[0]
    if not isinstance(mesh_out, pv.PolyData):
        mesh_pv = mesh_out.extract_geometry()
    else:
        mesh_pv = mesh_out 
    return mesh_pv