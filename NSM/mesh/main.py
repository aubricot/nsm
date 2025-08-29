"""
https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/mesh.py
"""

from skimage.measure import marching_cubes
import pyvista as pv
import os
import torch
import numpy as np
import pymskt as mskt
import vtk
import inspect

def scale_mesh_(mesh, scale=1.0, offset=(0.0, 0.0, 0.0), icp_transform=None, verbose=False):
    if not issubclass(type(mesh), mskt.mesh.Mesh):
        mesh = mskt.mesh.Mesh(mesh)

    if verbose is True:
        print("scale_mesh_. scale:", scale)

    pts = mesh.point_coords * scale
    pts += offset

    mesh.point_coords = pts

    if icp_transform is not None:
        transform = vtk.vtkTransform()
        transform.SetMatrix(icp_transform.GetMatrix())
        transform.Inverse()
        if verbose is True:
            print(icp_transform)
            print("INVERSE")
            print(transform)
        mesh.apply_transform_to_mesh(transform)

    return mesh


def scale_mesh(
    new_mesh,
    old_mesh=None,
    scale=1.0,
    offset=(0.0, 0.0, 0.0),
    scale_method="max_rad",
    icp_transform=None,
    verbose=False,
):

    if old_mesh is not None:
        old_mesh = mskt.mesh.Mesh(old_mesh)  # should handle vtk, pyvista, or string path to file
        old_pts = old_mesh.point_coords

        if not issubclass(type(new_mesh), mskt.mesh.Mesh):
            new_mesh = mskt.mesh.Mesh(
                new_mesh
            )  # should handle vtk, pyvista, or string path to file
        new_pts = new_mesh.point_coords

        offset = np.mean(old_pts, axis=0)
        old_pts -= offset

        if scale_method == "max_rad":
            scale = np.max(np.linalg.norm(old_pts, axis=-1), axis=-1)
        else:
            raise NotImplementedError

    mesh = scale_mesh_(
        new_mesh, scale=scale, offset=offset, icp_transform=icp_transform, verbose=verbose
    )
    return mesh


def create_mesh(
    decoder,
    latent_vector,
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
    batch_size=32**3,
    scale=1.0,
    offset=(0.0, 0.0, 0.0),
    path_save=None,
    filename="mesh_{mesh_idx}.vtk",
    path_original_mesh=None,
    scale_to_original_mesh=True,
    icp_transform=None,
    objects=1,
    verbose=False,
    device="cuda",
    use_vtk=True,
):

    if voxel_size is None:
        voxel_size = 2.0 / (n_pts_per_axis - 1)

    decoder.eval()

    samples = create_grid_samples(n_pts_per_axis, voxel_origin, voxel_size)
    sdf_values_ = get_sdfs(
        decoder, samples, latent_vector, batch_size, objects=objects, device=device
    )

    # resample SDFs into a grid:
    sdf_values = torch.zeros((n_pts_per_axis, n_pts_per_axis, n_pts_per_axis, objects))
    for i in range(objects):
        sdf_values[..., i] = sdf_values_[..., i].reshape(
            n_pts_per_axis, n_pts_per_axis, n_pts_per_axis
        )
    # sdf_values = sdf_values.reshape(n_pts_per_axis, n_pts_per_axis, n_pts_per_axis)

    # create mesh from gridded SDFs
    meshes = []
    for mesh_idx in range(objects):
        # iterate over all the meshes
        sdf_values_ = sdf_values[..., mesh_idx]

        # check if there is a surface
        if 0 < sdf_values_.min() or 0 > sdf_values_.max():
            if verbose is True:
                print("WARNING: SDF values do not span 0 - there is no surface")
                print("\tSDF min: ", sdf_values_.min())
                print("\tSDF max: ", sdf_values_.max())
                print("\tSDF mean: ", sdf_values_.mean())
            meshes.append(None)
        else:
            # if there is a surface, then extract it & post-process
            # for mesh_idx in range(objects):
            if use_vtk:
                mesh = sdf_grid_to_mesh_vtk(sdf_values_, voxel_origin, voxel_size)
            else:
                mesh = sdf_grid_to_mesh(sdf_values_, voxel_origin, voxel_size)
            meshes.append(mesh)

            if scale_to_original_mesh:
                if verbose is True:
                    print("Scaling mesh to original mesh... ")
                    print(icp_transform)
                # for mesh_idx, mesh in enumerate(meshes):
                mesh = scale_mesh(
                    meshes[mesh_idx],
                    old_mesh=path_original_mesh,
                    scale=scale,
                    offset=offset,
                    icp_transform=icp_transform,
                    verbose=verbose,
                )
                meshes[mesh_idx] = mesh

            # save the mesh (if desired)
            if path_save is not None:
                # for mesh_idx, mesh in enumerate(meshes):
                meshes[mesh_idx].save_mesh(
                    os.path.join(path_save, filename.format(mesh_idx=mesh_idx))
                )
    return meshes[0] if objects == 1 else meshes


def sdf_grid_to_mesh(
    sdf_values,
    voxel_origin,
    voxel_size,
    verbose=False,
    narrow_band=False,
    band_width=3.0,
    pad_voxels=2,
):
    sdf_values = sdf_values.cpu().numpy()

    if verbose is True:
        print("Starting marching cubes... ")

    # Apply narrow band optimization if requested
    if narrow_band:
        sub_sdf, crop_origin = crop_sdf_to_narrow_band(
            sdf_values, voxel_origin, voxel_size, band_width, pad_voxels, verbose
        )
    else:
        sub_sdf = sdf_values
        crop_origin = voxel_origin

    verts, faces, normals, values = marching_cubes(
        sub_sdf, level=0, spacing=(voxel_size, voxel_size, voxel_size)
    )

    if verbose is True:
        print("Starting vert/face conversion...")

    verts += crop_origin

    faces_ = []
    for face_idx in range(faces.shape[0]):
        face = np.insert(faces[face_idx, :], 0, faces.shape[1])
        faces_.append(face)

    faces = np.hstack(faces_)

    if verbose is True:
        print("Creating mesh... ")

    mesh = mskt.mesh.Mesh(mesh=pv.PolyData(verts, faces))

    return mesh


def crop_sdf_to_narrow_band(sdf_values, voxel_origin, voxel_size, band_width=3.0, pad_voxels=2, verbose=False):
    """
    Crop SDF volume to a narrow band around the surface for faster processing.
    
    Args:
        sdf_values: numpy array containing SDF values
        voxel_origin: Origin point of the voxel grid (x, y, z)
        voxel_size: Size of each voxel
        band_width: Width of narrow band in world units (multiplier of voxel_size)
        pad_voxels: Number of voxels to pad around cropped region
        verbose: Whether to print progress messages
    
    Returns:
        tuple: (cropped_sdf, new_origin) or (original_sdf, original_origin) if no cropping needed
    """
    orig_nx, orig_ny, orig_nz = sdf_values.shape
    
    if verbose:
        print(f"Applying narrow band optimization (band_width={band_width} * voxel_size)...")
    
    # Find voxels within the narrow band around the surface
    band = band_width * voxel_size
    mask = np.abs(sdf_values) <= band
    z, y, x = np.where(mask)
    
    if len(z) == 0:
        if verbose:
            print("WARNING: No voxels found within narrow band - using full volume")
        return sdf_values, voxel_origin
    
    # Calculate cropping bounds with padding
    xs = max(x.min() - pad_voxels, 0)
    xe = min(x.max() + pad_voxels + 1, orig_nz)
    ys = max(y.min() - pad_voxels, 0)
    ye = min(y.max() + pad_voxels + 1, orig_ny)
    zs = max(z.min() - pad_voxels, 0)
    ze = min(z.max() + pad_voxels + 1, orig_nx)
    
    # Extract subvolume
    sub_sdf = sdf_values[zs:ze, ys:ye, xs:xe]
    
    # Calculate new origin for the cropped volume
    crop_origin = (
        voxel_origin[0] + zs * voxel_size,
        voxel_origin[1] + ys * voxel_size,
        voxel_origin[2] + xs * voxel_size
    )
    
    if verbose:
        print(f"Cropped volume from {orig_nx}x{orig_ny}x{orig_nz} to {sub_sdf.shape}")
        print(f"Original origin: {voxel_origin}, Cropped origin: {crop_origin}")
    
    return sub_sdf, crop_origin


def sdf_grid_to_mesh_vtk(
    sdf_values,
    voxel_origin,
    voxel_size,
    verbose=False,
    narrow_band=True,
    band_width=3.0,
    pad_voxels=2,
):
    """
    Create mesh from SDF values using VTK Flying Edges algorithm instead of marching cubes.
    
    Args:
        sdf_values: PyTorch tensor containing SDF values
        voxel_origin: Origin point of the voxel grid (x, y, z)
        voxel_size: Size of each voxel
        verbose: Whether to print progress messages
        narrow_band: Whether to crop volume to narrow band around surface for speed
        band_width: Width of narrow band in world units (multiplier of voxel_size)
        pad_voxels: Number of voxels to pad around cropped region
    
    Returns:
        mskt.mesh.Mesh object
    """
    # Convert to numpy if needed
    if hasattr(sdf_values, 'cpu'):
        sdf_values = sdf_values.cpu().numpy()
    
    if verbose:
        print("Starting VTK Flying Edges mesh extraction...")
    
    # Apply narrow band optimization if requested
    if narrow_band:
        sub_sdf, crop_origin = crop_sdf_to_narrow_band(
            sdf_values, voxel_origin, voxel_size, band_width, pad_voxels, verbose
        )
    else:
        sub_sdf = sdf_values
        crop_origin = voxel_origin
    
    # Get grid dimensions (cropped or original)
    nx, ny, nz = sub_sdf.shape
    
    # Create PyVista ImageData (replaces deprecated UniformGrid)
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (voxel_size, voxel_size, voxel_size)
    grid.origin = crop_origin  # Use the cropped origin
    grid["sdf"] = sub_sdf.ravel(order="F")  # VTK likes column-major
    
    # Apply Flying Edges 3D algorithm
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(grid)
    fe.SetValue(0, 0.0)  # SDF iso-level
    fe.ComputeNormalsOff()  # compute later from SDF grads if desired
    fe.Update()
    
    # Wrap the output as PyVista mesh and create mskt mesh directly
    mesh = mskt.mesh.Mesh(mesh=fe.GetOutput())    
    if verbose:
        print(f"Extracted mesh with {mesh.n_points} vertices and {mesh.n_faces} faces")
        print("Creating final mesh object...")    

    return mesh


def find_object_bounds_random_sampling(
    decoder,
    latent_vector,
    n_random_samples=300_000,
    search_bounds=(-1.0, 1.0),
    objects=1,
    batch_size=300_000,
    device="cuda",
    verbose=False,
):
    """
    Find bounding box of objects by random sampling and detecting interior points (SDF < 0).
    
    Args:
        decoder: SDF decoder network
        latent_vector: Latent code for the objects
        n_random_samples: Number of random points to sample
        search_bounds: (min, max) bounds to sample within (assumes cube)
        objects: Number of objects
        batch_size: Batch size for SDF computation
        device: Device to run computation on
        verbose: Whether to print progress
    
    Returns:
        tuple: (bounds_min, bounds_max) where each is (x, y, z), or None if no objects found
    """
    if verbose:
        print(f"Finding object bounds with {n_random_samples} random samples...")
    
    # Generate random samples in the search space
    samples = torch.rand(n_random_samples, 3) * (search_bounds[1] - search_bounds[0]) + search_bounds[0]
    
    # Get SDF values for random samples
    sdf_values = get_sdfs(decoder, samples, latent_vector, batch_size, objects, device)
    
    # Find points that are inside any object (SDF < 0)
    interior_mask = torch.any(sdf_values < 0, dim=1)  # True if inside ANY object
    interior_points = samples[interior_mask]
    
    if len(interior_points) == 0:
        if verbose:
            print("WARNING: No interior points found in random sampling")
        return None
    
    # Calculate bounding box
    bounds_min = interior_points.min(dim=0)[0].cpu().numpy()
    bounds_max = interior_points.max(dim=0)[0].cpu().numpy()
    
    if verbose:
        print(f"Found {len(interior_points)} interior points")
        print(f"Object bounds: min={bounds_min}, max={bounds_max}")
        extent = bounds_max - bounds_min
        print(f"Object extent: {extent}")
    
    return bounds_min, bounds_max


def create_grid_samples_in_bounds(
    bounds_min,
    bounds_max,
    original_spacing,
    padding=0.1,
):
    """
    Create dense grid samples within discovered bounds using consistent spacing.
    
    Args:
        bounds_min: (x, y, z) minimum bounds
        bounds_max: (x, y, z) maximum bounds  
        original_spacing: Voxel spacing from original full grid
        padding: Extra space around bounds
    
    Returns:
        torch.Tensor: Grid samples of shape (N, 3)
    """
    # Add padding to bounds
    padded_min = bounds_min - padding
    padded_max = bounds_max + padding
    
    # Calculate grid dimensions to maintain original spacing
    nx = int((padded_max[0] - padded_min[0]) / original_spacing) + 1
    ny = int((padded_max[1] - padded_min[1]) / original_spacing) + 1
    nz = int((padded_max[2] - padded_min[2]) / original_spacing) + 1
    
    n_pts_total = nx * ny * nz
    
    indices = torch.arange(0, n_pts_total, out=torch.LongTensor())
    samples = torch.zeros(n_pts_total, 3)
    
    # Generate samples on the bounded grid
    samples[:, 2] = indices % nz
    samples[:, 1] = (indices // nz) % ny
    samples[:, 0] = ((indices // nz) // ny) % nx
    
    # Scale to actual coordinates within bounds
    samples[:, 0] = samples[:, 0] * original_spacing + padded_min[0]
    samples[:, 1] = samples[:, 1] * original_spacing + padded_min[1]
    samples[:, 2] = samples[:, 2] * original_spacing + padded_min[2]
    
    return samples, (nx, ny, nz), (padded_min[0], padded_min[1], padded_min[2])


def create_mesh_adaptive(
    decoder,
    latent_vector,
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
    n_random_samples=None,
    search_bounds=(-1.0, 1.0),
    bounds_padding=0.05,
    batch_size=300_000,
    scale=1.0,
    offset=(0.0, 0.0, 0.0),
    path_save=None,
    filename="mesh_{mesh_idx}.vtk",
    path_original_mesh=None,
    scale_to_original_mesh=True,
    icp_transform=None,
    objects=1,
    verbose=False,
    device="cuda",
    use_vtk=True,
    fallback_to_original=True,
):
    """
    Create mesh using adaptive two-pass sampling: random sampling to find bounds, 
    then dense sampling only in the bounded region.
    
    Falls back to original create_mesh if bounds detection fails.
    """
    if verbose:
        print("Starting adaptive mesh creation...")
    
    if n_random_samples is None:
        n_random_samples = batch_size
    
    # Calculate voxel size if not provided
    if voxel_size is None:
        original_extent = search_bounds[1] - search_bounds[0]
        voxel_size = original_extent / (n_pts_per_axis - 1)
    
    # Use voxel_size as the original spacing
    original_spacing = voxel_size
    
    decoder.eval()
    
    # Pass 1: Find object bounds with random sampling
    bounds_result = find_object_bounds_random_sampling(
        decoder, latent_vector, n_random_samples, search_bounds, 
        objects, batch_size, device, verbose
    )
    
    if bounds_result is None:
        if verbose:
            print("No objects found in random sampling.")
        if fallback_to_original:
            if verbose:
                print("Falling back to original create_mesh...")
            return create_mesh(
                decoder, latent_vector, n_pts_per_axis, 
                voxel_origin, voxel_size, batch_size, scale, offset, 
                path_save, filename, path_original_mesh, scale_to_original_mesh, 
                icp_transform, objects, verbose, device, use_vtk
            )
        else:
            return [None] * objects if objects > 1 else None
    
    bounds_min, bounds_max = bounds_result
    
    # Pass 2: Dense sampling in bounded region
    if verbose:
        print("Creating dense grid in bounded region...")
    
    samples, grid_dims, voxel_origin = create_grid_samples_in_bounds(
        bounds_min, bounds_max, original_spacing, bounds_padding
    )
    
    if verbose:
        print(f"Dense grid: {grid_dims} = {samples.shape[0]} points (vs {n_pts_per_axis**3} original)")
        print(f"Speedup: {n_pts_per_axis**3 / samples.shape[0]:.1f}x fewer points")
    
    # Get SDF values for dense grid
    sdf_values_ = get_sdfs(
        decoder, samples, latent_vector, batch_size, objects=objects, device=device
    )
    
    # Reshape SDF values into grid
    nx, ny, nz = grid_dims
    sdf_values = torch.zeros((nx, ny, nz, objects))
    for i in range(objects):
        sdf_values[..., i] = sdf_values_[..., i].reshape(nx, ny, nz)
    
    # Calculate voxel size for the bounded grid
    voxel_size = original_spacing
    
    # Create meshes from gridded SDFs (same as original pipeline)
    meshes = []
    for mesh_idx in range(objects):
        sdf_values_ = sdf_values[..., mesh_idx]
        
        # Check if there is a surface
        if 0 < sdf_values_.min() or 0 > sdf_values_.max():
            if verbose is True:
                print("WARNING: SDF values do not span 0 - there is no surface")
                print("\tSDF min: ", sdf_values_.min())
                print("\tSDF max: ", sdf_values_.max())
                print("\tSDF mean: ", sdf_values_.mean())
            meshes.append(None)
        else:
            # Extract surface using VTK or marching cubes
            if use_vtk:
                mesh = sdf_grid_to_mesh_vtk(sdf_values_, voxel_origin, voxel_size, verbose)
            else:
                mesh = sdf_grid_to_mesh(sdf_values_, voxel_origin, voxel_size, verbose)
            meshes.append(mesh)

            if scale_to_original_mesh:
                if verbose is True:
                    print("Scaling mesh to original mesh... ")
                    print(icp_transform)
                mesh = scale_mesh(
                    meshes[mesh_idx],
                    old_mesh=path_original_mesh,
                    scale=scale,
                    offset=offset,
                    icp_transform=icp_transform,
                    verbose=verbose,
                )
                meshes[mesh_idx] = mesh

            # Save the mesh (if desired)
            if path_save is not None:
                meshes[mesh_idx].save_mesh(
                    os.path.join(path_save, filename.format(mesh_idx=mesh_idx))
                )
    
    return meshes[0] if objects == 1 else meshes


def create_grid_samples(
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
):
    n_pts_total = n_pts_per_axis**3

    indices = torch.arange(0, n_pts_total, out=torch.LongTensor())
    samples = torch.zeros(n_pts_total, 3)

    # generate samples on a grid...
    samples[:, 2] = indices % n_pts_per_axis
    samples[:, 1] = (indices // n_pts_per_axis) % n_pts_per_axis
    samples[:, 0] = ((indices // n_pts_per_axis) // n_pts_per_axis) % n_pts_per_axis

    # scale & transform the grid as appropriate
    samples[:, :3] = samples[:, :3] * voxel_size
    for axis in range(3):
        samples[:, axis] = samples[:, axis] + voxel_origin[axis]

    return samples


def get_sdfs(decoder, samples, latent_vector, batch_size=32**3, objects=1, device="cuda"):
    """
    Get SDF values for samples with sticky cache optimization for triplanar models.
    
    Args:
        decoder: The decoder model
        samples: Sample points to evaluate
        latent_vector: Latent code for the shape
        batch_size: Batch size for processing points
        objects: Number of objects
        device: Device to run on
    """
    n_pts_total = samples.shape[0]
    current_idx = 0
    sdf_values = torch.zeros(samples.shape[0], objects)

    # Check if decoder is a triplanar model
    is_triplanar = hasattr(decoder, 'clear_sticky') and hasattr(decoder, '_sticky_latent')
    
    if is_triplanar:
        decoder.clear_sticky()  # Clear any existing cache
        print("Detected triplanar model - using sticky cache optimization")

    if batch_size > n_pts_total:
        print(
            "WARNING: batch_size is greater than the number of samples, setting batch_size to the number of samples"
        )
        batch_size = n_pts_total

    batch_num = 0
    while current_idx < n_pts_total:
        current_batch_size = min(batch_size, n_pts_total - current_idx)
        sampled_pts = samples[current_idx : current_idx + current_batch_size, :3].to(device)
        
        # Use sticky mode for triplanar models
        use_sticky = is_triplanar and latent_vector is not None
        sdf_values[current_idx : current_idx + current_batch_size, :] = (
            decode_sdf(decoder, latent_vector, sampled_pts, sticky=use_sticky).detach().cpu()
        )

        current_idx += current_batch_size
        if is_triplanar:
            batch_info = f" (batch {batch_num+1}: {'CNN+MLP' if batch_num == 0 else 'cached planes+MLP'}, size={current_batch_size})"
        else:
            batch_info = f" (batch {batch_num+1}, size={current_batch_size})"
        print(f"Processed {current_idx} / {n_pts_total} points{batch_info}")
        batch_num += 1
        
    return sdf_values


def decode_sdf(decoder, latent_vector, queries, sticky=True):
    """
    Decode SDF values for query points.
    
    Args:
        decoder: The decoder model
        latent_vector: Latent code for the shape
        queries: Query points (N, 3)
        sticky: If True, use sticky cache for triplanar models
    """
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
        return decoder(inputs)
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], dim=1)
        
        # Use sticky mode if requested and supported by the decoder
        if sticky and hasattr(decoder, 'forward'):
            # Check if decoder supports sticky parameter
            sig = inspect.signature(decoder.forward)
            if 'sticky' in sig.parameters:
                return decoder(inputs, sticky=True)
        
        return decoder(inputs)
