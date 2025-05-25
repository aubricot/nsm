import os
import pymskt as mskt
from pymskt.mesh import Mesh
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import torch
import hashlib
from datetime import datetime
import warnings
import time
import point_cloud_utils as pcu
from multiprocessing import Pool
import multiprocessing
import zipfile
import gc
try:
    from pympler import tracker, muppy #asizeof, summary, muppy, tracker
except ModuleNotFoundError:
    print('Pympler not installed, cannot use asizeof - if trying to debug memory usage, install pympler')



today_date = datetime.now().strftime("%b_%d_%Y")

def get_rand_uniform_pts(n_pts, mins=(-1, -1, -1), maxs=(1, 1, 1)):
    """
    Given a set of points, returns a set of points that are randomly sampled
    from a uniform distribution from min(s) to max(s).

    Args:
        n_pts (int): Number of points to sample
        mins (tuple, optional): Minimum value of the distribution. Defaults to (-1, -1, -1).
        maxs (tuple, optional): Maximum value of the distribution. Defaults to (1, 1, 1).

    Returns:
        np.ndarray: (n_pts, 3) array of points
    
    Tests: 
        - Ensure returns np array 
            - Ensure shape (n_pts, 3)
            - Ensure points within input mins/maxs
    """
    rand_gen = np.random.uniform

    pts = np.zeros((n_pts, len(mins)))
    mins = np.tile(mins, [n_pts, 1])
    pts[:,:] = rand_gen(mins, maxs)

    return pts

def get_pts_center_and_scale(
    pts,
    center=True,
    scale=True,
    scale_method='max_rad',
    return_pts=False,
    pts_center=None
):
    """
    Given a set of points, returns the center and scale of the points.

    Args:
        pts (np.ndarray): (n_pts, 3) array of points
        center (bool, optional): Whether to center the points. Defaults to True.
        scale (bool, optional): Whether to scale the points. Defaults to True.
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        return_pts (bool, optional): Whether to return the centered and scaled points. Defaults to False.
        pts_center (np.ndarray, optional): (n_pts, 3) array of points to center the points on. Defaults to None.
    
    Returns:
        tuple: (center, scale) of the points

    Raises:
        NotImplementedError: If scale_method is not implemented

    Notes:
        If pts_center is not None, then use that to center the points
        and use all of the points in pts to scale. This is used for
        the bone only, and then scaling based on bone + cartilage
    
    Tests:
        - Ensure returns tuple length 2
        - Ensure center is np array of shape (3,)
        - Ensure scale is float
    """

    if pts_center is None:
        center = np.mean(pts, axis=0)
    else:
        center = np.mean(pts_center, axis=0)
    pts -= center

    if scale_method == 'max_rad':
        scale = np.max(np.linalg.norm(pts, axis=-1), axis=-1)
        pts /= scale
    else:
        raise NotImplementedError(f'Scale Method ** {scale_method} ** Not Implemented')
   
    if return_pts is True:
        return center, scale, pts
    
    return center, scale

def is_zipfile(filename):
    try:
        return zipfile.is_zipfile(filename)
    except (IOError, zipfile.BadZipfile):
        return False
    
def meshfix(mesh, assert_=False, assert_error=0.01):
    """
    Fixes a mesh using meshfix.
    
    Args:
        mesh (mskt.mesh.Mesh): Mesh to fix
    
    Notes:
        Fixes the mesh in place.
    """
    n_pts_orig = mesh.point_coords.shape[0]
    mesh.fix_mesh()
    n_pts_fixed = mesh.point_coords.shape[0]
    # Asserting that no more than 1% of the mesh points were removed
    print(f'Fixed mesh, {n_pts_orig} -> {n_pts_fixed} ({(n_pts_fixed - n_pts_orig) / n_pts_orig * 100:.2f}%)')
    if assert_ is True:
        assert (n_pts_orig - n_pts_fixed) < (assert_error * n_pts_orig), f'Mesh dropped too many points, {n_pts_orig} -> {n_pts_fixed}'

def get_cube_mins_maxs(pts):
    """
    Given a set of points, returns the mins and maxs of the points
    in a cube.
    
    Args:
        pts (np.ndarray): (n_pts, 3) array of points
    
    Returns:
        tuple: (mins, maxs) of the points
    
    Raises:
        ValueError: If input is empty or has wrong dimensions
    
    Tests:
        - ensure returns tuple length 2
        - ensure mins and maxs are np arrays of shape (3,)
    """
    if pts.size == 0:
        raise ValueError("Input array is empty")
    if len(pts.shape) != 2 or pts.shape[1] != 3:
        raise ValueError("Input must be a 2D array with shape (n_pts, 3)")

    mean = np.mean(pts, axis=0)
    norm_pts = pts - mean
    radial_max = np.max(np.linalg.norm(norm_pts, axis=-1))
    mins = mean - radial_max
    maxs = mean + radial_max
    
    return mins, maxs

def read_mesh_get_sampled_pts(
    path, 
    mean=[0,0,0], 
    sigma=1, 
    n_pts=200000, 
    rand_function='normal', 
    center_pts=True,
    norm_pts=False,
    scale_method='max_rad',
    get_random=True,
    register_to_mean_first=False,
    mean_mesh=None,
    fix_mesh=True,
    include_surf_in_pts=False,
    uniform_pts_buffer=0.0,
    # Single mesh specific
    return_point_cloud=False,
    **kwargs,
):
    """
    Function to read in and sample points from a single mesh.
    
    Args:
        path (str): Path to mesh
        mean (list, optional): Mean to apply for random sample(s). Defaults to [0,0,0].
        sigma (float, optional): Standard deviation/scale to apply for random sample(s). Defaults to 1.
        n_pts (int, optional): Number of points to sample. Defaults to 200000.
        rand_function (str, optional): Distribution to sample from. Defaults to 'normal'. Also supports 'laplace'.
        center_pts (bool, optional): Whether to center the points. Defaults to True.
        norm_pts (bool, optional): Whether to normalize the points. Defaults to False.
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        get_random (bool, optional): Whether to sample random points. Defaults to True.
        return_orig_mesh (bool, optional): Whether to return the original mesh. Defaults to False.
        return_new_mesh (bool, optional): Whether to return the new mesh. Defaults to False.
        return_orig_pts (bool, optional): Whether to return the original points. Defaults to False.
        register_to_mean_first (bool, optional): Whether to register the mesh to the mean mesh first. Defaults to False.
        mean_mesh (vtkPolyData or mskt.mesh.Mesh, optional): Mean mesh to register to. Defaults to None.
        return_point_cloud (bool, optional): Whether to return the point cloud. Defaults to False.
        fix_mesh (bool, optional): Whether to fix the mesh (using meshfix). Defaults to True.
        include_surf_in_pts (bool, optional): Whether to include the surface points in the random points. Defaults to False.
    
    Returns:
        dict: Dictionary of results
    
    Notes:
        - Is this too big? Can this be broken into smaller functions and then
        called in this function? 
            - Its currently > 100 lines of code. 
            - Some of the individual components are lengthy enough to be a single function.
            - This will make it easier to test and debug
                - Each individual function could be more easily tested. 

    Tests:
        - 
    """
    list_deprecated = ['return_scale', 'return_center', 'return_orig_pts', 'return_orig_mesh', 'return_new_mesh']
    for kwarg in kwargs:
        if kwarg in list_deprecated:
            print(f'{kwarg} is deprecated and not used in this function - always True')
    
    results = {}

    # if mesh path does not exist, return None (skipping)
    if os.path.exists(path) is False:
        print(f'Skipping ... path does not exist, {path}')
        return None

    # read in mesh & "fix" using meshfix if requested
    orig_mesh = Mesh(path)
    if fix_mesh is True:
        meshfix(orig_mesh)
    
    new_mesh = orig_mesh.copy()
    
    # return orig_pts expanded dims for compatibility when storing
    # multiple meshes in the same dictionary
    results['orig_pts'] = [orig_mesh.point_coords]

    if (register_to_mean_first is True):
        print('Registering mesh to mean mesh')
        # Rigid + scaling alginment of the original mesh to the mean mesh
        # of the model. This allows all downstream scaling to occur as expected
        # it also aligns the new bone with the mean/expected bone of the shape model
        # to maximize fidelity of the reconstruction.

        if mean_mesh is None:
            raise Exception('Must provide mean mesh to register to')
        icp_transform = orig_mesh.rigidly_register(
            other_mesh=mean_mesh,
            as_source=True,
            apply_transform_to_mesh=False,
            return_transformed_mesh=False,
            max_n_iter=100,
            n_landmarks=1000,
            reg_mode='similarity',
            return_transform=True,
        )
        results['icp_transform'] = icp_transform
        new_mesh.apply_transform_to_mesh(icp_transform)
    else:
        print('No registration')
        results['icp_transform'] = None
        
    if (center_pts is True) or (norm_pts is True):
        print('Scaling and centering mesh')
        center, scale, new_pts = get_pts_center_and_scale(
            np.copy(new_mesh.point_coords),
            center=center_pts,
            scale=norm_pts,
            scale_method=scale_method,
            return_pts=True
        )
        new_mesh.point_coords = new_pts
    else:
        print('Not scaling or centering mesh')
        scale = 1
        center = np.zeros(3)
        new_pts = new_mesh.point_coords
    

    results['new_pts'] = [new_pts]

    if get_random is True:
        if sigma is not None:
            rand_pts = new_mesh.rand_pts_around_surface(n_pts=n_pts, surface_method='random', distribution=rand_function, sigma=sigma)
        else:
            mins, maxs = get_cube_mins_maxs(new_pts)
            mins = mins - uniform_pts_buffer/2 * (maxs-mins)
            maxs = maxs + uniform_pts_buffer/2 * (maxs-mins)
            rand_pts = get_rand_uniform_pts(n_pts, mins=mins, maxs=maxs)
        
        if include_surf_in_pts is True:
            rand_pts = np.concatenate([rand_pts, new_pts], axis=0)

        if norm_pts is True:
            clip_val = 1.0 + uniform_pts_buffer/2
            rand_pts = np.clip(rand_pts, -clip_val, clip_val)

        rand_sdf = new_mesh.get_sdf_pts(pts=rand_pts, method='pcu')

        results['xyz'] = rand_pts
        results['sdf'] = rand_sdf
        results['pts_surface'] = [0] * rand_pts.shape[0]
    else:
        results['pts'] = new_pts
        results['sdf'] = np.zeros(new_pts.shape[0])
        results['pts_surface'] = [0] * new_pts.shape[0]
        
    
    if return_point_cloud is True:
        results['point_cloud'] = new_pts

    # Theres no reason we shouldnt include these... they are always there and take up effectively no space. 
    results['scale'] = scale
    results['center'] = center

    results['orig_mesh'] = [orig_mesh]
    results['new_mesh'] = [new_mesh]
    
    return results

def unpack_pts(data, pts_name='orig_pts'):
    """
    Unpacks the points loaded from a numpy.npz cache file. The cache is a 
    dict of various params, e.g., xyz, sdf, indices of positive and negative
    sdfs, for each surface. 

    Args:
        data (dict): Dictionary of data loaded from a numpy.npz cache file.
        pts_name (str, optional): Name of the points to unpack. Defaults to 'orig_pts'.

    Returns:
        list: List of torch tensors of the points. The index position indicates
        what the surface the points are from.
    
    Notes:
        - 
    
    Tests:
        - Give path npz file, load, ensure:
            - returns list of torch tensors
            - the list length is the number of surfaces
    """
    # get original points...
    pts = []
    pts_arrays = [x for x in data.files if f'{pts_name}_' in x]
    if len(pts_arrays) > 0:
        for pts_idx in range(len(pts_arrays)):
            pts.append(torch.from_numpy(data[f'{pts_name}_{pts_idx}']))
        
    return pts

def unpack_numpy_data(
    data_,
    point_cloud=False,
    list_additional_keys=['orig_pts', 'new_pts', 'pos_idx', 'neg_idx', 'surf_idx']
):
    
    data = {}

    # Get points / xyz coords
    if 'pts' in data_:
        data['xyz'] = torch.from_numpy(data_['pts']).float()
    elif 'xyz' in data_:
        data['xyz'] = torch.from_numpy(data_['xyz']).float()
    else:
        raise ValueError('No pts or xyz in cached file')
    
    # Get SDFs of the original points
    if 'sdfs' in data_:
        data['gt_sdf'] = torch.from_numpy(data_['sdfs']).float()
    elif 'gt_sdf' in data_:
        data['gt_sdf'] = torch.from_numpy(data_['gt_sdf']).float()
    elif 'sdf' in data_:
        data['gt_sdf'] = torch.from_numpy(data_['sdf']).float()
    else:
        raise ValueError('No sdfs or gt_sdf or sdf in cached file')

    # Get random point cloud surface points... this was used for Diffusion SDFs model
    if point_cloud is True:
        data['point_cloud'] = torch.from_numpy(data_['point_cloud']).float()
    
    for key in list_additional_keys:
        key_data = unpack_pts(data_, pts_name=key)
        data[key] = key_data

    return data

def read_meshes_get_sampled_pts(
    paths, 
    mean=[0,0,0], 
    sigma=[1, 1], 
    n_pts=[200000, 200000], 
    rand_function='normal', 
    center_pts=True,
    norm_pts=False,
    scale_method='max_rad',
    get_random=True,
    register_to_mean_first=False,
    mean_mesh=None,
    fix_mesh=True,
    include_surf_in_pts=False,
    # Multiple mesh specific
    scale_all_meshes=True,
    center_all_meshes=False,
    mesh_to_scale=0,
    verbose=False,
    icp_transform=None,
    uniform_pts_buffer=0.0,
    **kwargs,
):
    """
    Function to read in and sample points from multiple meshes.
    """
    tic = time.time()
    list_deprecated = ['return_scale', 'return_center', 'return_orig_pts', 'return_orig_mesh', 'return_new_mesh']
    for kwarg in kwargs:
        if kwarg in list_deprecated:
            print(f'{kwarg} is deprecated and not used in this function - always True')
    
    # preallocate results dictionary
    results = {}

    # Read all meshes and store in list
    orig_meshes = []
    orig_pts = []
    for path in paths:
        if path is None:
            print(f'Mesh is None... returning None for meshes and pts, {path}')
            orig_meshes.append(None)
            orig_pts.append(None)
            continue
        if os.path.exists(path) is False:
            print(f'Skipping ... path does not exist, {path}')
            return None
        mesh = Mesh(path)
        # fixing meshes ensures they are not degenerate
        # degenerate meshes will cause issues fitting SDFs. 
        if fix_mesh is True:
            meshfix(mesh)
        orig_meshes.append(mesh)
        orig_pts.append(mesh.point_coords)
    
    # return orig_pts
    results['orig_pts'] = orig_pts

    toc = time.time()
    print(f'Finished reading meshes in {toc - tic:.3f}s')
    tic = time.time()

    # Copy all meshes & points to new lists
    new_meshes = []
    new_pts = []
    for mesh_idx, orig_mesh in enumerate(orig_meshes):
        if orig_mesh is None:
            print(f'Mesh is None... returning None for meshes and pts, {paths[mesh_idx]}')
            new_meshes.append(None)
            new_pts.append(None)
            continue
        new_mesh_ = orig_mesh.copy()
        new_pts.append(new_mesh_.point_coords)
        new_meshes.append(new_mesh_)

    if (register_to_mean_first is True):
        # Rigid + scaling (similarity) alginment of the original mesh to the mean mesh
        # of the model. This allows all downstream scaling to occur as expected
        # it also aligns the new bone with the mean/expected bone of the shape model
        # to maximize fidelity of the reconstruction.
        print('Registering meshes to mean mesh')
        if mean_mesh is None:
            raise ValueError('Must provide mean mesh to register to')

        if icp_transform is None:
            icp_transform = orig_meshes[mesh_to_scale].rigidly_register(
                other_mesh=mean_mesh,
                as_source=True,
                apply_transform_to_mesh=False,
                return_transformed_mesh=False,
                max_n_iter=100,
                n_landmarks=1000,
                reg_mode='similarity',
                return_transform=True,
            )
        results['icp_transform'] = icp_transform

        # apply transform to all meshes
        for idx, new_mesh in enumerate(new_meshes):
            if new_mesh is None:
                print(f'Mesh is None... returning None for meshes and pts, {paths[idx]}')
                new_pts[idx] = None
                continue
            new_mesh.apply_transform_to_mesh(icp_transform)
            new_pts[idx] = new_mesh.point_coords

    else:
        print('No registration')
        results['icp_transform'] = None
    
    toc = time.time()
    print(f'Finished registering meshes in {toc - tic:.3f}s')
    tic = time.time()

    if (center_pts is True) or (norm_pts is True):
        print('Scaling and centering meshes')
        if scale_all_meshes is True:
            if None in new_pts:
                new_pts_ = [x for x in new_pts if x is not None]
                pts_ = np.concatenate(new_pts_, axis=0)
            else:
                pts_ = np.concatenate(new_pts, axis=0)
            if center_all_meshes is True:
                # Set as None - becuasse scaling and centering on same data
                pts_center = None
            else:
                # set specific points to center becuase they are not the same
                # for centering as they are for scaling (pts_)
                pts_center = new_pts[mesh_to_scale]
        else:
            pts_ = new_pts[mesh_to_scale]
            if center_all_meshes is True:
                # set specific points to center because scale/center are not on
                # the same data
                if None in new_pts:
                    new_pts_ = [x for x in new_pts if x is not None]
                    pts_center = np.concatenate(new_pts_, axis=0)
                else:
                    pts_center = np.concatenate(new_pts, axis=0)
            else:
                # Set as None - becuasse scaling and centering on same data
                pts_center = None

        center, scale = get_pts_center_and_scale(
            np.copy(pts_),
            center=center_pts,
            scale=norm_pts,
            scale_method=scale_method,
            return_pts=False,
            pts_center=pts_center,
        )

        for pts_idx, new_pts_ in enumerate(new_pts):
            if new_pts_ is None:
                continue
            new_pts[pts_idx] = (new_pts_ - center)/scale
    else:
        # Do nothing to the points because they are left the same.
        scale = 1
        center = np.zeros(3)
    
    toc = time.time()
    print(f'Finished centering and scaling meshes in {toc - tic:.3f}s')
    tic = time.time()

    for mesh_idx, new_mesh in enumerate(new_meshes):
        if new_mesh is None:
            continue
        new_mesh.point_coords = new_pts[mesh_idx]
    
    results['new_pts'] = new_pts

    if get_random is True:
        rand_pts = []
        rand_sdf = []
        pts_surface = []

        if None in sigma:
            new_pts_ = [x for x in new_pts if x is not None]
            pts_cube = np.concatenate(new_pts_, axis=0)
            mins, maxs = get_cube_mins_maxs(pts_cube)
            mins = mins - uniform_pts_buffer/2 * (maxs-mins)
            maxs = maxs + uniform_pts_buffer/2 * (maxs-mins)

        for new_pts_idx, new_mesh_ in enumerate(new_meshes):
            if new_mesh_ is None:
                continue
            if n_pts[new_pts_idx]  > 0:
                if sigma[new_pts_idx] is not None:
                    rand_pts_ = new_mesh_.rand_pts_around_surface(n_pts=n_pts[new_pts_idx], surface_method='random', distribution=rand_function, sigma=sigma[new_pts_idx])
                else:
                    rand_pts_ = get_rand_uniform_pts(n_pts[new_pts_idx], mins=mins, maxs=maxs)
                
                if include_surf_in_pts is True:
                    rand_pts_ = np.concatenate([rand_pts_, new_pts_], axis=0)

                rand_pts.append(rand_pts_)
                pts_surface.append([new_pts_idx] * rand_pts_.shape[0])
            else:
                rand_pts.append(np.zeros((0,3)))
                pts_surface.append(np.zeros((0,3)))
        
        rand_pts = np.concatenate(rand_pts, axis=0)
        # NOTE: pts_surface indices correspond to original mesh positions in the input list,
        # not contiguous indices. If meshes [mesh0, None, mesh2] are passed, points will be
        # labeled as surface 0 and surface 2 (no surface 1 points exist).
        # This works correctly with reconstruction code which handles missing surfaces.
        pts_surface = np.concatenate(pts_surface, axis=0)

        
        for new_mesh in new_meshes:
            if new_mesh is None:
                rand_sdf.append(None)
                continue
            tic_ = time.time()
            print(rand_pts.shape, new_mesh.point_coords.shape)
            print(rand_pts.dtype, new_mesh.point_coords.dtype)
            print(type(rand_pts))
            rand_sdf.append(new_mesh.get_sdf_pts(pts=rand_pts, method='pcu'))
            toc_ = time.time()
            print(f'Finished calculating SDFs in {toc_ - tic_:.3f}s')

        results['pts'] = rand_pts
        results['sdf'] = rand_sdf
        results['pts_surface'] = pts_surface
    else:
        sdfs = []
        # Need to set SDFs for the same mesh to be 0
        # but need to actually calculate the SDFs for the other
        # meshes.
        for mesh_idx, new_mesh in enumerate(new_meshes):
            if new_mesh is None:
                sdfs.append(None)
                continue
            sdfs_ = []

            for pts_idx, new_pts_ in enumerate(new_pts):
                if new_pts_ is None:
                    continue
                if verbose is True:
                    print('mesh_idx, new_mesh point_coords shape', mesh_idx, new_mesh.point_coords.shape)
                if pts_idx == mesh_idx:
                    if verbose is True:
                        print('adding zeros new_pts_ shape (zero)', new_pts_.shape)
                    # same mesh, set SDFs to 0
                    sdfs_.append(np.zeros(new_pts_.shape[0]))
                else:
                    # different mesh, calculate SDFs
                    _sdfs_ = new_mesh.get_sdf_pts(pts=new_pts_, method='pcu')
                    if verbose is True:
                        print('caculating SDFs for new_pts_ ', _sdfs_.shape)
                    sdfs_.append(_sdfs_)

            sdfs.append(np.concatenate(sdfs_, axis=0))

        pts_surface = []
        # NOTE: pts_surface indices correspond to original mesh positions in the input list,
        # not contiguous indices. If meshes [mesh0, None, mesh2] are passed, points will be
        # labeled as surface 0 and surface 2 (no surface 1 points exist).
        # This works correctly with reconstruction code which handles missing surfaces.
        for pts_idx, new_pts_ in enumerate(new_pts):
            if new_pts_ is None:
                continue
            pts_surface.append([pts_idx] * new_pts_.shape[0])
        pts_surface = np.concatenate(pts_surface, axis=0)

        new_pts_filtered = [x for x in new_pts if x is not None]
        results['pts'] = np.concatenate(new_pts_filtered, axis=0)
            
        results['sdf'] = sdfs
        results['pts_surface'] = pts_surface
    
    toc = time.time()
    print(f'Finished getting random points and SDFs in {toc - tic:.3f}s')


    results['new_pts'] = new_pts

    results['scale'] = scale
    results['center'] = center

    results['orig_mesh'] = orig_meshes
    results['new_mesh'] = new_meshes

    return results


# check that probabilities 0-1
def check_probabilities(p_):
    if (p_<0) or (p_>1):
        raise ValueError('Probabilities must be between 0 and 1')

def check_probabilities_sum(p_near_, p_far_):
    if (p_near_ + p_far_ > 1):
        raise ValueError('sum of p_near_ & p_far_ must be <=1')

class SDFSamples(torch.utils.data.Dataset):
    """
    Dataset class for sampling SDFs from meshes.
    
    Args:
        list_mesh_paths (list): List of paths to meshes
        subsample (int, optional): Number of points to subsample. Defaults to None.
        n_pts (int, optional): Number of points to sample. Defaults to 500000.
        p_near_surface (float, optional): Proportion of points to sample near the surface. Defaults to 0.4.
        p_further_from_surface (float, optional): Proportion of points to sample further from the surface. Defaults to 0.4.
        sigma_near (float, optional): Standard deviation/scale of the distribution for points near the surface. Defaults to 0.01.
        sigma_far (float, optional): Standard deviation/scale of the distribution for points further from the surface. Defaults to 0.1.
        rand_function (str, optional): Distribution to sample from. Defaults to 'normal'. Also supports 'laplace'.
        center_pts (bool, optional): Whether to center the points. Defaults to True.
        norm_pts (bool, optional): Whether to normalize the points. Defaults to False.
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        loc_save (str, optional): Location to save the cached files. Defaults to os.environ['LOC_SDF_CACHE'].
        include_seed_in_hash (bool, optional): Whether to include the random seed in the hash. Defaults to True.
        save_cache (bool, optional): Whether to save the cached files. Defaults to True.
        load_cache (bool, optional): Whether to load the cached files. Defaults to True.
        random_seed (int, optional): Random seed. Defaults to None.
        reference_mesh (vtkPolyData or mskt.mesh.Mesh, optional): Reference mesh to register to. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        equal_pos_neg (bool, optional): Whether to have equal positive and negative SDFs. Defaults to True.
        fix_mesh (bool, optional): Whether to fix the meshes (sing meshfix). Defaults to True.
        print_filename (bool, optional): Whether to print the filename when loading. Defaults to False.
        
        Notes:
            If reference_mesh is not None, then all meshes will be registered to the reference mesh.
            If equal_pos_neg is True, then the number of positive and negative SDFs will be equal.
            If fix_mesh is True, then the meshes will be fixed using meshfix.
            If print_filename is True, then the filename will be printed when loading.    
        """
    
    def __init__(
        self,
        list_mesh_paths,
        subsample,
        n_pts=500000,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function='normal', 
        center_pts=True,
        norm_pts=False,
        scale_method='max_rad',
        scale_jointly=False,
        joint_scale_buffer=0.1,
        loc_save=os.environ.get('LOC_SDF_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'nsm_sdf_cache')),
        include_seed_in_hash=True,
        save_cache=True,
        load_cache=True,
        random_seed=None,
        reference_mesh=None,
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
        print_filename=False,
        multiprocessing=True,
        n_processes=2,
        store_data_in_memory=False,
        debug_memory=False,
        test_load_times=True,
        uniform_pts_buffer=0.0,
    ):
        
        # p_near_surface & p_further_from_surface must be >=0, <=1
        # sum of p_near_surface & p_further_from_surface must be <=1
        if isinstance(p_near_surface, (list, tuple)) & isinstance(p_further_from_surface, (list, tuple)):
            for p_near, p_far in zip(p_near_surface, p_further_from_surface):
                check_probabilities(p_near)
                check_probabilities(p_far)
                check_probabilities_sum(p_near, p_far)
        elif isinstance(p_near_surface, float) & isinstance(p_further_from_surface, float):
            check_probabilities(p_near_surface)
            check_probabilities(p_further_from_surface)
            check_probabilities_sum(p_near_surface, p_further_from_surface)
        else:
            raise ValueError('p_near_surface & p_further_from_surface must be floats or lists/tuples of floats')
        
        self.list_mesh_paths = list_mesh_paths
        self.subsample = subsample
        self.n_pts = n_pts
        self.p_near_surface = p_near_surface
        self.p_further_from_surface = p_further_from_surface
        self.sigma_near = sigma_near
        self.sigma_far = sigma_far
        self.rand_function = rand_function
        self.center_pts = center_pts
        self.norm_pts = norm_pts
        self.scale_method = scale_method
        self.scale_jointly = scale_jointly
        self.joint_scale_buffer = joint_scale_buffer
        self.loc_save = loc_save
        self.include_seed_in_hash = include_seed_in_hash
        self.random_seed = random_seed
        self.reference_mesh = reference_mesh
        self.verbose = verbose
        self.equal_pos_neg = equal_pos_neg
        self.fix_mesh = fix_mesh
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.print_filename = print_filename
        self.multiprocessing = multiprocessing
        self.n_processes = n_processes
        self.store_data_in_memory = store_data_in_memory
        self.debug_memory = debug_memory
        self._memory_tracker = None
        self._memory_counter = 0
        self.test_load_times = test_load_times
        self.uniform_pts_buffer = uniform_pts_buffer
        
        # if store_data_in_memory is False & save_cache is False, then raise error
        if (self.store_data_in_memory is False) and (self.save_cache is False):
            raise ValueError('If store_data_in_memory is False, then save_cache must be True.'
                             'when data not stored in memory, it is loaded from disk - but data is'
                             'not saved to disk when save_cache is False.')

        # set defaults so can use same 'norm_and_scale_all_meshes' function
        # for single and multiple meshes
        if not hasattr(self, 'reference_object'):
            self.reference_object = 0
        if not hasattr(self, 'n_meshes'):
            self.n_meshes = 1

        self.max_radius = None
        self.center = None

        # preprocess inputs before proceeding
        self.preprocess_inputs()

        self.list_hash_params = self.get_hash_params()

        if save_cache is True:
            self.cache_folder = os.path.join(self.loc_save, today_date)
            os.makedirs(self.cache_folder, exist_ok=True)

        # get the combinations of points and sigmas to sample
        self.pt_sample_combos = self.get_pt_sample_combos()
        
        # preallocate reference mesh path to None
        self.reference_mesh_path = None
        
        if self.reference_mesh is not None:
            self.load_reference_mesh()

        # function to allow calling additional internal functions from subclasses. 
        self.run_before_loading_data()

        self.data = []
        # Wrap this loading loop in a multiprocessing pool
        if self.verbose is True:
            try:
                print(f'CPU affinity:{os.sched_getaffinity(0)}')
            except AttributeError:
                # sched_getaffinity is not available on all platforms (eg., mac/windows)
                print('CPU affinity not available on this platform')
        if self.multiprocessing is True:
            list_inputs = [(loc_mesh, self.verbose) for loc_mesh in self.list_mesh_paths]
            with Pool(processes=self.n_processes) as pool:
                self.data = pool.starmap(self.load_mesh_step, list_inputs)
        else:
            self.data = [self.load_mesh_step(loc_mesh, self.verbose) for loc_mesh in self.list_mesh_paths]

        # remove mesh paths that failed to load
        self.list_mesh_paths = [x for idx, x in enumerate(self.list_mesh_paths) if self.data[idx] is not None]
        # remove data that failed to load
        self.data = [x for x in self.data if x is not None]

        if self.scale_jointly is True:
            self.norm_and_scale_all_meshes()    
    
    
    def print_memory_summary(self):
        if self._memory_tracker is None:
            self._memory_tracker = tracker.SummaryTracker()
        
        # every 100th iteration, print the memory summary
        if self._memory_counter % 100 == 0:
            self._memory_tracker.print_diff()

            # all_objects = muppy.get_objects()
            # numpy_arrays = [obj for obj in all_objects if isinstance(obj, np.ndarray)]
            # refs = gc.get_referrers(numpy_arrays[0])
            # print('REFERENCES TO NUMPY ARRAY')
            # print(refs)
        # size_info = asizeof.asized(self, detail=1)
        # print(size_info)
        # all_objects = muppy.get_objects()
        # memory_summary = summary.summarize(all_objects)
        # if self._memory_summary is not None:
        # self._memory_summary = memory_summary

        self._memory_counter += 1

    
    def run_before_loading_data(self):
        pass

    def load_mesh_step(self, loc_mesh, verbose):
        if verbose is True:
            print('Loading mesh:', loc_mesh)
        
        if self.debug_memory is True:
            self.print_memory_summary()
        
        if self.multiprocessing is True:
            os.sched_setaffinity(0, range(multiprocessing.cpu_count()))
        data = self.get_sample_data_dict(loc_mesh)
        
        if data is None:
            print('Skipping mesh:', loc_mesh)
            print('Error in loading')
        
        if verbose is True:
            print('Data type:', type(data))
            print('Finished loading mesh:', loc_mesh)

        gc.collect()

        return data

    def norm_and_scale_all_meshes(self):
        """
        Normalize and scale all of the meshes.

        Take the average of the center of each mesh and uses it to center all of the meshes.
        Then, takes the max radius of all of the meshes (after centering) and uses it to 
        scale all of the meshes.

        Now, all of the meshes are centered and scaled jointly so the anatomical surfaces should
        roughly be aligned, removing this as a source of variation.
        """
        print('Computing centering and scaling...')
        # if not stored in memory, then get the centers and max radii from the data in memory
        if self.store_data_in_memory is False:
            print('Data not stored in memory... loading from disk')
            tic = time.time()
            centers = []
            for data in self.data:
                # load in the npz dict
                data_ = np.load(data)
                centers.append(np.mean(data_[f'new_pts_{self.reference_object}'], axis=0))
            # new center:
            center = np.mean(centers, axis=0)
            
            print('Done computing centers')
            
            max_radii = []
            # for each data, comput the max radius (from the new/global center)
            for data in self.data:
                data_ = np.load(data)
                max_radius = 0
                for mesh_idx in range(self.n_meshes):
                    xyz = data_[f'new_pts_{mesh_idx}']
                    centered_xyz = xyz - center
                    radii = np.linalg.norm(centered_xyz, axis=-1)
                    max_radius_ = np.max(radii)
                    if max_radius_ > max_radius:
                        max_radius = max_radius_
                max_radii.append(max_radius)
            max_radius = np.max(max_radii)
            # make the biggest radius a bit bigger than observed to enable model to 
            # generalize to unseen data that is slightly larger than the observed data.
            max_radius = max_radius * (1 + self.joint_scale_buffer)
            print('Done computing max radii')
            
            self.max_radius = max_radius.astype(np.float32)
            self.center = center.astype(np.float32)
            toc = time.time()
            print(f'Finished computing centering and scaling in {toc - tic:.3f}s')
            print(f'\tMax radius: {self.max_radius}')
            print(f'\tCenter: {self.center}')
            
        else: 
            # get the center of all of the meshes
            centers = []
            for data in self.data:
                # center around the reference object
                xyz = data[f'new_pts_{self.reference_object}']
                center = np.mean(xyz, axis=0)
                centers.append(center)
            centers = np.stack(centers, axis=0)
            center = np.mean(centers, axis=0)

            # subtract the center from all of the meshes
            for idx, data in enumerate(self.data):
                self.data[idx]['xyz'] -= center
                # iterate over all of the meshes and subtract the center
                for mesh_idx in range(self.n_meshes):
                    self.data[idx][f'new_pts_{mesh_idx}'] -= center
            
            # get the max radius of all of the meshes
            max_radii = 0
            for data in self.data:
                for mesh_idx in range(self.n_meshes):
                    xyz = data[f'new_pts_{mesh_idx}']
                    max_radius = np.max(np.linalg.norm(xyz, axis=-1))
                    if max_radius > max_radii:
                        max_radii = max_radius
            
                    
            # divide all of the meshes by the max radius
            for idx, data in enumerate(self.data):
                self.data[idx]['xyz'] /= max_radii
                # do the same for the sdf of each point
                self.data[idx]['gt_sdf'] /= max_radii
                # do the same for the original points
                for mesh_idx in range(self.n_meshes):
                    self.data[idx][f'new_pts_{mesh_idx}'] /= max_radii

    def preprocess_inputs(self):
        """
        Preprocess inputs to ensure they are in the correct format.
        """

        if self.scale_jointly is True:
            if self.center_pts is True:
                raise ValueError('Scale jointly assumes centering at end... so center should be False')
            if self.norm_pts is True:
                raise ValueError('Scale jointly assumes normalizing at end... so norm should be False')
    
    def get_dict_pts(self, data, pts_name):
        dict_pts = {}
        if isinstance(data[pts_name], list):
            for idx_, orig_pts_ in enumerate(data[pts_name]):
                dict_pts[f'{pts_name}_{idx_}'] = orig_pts_
        else:
            dict_pts[f'{pts_name}_0'] = data[pts_name]
        return dict_pts

    def save_data_to_cache(self, data, file_hash, filepath=None):
        """
        Save the data to the cache.
        
        Args:
            data (dict): Dictionary of data to save
            file_hash (str): Hash of the file
        """
        # if want to cache, and new... then save. 
        if filepath is None:
            filepath = os.path.join(self.cache_folder, f'{file_hash}.npz')
        dict_pts = {}
        dict_pts.update(self.get_dict_pts(data, 'orig_pts'))
        dict_pts.update(self.get_dict_pts(data, 'new_pts'))

        additional_keys = ['pos_idx', 'neg_idx', 'surf_idx', 'center', 'max_radius', 'max_radius_xyz']
        for key in additional_keys:
            if key in data:
                dict_pts.update(self.get_dict_pts(data, key))
                # dict_pts[key] = data[key]

        # add pos/negative point indices

        np.savez(filepath, pts=data['xyz'], sdfs=data['gt_sdf'], **dict_pts)

    def get_sample_data_dict(self, loc_mesh):
        """
        Given a mesh path, return a dictionary of the sampled points and SDFs.

        Args:
            loc_mesh (str): Path to mesh

        Returns:
            dict: Dictionary of sampled points and SDFs
        """
        

        # Create hash and filename
        file_hash = self.create_hash(loc_mesh)
        cached_file = self.find_hash(filename=f'{file_hash}.npz')

        file_loaded = False

        if (len(cached_file) > 0) and (self.load_cache is True):
            for cached_file_ in cached_file:
                if not is_zipfile(cached_file_):
                    print('DELETING BAD ZIP FILE:', cached_file_)
                    os.remove(cached_file_)
                    continue
                
                # if hashed file exists, load it. 
                try:
                    data_ = np.load(cached_file_)
                    data = unpack_numpy_data(data_)
                except zipfile.BadZipFile:
                    print('DELETING BAD ZIP FILE:', cached_file_)
                    os.remove(cached_file_)
                    continue
                
                if ('pos_idx' not in data) or ('neg_idx' not in data) or ('surf_idx' not in data):
                    pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
                    data['pos_idx'] = pos_idx
                    data['neg_idx'] = neg_idx
                    data['surf_idx'] = surf_idx
                    self.save_data_to_cache(data, file_hash, filepath=cached_file_)
                
                file_loaded = True
                cache_path = cached_file_
                break
            
        if file_loaded is False:
            # otherwise, load the mesh and create SDF samples. 
            print('Creating SDF Samples')
            if  self.print_filename is True:
                print(loc_mesh)
            data = {
                'xyz': torch.zeros((self.n_pts, 3)),
                'gt_sdf': torch.zeros((self.n_pts)),
            }
            pts_idx = 0

            if self.multiprocessing is True:
                if self.reference_mesh_path is not None:
                    reference_mesh = Mesh(self.reference_mesh_path)
                else:
                    reference_mesh = None
            else:
                reference_mesh = self.reference_mesh
            
            if self.verbose is True:
                print('type of reference mesh:', type(reference_mesh))
                print('ref mesh path:', self.reference_mesh_path)

            for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
                result_ = read_mesh_get_sampled_pts(
                    loc_mesh, 
                    mean=[0,0,0], 
                    sigma=sigma_, 
                    n_pts=n_pts_, 
                    rand_function=self.rand_function, 
                    center_pts=self.center_pts,
                    norm_pts=self.norm_pts,
                    scale_method=self.scale_method,
                    get_random=True,
                    return_orig_mesh=False,
                    return_new_mesh=False,
                    fix_mesh=self.fix_mesh,
                    register_to_mean_first=False if reference_mesh is None else True,
                    mean_mesh=reference_mesh,
                    uniform_pts_buffer=self.uniform_pts_buffer,
                )

                if result_ is None:
                    return None
            
                xyz_ = result_['pts'] if 'pts' in result_ else result_['xyz']
                sdfs_ = result_['sdf'] if 'sdf' in result_ else result_['gt_sdf']

                data['xyz'][pts_idx:pts_idx + n_pts_, :] = torch.from_numpy(xyz_).float()
                data['gt_sdf'][pts_idx:pts_idx + n_pts_] = torch.from_numpy(sdfs_).float()
                pts_idx += n_pts_

                if idx_ == 0:
                    # Convert list of arrays to tensors
                    data['orig_pts'] = [torch.from_numpy(pts).float() for pts in result_['orig_pts']]
                    data['new_pts'] = [torch.from_numpy(pts).float() for pts in result_['new_pts']]

            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data['pos_idx'] = pos_idx
            data['neg_idx'] = neg_idx
            data['surf_idx'] = surf_idx

            if self.save_cache is True:
                self.save_data_to_cache(data, file_hash)
                cache_path = os.path.join(self.cache_folder, f'{file_hash}.npz')
        
        if self.store_data_in_memory is False:
            if self.verbose is True:
                print('updating data to be cache path')
            # change the data to be the path to the saved cache file
            data = cache_path

        return data
    
    def get_pt_sample_combos(self):
        """
        Get the combinations of points and sigmas to sample from each mesh.

        Returns:
            list: List of lists of [n_pts, sigma] to sample
        """

        n_p_near_surface = int(self.n_pts * self.p_near_surface)
        n_p_further_from_surface = int(self.n_pts * self.p_further_from_surface)
        n_p_random = self.n_pts - n_p_near_surface - n_p_further_from_surface

        pt_sample_combos = [
            [n_p_near_surface, self.sigma_near],
            [n_p_further_from_surface, self.sigma_far],
            [n_p_random, None]
        ]

        return pt_sample_combos
    
    def sdf_pos_neg_idx(self, data):
        """
        Get the indices of the positive, negative, and surface SDFs.
        
        Args:
            data (dict): Dictionary of sampled points and SDFs
        
        Returns:
            tuple: (pos_idx, neg_idx, surf_idx) of indices
        """

        pos_idx = (data['gt_sdf'] > 0).nonzero(as_tuple=True)[0]
        neg_idx = (data['gt_sdf'] < 0).nonzero(as_tuple=True)[0]
        surf_idx = (data['gt_sdf'] == 0).nonzero(as_tuple=True)[0]

        # Repeat +/- indices if either of them does not have enough for a full batch. 
        samples_per_sign = int(self.subsample/2)
        pos_idx = pos_idx.repeat(samples_per_sign//pos_idx.size(0) + 1)
        neg_idx = neg_idx.repeat(samples_per_sign//neg_idx.size(0) + 1)
        
        return pos_idx, neg_idx, surf_idx
        
    def find_hash(self, filename='hashed_filename.npz'):
        """
        Find the hashed filename in the cache.
        
        Args:
            filename (str, optional): Hashed filename. Defaults to 'hashed_filename.npz'.
        
        Returns:
            list: List of paths to the hashed filename
        """

        files = []
        for p, d, f in os.walk(self.loc_save):
            for filename_ in f:
                if filename_ == filename:
                    files.append(os.path.join(p, filename_))
                    print('File found in cache:', files[-1])
                    return files
                    
        return files

    def load_reference_mesh(self):
        """
        Load the reference mesh.
        
        Raises:
            TypeError: If reference mesh is not a string, list of strings, or mesh.Mesh object
        """
        
        if self.verbose is True:
            print('Loading reference mesh: ', self.reference_mesh)

        if issubclass(type(self.reference_mesh), Mesh):
            pass
        elif isinstance(self.reference_mesh, int):
            if isinstance(self.list_mesh_paths[0], (str, Mesh)):
                mesh = self.list_mesh_paths[self.reference_mesh]
            elif isinstance(self.list_mesh_paths[0], (list, tuple)):
                
                mesh = self.list_mesh_paths[self.reference_mesh][self.mesh_to_scale]
            else:
                raise TypeError('provided list_meshes wrong type')
            self.reference_mesh = Mesh(mesh)
        elif isinstance(self.reference_mesh, str):
            self.reference_mesh = Mesh(self.reference_mesh)
        elif isinstance(self.reference_mesh, list):
            # below will throw error in SDFSamples, but will work in MultiSurfaceSDFSamples
            # where self.mesh_to_scale is defined & a list/tuple type likely
            # TODO: Why is reference_object different from mesh_to_scale?
            self.reference_mesh = Mesh(self.reference_mesh[self.reference_object])
        else:
            raise TypeError('Reference mesh must be a string, list of strings, or mesh.Mesh object, not', type(self.reference_mesh))
        
        if self.verbose is True:
            print('type of reference mesh:', type(self.reference_mesh))
        
        if self.multiprocessing is True:
            # update reference mesh path to be a has on the current time - so as to not end up with 
            # multiple training runs of different tissues using the same reference mesh.
            # this happens because the random seed is set - so all models get the same random number.
            hashed_time = hashlib.md5(str(int(time.time())).encode()).hexdigest()
            self.reference_mesh_path = os.path.join(self.cache_folder, f'REFERENCE_MESH_{hashed_time}.vtk')
            self.reference_mesh.save_mesh(self.reference_mesh_path)
            self.reference_mesh = None

    def get_hash_params(self):
        """
        Get the parameters to hash for saving/loading the cache.
        
        Returns:
            list: List of parameters to hash
        """

        list_hash_params = [
            self.n_pts,
            self.p_near_surface, self.sigma_near,
            self.p_further_from_surface, self.sigma_far,
            self.center_pts,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.reference_mesh,
            self.fix_mesh,
            self.scale_jointly
        ]

        return list_hash_params
    
    def create_hash(self, loc_mesh):
        """
        Create the hash for the cache.
        
        Args:
            loc_mesh (str): Path to mesh
            
        Returns:
            str: Hashed string
        """

        list_hash_params = self.list_hash_params.copy()
        if isinstance(loc_mesh, str):
            list_hash_params.insert(0, loc_mesh)
        elif isinstance(loc_mesh, (list, tuple)):
            for path in loc_mesh:
                if self.verbose is True:
                    print(loc_mesh)
                list_hash_params.insert(0, path)
    
        if (self.include_seed_in_hash is True) and (self.random_seed is not None):
            list_hash_params.append(self.random_seed) # random seed state
        if self.verbose is True:
            print('List Params', list_hash_params)
        list_hash_params = [str(x) for x in list_hash_params]
        file_params_string = '_'.join(list_hash_params)
        hash_str = hashlib.md5(file_params_string.encode()).hexdigest()
        return hash_str
        
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Length of the dataset
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the item at the index.
        
        Args:
            idx (int): Index of the item
        
        Returns:
            dict: Dictionary of the item
            idx (int): Index of the item
        """
        
        tic_whole_load = time.time()

        if self.store_data_in_memory is False:
            # if not storing in memory, then load from cache
            tic = time.time()
            data_ = np.load(self.data[idx])
            toc = time.time()
            time_ = toc - tic
            
            # get size of the numpy file in mb
            size = os.path.getsize(self.data[idx]) / 1e6

            if self.equal_pos_neg is True:
                list_keys_unpack = ['pos_idx', 'neg_idx']
            else:
                list_keys_unpack = []
            data_ = unpack_numpy_data(data_, list_additional_keys=list_keys_unpack)
        elif self.store_data_in_memory is True:
            # if storing in memory, then just get the data
            data_ = self.data[idx]
        else:
            raise ValueError('store_data_in_memory must be True or False')

        if self.subsample is not None:
            if self.equal_pos_neg is True:
                tic_rand_sample = time.time()
                samples_per_sign = int(self.subsample/2)
                
                # idx_pos = data_['pos_idx'].repeat(data_['pos_idx'].size(0)//samples_per_sign + 1)
                # perm_pos = torch.randperm(idx_pos.size(0))
                if isinstance(data_['pos_idx'], list):
                    perm_pos = torch.randperm(data_['pos_idx'][0].size(0))[:samples_per_sign]
                    idx_pos = data_['pos_idx'][0][perm_pos]
                elif isinstance(data_['pos_idx'], torch.Tensor):
                    perm_pos = torch.randperm(data_['pos_idx'].size(0))[:samples_per_sign]
                    idx_pos = data_['pos_idx'][perm_pos]
                else:
                    raise ValueError('pos_idx must be a list or tensor')

                # idx_neg = data_['neg_idx'].repeat(data_['neg_idx'].size(0)//samples_per_sign + 1)
                # perm_neg = torch.randperm(idx_neg.size(0))
                # idx_neg = perm_neg[:samples_per_sign]
                if isinstance(data_['neg_idx'], list):
                    perm_neg = torch.randperm(data_['neg_idx'][0].size(0))[:samples_per_sign]
                    idx_neg = data_['neg_idx'][0][perm_neg]
                elif isinstance(data_['neg_idx'], torch.Tensor):
                    perm_neg = torch.randperm(data_['neg_idx'].size(0))[:samples_per_sign]
                    idx_neg = data_['neg_idx'][perm_neg]
                else:
                    raise ValueError('neg_idx must be a list or tensor')
                toc_rand_sample = time.time()
                if self.verbose is True:
                    print(f'rand sample time: {toc_rand_sample - tic_rand_sample}s')

                tic_cat = time.time()
                idx_ = torch.cat((idx_pos, idx_neg), dim=0)
                toc_cat = time.time()
                if self.verbose is True:
                    print(f'concat time: {toc_cat - tic_cat}s')

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    tic_rand = time.time()
                    perm = torch.randperm(data_['xyz'].size(0))
                    _idx_ = perm[:self.subsample-len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
                    toc_rand = time.time()
                    if self.verbose is True:
                        print(f'rand additional sub sample time: {toc_rand - tic_rand}s')
            
            else:
                perm = torch.randperm(data_['xyz'].size(0))
                idx_ = perm[:self.subsample]

            if self.verbose is True:
                print('idx_ size:', idx_.size(), 'idx_ min:', idx_.min(),'idx_ max:', idx_.max())
                print('equal neg pos', self.equal_pos_neg)
            
            # unpack the data
            xyz = data_['xyz'][idx_, :]
            sdf = data_['gt_sdf'][idx_]

            if (self.max_radius is not None) and (self.center is not None):
                # if normalizing at the group level, then normalize here.
                tic_norm = time.time()
                xyz = (xyz - self.center) / self.max_radius
                sdf = sdf / self.max_radius
                toc_norm = time.time()
                if self.verbose is True:
                    print(f'norm time: {toc_norm - tic_norm}s')
                
            data_ = {
                'xyz': xyz,
                'gt_sdf': sdf,
            }
            
            toc_whole_load = time.time()
            time_whole_load = toc_whole_load - tic_whole_load
            
            if (self.test_load_times is True) and (self.store_data_in_memory is False):
                data_['time'] = time_
                data_['size'] = size
                data_['mb_per_sec'] = size / time_
                data_['whole_load_time'] = time_whole_load
        
        return data_, idx
    
class MultiSurfaceSDFSamples(SDFSamples):
    def __init__(
        self,
        list_mesh_paths,
        subsample=None,
        n_pts=500000,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function='normal', 
        center_pts=True,
        norm_pts=False,
        scale_method='max_rad',
        scale_jointly=False,
        loc_save=os.environ.get('LOC_SDF_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'nsm_sdf_cache')),
        include_seed_in_hash=True,
        save_cache=True,
        load_cache=True,
        random_seed=None,
        reference_mesh=None,    
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
        print_filename=False,
        test_load_times=True,
        uniform_pts_buffer=0.0,

        # Multi surface specific 
        scale_all_meshes=True,                  
        center_all_meshes=False,                
        mesh_to_scale=0,
        reference_object=0,
        store_data_in_memory=False,
        multiprocessing=True,     
        debug_memory=False,
        n_processes=2,
        
    ):
        # if n_pts is not a list, then make it a list that is 
        # the same length as the number of meshes.
        if not isinstance(n_pts, (list, tuple)):
            n_pts = [n_pts] * len(list_mesh_paths[0])
        if len(n_pts) == 1 and len(list_mesh_paths[0]) > 1:
            n_pts = n_pts * len(list_mesh_paths[0])

        self.times = []
        self.data_size = []
        self.mb_per_sec = []
        self.test_load_times = test_load_times
        # Multi surface specific
        self.mesh_to_scale = mesh_to_scale
        self.total_n_pts = sum(n_pts)
        self.scale_all_meshes = scale_all_meshes
        self.center_all_meshes = center_all_meshes
        self.n_meshes = len(list_mesh_paths[0])
        self.reference_object = reference_object

        super().__init__(
            list_mesh_paths=list_mesh_paths,
            subsample=subsample,
            n_pts=n_pts,
            p_near_surface=p_near_surface,
            p_further_from_surface=p_further_from_surface,
            sigma_near=sigma_near,
            sigma_far=sigma_far,
            rand_function=rand_function,
            center_pts=center_pts,
            norm_pts=norm_pts,
            scale_method=scale_method,
            scale_jointly=scale_jointly,
            loc_save=loc_save,
            include_seed_in_hash=include_seed_in_hash,
            save_cache=save_cache,
            load_cache=load_cache,
            random_seed=random_seed,
            reference_mesh=reference_mesh,
            verbose=verbose,
            equal_pos_neg=equal_pos_neg,
            fix_mesh=fix_mesh,
            print_filename=print_filename,
            store_data_in_memory=store_data_in_memory,
            multiprocessing=multiprocessing,
            n_processes=n_processes,
            debug_memory=debug_memory,
            test_load_times=test_load_times,
            uniform_pts_buffer=uniform_pts_buffer,
        )
    
    def preprocess_inputs(self):
        super().preprocess_inputs()

        if isinstance(self.list_mesh_paths[0], (list, tuple)):
            self.n_meshes = len(self.list_mesh_paths[0])
        elif isinstance(self.list_mesh_paths[0], (str, Mesh)):
            self.n_meshes = len(self.list_mesh_paths)
        
        if not isinstance(self.p_near_surface, (list, int)):
            self.p_near_surface = [self.p_near_surface] * self.n_meshes
        if not isinstance(self.p_further_from_surface, (list, int)):
            self.p_further_from_surface = [self.p_further_from_surface] * self.n_meshes
        if not isinstance(self.sigma_near, (list, int)):
            self.sigma_near = [self.sigma_near] * self.n_meshes
        if not isinstance(self.sigma_far, (list, int)):
            self.sigma_far = [self.sigma_far] * self.n_meshes
        if not isinstance(self.n_pts, (list, int)):
            self.n_pts = [self.n_pts] * self.n_meshes
    
    def run_before_loading_data(self):
        self.get_samples_per_sign()
    
    def test_if_idx_in_range(self, data):
        n_pts = data['xyz'].shape[0]
        
        for name in ['pos_idx', 'neg_idx']:
            indices = data[name]
            max_idx = 0
            for tensor in indices:
                max_idx = torch.max(tensor)
                if max_idx >= n_pts:
                    return False
        
        return True
                
    def get_sample_data_dict(self, loc_meshes):
        #with open(os.path.expanduser('~/test.txt'), 'a') as f:
            # Use the print function with the `file` argument to write to the file.
            # print(f'inside get_sample_data_dict, affinity: {os.sched_getaffinity(0)}', file=f)        
        
        if type(loc_meshes) not in (tuple, list):
            loc_meshes = [loc_meshes]
        
        with open(os.path.join(self.loc_save,'list_meshes_started_loading.log'), 'a') as f:
            f.write(str(loc_meshes) + '\n')
        
        # get the number of points to sample per mesh / sign(in/out or pos/neg)
        self.get_samples_per_sign()
        
        # Create hash and filename 
        file_hash = self.create_hash(loc_meshes)
        cached_file = self.find_hash(filename=f'{file_hash}.npz')

        file_loaded = False

        if (len(cached_file) > 0) and (self.load_cache is True):
            if self.verbose is True:
                print('Loading cached file')
            for cache_path in cached_file:
                if not is_zipfile(cache_path):
                    print('DELETEING BAD ZIP FILE:', cache_path)
                    os.remove(cache_path)
                    continue
                
                try:
                    data = np.load(cache_path)
                    data = unpack_numpy_data(data)
                except zipfile.BadZipFile:
                    print('DELETEING BAD ZIP FILE:', cache_path)
                    os.remove(cache_path)
                    continue

                # if previous pre-processing not yet done, do it now 
                # and update/resave the data to cache. 
                resave_data = False

                data, in_in = self.remove_overlapping_points(data)

                if in_in > 0:
                    resave_data = True

                if (
                        ('pos_idx' not in data) or (len(data['pos_idx']) != self.n_meshes) 
                        or ('neg_idx' not in data) or (len(data['neg_idx']) != self.n_meshes)
                        or ('surf_idx' not in data) or (len(data['surf_idx']) != self.n_meshes)
                    ):
                    print('getting pos/neg')
                    pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
                    data['pos_idx'] = pos_idx
                    data['neg_idx'] = neg_idx
                    data['surf_idx'] = surf_idx

                    resave_data = True
                
                if self.test_if_idx_in_range(data) is False:
                    print('Indices out of range!', cache_path)
                    print('\tDeleting file...')
                    os.remove(cache_path)
                    break

                if resave_data is True:
                    self.save_data_to_cache(data, file_hash, filepath=cache_path) #resave data to cache - overwriting original. 
                
                file_loaded = True
                #TODO: crat
                break
            
        if file_loaded is False:
            # otherwise, load the mesh and create SDF samples. 
            print('Creating SDF Samples')
            if self.print_filename is True:
                print(loc_meshes)

            data = {
                'xyz': torch.zeros((sum(self.n_pts), 3)),
                'gt_sdf': torch.zeros((sum(self.n_pts), len(loc_meshes))),
            }
            pts_idx = 0
            icp_transform = None

            if self.multiprocessing is True:
                if self.reference_mesh_path is not None:
                    reference_mesh = Mesh(self.reference_mesh_path)
                else:
                    reference_mesh = None
            else:
                reference_mesh = self.reference_mesh
            
            if self.verbose is True:
                print('type of reference mesh:', type(reference_mesh))
                print('ref mesh path:', self.reference_mesh_path)

            for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
                tic = time.time()
                result_ = read_meshes_get_sampled_pts(
                    loc_meshes, 
                    mean=[0,0,0], 
                    sigma=sigma_, 
                    n_pts=n_pts_, 
                    rand_function=self.rand_function, 
                    center_pts=self.center_pts,
                    norm_pts=self.norm_pts,
                    scale_method=self.scale_method,
                    get_random=True,
                    fix_mesh=self.fix_mesh,
                    register_to_mean_first=False if reference_mesh is None else True,  #
                    mean_mesh=reference_mesh,  #
                    uniform_pts_buffer=self.uniform_pts_buffer,
                    
                    # Multi surface specific
                    mesh_to_scale=self.mesh_to_scale,
                    scale_all_meshes=self.scale_all_meshes,
                    center_all_meshes=self.center_all_meshes,

                    icp_transform=icp_transform,
                )

                if result_ is None:
                    return None
                
                icp_transform = result_['icp_transform']

                toc = time.time()
                print(f'{idx_} - {sigma_}: {toc - tic}s')
                
                if idx_ == 0:
                    data['orig_pts'] = result_['orig_pts']
                    data['new_pts'] = result_['new_pts']
                
                xyz_ = result_['pts'] if 'pts' in result_ else result_['xyz']
                sdfs_ = result_['sdf'] if 'sdf' in result_ else result_['gt_sdf']

                data['xyz'][pts_idx:pts_idx + sum(n_pts_), :] = torch.from_numpy(xyz_).float()

                for mesh_idx, _sdfs_ in enumerate(sdfs_):
                    if _sdfs_ is None:
                        # If mesh was None, fill with NaN to indicate missing data
                        # don't need this now.. but can handle training on incomplete data in the future. 
                        data['gt_sdf'][pts_idx:pts_idx + sum(n_pts_), mesh_idx] = float('nan')
                    else:
                        data['gt_sdf'][pts_idx:pts_idx + sum(n_pts_), mesh_idx] = torch.from_numpy(_sdfs_).float()
                pts_idx += sum(n_pts_)
            
            # Drop points that have are labeled as being inside
            # 2 objects - clearly this is an error.
            data, in_in = self.remove_overlapping_points(data)

            print('getting pos/neg')
            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data['pos_idx'] = pos_idx
            data['neg_idx'] = neg_idx
            data['surf_idx'] = surf_idx
            
            if (data is not None) and (self.save_cache is True):
                self.save_data_to_cache(data, file_hash)
                cache_path = os.path.join(self.cache_folder, f'{file_hash}.npz')
        
        if self.store_data_in_memory is False:
            if self.verbose is True:
                print('updating data to be cache path')
            # change the data to be the path to the saved cache file
            data = cache_path
            
        return data

    def get_samples_per_sign(self):
        samples_per_mesh = [int((n_pts_/self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts]

        # setup samples per sign 
        self.samples_per_sign_ = []
        for subsample_ in samples_per_mesh:
            samples_per_sign = int(subsample_/2)
            if self.verbose is True:
                print(samples_per_sign)
            self.samples_per_sign_.append(samples_per_sign)

    def remove_overlapping_points(self, data):
        sdf_ = data['gt_sdf'].clone()
        
        # Check if we have None values (represented as NaN)
        non_none_mask = ~torch.isnan(sdf_).all(dim=0)
        
        if non_none_mask.sum() < 2:
            return data, 0  # Can't have overlaps with fewer than 2 surfaces

        # Only process non-None columns for overlap detection
        sdf_filtered = sdf_[:, non_none_mask]
        
        sdf_filtered[sdf_filtered < 0] = -1
        sdf_filtered[sdf_filtered > 0] = 1
        sdf_filtered[sdf_filtered == 0] = 0
        total = torch.sum(sdf_filtered, axis=1)

        out_out = torch.sum(total == 2)
        out_in = torch.sum(total == 0)
        in_in = torch.sum(total == -2)

        # Create mask for points to keep (not overlapping)
        keep_mask = total != -2
        
        # Apply the mask to remove overlapping points from the full dataset
        # This preserves the None columns while removing problematic points
        data['gt_sdf'] = data['gt_sdf'][keep_mask, :]
        data['xyz'] = data['xyz'][keep_mask, :]

        if self.verbose is True:
            print('total shape', total.shape)
            print('total', total)
            print('out_out', out_out)
            print('out_in', out_in)
            print('in_in', in_in)
            print(f'Removed {in_in} overlapping points')
        
        return data, in_in

    def get_pt_sample_combos(self):
        n_p_near_surface = [int(n_pts_ * p_near) for n_pts_, p_near in zip(self.n_pts, self.p_near_surface)]
        n_p_further_from_surface = [int(n_pts_ * p_far) for n_pts_, p_far in zip(self.n_pts, self.p_further_from_surface)]
        n_p_random = [n_pts_ - n_p_near - n_p_far for n_pts_, n_p_near, n_p_far in zip(self.n_pts, n_p_near_surface, n_p_further_from_surface)]

        pt_sample_combos = [
            [n_p_near_surface, self.sigma_near],
            [n_p_further_from_surface, self.sigma_far],
            [n_p_random, [None,] * self.n_meshes]
        ]

        return pt_sample_combos

    def get_hash_params(self):
        list_hash_params = [
            self.center_pts,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.scale_all_meshes,
            self.center_all_meshes,
            self.reference_mesh,
            self.reference_object,
            False,
            self.fix_mesh,
            self.scale_jointly
        ]

        for n_pts_ in self.n_pts:
            list_hash_params.append(n_pts_)
        for p_near in self.p_near_surface:
            list_hash_params.append(p_near)
        for p_far in self.p_further_from_surface:
            list_hash_params.append(p_far)
        for sigma_near in self.sigma_near:
            list_hash_params.append(sigma_near)
        for sigma_far in self.sigma_far:
            list_hash_params.append(sigma_far)
        
        return list_hash_params


    def sdf_pos_neg_idx(self, data):
        '''
        - iterate over each mesh
        - get number of points for that mesh and get: 
            - points positive (outside mesh)
            - points negative (inside mesh)
        - return list of indices
        '''

        pos_idx = []
        neg_idx = []
        surf_idx = []
        pts_idx_ = 0
        if self.verbose is True:
            print('data', data['xyz'].shape, data['gt_sdf'].shape)
        
        for mesh_idx in range(self.n_meshes):
            
            samples_per_sign = self.samples_per_sign_[mesh_idx]

            # BELOW NEEDS LOGIC TO UNPACK  1/2 pos/neg pts for each mesh
            # mesh_sdfs = data['gt_sdf'][pts_idx_:pts_idx_ + n_pts_, mesh_idx]
            mesh_sdfs = data['gt_sdf'][:, mesh_idx].clone()
            pos_idx_ = (mesh_sdfs > 0).nonzero(as_tuple=True)[0] #+ pts_idx_
            neg_idx_ = (mesh_sdfs < 0).nonzero(as_tuple=True)[0] #+ pts_idx_
            surf_idx_ = (mesh_sdfs == 0).nonzero(as_tuple=True)[0] #+ pts_idx_

            # Repeat +/- indices if either of them does not have enough for a full batch. 
            pos_idx_ = pos_idx_.repeat(samples_per_sign//pos_idx_.size(0) + 1)
            neg_idx_ = neg_idx_.repeat(samples_per_sign//neg_idx_.size(0) + 1)

            pos_idx.append(pos_idx_)
            neg_idx.append(neg_idx_)
            surf_idx.append(surf_idx_)
                
        return pos_idx, neg_idx, surf_idx
    
    def __getitem__(self, idx):
        #TODO: get rid of pts_array from above
        # replace with self.data[idx] = [pts, sdfs_]
        # this will simplify everything downstream because this is something that we are
        # constantly undoing/re-doing elsewhere in the code - and it even lookts like
        # the sdfs/pts are stroed separately in the npy files. 

        tic_whole_load = time.time()
        if self.store_data_in_memory is False:
            # if not storing in memory, then load from cache

            # if self.test_load_times is True:
            tic = time.time()
            data_ = np.load(self.data[idx])
            toc = time.time()
            time_ = toc - tic
            # self.times.append(time_)
    
            # get size of the numpy file in mb
            size = os.path.getsize(self.data[idx]) / 1e6
            # self.sizes.append(size)

            # self.mb_per_sec.append(size / time_)

            if self.verbose is True:
                print(f'size: {size}mb, time: {time_}s, mb/s: {size / time_}mb/s')

            if self.equal_pos_neg is True:
                list_keys_unpack = ['pos_idx', 'neg_idx']
            else:
                list_keys_unpack = []
            tic_unpack = time.time()
            data_ = unpack_numpy_data(data_, list_additional_keys=list_keys_unpack)
            toc_unpack = time.time()
            if self.verbose is True:
                print(f'unpack time: {toc_unpack - tic_unpack}s')

        elif self.store_data_in_memory is True:
            # if storing in memory, then just get the data
            data_ = self.data[idx]
        else:
            raise ValueError('store_data_in_memory must be True or False')

        if self.subsample is not None:
            if self.equal_pos_neg is True:
                # get number of points for each mesh
                # this is weighted by the number of points in the mesh 
                # relative to the total number of points in the dataset
                # samples_per_mesh = [int((n_pts_/self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts]
                idx_ = []
                for mesh_idx, samples_per_sign in enumerate(self.samples_per_sign_):
                    tic_mesh = time.time()
                    # get number of positive and negative points for this mesh
                    # samples_per_sign = int(subsample_/2)
                    if self.verbose is True:
                        print('samples_per_sign', samples_per_sign)
                        print('mesh idx', mesh_idx)
                        print('data_ pos', data_['pos_idx'])
                    
                    if samples_per_sign == 0:
                        continue

                    # get random indices for positive and negative points
                    # idx_pos = data_['pos_idx'][mesh_idx].repeat(data_['pos_idx'][mesh_idx].size(0)//samples_per_sign + 1)
                    perm_pos = torch.randperm(data_['pos_idx'][mesh_idx].size(0))[:samples_per_sign]
                    idx_pos = data_['pos_idx'][mesh_idx][perm_pos]

                    # idx_neg = data_['neg_idx'][mesh_idx].repeat(data_['neg_idx'][mesh_idx].size(0)//samples_per_sign + 1)
                    perm_neg = torch.randperm(data_['neg_idx'][mesh_idx].size(0))[:samples_per_sign]
                    idx_neg = data_['neg_idx'][mesh_idx][perm_neg]

                    # combine positive and negative indices
                    idx_ += [idx_pos, idx_neg]
                    toc_mesh = time.time()
                    if self.verbose is True:
                        print(f'mesh {mesh_idx} time: {toc_mesh - tic_mesh}s')
                
                tic_cat = time.time()
                # combine indices for all meshes
                idx_ = torch.cat(idx_, dim=0)
                toc_cat = time.time()
                if self.verbose is True:
                    print(f'cat time: {toc_cat - tic_cat}s')

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    tic_rand = time.time()
                    perm = torch.randperm(data_['xyz'].size(0))
                    _idx_ = perm[:self.subsample-len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
                    toc_rand = time.time()
                    if self.verbose is True:
                        print(f'rand additional sub sample time: {toc_rand - tic_rand}s')
                
            else:
                perm = torch.randperm(data_['xyz'].size(0))
                idx_ = perm[: self.subsample]
            
            if self.verbose is True:
                print('idx_ size:', idx_.size(), 'idx_ min:', idx_.min(),'idx_ max:', idx_.max())
                print('equal neg pos', self.equal_pos_neg)
            
            xyz = data_['xyz'][idx_, :]
            sdf = data_['gt_sdf'][idx_, :]

            if (self.max_radius is not None) and (self.center is not None):
                tic_scaling = time.time()
                xyz = (xyz - self.center) / self.max_radius
                sdf = sdf / self.max_radius
                toc_scaling = time.time()
                if self.verbose is True:
                    print(f'scaling time: {toc_scaling - tic_scaling}s')

            data_ = {
                'xyz': xyz,
                'gt_sdf': sdf,
            }

            toc_whole_load = time.time()

            if self.test_load_times is True:
                data_['time'] = time_
                data_['size'] = size
                data_['mb_per_sec'] = size / time_
                data_['whole_load_time'] = toc_whole_load - tic_whole_load
        
        return data_, idx
