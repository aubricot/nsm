import torch
import numpy as np
import scipy
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk

from NSM.utils import print_gpu_memory

EPS = 1e-8
def assert_finite(tensor, name):
    """Helper function to check for NaN/Inf values"""
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN/Inf")

def add_cell_idx(mesh):
    if 'cell_idx' not in mesh.scalar_names:
        n_cells = mesh.mesh.GetNumberOfCells()
        cells = np.arange(n_cells)
        cells_ = numpy_to_vtk(cells)
        cells_.SetName('cell_idx')
        mesh.mesh.GetCellData().AddArray(cells_)

def sdf_gradients(sdf_model, points, latent, verbose=False):
    '''
    Function that computes gradients for a set of points/vertices. If the 
    points are on the surface of the specific latent, then they are 
    equivalent to the normal vectors of the surface. If they are not
    on the surface, then they are the gradient of the SDF at that point and
    indicate the direction of the steepest ascent.

    Args:
    - sdf_model (nn.Module): The model that computes the SDF
    - points (np.ndarray or torch.tensor): The points for which to compute the gradients
    - latent (np.ndarray or torch.tensor): The latent vector for the specific shape
    - verbose (bool): If True, print the GPU memory usage after each gradient step

    Returns:
    - gradients (list): A list of gradients for each point
    - sdf_values (np.ndarray): The SDF values for each point
    '''
    if isinstance(points, np.ndarray):
        points = torch.tensor(points)

    if isinstance(latent, np.ndarray):
        latent = torch.tensor(latent) 
    elif isinstance(latent, torch.Tensor):
        pass
    else:
        raise Exception(f'unknown data type {type(latent)}')
    
    # Repeat latent vector for each point
    vecs = latent.repeat(points.shape[0], 1)
    # concatenate latent vector with points, and add 
    # to same device as model
    device = next(sdf_model.parameters()).device  # Get the device from the model
    p = torch.cat([vecs, points], axis=1).to(device, dtype=torch.float).detach().requires_grad_(True)

    # Forward pass
    sdf_values = sdf_model(p)
    assert_finite(sdf_values, "SDF values")

    # Initialize a zero gradient tensor
    grad_output = torch.zeros_like(sdf_values)

    # Container for gradients
    gradients = []

    # Loop through each SDF output
    for i in range(sdf_values.shape[1]):
        # Set the gradient for the current SDF to 1
        grad_output[:, i] = 1.0

        # Backward pass for gradient computation
        sdf_values.backward(gradient=grad_output, retain_graph=True)

        # Extract and store the gradient
        grad = p.grad.clone().detach().cpu()
        assert_finite(grad, f"Gradients for surface {i}")
        gradients.append(grad)

        # Reset the gradients of input and grad_output for the next loop
        p.grad.zero_()
        grad_output[:, i] = 0.0
        if verbose is True:
            print_gpu_memory()

    return gradients, sdf_values.detach().cpu()


def slerp_latent(latent1, latent2, step):
    """
    Spherical linear interpolation of two latent vectors

    Args:
    - latent1 (np.ndarray): The first latent vector
    - latent2 (np.ndarray): The second latent vector
    - step (float): The interpolation step

    Returns:
    - new_latent (np.ndarray): The new latent vector
    """
    assert (step > 0) and (step <= 1)
    
    latent1_mag = np.linalg.norm(latent1)
    latent2_mag = np.linalg.norm(latent2)
    
    # Protect against zero magnitude latents
    if latent1_mag < EPS or latent2_mag < EPS:
        # Fall back to linear interpolation if either vector has near-zero magnitude
        return linear_interp_latent(latent1, latent2, step)

    latent1_norm = latent1 / latent1_mag
    latent2_norm = latent2 / latent2_mag
    
    latent_norm = scipy.spatial.geometric_slerp(latent1_norm, latent2_norm, step)
    latent_mag = (1-step) * latent1_mag + step * latent2_mag
    
    new_latent = latent_norm * latent_mag
    
    return new_latent

def linear_interp_latent(latent1, latent2, step):
    """
    Linear interpolation of two latent vectors

    Args:
    - latent1 (np.ndarray): The first latent vector
    - latent2 (np.ndarray): The second latent vector
    - step (float): The interpolation step

    Returns:
    - new_latent (np.ndarray): The new latent vector
    """
    assert (step > 0) and (step <= 1)
    
    new_latent = ((1-step) * latent1) + (step * latent2)
    
    return new_latent

def update_positions(model, new_latent, current_points, surface_idx=0, verbose=True):
    """
    Function that updates the positions of a set of points based on the
    gradients of the SDF at those points. Assume that the points are on the
    old/original surface and the new latent vector is the new shape. Therefore, 
    the points are moved in the direction of the steepest descent of the SDF.

    Args:
    - model (nn.Module): The model that computes the SDF
    - new_latent (np.ndarray or torch.tensor): The new latent vector
    - current_points (np.ndarray or torch.tensor): The current points
    - surface_idx (int): The index of the surface in the SDF output (if SDF has multiple surfaces)
    - verbose (bool): If True, print the GPU memory usage after each gradient step

    Returns:
    - new_points (np.ndarray): The new points

    """
    # Ensure both new_latent and current_points are tensors and on the same device as model
    device = next(model.parameters()).device  # Get the device from the model
    
    if not torch.is_tensor(new_latent):
        new_latent = torch.tensor(new_latent).to(device)
    else:
        new_latent = new_latent.to(device)

    if not torch.is_tensor(current_points):
        current_points = torch.tensor(current_points).to(device)
    else:
        current_points = current_points.to(device)
        
    grads, sdfs = sdf_gradients(model, current_points, new_latent, verbose=verbose)

    grads = grads[surface_idx][:,-3:]
    
    # Safe normalization with eps clamping
    grad_norm = torch.norm(grads, dim=1, keepdim=True)
    zero_mask = grad_norm < EPS
    grad_norm = grad_norm.clamp_min(EPS)  # avoid division by zero
    grads = grads / grad_norm
    grads[zero_mask.squeeze()] = 0.0  # leave flat points unchanged
    
    assert_finite(grads, "Normalized gradients")
    
    sdfs = sdfs[:, surface_idx]
    assert_finite(sdfs, "SDF values")
    
    points_step = grads * sdfs[:,None]
    assert_finite(points_step, "Point step")
    
    new_points = current_points.cpu() - points_step
    assert_finite(new_points, "New points")
    
    return new_points


def interpolate_common(model, latent1, latent2, n_steps=100, data=None, surface_idx=0, verbose=False, spherical=True, is_mesh=False, max_edge_len=0.04, adaptive=False, smooth=True, smooth_type='laplacian'):
    if data is None:
        raise Exception('Not implemented')
        # create function that gets the surface points for latent1 as a starting point. 

    if is_mesh:
        if not isinstance(data.mesh, pv.PolyData):
            data.mesh = pv.PolyData(data.mesh)
        add_cell_idx(data)   

    device = next(model.parameters()).device  # Get the device from the model

    for idx, step in enumerate(np.linspace(1/n_steps, 1, n_steps)):
        if verbose is True:
            print(f'{idx+1}/{n_steps}')
        
        new_latent = slerp_latent(latent1, latent2, step) if spherical else linear_interp_latent(latent1, latent2, step)

        if is_mesh:
            new_points = torch.tensor(data.point_coords.copy(), dtype=torch.float).to(device)
            new_points = update_positions(model, new_latent, new_points, surface_idx=surface_idx, verbose=verbose).detach().cpu().numpy()
            data.point_coords = new_points
            if adaptive:
                data.mesh.subdivide_adaptive(
                    max_edge_len=max_edge_len, 
                    max_tri_area=None, 
                    max_n_tris=None, 
                    max_n_passes=3, 
                    inplace=True, 
                    progress_bar=False
                )
            if smooth:
                # meshes should start as well spaced/regular as possible. Then, 
                # each step is a small change in the shape, so the mesh should
                # remain well spaced/regular and should allow small n_iter and
                # large relaxation_factor
                if smooth_type == 'laplacian':
                    data.mesh.smooth(inplace=True, relaxation_factor=0.01, n_iter=2)
                elif smooth_type == 'taubin':
                    data.mesh.smooth_taubin(inplace=True, n_iter=2, pass_band=0.1)
                else:
                    raise Exception(f'Unknown smoothing type: {smooth_type}')
        else:
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float).to(device)
            elif not torch.is_tensor(data):
                raise Exception(f'Unknown data type: {type(data)}')
      
            data = data.to(device)
            data = update_positions(model, new_latent, data, surface_idx=surface_idx, verbose=verbose)
    
    if not is_mesh:
        data = data.detach().cpu().numpy()
    
    return data



def interpolate_points(model, latent1, latent2, n_steps=100, points1=None, surface_idx=0, verbose=False, spherical=True):
    return interpolate_common(model, latent1, latent2, n_steps, points1, surface_idx, verbose, spherical, is_mesh=False)

def interpolate_mesh(model, latent1, latent2, n_steps=100, mesh=None, surface_idx=0, verbose=False, spherical=True, max_edge_len=0.04, adaptive=False, smooth=True, smooth_type='laplacian'):
    return interpolate_common(model, latent1, latent2, n_steps, mesh, surface_idx, verbose, spherical, is_mesh=True, max_edge_len=max_edge_len, adaptive=adaptive, smooth=smooth, smooth_type=smooth_type)