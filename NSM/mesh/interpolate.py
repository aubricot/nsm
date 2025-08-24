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
    if "cell_idx" not in mesh.scalar_names:
        n_cells = mesh.mesh.GetNumberOfCells()
        cells = np.arange(n_cells)
        cells_ = numpy_to_vtk(cells)
        cells_.SetName("cell_idx")
        mesh.mesh.GetCellData().AddArray(cells_)


def sdf_gradients(sdf_model, points, latent, surface_idx=None, verbose=False):
    """
    Computes gradients of SDF with respect to 3D positions (not latent).
    If surface_idx is provided, computes only that surface's gradient (fastest).
    Otherwise returns gradients for all surfaces.

    If the points are on the surface of the specific latent, then the gradients
    are equivalent to the normal vectors of the surface. If they are not on the
    surface, then they are the gradient of the SDF at that point and indicate
    the direction of the steepest ascent.

    Args:
    - sdf_model (nn.Module): The model that computes the SDF
    - points (np.ndarray or torch.tensor): The points for which to compute gradients (B, 3)
    - latent (np.ndarray or torch.tensor): The latent vector for the specific shape
    - surface_idx (int, optional): If provided, only compute gradients for this surface (0-based)
    - verbose (bool): If True, print the GPU memory usage after gradient computation

    Returns:
    - gradients (torch.Tensor):
        - If surface_idx provided: gradients for that surface only (B, latent_dim + 3)
        - If surface_idx is None: list of gradients for each surface
    - sdf_values (torch.Tensor): The SDF values for each point (B, num_surfaces)
    """
    # Convert to tensors
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    if isinstance(latent, np.ndarray):
        latent = torch.from_numpy(latent)

    # Get device and dtype from model
    device = next(sdf_model.parameters()).device
    dtype = next(sdf_model.parameters()).dtype
    points = points.to(device=device, dtype=dtype)
    latent = latent.to(device=device, dtype=dtype)

    B = points.shape[0]
    D_lat = latent.shape[-1]
    assert points.shape[-1] == 3, "points must be (B, 3)"

    # Handle latent vector shape
    if latent.ndim == 1:
        latent = latent.unsqueeze(0)  # (1, D_lat)
    if latent.shape[0] == 1:
        latent = latent.expand(B, -1)  # (B, D_lat)

    # Only positions need gradients (more efficient than full input)
    pos = points.detach().requires_grad_(True)  # (B, 3)
    vecs = latent.detach()  # (B, D_lat) no grad needed

    # Concatenate for model input
    p = torch.cat([vecs, pos], dim=1)  # (B, D_lat + 3)

    # Set model to eval mode for stability during gradient computation
    was_training = sdf_model.training
    sdf_model.eval()

    # Forward pass
    sdf_values = sdf_model(p)  # (B, Ns)
    assert_finite(sdf_values, "SDF values")

    def _finish(g):
        if verbose:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print_gpu_memory()
        return g.detach().cpu(), sdf_values.detach().cpu()

    # Fast path: single surface only
    if surface_idx is not None:
        y = sdf_values[:, surface_idx]  # (B,)
        # Use sum() trick - equivalent to one-hot grad_outputs but more efficient
        (grad_pos,) = torch.autograd.grad(
            y.sum(), pos, create_graph=False, retain_graph=False, allow_unused=False
        )
        sdf_model.train(was_training)
        assert_finite(grad_pos, f"Gradients for surface {surface_idx}")

        # Reconstruct full gradient (latent + position) for backward compatibility
        grad_latent_zeros = torch.zeros(B, D_lat, device=device, dtype=dtype)
        full_grad = torch.cat([grad_latent_zeros, grad_pos], dim=1)
        return _finish(full_grad)  # (B, D_lat + 3), (B, Ns)

    # All surfaces (for backward compatibility)
    Ns = sdf_values.shape[1]
    gradients = []

    for i in range(Ns):
        y = sdf_values[:, i]
        (grad_pos,) = torch.autograd.grad(
            y.sum(), pos, create_graph=False, retain_graph=(i < Ns - 1)
        )
        assert_finite(grad_pos, f"Gradients for surface {i}")

        # Reconstruct full gradient for backward compatibility
        grad_latent_zeros = torch.zeros(B, D_lat, device=device, dtype=dtype)
        full_grad = torch.cat([grad_latent_zeros, grad_pos], dim=1)
        gradients.append(full_grad.detach().cpu())

    sdf_model.train(was_training)
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
    latent_mag = (1 - step) * latent1_mag + step * latent2_mag

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

    new_latent = ((1 - step) * latent1) + (step * latent2)

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

    # Use optimized single-surface gradient computation
    grads, sdfs = sdf_gradients(
        model, current_points, new_latent, surface_idx=surface_idx, verbose=verbose
    )

    # Extract spatial gradients (last 3 dimensions)
    grads = grads[:, -3:]

    # Safe normalization with eps clamping
    grad_norm = torch.norm(grads, dim=1, keepdim=True)
    zero_mask = grad_norm < EPS
    grad_norm = grad_norm.clamp_min(EPS)  # avoid division by zero
    grads = grads / grad_norm
    grads[zero_mask.squeeze()] = 0.0  # leave flat points unchanged

    assert_finite(grads, "Normalized gradients")

    # Extract SDF values for the specific surface
    sdfs = sdfs[:, surface_idx]
    assert_finite(sdfs, "SDF values")

    points_step = grads * sdfs[:, None]
    assert_finite(points_step, "Point step")

    new_points = current_points.cpu() - points_step
    assert_finite(new_points, "New points")

    return new_points


def interpolate_common(
    model,
    latent1,
    latent2,
    n_steps=100,
    data=None,
    surface_idx=0,
    verbose=False,
    spherical=True,
    is_mesh=False,
    max_edge_len=0.04,
    adaptive=False,
    smooth=True,
    smooth_type="laplacian",
):
    if data is None:
        raise Exception("Not implemented")
        # create function that gets the surface points for latent1 as a starting point.

    if is_mesh:
        if not isinstance(data.mesh, pv.PolyData):
            data.mesh = pv.PolyData(data.mesh)
        add_cell_idx(data)

    device = next(model.parameters()).device  # Get the device from the model

    for idx, step in enumerate(np.linspace(1 / n_steps, 1, n_steps)):
        if verbose is True:
            print(f"{idx+1}/{n_steps}")

        new_latent = (
            slerp_latent(latent1, latent2, step)
            if spherical
            else linear_interp_latent(latent1, latent2, step)
        )

        if is_mesh:
            new_points = torch.tensor(data.point_coords.copy(), dtype=torch.float).to(device)
            new_points = (
                update_positions(
                    model, new_latent, new_points, surface_idx=surface_idx, verbose=verbose
                )
                .detach()
                .cpu()
                .numpy()
            )
            data.point_coords = new_points
            if adaptive:
                data.mesh.subdivide_adaptive(
                    max_edge_len=max_edge_len,
                    max_tri_area=None,
                    max_n_tris=None,
                    max_n_passes=3,
                    inplace=True,
                    progress_bar=False,
                )
            if smooth:
                # meshes should start as well spaced/regular as possible. Then,
                # each step is a small change in the shape, so the mesh should
                # remain well spaced/regular and should allow small n_iter and
                # large relaxation_factor
                if smooth_type == "laplacian":
                    data.mesh.smooth(inplace=True, relaxation_factor=0.01, n_iter=2)
                elif smooth_type == "taubin":
                    data.mesh.smooth_taubin(inplace=True, n_iter=2, pass_band=0.1)
                else:
                    raise Exception(f"Unknown smoothing type: {smooth_type}")
        else:
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float).to(device)
            elif not torch.is_tensor(data):
                raise Exception(f"Unknown data type: {type(data)}")

            data = data.to(device)
            data = update_positions(
                model, new_latent, data, surface_idx=surface_idx, verbose=verbose
            )

    if not is_mesh:
        data = data.detach().cpu().numpy()

    return data


def interpolate_points(
    model, latent1, latent2, n_steps=100, points1=None, surface_idx=0, verbose=False, spherical=True
):
    return interpolate_common(
        model, latent1, latent2, n_steps, points1, surface_idx, verbose, spherical, is_mesh=False
    )


def interpolate_mesh(
    model,
    latent1,
    latent2,
    n_steps=100,
    mesh=None,
    surface_idx=0,
    verbose=False,
    spherical=True,
    max_edge_len=0.04,
    adaptive=False,
    smooth=True,
    smooth_type="laplacian",
):
    return interpolate_common(
        model,
        latent1,
        latent2,
        n_steps,
        mesh,
        surface_idx,
        verbose,
        spherical,
        is_mesh=True,
        max_edge_len=max_edge_len,
        adaptive=adaptive,
        smooth=smooth,
        smooth_type=smooth_type,
    )
