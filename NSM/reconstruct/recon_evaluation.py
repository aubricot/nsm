import numpy as np

try:
    from NSM.dependencies import sinkhorn

    __emd__ = True
except:
    print("Error importing `sinkhorn` from NSM.dependencies")
    __emd__ = False

from .utils import compute_chamfer  # , compute_assd


def compute_recon_loss(
    meshes,
    # orig_pts,
    orig_meshes,
    n_samples_chamfer=None,
    n_samples_assd=None,
    chamfer_norm=1,
    calc_symmetric_chamfer=False,
    calc_assd=False,
    calc_emd=False,
):
    """
    Computes the reconstruction loss between the predicted meshes and the ground truth meshes.

    Args:
        meshes (list): A list of predicted meshes.
        orig_pts (list): A list of pts from ground truth meshes.
        n_samples_chamfer (int, optional): The number of samples to use for the chamfer distance calculation. Defaults to None.
        chamfer_norm (int, optional): The power to which the chamfer distance is raised. Defaults to 1.
        calc_symmetric_chamfer (bool, optional): Whether to calculate the symmetric chamfer distance. Defaults to False.
        calc_emd (bool, optional): Whether to calculate the earth mover's distance. Defaults to False.

    Returns:
        dict: A dictionary containing the reconstruction loss for each mesh.
    """

    result = {}

    if not isinstance(meshes, list):
        meshes = [meshes]
    if not isinstance(orig_meshes, list):
        orig_meshes = [orig_meshes]

    assert len(meshes) == len(
        orig_meshes
    ), "Number of meshes and number of original points must be equal"

    for mesh_idx, mesh in enumerate(meshes):
        if mesh is not None:
            pts_recon_ = mesh.point_coords
        else:
            pts_recon_ = None

        xyz_orig_ = orig_meshes[mesh_idx].point_coords

        if calc_symmetric_chamfer:
            # if __chamfer__ is True:
            if pts_recon_ is None:
                chamfer_loss_ = np.nan
            else:
                chamfer_loss_ = compute_chamfer(
                    xyz_orig_, pts_recon_, num_samples=n_samples_chamfer, power=chamfer_norm
                )
            result[f"chamfer_{mesh_idx}"] = chamfer_loss_
            # elif __chamfer__ is False:
            #     raise ImportError('Cannot calculate symmetric chamfer distance without chamfer_pytorch module')

        if calc_assd:
            if pts_recon_ is None:
                assd_loss_ = np.nan
            else:
                assd_loss_ = mesh.get_assd_mesh(orig_meshes[mesh_idx])
                #     xyz_orig_,
                #     pts_recon_,
                #     num_samples=n_samples_assd,
                # )
            result[f"assd_{mesh_idx}"] = assd_loss_

        if calc_emd:
            if __emd__ is True:
                if pts_recon_ is None:
                    emd_loss_ = np.nan
                else:
                    emd_loss_, _, _ = sinkhorn(xyz_orig_, pts_recon_)
                result[f"emd_{mesh_idx}"] = emd_loss_
            elif __emd__ is False:
                raise ImportError("Cannot calculate EMD without emd module")

    return result
