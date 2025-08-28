import logging
import numpy as np

try:
    from NSM.dependencies import sinkhorn

    __emd__ = True
except:
    print("Error importing `sinkhorn` from NSM.dependencies")
    __emd__ = False

from .utils import compute_chamfer  # , compute_assd

# Set up logger for this module
logger = logging.getLogger(__name__)


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
    logger.info("Starting reconstruction loss computation")
    logger.debug(f"Computing loss for {len(meshes) if isinstance(meshes, list) else 1} meshes")
    logger.debug(f"Loss calculation settings: chamfer={calc_symmetric_chamfer}, assd={calc_assd}, emd={calc_emd}")

    result = {}

    if not isinstance(meshes, list):
        meshes = [meshes]
    if not isinstance(orig_meshes, list):
        orig_meshes = [orig_meshes]

    assert len(meshes) == len(
        orig_meshes
    ), "Number of meshes and number of original points must be equal"
    
    logger.debug(f"Processing {len(meshes)} mesh pairs")

    for mesh_idx, mesh in enumerate(meshes):
        logger.debug(f"Processing mesh {mesh_idx + 1}/{len(meshes)}")
        
        if mesh is not None:
            pts_recon_ = mesh.point_coords
            logger.debug(f"Mesh {mesh_idx}: {len(pts_recon_)} reconstructed points")
        else:
            pts_recon_ = None
            logger.warning(f"Mesh {mesh_idx}: No reconstructed mesh provided (None)")

        xyz_orig_ = orig_meshes[mesh_idx].point_coords
        logger.debug(f"Mesh {mesh_idx}: {len(xyz_orig_)} original points")

        if calc_symmetric_chamfer:
            logger.debug(f"Computing Chamfer distance for mesh {mesh_idx}")
            # if __chamfer__ is True:
            if pts_recon_ is None:
                chamfer_loss_ = np.nan
                logger.warning(f"Mesh {mesh_idx}: Chamfer distance set to NaN (no reconstructed mesh)")
            else:
                chamfer_loss_ = compute_chamfer(
                    xyz_orig_, pts_recon_, num_samples=n_samples_chamfer, power=chamfer_norm
                )
                logger.debug(f"Mesh {mesh_idx}: Chamfer distance = {chamfer_loss_:.6f}")
            result[f"chamfer_{mesh_idx}"] = chamfer_loss_
            # elif __chamfer__ is False:
            #     raise ImportError('Cannot calculate symmetric chamfer distance without chamfer_pytorch module')

        if calc_assd:
            logger.debug(f"Computing ASSD for mesh {mesh_idx}")
            if pts_recon_ is None:
                assd_loss_ = np.nan
                logger.warning(f"Mesh {mesh_idx}: ASSD set to NaN (no reconstructed mesh)")
            else:
                assd_loss_ = mesh.get_assd_mesh(orig_meshes[mesh_idx])
                logger.debug(f"Mesh {mesh_idx}: ASSD = {assd_loss_:.6f}")
                #     xyz_orig_,
                #     pts_recon_,
                #     num_samples=n_samples_assd,
                # )
            result[f"assd_{mesh_idx}"] = assd_loss_

        if calc_emd:
            logger.debug(f"Computing EMD for mesh {mesh_idx}")
            if __emd__ is True:
                if pts_recon_ is None:
                    emd_loss_ = np.nan
                    logger.warning(f"Mesh {mesh_idx}: EMD set to NaN (no reconstructed mesh)")
                else:
                    emd_loss_, _, _ = sinkhorn(xyz_orig_, pts_recon_)
                    logger.debug(f"Mesh {mesh_idx}: EMD = {emd_loss_:.6f}")
                result[f"emd_{mesh_idx}"] = emd_loss_
            elif __emd__ is False:
                logger.error("Cannot calculate EMD: sinkhorn module not available")
                raise ImportError("Cannot calculate EMD without emd module")

    logger.info(f"Reconstruction loss computation completed. Computed {len(result)} loss values.")
    logger.debug(f"Result keys: {list(result.keys())}")
    return result
