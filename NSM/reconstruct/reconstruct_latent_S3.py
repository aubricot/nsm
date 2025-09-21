import torch
import numpy as np
from NSM.datasets import (
    read_mesh_get_sampled_pts,
    get_pts_center_and_scale,
)

from .utils import adjust_learning_rate

# TODO: Add scaling, translation, and rotation to optimization
# DeepSDF x Sim(3)
# Extending DeepSDF for automatic 3D shape retrieval and similarity transform estimation
# https://arxiv.org/abs/2004.09048
#


def get_w(w1, w2, w3):
    return torch.Tensor([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])


def get_axis_angle_rotation_matrix(polar_angle, azimuthal_angle, theta, epsilon=1e-6):
    """ """
    if type(polar_angle) is not torch.Tensor:
        if type(polar_angle) is np.ndarray:
            polar_angle = torch.from_numpy(polar_angle)
        else:
            polar_angle = torch.Tensor([polar_angle])
    if type(azimuthal_angle) is not torch.Tensor:
        if type(azimuthal_angle) is np.ndarray:
            azimuthal_angle = torch.from_numpy(azimuthal_angle)
        else:
            azimuthal_angle = torch.Tensor([azimuthal_angle])
    if type(theta) is not torch.Tensor:
        if type(theta) is np.ndarray:
            theta = torch.from_numpy(theta)
        else:
            theta = torch.Tensor([theta])

    # use spherical coordinates to define the rotation axis
    w1 = torch.sin(polar_angle) * torch.cos(azimuthal_angle)
    w2 = torch.sin(polar_angle) * torch.sin(azimuthal_angle)
    w3 = torch.cos(polar_angle)

    # Normalize - becuase the spherical conversion is not perfect
    norm = torch.sqrt(w1**2 + w2**2 + w3**2)
    w1 = w1 / norm
    w2 = w2 / norm
    w3 = w3 / norm

    w = get_w(w1, w2, w3)
    I = torch.eye(3)
    # See section 5.1 of paper:
    # https://arxiv.org/pdf/2004.09048.pdf
    R = I + w * torch.sin(theta) + w @ w * (1 - torch.cos(theta))
    return R


def reconstruct_latent_S3(
    decoder,
    num_iterations,
    latent_size,
    new_sdf,  # Nx4
    loss_type="l1",
    lr=5e-4,
    loss_weight=1.0,
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    convergence="num_iterations",
    convergence_patience=50,
    log_wandb=False,
    soft_contrain_scale=True,
    scale_constrain_deviation=0.05,
    # scale_constrain_max=200,
    soft_constrain_theta=True,
    theta_constrain_deviation=torch.pi / 36,
    soft_constrain_translation=True,
    translation_constrain_deviation=0.05,
    init_scale_method="max_rad",
    transform_update_patience=500,
    verbose=False,
    device="cuda",
):
    """
    DeepSDF x Sim(3)
    Extending DeepSDF for automatic 3D shape retrieval and similarity transform estimation
    https://arxiv.org/abs/2004.09048
    """

    if convergence != "num_iterations":
        num_iterations = max(num_iterations, 10000)

    adjust_lr_every = num_iterations // n_lr_updates

    # Parse the inputted SDF
    if type(new_sdf) is str:
        # if string, assume path to mesh
        # TODO: Do we want to add option to sample points in/out of mesh?
        #      Will this speed up convergence?
        result_ = read_mesh_get_sampled_pts(
            new_sdf,
            get_random=False,
            center_pts=False,
            norm_pts=False,
            scale_method=None,
            return_orig_mesh=False,
            return_new_mesh=False,
            return_orig_pts=False,
        )
        xyz = result_["pts"]
        sdf_gt = result_["sdf"]
        xyz = torch.from_numpy(xyz).float()
        sdf_gt = torch.from_numpy(sdf_gt).float().unsqueeze(1)
    elif isinstance(new_sdf, (torch.Tensor, np.ndarray)):
        if new_sdf.shape[1] == 3:
            # if
            xyz = new_sdf
            sdf_gt = torch.zeros(xyz.shape[0], 1)
        elif new_sdf.shape[1] == 4:
            xyz = new_sdf[:, :3]
            sdf_gt = new_sdf[:, 3].unsqueeze(1)
        else:
            raise ValueError(f"Inputted SDF must have shape Nx3 or Nx4 got: {new_s}")

    # Sensibly initilize the registration parameters
    init_center, init_scale = get_pts_center_and_scale(
        np.copy(xyz), center=True, scale=True, scale_method=init_scale_method, return_pts=False
    )
    init_center = torch.from_numpy(init_center).float()

    if verbose is True:
        print(f"init_center: {init_center}")
        print(f"init_scale: {init_scale}")

    # constrain the scale to be within a fraction of the initial scale
    scale_constrain = init_scale * scale_constrain_deviation
    # scale_constrain_min = init_scale - (init_scale * scale_constrain_deviation)
    # scale_constrain_max = init_scale + (init_scale * scale_constrain_deviation)

    # constaint for translation is a fraction of the diameter of the object
    # the init_scale = radius, so the diameter is 2 * init_scale
    translation_constrain = 2 * init_scale * translation_constrain_deviation

    # constrain theta to be within a certain range that should be relatively small
    # we should still have all knees pointing in the same direction in space based on MRI alignment
    # therefore, they should only need to change a small amount to be aligned but the axis they should
    # rotate around could be anything. Therefore, we will leave the axis of rotation completely
    # learnable but the rotation angle should be relatively small (10 degrees?)
    # in the real world application of this, we could apply a registration of images before creating meshes
    # as a starting point - or rigidly register any created point could to the mean mesh as a starting point
    # theta_constrain_max = theta_constrain_deviation
    # theta_constrain_min = -theta_constrain_deviation

    n_samples = xyz.shape[0]

    if clamp_dist is not None:
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
    sdf_gt = sdf_gt.to(device)
    # Figure out information about LR updates

    # Initialize latent vector
    latent = torch.ones(1, latent_size).normal_(mean=latent_init_mean, std=latent_init_std)

    # Initialize similarity transform parameters
    polar_angle = torch.ones(1).normal_(mean=0.0, std=0.01)
    azimuthal_angle = torch.ones(1).normal_(mean=0.0, std=0.01)
    theta = torch.ones(1).normal_(mean=0.0, std=0.01)
    scale = torch.ones(1).normal_(mean=100, std=0.01)
    translation = torch.ones(3).normal_(mean=0.0, std=0.01)

    list_to_optimize = [latent, polar_angle, azimuthal_angle, theta, scale, translation]

    for item in list_to_optimize:
        item.requires_grad = True

    # Initialize optimizer
    optimizer = torch.optim.Adam(list_to_optimize, lr=lr)

    # Initialize loss
    loss = 0
    if loss_type == "l1":
        loss_fn = torch.nn.L1Loss()
    elif loss_type == "l2":
        loss_fn = torch.nn.MSELoss()

    # Initialize convergence tracking
    patience = 0
    loss = 10

    for step in range(num_iterations):
        decoder.to(device)
        decoder.eval()

        # update LR
        adjust_learning_rate(
            initial_lr=lr,
            optimizer=optimizer,
            iteration=step,
            decreased_by=lr_update_factor,
            adjust_lr_every=adjust_lr_every,
        )

        optimizer.zero_grad()

        # use inverse transform information to
        # transform xyz to the canonical space

        # EQUATION 15: https://arxiv.org/pdf/2004.09048.pdf
        translated = xyz - translation
        # torch.linalg.inv
        # torch.linalg.inv documentation:
        # It is always prefered to use solve() when possible,
        # as it is faster and more numerically stable than
        # computing the inverse explicitly.
        # ITS NOT CLEAR IF THIS WILL WORK BECUASE ITS NOT A SQUARE SYSTEM
        # OF EQUATIONS: e.g., https://www.mathwords.com/s/square_system_of_equations.htm#:~:text=Mathwords%3A%20Square%20System%20of%20Equations&text=A%20linear%20system%20of%20equations,system%20is%20a%20square%20matrix.

        R = get_axis_angle_rotation_matrix(polar_angle, azimuthal_angle, theta)
        Rinv = torch.linalg.inv(R)
        rotated = (Rinv @ translated.T).T
        scaled = rotated / scale

        if verbose is True:
            print("scale", scale)
            print("R", R)
            print("translation", translation)
            print("theta_contrain_deviation", theta_constrain_deviation)
            print("scale_constrain", scale_constrain)
            print("translation_constrain", translation_constrain)

        latent_input = latent.expand(n_samples, -1)
        inputs = torch.cat([latent_input, scaled], dim=1).to(device)

        pred_sdf = decoder(inputs)

        if clamp_dist is not None:
            pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        _loss_ = loss_fn(pred_sdf, sdf_gt) * loss_weight

        if l2reg is True:
            latent_loss_ = latent_reg_weight * torch.mean(latent**2)
            loss_ = _loss_ + latent_loss_
        else:
            loss_ = _loss_

        if (transform_update_patience is not None) & (step > transform_update_patience):
            # Applying soft constraints to the scale
            # and theta parameters to keep them in a reasonable range
            # can probably tighten up the scale parameter
            if soft_contrain_scale is True:
                scale_diff = torch.abs(scale - init_scale)
                if verbose is True:
                    print("scale diff:", scale_diff)
                    print("scale constrain:", scale_constrain)
                if scale_diff > scale_constrain:
                    update = 1000 * (scale_diff - scale_constrain) ** 2
                # if (scale - scale_constrain) > init_scale:
                #     loss_ += (scale - scale_constrain + init_scale) * 1000
                # elif (scale + scale_constrain) < init_scale:
                #     loss_ += (scale - init_scale - scale_constrain) * 1000
                else:
                    # normalize this to the initalization scale
                    update = (scale_diff / init_scale) * 1e-6
                if verbose is True:
                    print("scale update:", update)
                loss_ += torch.squeeze(update)

            if soft_constrain_theta is True:
                # if (theta > theta_constrain_max):
                #     loss_ += (theta - theta_constrain_max) * 1000
                # elif (theta < theta_constrain_min):
                #     loss_ += (theta_constrain_min - theta) * 1000
                if verbose is True:
                    print("theta:", theta)
                if torch.abs(theta) > theta_constrain_deviation:
                    update = 1000 * (theta - theta_constrain_deviation) ** 2
                else:
                    update = torch.abs(theta) * 1e-8
                if verbose is True:
                    print("theta update:", update)
                loss_ += torch.squeeze(update)

            if soft_constrain_translation is True:
                if verbose is True:
                    print("translation:", translation)

                if torch.linalg.norm(translation - init_center) > translation_constrain:
                    update = (
                        1000
                        * (torch.linalg.norm(translation - init_center) - translation_constrain)
                        ** 2
                    )
                else:
                    # normalize this to the initalization scale
                    update = (torch.linalg.norm(translation - init_center) / init_scale) * 1e-6

                if verbose is True:
                    print("translation update:", update)

                loss_ += torch.squeeze(update)

        loss_.backward()
        optimizer.step()

        if step % 50 == 0:
            print("Step: ", step, "Loss: ", loss_.item())
            if verbose is True:
                print("\tLatent norm: ", latent.norm)

        if log_wandb is True:
            wandb.log(
                {
                    "l1_loss": loss_.item(),
                    "recon_loss": _loss_.item(),
                    "latent_loss": latent_loss_.item(),
                    "latent_norm": latent.norm().item(),
                }
            )

        if convergence == "overall_loss":
            if loss_ < loss:
                loss = loss_
                latent_ = torch.clone(latent)
                patience = 0
            else:
                patience += 1

            if patience > convergence_patience:
                print("Converged!")
                print("Step: ", step)
                break
        elif convergence == "recon_loss":
            raise Exception("Not implemented yet")
        else:
            loss = loss_
            latent_ = torch.clone(latent)

    latent_return = {
        "latent": latent_,
        "R": R,
        "s": scale,
        "t": translation,
    }

    return loss, latent_return
