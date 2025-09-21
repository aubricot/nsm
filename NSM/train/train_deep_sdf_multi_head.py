from NSM.utils import (
    get_learning_rate_schedules,
    adjust_learning_rate,
    save_latent_vectors,
    save_model,
    get_optimizer,
    get_latent_vecs,
    get_checkpoints,
)
from NSM.reconstruct import get_mean_errors

from NSM.train.utils import get_kld, cyclic_anneal_linear, calc_weight, add_plain_lr_to_config

import wandb
import os
import torch
import time
import numpy as np

loss_l1 = torch.nn.L1Loss(reduction="none")


def train_deep_sdf(config, models: tuple, sdf_dataset, use_wandb=False):

    config = add_plain_lr_to_config(config)

    config["checkpoints"] = get_checkpoints(config)
    config["lr_schedules"] = get_learning_rate_schedules(config)

    for model in models:
        model = model.to(config["device"])

    if use_wandb is True:
        wandb.login(key=os.environ["WANDB_KEY"])
        wandb.init(
            # Set the project where this run will be logged
            project=config["project_name"],  # "diffusion-net-predict-sex",
            entity=config["entity_name"],  # "bone-modeling",
            # Track hyperparameters and run metadata
            config=config,
            name=config["run_name"],
            tags=config["tags"],
        )
        wandb.watch(models, log="all")

    data_loader = torch.utils.data.DataLoader(
        sdf_dataset,
        batch_size=config["objects_per_batch"],
        shuffle=True,
        num_workers=config["num_data_loader_threads"],
        drop_last=False,
    )

    latent_vecs = get_latent_vecs(len(data_loader.dataset), config)

    optimizer = get_optimizer(
        model,
        latent_vecs,
        lr_schedules=config["lr_schedules"],
        optimizer=config["optimizer"],
        weight_decay=config["weight_decay"],
    )

    for epoch in range(1, config["n_epochs"] + 1):
        # not passing latent_vecs because presumably they are being tracked by the
        # and updated in memory?
        log_dict = train_epoch(
            models,
            data_loader,
            latent_vecs,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            return_loss=True,
        )

        if epoch in config["checkpoints"]:
            save_latent_vectors(
                config=config,
                epoch=epoch,
                latent_vec=latent_vecs,
            )
            save_model(
                config=config,
                epoch=epoch,
                decoder=models,
            )
            if ("val_paths" in config) & (config["val_paths"] is not None):
                # if cuda is being used, empty the cache
                if config["device"] == "cuda":
                    torch.cuda.empty_cache()
                elif config["device"] == "cpu":
                    torch.mps.empty_cache()

                dict_loss = get_mean_errors(
                    mesh_paths=config["val_paths"],
                    num_iterations=config["num_iterations_recon"],
                    decoders=models,
                    latent_size=config["latent_size"],
                    calc_symmetric_chamfer=config["chamfer"],
                    calc_assd=config["assd"],
                    calc_emd=config["emd"],
                    register_similarity=True,
                    scale_all_meshes=True,
                    verbose=config["verbose"],
                    get_rand_pts=config["get_rand_pts_recon"],
                    n_pts_random=config["n_pts_random_recon"],
                    lr=config["lr_recon"],
                    l2reg=config["l2reg_recon"],
                    clamp_dist=config["clamp_dist_recon"],
                    n_lr_updates=config["n_lr_updates_recon"],
                    lr_update_factor=config["lr_update_factor_recon"],
                    convergence_patience=config["convergence_patience_recon"],
                    batch_size_latent_recon=config["batch_size_latent_recon"],
                    convergence=config["convergence_type_recon"],
                    sigma_rand_pts=config["sigma_rand_pts_recon"],
                    n_samples_latent_recon=config["n_samples_latent_recon"],
                    device=config["device"],
                )

                log_dict.update(dict_loss)

        if use_wandb is True:
            wandb.log(log_dict)

    return


def train_epoch(
    models,
    data_loader,
    latent_vecs,
    optimizer,
    config,
    epoch,
    return_loss=True,
    verbose=False,
):
    n_surfaces = len(models)
    start = time.time()
    for model in models:
        model.train()

    adjust_learning_rate(config["lr_schedules"], optimizer, epoch)

    step_losses = 0
    step_l1_loss = 0
    step_code_reg_loss = 0
    step_l1_losses = [0.0 for _ in range(n_surfaces)]

    # if config['code_regularization_type_prior'] == 'kld_diagonal':
    #     kld_loss = get_kld(latent_vecs)

    for sdf_data, indices in data_loader:
        if config["verbose"] is True:
            print("sdf index size:", indices.size())
            print("xyz data size:", sdf_data["xyz"].size())
            print("sdf gt size:", sdf_data["gt_sdf"].size())

        xyz = sdf_data["xyz"]
        xyz = xyz.reshape(-1, 3)

        num_sdf_samples = xyz.shape[0]
        xyz.requires_grad = False

        sdf_gt = []
        for surf_idx in range(n_surfaces):
            sdf_gt_ = sdf_data["gt_sdf"][:, :, surf_idx].reshape(-1, 1)
            if config["enforce_minmax"] is True:
                sdf_gt_ = torch.clamp(sdf_gt_, -config["clamp_dist"], config["clamp_dist"])
            sdf_gt_.requires_grad = False
            sdf_gt.append(sdf_gt_)

        if config["verbose"] is True:
            print("sdf gt size:", sdf_gt[0].size(), sdf_gt[1].size())

        xyz = torch.chunk(xyz, config["batch_split"])
        indices = torch.chunk(
            indices.unsqueeze(-1)
            .repeat(1, config["samples_per_object_per_batch"])
            .view(-1),  # repeat the index for every sample
            config[
                "batch_split"
            ],  # split the data into the appropriate number of batches - so can fit in ram.
        )

        for surf_idx in range(n_surfaces):
            sdf_gt[surf_idx] = torch.chunk(sdf_gt[surf_idx], config["batch_split"])

        if config["verbose"] is True:
            print("len sdf_gt", len(sdf_gt))
            print("len sdf_gt chunks", len(sdf_gt[0]), len(sdf_gt[1]))
            print("len xyz chunks", len(xyz))

        batch_loss = 0.0
        batch_l1_loss = 0.0
        batch_l1_losses = [0.0 for _ in range(n_surfaces)]
        batch_code_reg_loss = 0.0

        optimizer.zero_grad()

        for split_idx in range(config["batch_split"]):
            if config["verbose"] is True:
                print("Split idx: ", split_idx)

            batch_vecs = latent_vecs(indices[split_idx])
            inputs = torch.cat([batch_vecs, xyz[split_idx]], dim=1)
            inputs = inputs.to(config["device"])

            pred_sdfs = []
            for model in models:
                pred_sdf = model(inputs)
                if config["enforce_minmax"] is True:
                    pred_sdf = torch.clamp(pred_sdf, -config["clamp_dist"], config["clamp_dist"])
                pred_sdfs.append(pred_sdf)

            if config["verbose"] is True:
                print("len pred_sdfs", len(pred_sdfs))
                print("split idx", split_idx)
            l1_losses = []
            for surf_idx, pred_sdf in enumerate(pred_sdfs):
                if config["verbose"] is True:
                    print("surf idx", surf_idx)
                    print(len(sdf_gt))
                    print(len(sdf_gt[surf_idx]))
                l1_losses.append(loss_l1(pred_sdf, sdf_gt[surf_idx][split_idx].cuda()))
            # l1_loss = loss_l1(pred_sdf, sdf_gt[split_idx].cuda())

            if "multi_object_overlap" in config and config["multi_object_overlap"] is True:
                raise Exception("Not implemented yet")
                # Should add some weighted penalty to the l1 loss
                # this is similar to surface_accuracy_e below
                # The idea being - the signs of the objects should never have 2 objects as (-) becuase it
                # means one object is inside the other.
                # However, we dont want there to be any gaps between them - so we need to be intelligent about
                # how to penalize this so that it doesnt end up with weird artefacts.

            # curriculum SDF equation 5
            # progressively fine-tune the regions of surface cared about by the network.
            if config["surface_accuracy_e"] is not None:
                weight_schedule = 1 - calc_weight(
                    epoch,
                    config["n_epochs"],
                    config["surface_accuracy_schedule"],
                    config["surface_accuracy_cooldown"],
                )
                for l1_idx, l1_loss in enumerate(l1_losses):
                    l1_losses[l1_idx] = torch.maximum(
                        l1_loss - (weight_schedule * config["surface_accuracy_e"]),
                        torch.zeros_like(l1_loss),
                    )

            # curriculum SDF equation 6
            # progressively fine-tune the regions of surface cared about by the network.
            # weighting gives higher preference to regions closer to surface / with opposite sign.
            if config["sample_difficulty_weight"] is not None:
                weight_schedule = calc_weight(
                    epoch,
                    config["n_epochs"],
                    config["sample_difficulty_weight_schedule"],
                    config["sample_difficulty_cooldown"],
                )
                difficulty_weight = weight_schedule * config["sample_difficulty_weight"]
                for surf_idx, surf_gt_ in enumerate(sdf_gt):
                    # Weights points independently
                    # so, if hard for one surface - then we weight it heavily, but if
                    # easy for another surface - then we weight it less.
                    error_sign = torch.sign(surf_gt_[split_idx].cuda() - pred_sdfs[surf_idx])
                    sdf_gt_sign = torch.sign(surf_gt_[split_idx].cuda())
                    sample_weights = 1 + difficulty_weight * sdf_gt_sign * error_sign
                    l1_losses[surf_idx] = l1_losses[surf_idx] * sample_weights
            elif config["sample_difficulty_lx"] is not None:
                weight_schedule = calc_weight(
                    epoch,
                    config["n_epochs"],
                    config["sample_difficulty_lx_schedule"],
                    config["sample_difficulty_lx_cooldown"],
                )
                for surf_idx, surf_gt_ in enumerate(sdf_gt):
                    difficulty_weight = 1 / (
                        l1_losses[surf_idx] ** config["sample_difficulty_lx"]
                        + config["sample_difficulty_lx_epsilon"]
                    )
                    difficulty_weight = difficulty_weight * weight_schedule
                    l1_losses[surf_idx] = l1_losses[surf_idx] * difficulty_weight

            # Weight each surface loss by the number of samples it has
            # so that the sum of them all is the same as the mean loss.
            for idx, l1_loss_ in enumerate(l1_losses):
                l1_losses[idx] = l1_loss_ / num_sdf_samples

            # l1_loss = l1_loss / num_sdf_samples

            # Comput total loss for each mesh (which is the same as the
            # mean).
            l1_loss = 0

            # Create weights for each surface
            if ("surface_weighting" in config) & (
                isinstance(config["surface_weighting"], (list, tuple))
            ):
                assert len(config["surface_weighting"]) == n_surfaces
                weights_total = n_surfaces
                weights_sum = sum(config["surface_weighting"])
                weights = []
                for weight in config["surface_weighting"]:
                    weights.append(weight / weights_sum * weights_total)
            else:
                weights = [
                    1,
                ] * n_surfaces

            # Apple the weights to each surface loss before adding
            # to the total l1_loss.
            # This does not apply the weights to the individual surfaces
            # l1_losses, this is why they are non-zero in the saved results
            for l1_idx, l1_loss_ in enumerate(l1_losses):
                l1_loss += l1_loss_.sum() * weights[l1_idx]

            # Normalize by number of surfaces
            l1_loss = l1_loss / len(l1_losses)

            if config["verbose"] is True:
                print(f"l1 losses: {[l1_loss_.sum().item() for l1_loss_ in l1_losses]}")
                print(f"l1 loss: {l1_loss.item()}")

            batch_l1_loss += l1_loss.item()
            for l1_idx, l1_loss_ in enumerate(l1_losses):
                batch_l1_losses.append(l1_loss_.sum().item())
            chunk_loss = l1_loss

            if config["code_regularization"] is True:
                if config["code_regularization_type_prior"] == "spherical":
                    # spherical prior
                    # all latent vectors should have the same unit length
                    # therefore, the latent dimensions will be correlated
                    # with one another - this is as opposed to PCA (and below).
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                elif config["code_regularization_type_prior"] == "identity":
                    # independently penalize each dimension/value of latent code
                    # therefore latent code ends up having identity covariance matrix
                    l2_size_loss = torch.sum(torch.square(batch_vecs))
                elif config["code_regularization_type_prior"] == "kld_diagonal":
                    l2_size_loss = get_kld(batch_vecs)
                else:
                    raise ValueError(
                        "Unknown code regularization type prior: {}".format(
                            config["code_regularization_type_prior"]
                        )
                    )
                reg_loss = (
                    config["code_regularization_weight"] * min(1, epoch / 100) * l2_size_loss
                ) / num_sdf_samples

                if config["code_cyclic_anneal"] is True:
                    anneal_weight = cyclic_anneal_linear(epoch=epoch, n_epochs=config["n_epochs"])
                    reg_loss = reg_loss * anneal_weight

                chunk_loss = chunk_loss + reg_loss.cuda()
                batch_code_reg_loss += reg_loss.item()

            chunk_loss.backward()

            batch_loss += chunk_loss.item()

        step_losses += batch_loss
        step_l1_loss += batch_l1_loss
        step_code_reg_loss += batch_code_reg_loss
        for l1_idx, l1_loss_ in enumerate(l1_losses):
            step_l1_losses[l1_idx] += l1_loss_.sum().item()  # l1_loss_

        if config["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        optimizer.step()
    end = time.time()

    seconds_elapsed = end - start

    save_loss = step_losses / len(data_loader)
    save_l1_loss = step_l1_loss / len(data_loader)
    save_code_reg_loss = step_code_reg_loss / len(data_loader)
    save_l1_losses = [l1_loss_ / len(data_loader) for l1_loss_ in step_l1_losses]
    print("save loss: ", save_loss)
    print("\t save l1 loss: ", save_l1_loss)
    print("\t save code loss: ", save_code_reg_loss)
    print("\t save l1 losses: ", save_l1_losses)

    log_dict = {
        "loss": save_loss,
        "epoch_time_s": seconds_elapsed,
        "l1_loss": save_l1_loss,
        "latent_code_regularization_loss": save_code_reg_loss,
    }
    for l1_idx, l1_loss_ in enumerate(save_l1_losses):
        log_dict["l1_loss_{}".format(l1_idx)] = l1_loss_

    return log_dict
