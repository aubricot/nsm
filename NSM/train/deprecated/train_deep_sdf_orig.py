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


def train_deep_sdf(config, model, sdf_dataset, use_wandb=False):

    config = add_plain_lr_to_config(config)
    config["checkpoints"] = get_checkpoints(config)
    config["lr_schedules"] = get_learning_rate_schedules(config)

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
        wandb.watch(model, log="all")

    data_loader = torch.utils.data.DataLoader(
        sdf_dataset,
        batch_size=config["objects_per_batch"],
        shuffle=True,
        num_workers=config["num_data_loader_threads"],
        drop_last=False,
        prefetch_factor=config["prefetch_factor"],
        pin_memory=True,
    )

    latent_vecs = get_latent_vecs(len(data_loader.dataset), config).cuda()

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
            model,
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
                decoder=model,
            )
            if ("val_paths" in config) and (config["val_paths"] is not None):

                torch.cuda.empty_cache()

                dict_loss = get_mean_errors(
                    mesh_paths=config["val_paths"],
                    decoders=model,
                    num_iterations=config["num_iterations_recon"],
                    register_similarity=True,
                    latent_size=config["latent_size"],
                    lr=config["lr_recon"],
                    l2reg=config["l2reg_recon"],
                    clamp_dist=config["clamp_dist_recon"],
                    n_lr_updates=config["n_lr_updates_recon"],
                    lr_update_factor=config["lr_update_factor_recon"],
                    calc_symmetric_chamfer=config["chamfer"],
                    calc_assd=config["assd"],
                    calc_emd=config["emd"],
                    convergence=config["convergence_type_recon"],
                    convergence_patience=config["convergence_patience_recon"],
                    verbose=config["verbose"],
                    objects_per_decoder=1,
                    batch_size_latent_recon=config["batch_size_latent_recon"],
                    get_rand_pts=config["get_rand_pts_recon"],
                    n_pts_random=config["n_pts_random_recon"],
                    sigma_rand_pts=config["sigma_rand_pts_recon"],
                    n_samples_latent_recon=config["n_samples_latent_recon"],
                    # recon_func=None if (('recon_val_func_name' not in config)) else DICT_VALIDATION_FUNCS[config['recon_val_func_name']],
                    # predict_val_variables=None if ('predict_val_variables' not in config) else config['predict_val_variables'],
                )

                log_dict.update(dict_loss)

        if use_wandb is True:
            wandb.log(log_dict)

    return loss


def train_epoch(
    model,
    data_loader,
    latent_vecs,
    optimizer,
    config,
    epoch,
    return_loss=True,
    verbose=False,
):
    start = time.time()
    model.train()

    adjust_learning_rate(config["lr_schedules"], optimizer, epoch)

    step_losses = 0
    step_l1_loss = 0
    step_code_reg_loss = 0

    # if config['code_regularization_type_prior'] == 'kld_diagonal':
    #     kld_loss = get_kld(latent_vecs)

    for sdf_data, indices in data_loader:
        if config["verbose"] is True:
            print("sdf index size:", indices.size())
            print("xyz data size:", sdf_data["xyz"].size())
            print("sdf gt size:", sdf_data["gt_sdf"].size())

        # sdf_data = sdf_data.reshape(-1, 4)

        xyz = sdf_data["xyz"].cuda()
        xyz = xyz.reshape(-1, 3)

        num_sdf_samples = xyz.shape[0]
        xyz.requires_grad = False

        indices = indices.cuda()

        sdf_gt = sdf_data["gt_sdf"]
        sdf_gt = sdf_gt.reshape(-1, 1)

        sdf_gt.requires_grad = False

        if config["enforce_minmax"] is True:
            sdf_gt = torch.clamp(sdf_gt, -config["clamp_dist"], config["clamp_dist"])

        xyz = torch.chunk(xyz, config["batch_split"])
        indices = torch.chunk(
            indices.unsqueeze(-1)
            .repeat(1, config["samples_per_object_per_batch"])
            .view(-1),  # repeat the index for every sample
            config[
                "batch_split"
            ],  # split the data into the appropriate number of batches - so can fit in ram.
        )
        sdf_gt = torch.chunk(sdf_gt, config["batch_split"])

        batch_loss = 0.0
        batch_l1_loss = 0.0
        batch_code_reg_loss = 0.0

        optimizer.zero_grad()

        for split_idx in range(config["batch_split"]):
            if config["verbose"] is True:
                print("Split idx: ", split_idx)

            batch_vecs = latent_vecs(indices[split_idx])
            inputs = torch.cat([batch_vecs, xyz[split_idx]], dim=1)
            # inputs = inputs.to(config['device'])

            pred_sdf = model(inputs, epoch=epoch)

            if config["enforce_minmax"] is True:
                pred_sdf = torch.clamp(pred_sdf, -config["clamp_dist"], config["clamp_dist"])

            l1_loss = loss_l1(pred_sdf, sdf_gt[split_idx].cuda())

            # curriculum SDF equation 5
            # progressively fine-tune the regions of surface cared about by the network.
            if config["surface_accuracy_e"] is not None:
                weight_schedule = 1 - calc_weight(
                    epoch,
                    config["n_epochs"],
                    config["surface_accuracy_schedule"],
                    config["surface_accuracy_cooldown"],
                )
                l1_loss = torch.maximum(
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
                error_sign = torch.sign(sdf_gt[split_idx].cuda() - pred_sdf)
                sdf_gt_sign = torch.sign(sdf_gt[split_idx].cuda())
                sample_weights = 1 + difficulty_weight * sdf_gt_sign * error_sign
                l1_loss = l1_loss * sample_weights
            elif config["sample_difficulty_lx"] is not None:
                weight_schedule = calc_weight(
                    epoch,
                    config["n_epochs"],
                    config["sample_difficulty_lx_schedule"],
                    config["sample_difficulty_lx_cooldown"],
                )
                difficulty_weight = 1 / (
                    l1_loss ** config["sample_difficulty_lx"]
                    + config["sample_difficulty_lx_epsilon"]
                )
                difficulty_weight = difficulty_weight * weight_schedule
                l1_loss = l1_loss * difficulty_weight

            l1_loss = l1_loss / num_sdf_samples

            l1_loss = l1_loss.sum()

            batch_l1_loss += l1_loss.item()
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
                    config["code_regularization_weight"]
                    * min(1, epoch / config["code_regularization_warmup"])
                    * l2_size_loss
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

        if config["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        optimizer.step()
    end = time.time()

    seconds_elapsed = end - start

    save_loss = step_losses / len(data_loader)
    save_l1_loss = step_l1_loss / len(data_loader)
    save_code_reg_loss = step_code_reg_loss / len(data_loader)
    print("save loss: ", save_loss)
    print("\t save l1 loss: ", save_l1_loss)
    print("\t save code loss: ", save_code_reg_loss)

    log_dict = {
        "loss": save_loss,
        "epoch_time_s": seconds_elapsed,
        "l1_loss": save_l1_loss,
        "latent_code_regularization_loss": save_code_reg_loss,
    }

    return log_dict
