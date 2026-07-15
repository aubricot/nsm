from NSM.utils import (
    get_learning_rate_schedules,
    adjust_learning_rate,
    save_latent_vectors,
    save_model,
    save_model_params,
    get_optimizer,
    get_latent_vecs,
    get_checkpoints,
    clear_gpu_cache,
    rename_optimizer_param_groups,
)
from NSM.losses import eikonal_loss
from NSM.reconstruct import (
    get_mean_errors,
    compare_cart_thickness,
    compare_cart_thickness_tibia,
    compare_cart_thickness_patella,
    compare_cart_thickness_femur,
    compare_cart_thickness_whole_joint,
)
from NSM.train.utils import (
    get_kld,
    cyclic_anneal_linear,
    calc_weight,
    add_plain_lr_to_config,
    get_profiler,
)

from hierarchy.losses import (
    TaxonomyLabelEncoder,
    HierarchyContrastiveLoss,
    TaxonomyClassificationHeads,
    compute_classification_head_loss,
)

import wandb
import csv
import os
import torch
import time
import numpy as np
import itertools
import json


DICT_VALIDATION_FUNCS = {
    "compare_cart_thickness": compare_cart_thickness,
    "compare_cart_thickness_tibia": compare_cart_thickness_tibia,
    "compare_cart_thickness_patella": compare_cart_thickness_patella,
    "compare_cart_thickness_femur": compare_cart_thickness_femur,
    "compare_cart_thickness_whole_joint": compare_cart_thickness_whole_joint,
    None: None,
}

loss_l1 = torch.nn.L1Loss(reduction="none")


def _write_epoch_losses_csv(log_dict, fpath):
    total = log_dict.get("loss", 1.0) or 1.0
    loss_keys = [
        "l1_loss",
        "latent_code_regularization_loss",
        "eikonal_loss",
        "hierarchy_loss",
        "classification_head_loss",
    ]
    surf_keys = sorted([k for k in log_dict if k.startswith("l1_loss_")])

    row = {"epoch": log_dict["epoch"], "total_loss": total}
    for k in loss_keys + surf_keys:
        if k in log_dict:
            row[k] = log_dict[k]
            row[k + "_pct"] = round(log_dict[k] / total * 100, 2)
    row["epoch_time_s"] = log_dict.get("epoch_time_s", "")

    write_header = not os.path.exists(fpath)
    with open(fpath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_deep_sdf(config, model, sdf_dataset, use_wandb=False):
    config.setdefault("objects_per_decoder", 1)
    config.setdefault("resume_epoch", 0)
    config.setdefault("scale_jointly", False)
    config.setdefault("fix_mesh_recon", False)
    config.setdefault("log_latent", None)

    # hierarchy defaults
    config.setdefault("hierarchy_loss_enabled", True)
    config.setdefault("hierarchy_weight", config.get("hierarchy_weight", 0.01))
    config.setdefault("hierarchy_warmup", config.get("hierarchy_warmup", 200))
    config.setdefault(
        "hierarchy_margins",
        config.get("hierarchy_margins", {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}),
    )
    config.setdefault("classification_heads_enabled", True)
    config.setdefault("classification_head_weight", 0.005)
    config.setdefault("classification_head_warmup", 100)
    config.setdefault("classification_head_hidden_dim", 256)
    config.setdefault(
        "classification_level_weights",
        {"species": 1.0, "genus": 0.5, "family": 0.25, "position": 0.75},
    )

    config = add_plain_lr_to_config(config)
    config["checkpoints"] = get_checkpoints(config)
    config["lr_schedules"] = get_learning_rate_schedules(config)

    model = model.to(config["device"])

    taxonomy_encoder = TaxonomyLabelEncoder(sdf_dataset.list_mesh_paths)
    taxonomy_info = taxonomy_encoder.get_taxonomy_info()

    hierarchy_contrastive = None
    if config.get("hierarchy_loss_enabled", False):
        hierarchy_contrastive = HierarchyContrastiveLoss(
            margins=config["hierarchy_margins"]
        )

    classification_heads = None
    if config.get("classification_heads_enabled", False):
        classification_heads = TaxonomyClassificationHeads(
            latent_dim=config["latent_size"],
            num_species=taxonomy_encoder.num_classes("species"),
            num_genera=taxonomy_encoder.num_classes("genus"),
            num_families=taxonomy_encoder.num_classes("family"),
            num_positions=taxonomy_encoder.num_classes("position"),
            hidden_dim=config["classification_head_hidden_dim"],
        ).to(config["device"])

    _train_logs_dir = "./train_logs"
    os.makedirs(_train_logs_dir, exist_ok=True)
    losses_log_fpath = os.path.join(
        _train_logs_dir,
        os.path.split(config["experiment_directory"])[1] + "_epoch_losses.csv",
    )

    if use_wandb is True:
        wandb.login(key=os.environ["WANDB_KEY"])
        wandb.init(
            project=config["project_name"],
            entity=config["entity_name"],
            config=config,
            name=config["run_name"],
            tags=config["tags"],
        )
        wandb.watch(model, log="all")
    else:
        cwd = config["experiment_directory"]
        log_fpath = os.path.split(cwd)[0] + "/train_logs/" + os.path.split(cwd)[1] + "_train_log.csv"
        if not os.path.exists(os.path.split(cwd)[0] + "/train_logs/"):
            os.makedirs(os.path.split(cwd)[0] + "/train_logs/")

    data_loader = torch.utils.data.DataLoader(
        sdf_dataset,
        batch_size=config["objects_per_batch"],
        shuffle=True,
        num_workers=config["num_data_loader_threads"],
        drop_last=False,
        prefetch_factor=config["prefetch_factor"],
        pin_memory=True,
    )

    latent_vecs = get_latent_vecs(len(data_loader.dataset), config).to(config["device"])
    optimizer = get_optimizer(
        model,
        latent_vecs,
        lr_schedules=config["lr_schedules"],
        optimizer=config["optimizer"],
        weight_decay=config["weight_decay"],
    )

    if classification_heads is not None:
        optimizer.add_param_group(
            {
                "name": "classification_heads",
                "params": classification_heads.parameters(),
                "lr": config["lr_schedules"][0].get_learning_rate(0),
            }
        )

    if config["resume_epoch"] > 1:
        print("Loading model, optimizer, and latent states from epoch", config["resume_epoch"])

        model_ckpt_path = os.path.join(
            config["experiment_directory"], "model", f'{config["resume_epoch"]}.pth'
        )
        latent_ckpt_path = os.path.join(
            config["experiment_directory"], "latent_codes", f'{config["resume_epoch"]}.pth'
        )

        model_checkpoint = torch.load(model_ckpt_path, map_location=config["device"])
        latent_checkpoint = torch.load(latent_ckpt_path, map_location=config["device"])

        model.load_state_dict(model_checkpoint["model"])

        optimizer_state = model_checkpoint.get("optimizer")
        if optimizer_state is None:
            raise ValueError(f"No optimizer state found in checkpoint: {model_ckpt_path}")
        optimizer.load_state_dict(optimizer_state)

        group_names = model_checkpoint.get("optimizer_group_names")
        if group_names is not None:
            if len(group_names) != len(optimizer.param_groups):
                raise ValueError(
                    f"optimizer_group_names length mismatch: "
                    f"{len(group_names)} names for {len(optimizer.param_groups)} param groups"
                )
            for group, name in zip(optimizer.param_groups, group_names):
                if name is None:
                    raise ValueError("Checkpoint contains optimizer param group with missing name")
                group["name"] = name
        else:
            n_model_groups = len(model) if isinstance(model, (list, tuple)) else 1
            rename_optimizer_param_groups(
                optimizer,
                n_model_groups=n_model_groups,
                has_classification_heads=classification_heads is not None,
            )

        latent_vecs.load_state_dict(latent_checkpoint["latent_codes"])

        heads_path = os.path.join(
            config["experiment_directory"], "classification_heads", f'{config["resume_epoch"]}.pth'
        )
        if classification_heads is not None and os.path.exists(heads_path):
            classification_heads.load_state_dict(torch.load(heads_path)["classification_heads"])

    with get_profiler(config) as profiler:
        for epoch in range(config["resume_epoch"] + 1, config["n_epochs"] + 1):
            print(f'\033[92m\n\n\nEpoch: {epoch}\033[0m')

            log_dict = train_epoch(
                model,
                data_loader,
                latent_vecs,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                return_loss=True,
                n_surfaces=config["objects_per_decoder"],
                taxonomy_encoder=taxonomy_encoder,
                hierarchy_contrastive=hierarchy_contrastive,
                classification_heads=classification_heads,
            )

            val_epoch = (
                (epoch in config["checkpoints"])
                and ("val_paths" in config)
                and (config["val_paths"] is not None)
            )
            checkpoint_epoch = (
                epoch in config["checkpoints"] or epoch % config["save_frequency"] == 0
            )

            if val_epoch or checkpoint_epoch:
                if "schedule_free" in config["optimizer"]:
                    optimizer.eval()
                    with torch.no_grad():
                        for batch in itertools.islice(data_loader, 50):
                            model(batch)

            if checkpoint_epoch:
                print("\nCheckpoint epoch...")
                save_model_params(config=config, list_mesh_paths=sdf_dataset.list_mesh_paths)
                save_latent_vectors(config=config, epoch=epoch, latent_vec=latent_vecs)
                save_model(config=config, epoch=epoch, decoder=model, optimizer=optimizer)

                if classification_heads is not None:
                    heads_dir = os.path.join(config["experiment_directory"], "classification_heads")
                    os.makedirs(heads_dir, exist_ok=True)
                    torch.save(
                        {"epoch": epoch, "classification_heads": classification_heads.state_dict()},
                        os.path.join(heads_dir, f"{epoch}.pth"),
                    )

                tax_dir = os.path.join(config["experiment_directory"], "taxonomy")
                os.makedirs(tax_dir, exist_ok=True)
                with open(os.path.join(tax_dir, "taxonomy_info.json"), "w") as f:
                    json.dump(taxonomy_info, f, indent=2)

            if val_epoch:
                print("\nValidation epoch...")
                clear_gpu_cache(config["device"])

                # 1) Write losses + relative percentages percentages to CSV post-run analysis
                _write_epoch_losses_csv(log_dict, losses_log_fpath)

                # 2) Write train losses
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
                    objects_per_decoder=config["objects_per_decoder"],
                    batch_size_latent_recon=config["batch_size_latent_recon"],
                    get_rand_pts=config["get_rand_pts_recon"],
                    n_pts_random=config["n_pts_random_recon"],
                    sigma_rand_pts=config["sigma_rand_pts_recon"],
                    n_samples_latent_recon=config["n_samples_latent_recon"],
                    scale_all_meshes=True,
                    recon_func=(
                        None
                        if (("recon_val_func_name" not in config))
                        else DICT_VALIDATION_FUNCS[config["recon_val_func_name"]]
                    ),
                    predict_val_variables=(
                        None
                        if ("predict_val_variables" not in config)
                        else config["predict_val_variables"]
                    ),
                    scale_jointly=config["scale_jointly"],
                    fix_mesh=config["fix_mesh_recon"],
                    device=config["device"],
                )

                log_dict.update(dict_loss)
                write_header = not os.path.exists(log_fpath)
                with open(log_fpath, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(log_dict)

            if use_wandb is True:
                wandb.log(log_dict, step=epoch - 1)

            profiler.step()
            clear_gpu_cache(config["device"])


def train_epoch(
    model,
    data_loader,
    latent_vecs,
    optimizer,
    config,
    epoch,
    taxonomy_encoder,
    hierarchy_contrastive=None,
    classification_heads=None,
    return_loss=True,
    verbose=False,
    n_surfaces=2,
):
    start = time.time()
    model.train()
    if classification_heads is not None:
        classification_heads.train()

    if not ("schedule_free" in config["optimizer"]):
        adjust_learning_rate(config["lr_schedules"], optimizer, epoch)
    else:
        optimizer.train()

    step_losses = 0
    step_l1_loss = 0
    step_code_reg_loss = 0
    step_eikonal_loss = 0
    step_hierarchy_loss = 0
    step_cls_loss = 0
    step_cls_losses = {}
    step_l1_losses = [0.0 for _ in range(n_surfaces)]
    step_mean_vec_length = 0
    step_std_vec_length = 0

    step_mean_size = 0
    step_mean_load_time = 0
    step_mean_load_rate = 0
    step_whole_load_time = 0

    for sdf_data, indices in data_loader:
        xyz = sdf_data["xyz"].to(config["device"])
        xyz = xyz.reshape(-1, 3)
        num_sdf_samples = xyz.shape[0]
        xyz.requires_grad = False

        original_object_indices = indices.to(config["device"])
        expanded_indices = torch.chunk(
            original_object_indices.unsqueeze(-1).repeat(1, config["samples_per_object_per_batch"]).view(-1),
            config["batch_split"],
        )
        xyz = torch.chunk(xyz, config["batch_split"])

        sdf_gt = []
        if n_surfaces == 1:
            sdf_gt_ = sdf_data["gt_sdf"].reshape(-1, 1)
            if config["enforce_minmax"] is True:
                sdf_gt_ = torch.clamp(sdf_gt_, -config["clamp_dist"], config["clamp_dist"])
            sdf_gt_.requires_grad = False
            sdf_gt.append(torch.chunk(sdf_gt_, config["batch_split"]))
        else:
            for surf_idx in range(n_surfaces):
                sdf_gt_ = sdf_data["gt_sdf"][:, :, surf_idx].reshape(-1, 1)
                if config["enforce_minmax"] is True:
                    sdf_gt_ = torch.clamp(sdf_gt_, -config["clamp_dist"], config["clamp_dist"])
                sdf_gt_.requires_grad = False
                sdf_gt.append(torch.chunk(sdf_gt_, config["batch_split"]))

        batch_loss = 0.0
        batch_l1_loss = 0.0
        batch_l1_losses = [0.0 for _ in range(n_surfaces)]
        batch_code_reg_loss = 0.0
        batch_eikonal_loss = 0.0
        batch_hierarchy_loss = 0.0
        batch_cls_loss = 0.0
        batch_cls_losses = {}

        optimizer.zero_grad()

        # compute hierarchy/head terms once per batch (avoids scaling by batch_split)
        hierarchy_term = None
        if config.get("hierarchy_loss_enabled", False) and hierarchy_contrastive is not None:
            per_object_vecs = latent_vecs(original_object_indices)
            warmup = min(1.0, epoch / config["hierarchy_warmup"])
            h_loss = hierarchy_contrastive(
                per_object_vecs,
                original_object_indices,
                taxonomy_encoder,
                device=config["device"],
            )
            hierarchy_term = config["hierarchy_weight"] * warmup * h_loss
            batch_hierarchy_loss += hierarchy_term.item()

        cls_term = None
        if config.get("classification_heads_enabled", False) and classification_heads is not None:
            per_object_vecs_cls = latent_vecs(original_object_indices)
            logits = classification_heads(per_object_vecs_cls)
            warmup = min(1.0, epoch / config["classification_head_warmup"])
            cls_loss, cls_loss_dict = compute_classification_head_loss(
                logits,
                original_object_indices,
                taxonomy_encoder,
                level_weights=config["classification_level_weights"],
                device=config["device"],
            )
            cls_term = config["classification_head_weight"] * warmup * cls_loss
            batch_cls_loss += cls_term.item()
            for k, v in cls_loss_dict.items():
                batch_cls_losses[k] = batch_cls_losses.get(k, 0) + v

        for split_idx in range(config["batch_split"]):
            batch_vecs = latent_vecs(expanded_indices[split_idx])

            if "variational" in config and config["variational"] is True:
                mu = batch_vecs[:, : config["latent_size"]]
                logvar = batch_vecs[:, config["latent_size"] :]
                std = torch.exp(0.5 * logvar)
                err = torch.randn_like(std)
                batch_vecs = std * err + mu

            inputs = torch.cat([batch_vecs, xyz[split_idx]], dim=1)
            pred_sdf = model(inputs, epoch=epoch)

            if n_surfaces == 1 and not (pred_sdf.dim() == 2 and pred_sdf.shape[1] == 1):
                pred_sdf = pred_sdf.unsqueeze(1)

            if config["enforce_minmax"] is True:
                pred_sdf = torch.clamp(pred_sdf, -config["clamp_dist"], config["clamp_dist"])

            l1_losses = []
            for surf_idx in range(n_surfaces):
                l1_losses.append(
                    loss_l1(
                        pred_sdf[:, surf_idx],
                        sdf_gt[surf_idx][split_idx].squeeze(1).to(config["device"]),
                    )
                )

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

            if config["sample_difficulty_weight"] is not None:
                weight_schedule = calc_weight(
                    epoch,
                    config["n_epochs"],
                    config["sample_difficulty_weight_schedule"],
                    config["sample_difficulty_cooldown"],
                )
                difficulty_weight = weight_schedule * config["sample_difficulty_weight"]
                for surf_idx, surf_gt_ in enumerate(sdf_gt):
                    error_sign = torch.sign(
                        surf_gt_[split_idx].squeeze(1).to(config["device"]) - pred_sdf[:, surf_idx]
                    )
                    sdf_gt_sign = torch.sign(surf_gt_[split_idx].squeeze(1).to(config["device"]))
                    sample_weights = 1 + difficulty_weight * sdf_gt_sign * error_sign
                    l1_losses[surf_idx] = l1_losses[surf_idx] * sample_weights

            for idx, l1_loss_ in enumerate(l1_losses):
                l1_losses[idx] = l1_loss_ / num_sdf_samples

            l1_loss = 0
            if isinstance(config.get("surface_weighting", None), (list, tuple)):
                assert len(config["surface_weighting"]) == n_surfaces
                weights_total = n_surfaces
                weights_sum = sum(config["surface_weighting"])
                weights = []
                for weight in config["surface_weighting"]:
                    weights.append(weight / weights_sum * weights_total)
            else:
                weights = [1] * n_surfaces

            for l1_idx, l1_loss_ in enumerate(l1_losses):
                l1_loss += l1_loss_.sum() * weights[l1_idx]
            l1_loss = l1_loss / len(l1_losses)

            batch_l1_loss += l1_loss.item()
            for l1_idx, l1_loss_ in enumerate(l1_losses):
                batch_l1_losses[l1_idx] += l1_loss_.sum().item()

            chunk_loss = l1_loss

            if config.get("eikonal_weight", 0) > 0:
                xyz_grad = xyz[split_idx].detach().requires_grad_(True)
                inputs_grad = torch.cat([batch_vecs, xyz_grad], dim=1)
                pred_sdf_grad = model(inputs_grad, epoch=epoch)
                eik_loss = eikonal_loss(pred_sdf_grad, xyz_grad, reduction="mean")
                batch_eikonal_loss += eik_loss.item()
                chunk_loss = chunk_loss + config["eikonal_weight"] * eik_loss

            if config["code_regularization"] is True:
                if "variational" in config and config["variational"] is True:
                    kld = torch.mean(
                        -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
                    )
                    reg_loss = kld
                    code_reg_norm = 1
                else:
                    if config["code_regularization_type_prior"] == "spherical":
                        reg_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    elif config["code_regularization_type_prior"] == "identity":
                        reg_loss = torch.sum(torch.square(batch_vecs))
                    elif config["code_regularization_type_prior"] == "kld_diagonal":
                        reg_loss = get_kld(batch_vecs)
                    else:
                        raise ValueError(
                            f'Unknown code regularization type prior: {config["code_regularization_type_prior"]}'
                        )
                    code_reg_norm = num_sdf_samples

                reg_loss = (
                    config["code_regularization_weight"]
                    * min(1, epoch / config["code_regularization_warmup"])
                    * reg_loss
                ) / code_reg_norm

                if config["code_cyclic_anneal"] is True:
                    anneal_weight = cyclic_anneal_linear(epoch=epoch, n_epochs=config["n_epochs"])
                    reg_loss = reg_loss * anneal_weight

                chunk_loss = chunk_loss + reg_loss.to(config["device"])
                batch_code_reg_loss += reg_loss.item()

            if split_idx == 0 and hierarchy_term is not None:
                chunk_loss = chunk_loss + hierarchy_term
            if split_idx == 0 and cls_term is not None:
                chunk_loss = chunk_loss + cls_term

            mean_vec_length = torch.mean(torch.norm(batch_vecs, dim=1))
            std_vec_length = torch.std(torch.norm(batch_vecs, dim=1))

            chunk_loss.backward()
            batch_loss += chunk_loss.item()

        step_losses += batch_loss
        step_l1_loss += batch_l1_loss
        step_code_reg_loss += batch_code_reg_loss
        step_eikonal_loss += batch_eikonal_loss
        step_hierarchy_loss += batch_hierarchy_loss
        step_cls_loss += batch_cls_loss
        for k, v in batch_cls_losses.items():
            step_cls_losses[k] = step_cls_losses.get(k, 0) + v
        for l1_idx, l1_loss_ in enumerate(batch_l1_losses):
            step_l1_losses[l1_idx] += l1_loss_

        step_mean_vec_length = mean_vec_length.item()
        step_std_vec_length = std_vec_length.item()

        if config["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        step_mean_size += torch.mean(sdf_data["size"]).item()
        step_mean_load_time += torch.mean(sdf_data["time"]).item()
        step_mean_load_rate += torch.mean(sdf_data["mb_per_sec"]).item()
        step_whole_load_time += torch.mean(sdf_data["whole_load_time"]).item()

        optimizer.step()

    seconds_elapsed = time.time() - start
    n_batches = len(data_loader)

    save_loss = step_losses / n_batches
    save_l1_loss = step_l1_loss / n_batches
    save_code_reg_loss = step_code_reg_loss / n_batches
    save_eikonal_loss = step_eikonal_loss / n_batches
    save_hierarchy_loss = step_hierarchy_loss / n_batches
    save_cls_loss = step_cls_loss / n_batches
    save_l1_losses = [l1_loss_ / n_batches for l1_loss_ in step_l1_losses]
    save_mean_vec_length = step_mean_vec_length / n_batches
    save_std_vec_length = step_std_vec_length / n_batches

    print(f"[Epoch {epoch:04d}]")
    print("save loss: ", save_loss)
    print("\t save l1 loss: ", save_l1_loss)
    print("\t save code loss: ", save_code_reg_loss)
    if config.get("eikonal_weight", 0) > 0:
        print(f"\t save eikonal loss: {save_eikonal_loss:.6f}")
    print(f"\t save hierarchy loss: {save_hierarchy_loss:.6f}")
    print(f"\t save classification head loss: {save_cls_loss:.6f}")
    for k, v in step_cls_losses.items():
        print(f"\t\t {k}: {v / n_batches:.6f}")
    print("\t save l1 losses: ", save_l1_losses)

    log_dict = {
        "epoch": epoch,
        "loss": save_loss,
        "epoch_time_s": seconds_elapsed,
        "l1_loss": save_l1_loss,
        "latent_code_regularization_loss": save_code_reg_loss,
        "hierarchy_loss": save_hierarchy_loss,
        "classification_head_loss": save_cls_loss,
        "mean_size": step_mean_size / n_batches,
        "mean_load_time": step_mean_load_time / n_batches,
        "mean_load_rate": step_mean_load_rate / n_batches,
        "whole_load_time": step_whole_load_time / n_batches,
        "mean_vec_length": save_mean_vec_length,
        "std_vec_length": save_std_vec_length,
    }
    if config.get("eikonal_weight", 0) > 0:
        log_dict["eikonal_loss"] = save_eikonal_loss
    for l1_idx, l1_loss_ in enumerate(save_l1_losses):
        log_dict["l1_loss_{}".format(l1_idx)] = l1_loss_
    for k, v in step_cls_losses.items():
        log_dict[k] = v / n_batches

    if config["log_latent"] is not None:
        vecs = latent_vecs.weight.data.cpu().numpy()
        for latent_idx in range(config["log_latent"]):
            latent_values = vecs[:, latent_idx]
            log_dict[f"latent_{latent_idx}_mean"] = float(latent_values.mean())
            log_dict[f"latent_{latent_idx}_std"] = float(latent_values.std())
            log_dict[f"latent_{latent_idx}_min"] = float(latent_values.min())
            log_dict[f"latent_{latent_idx}_max"] = float(latent_values.max())

    return log_dict