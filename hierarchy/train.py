# train_model_hierarchy.py
# =======================
# Training script for NSM with hierarchy-aware loss.
#
# Extends base NSM training (train_model.py) with:
#   1. Hierarchical contrastive loss on latent codes — encourages
#      taxonomically similar specimens to cluster in latent space.
#   2. Classification heads predicting species, genus, family, and
#      spinal position from latent codes during training.
#
# Usage:
#   conda activate NSM
#   python train_model_hierarchy.py
#
# The resulting latent space encodes both shape reconstruction quality
# and taxonomic structure, improving downstream classification of
# species, genus, and spinal position.
# =======================

import argparse
import sys
import torch
import numpy as np
import json
import os
import random
import time
import csv
import itertools
import warnings

from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
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
)
from NSM.losses import eikonal_loss
from NSM.train.utils import (
    get_kld,
    cyclic_anneal_linear,
    calc_weight,
    add_plain_lr_to_config,
    get_profiler,
)
from .losses import (
    TaxonomyLabelEncoder,
    HierarchyContrastiveLoss,
    TaxonomyClassificationHeads,
    compute_classification_head_loss,
)

# --- Monkey-patch for type conversion (same as train_model.py) ---
import pymskt.mesh.meshTools as meshTools
old_sdf_fn = meshTools.pcu.signed_distance_to_mesh
def new_sdf_fn(pts, points, faces):
    pts = pts.astype(np.float64)
    points = points.astype(np.float64)
    return old_sdf_fn(pts, points, faces)
meshTools.pcu.signed_distance_to_mesh = new_sdf_fn

loss_l1 = torch.nn.L1Loss(reduction="none")


def _pick_next_run_name(base: str = "hierarchy_v") -> str:
    """Return `hierarchy_v<N>` where N is one higher than the largest
    existing `hierarchy_v<int>` directory in CWD.

    Falls back to a timestamped name `hierarchy_v<YYYYMMDD_HHMMSS>` if the
    auto-increment scan fails for any reason (unreadable CWD, race against
    another process, unexpected entries, etc.) — guarantees we never
    silently overwrite a previous run's outputs."""
    try:
        nums = []
        for entry in os.listdir('.'):
            if os.path.isdir(entry) and entry.startswith(base):
                suffix = entry[len(base):]
                if suffix.isdigit():
                    nums.append(int(suffix))
        next_n = max(nums) + 1 if nums else 1
        candidate = f"{base}{next_n}"
        if os.path.exists(candidate):
            raise RuntimeError(f"selected name '{candidate}' raced with another process")
        return candidate
    except Exception as e:
        fallback = f"{base}{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"WARNING: auto-increment naming failed ({e}); using timestamped fallback '{fallback}'")
        return fallback


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train NSM with hierarchy-aware loss. CLI args override "
                    "values in vertebrae_config.json."
    )
    parser.add_argument('--run-name', type=str, default=None,
                        help='Override the auto-incremented hierarchy_v<N> run directory.')
    # Hierarchy contrastive loss
    parser.add_argument('--contrastive-weight', type=float, default=None,
                        help='Weight on the hierarchy contrastive loss (default 0.01).')
    parser.add_argument('--contrastive-warmup', type=int, default=None,
                        help='Warmup epochs before contrastive loss reaches full weight (default 200).')
    # Classification heads
    parser.add_argument('--head-weight', type=float, default=None,
                        help='Weight on the classification-head loss (default 0.005).')
    parser.add_argument('--head-warmup', type=int, default=None,
                        help='Warmup epochs for classification head loss (default 100).')
    parser.add_argument('--head-hidden-dim', type=int, default=None,
                        help='Hidden dim of the classification heads (default 256).')
    # Per-level weights inside classification head loss
    parser.add_argument('--species-weight', type=float, default=None,
                        help='Weight on species-level head loss (default 1.0).')
    parser.add_argument('--genus-weight', type=float, default=None,
                        help='Weight on genus-level head loss (default 0.5).')
    parser.add_argument('--family-weight', type=float, default=None,
                        help='Weight on family-level head loss (default 0.25).')
    parser.add_argument('--position-weight', type=float, default=None,
                        help='Weight on spinal-position head loss (default 0.75).')
    # Disable switches
    parser.add_argument('--no-contrastive', action='store_true',
                        help='Disable hierarchy contrastive loss entirely.')
    parser.add_argument('--no-heads', action='store_true',
                        help='Disable classification heads entirely.')
    return parser


def main():

    args = _build_arg_parser().parse_args()

    # ======================================================================
    # Configuration
    # ======================================================================
    CACHE = True
    USE_WANDB = False
    PROJECT_NAME = 'Classification_Hierarchy'  # TO DO: change name
    ENTITY_NAME = 'GATECH'
    RUN_NAME = args.run_name if args.run_name else _pick_next_run_name()
    print(f"Training run directory: {RUN_NAME}")
    LOC_SDF_CACHE = 'cache'
    LOC_SAVE_NEW_MODELS = RUN_NAME

    if (USE_WANDB is True) and ('WANDB_KEY' not in os.environ):
        raise ValueError('WANDB_KEY is not in the environment variables. Please set it or set USE_WANDB to False.')

    if CACHE is True:
        if not os.path.exists(LOC_SDF_CACHE):
            os.makedirs(LOC_SDF_CACHE)
        LOC_SDF_CACHE = os.path.abspath(LOC_SDF_CACHE)
        os.environ['LOC_SDF_CACHE'] = LOC_SDF_CACHE

    path_config = 'vertebrae_config.json'
    with open(path_config, 'r') as f:
        config = json.load(f)

    # ======================================================================
    # Device detection — CUDA > MPS > CPU
    # ======================================================================
    if torch.cuda.is_available():
        config['device'] = 'cuda:0'
        print("Using CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        config['device'] = 'mps'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("Using Apple MPS (M1/M2/M3 GPU)")
    else:
        config['device'] = 'cpu'
        print("Using CPU")

    if USE_WANDB is True:
        config['project_name'] = PROJECT_NAME
        config['entity_name'] = ENTITY_NAME
        config['entity'] = ENTITY_NAME
        config['run_name'] = RUN_NAME

    config['experiment_directory'] = os.path.abspath(LOC_SAVE_NEW_MODELS)

    slurm_id = os.environ.get('SLURM_JOB_ID', '')
    sentinel = f'.last_train_run.{slurm_id}' if slurm_id else '.last_train_run'
    with open(sentinel, 'w') as f:
        f.write(RUN_NAME)

    # --- Hierarchy loss configuration (CLI > config file > hardcoded default) ---
    def _override(key, cli_val, default):
        if cli_val is not None:
            config[key] = cli_val
        else:
            config.setdefault(key, default)

    config['hierarchy_loss_enabled'] = not args.no_contrastive
    config['classification_heads_enabled'] = not args.no_heads
    _override('hierarchy_contrastive_weight', args.contrastive_weight, 0.01)
    _override('hierarchy_contrastive_warmup', args.contrastive_warmup, 200)
    config.setdefault('hierarchy_contrastive_margins', {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0})
    _override('classification_head_weight', args.head_weight, 0.005)
    _override('classification_head_warmup', args.head_warmup, 100)
    _override('classification_head_hidden_dim', args.head_hidden_dim, 256)

    default_level_weights = {'species': 1.0, 'genus': 0.5, 'family': 0.25, 'position': 0.75}
    config.setdefault('classification_level_weights', default_level_weights)
    for lvl, cli_val in [('species', args.species_weight), ('genus', args.genus_weight),
                         ('family', args.family_weight), ('position', args.position_weight)]:
        if cli_val is not None:
            config['classification_level_weights'][lvl] = cli_val

    print("Hierarchy loss config:")
    print(f"  contrastive: enabled={config['hierarchy_loss_enabled']}, "
          f"weight={config['hierarchy_contrastive_weight']}, "
          f"warmup={config['hierarchy_contrastive_warmup']}")
    print(f"  heads:       enabled={config['classification_heads_enabled']}, "
          f"weight={config['classification_head_weight']}, "
          f"warmup={config['classification_head_warmup']}, "
          f"hidden_dim={config['classification_head_hidden_dim']}")
    print(f"  level weights: {config['classification_level_weights']}")

    # Create the run directory and write the focused hyperparams manifest
    # *before* heavy work (dataset loading, model build), so even an early
    # crash leaves a record of what was attempted at this directory.
    os.makedirs(config['experiment_directory'], exist_ok=True)
    hyperparams_manifest = {
        'run_name': RUN_NAME,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'cli_args': sys.argv[1:],
        'hierarchy_loss_enabled': config['hierarchy_loss_enabled'],
        'hierarchy_contrastive_weight': config['hierarchy_contrastive_weight'],
        'hierarchy_contrastive_warmup': config['hierarchy_contrastive_warmup'],
        'hierarchy_contrastive_margins': config['hierarchy_contrastive_margins'],
        'classification_heads_enabled': config['classification_heads_enabled'],
        'classification_head_weight': config['classification_head_weight'],
        'classification_head_warmup': config['classification_head_warmup'],
        'classification_head_hidden_dim': config['classification_head_hidden_dim'],
        'classification_level_weights': config['classification_level_weights'],
    }
    hp_path = os.path.join(config['experiment_directory'], 'hyperparams.json')
    with open(hp_path, 'w') as f:
        json.dump(hyperparams_manifest, f, indent=2, default=str)
    print(f"Wrote hyperparams manifest: {hp_path}")

    # ======================================================================
    # Data loading (same as train_model.py)
    # ======================================================================
    folder_vtk = os.path.abspath('vertebrae_meshes')  # TO DO: change path
    all_vtk_files = [os.path.join(folder_vtk, f) for f in os.listdir(folder_vtk) if f.lower().endswith('.vtk')]

    total_files = len(all_vtk_files)
    N_TRAIN = int(0.8 * total_files)  # TO DO
    N_TEST = int(0.15 * total_files)  # TO DO
    N_VAL = total_files - N_TRAIN - N_TEST

    random.seed(42)
    random.shuffle(all_vtk_files)

    if len(all_vtk_files) < N_TRAIN + N_VAL + N_TEST:
        raise ValueError("Not enough .vtk files in vertebrae_meshes folder.")
    list_mesh_paths = sorted(all_vtk_files[:N_TRAIN])
    list_val_paths = sorted(all_vtk_files[N_TRAIN:N_TRAIN + N_VAL])
    list_test_paths = sorted(all_vtk_files[N_TRAIN + N_VAL:])

    config['test_paths'] = list_test_paths
    config['val_paths'] = list_val_paths
    config['list_mesh_paths'] = list_mesh_paths

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # ======================================================================
    # Build SDF dataset
    # ======================================================================
    sdf_dataset = SDFSamples(
        list_mesh_paths=list_mesh_paths,
        subsample=config["samples_per_object_per_batch"],
        print_filename=True,
        n_pts=config["n_pts_per_object"],
        p_near_surface=config['percent_near_surface'],
        p_further_from_surface=config['percent_further_from_surface'],
        sigma_near=config['sigma_near'],
        sigma_far=config['sigma_far'],
        rand_function=config['random_function'],
        center_pts=config['center_pts'],
        norm_pts=config['normalize_pts'],
        scale_method=config['scale_method'],
        scale_jointly=config['scale_jointly'],
        random_seed=config['seed'],
        reference_mesh=None,
        verbose=config['verbose'],
        save_cache=config['cache'],
        equal_pos_neg=config['equal_pos_neg'],
        fix_mesh=config['fix_mesh'],
        load_cache=config['load_cache'],
        store_data_in_memory=config['store_data_in_memory'],
        multiprocessing=config['multiprocessing'],
        n_processes=config['n_processes'],
    )
    print('sdf_dataset:', sdf_dataset)
    print('len sdf_dataset:', len(sdf_dataset))

    # ======================================================================
    # Build taxonomy encoder
    # ======================================================================
    taxonomy_encoder = TaxonomyLabelEncoder(list_mesh_paths)
    taxonomy_info = taxonomy_encoder.get_taxonomy_info()

    print("\nTaxonomy summary:")
    for level in taxonomy_encoder.levels:
        n_cls = taxonomy_encoder.num_classes(level)
        n_valid = (taxonomy_encoder.labels[level] >= 0).sum()
        print(f"  {level}: {n_cls} classes, {n_valid}/{taxonomy_encoder.n_objects} labeled")

    # ======================================================================
    # Build models
    # ======================================================================
    triplane_args = {
        'latent_dim': config['latent_size'],
        'n_objects': config['objects_per_decoder'],
        'conv_hidden_dims': config['conv_hidden_dims'],
        'conv_deep_image_size': config['conv_deep_image_size'],
        'conv_norm': config['conv_norm'],
        'conv_norm_type': config['conv_norm_type'],
        'conv_start_with_mlp': config['conv_start_with_mlp'],
        'sdf_latent_size': config['sdf_latent_size'],
        'sdf_hidden_dims': config['sdf_hidden_dims'],
        'sdf_weight_norm': config['weight_norm'],
        'sdf_final_activation': config['final_activation'],
        'sdf_activation': config['activation'],
        'sdf_dropout_prob': config['dropout_prob'],
        'sum_sdf_features': config['sum_conv_output_features'],
        'conv_pred_sdf': config['conv_pred_sdf'],
    }

    model = TriplanarDecoder(**triplane_args)

    classification_heads = None
    if config['classification_heads_enabled']:
        classification_heads = TaxonomyClassificationHeads(
            latent_dim=config['latent_size'],
            num_species=taxonomy_encoder.num_classes('species'),
            num_genera=taxonomy_encoder.num_classes('genus'),
            num_families=taxonomy_encoder.num_classes('family'),
            num_positions=taxonomy_encoder.num_classes('position'),
            hidden_dim=config['classification_head_hidden_dim'],
        )

    hierarchy_contrastive = None
    if config['hierarchy_loss_enabled']:
        hierarchy_contrastive = HierarchyContrastiveLoss(
            margins=config['hierarchy_contrastive_margins'],
        )


    # ======================================================================
    # Training functions
    # ======================================================================

    def train_deep_sdf_hierarchy(
        config, model, sdf_dataset, taxonomy_encoder, taxonomy_info,
        classification_heads=None, hierarchy_contrastive=None, use_wandb=False,
    ):
        """
        Extended training loop that adds hierarchy-aware losses to standard
        NSM SDF training. Structure mirrors NSM/train/train_deep_sdf.py with
        the addition of contrastive and classification head losses.
        """
        config.setdefault("objects_per_decoder", 1)
        config.setdefault("resume_epoch", 0)
        config.setdefault("scale_jointly", False)
        config.setdefault("fix_mesh_recon", False)
        config.setdefault("log_latent", None)

        config = add_plain_lr_to_config(config)
        config["checkpoints"] = get_checkpoints(config)
        config["lr_schedules"] = get_learning_rate_schedules(config)

        if "resume_epoch" not in config:
            config["resume_epoch"] = 0

        model = model.to(config["device"])

        if classification_heads is not None:
            classification_heads = classification_heads.to(config["device"])

        # CSV logging setup
        if use_wandb is True:
            import wandb
            wandb.login(key=os.environ["WANDB_KEY"])
            wandb.init(
                project=config["project_name"],
                entity=config["entity_name"],
                config=config,
                name=config["run_name"],
                tags=config.get("tags", []),
            )
            wandb.watch(model, log="all")
        else:
            log_fpath = os.path.join(config['experiment_directory'], 'train_log.csv')
            os.makedirs(config['experiment_directory'], exist_ok=True)

        use_pin_memory = ("cuda" in str(config["device"]))
        data_loader = torch.utils.data.DataLoader(
            sdf_dataset,
            batch_size=config["objects_per_batch"],
            shuffle=True,
            num_workers=config["num_data_loader_threads"],
            drop_last=False,
            prefetch_factor=config["prefetch_factor"],
            pin_memory=use_pin_memory,
        )

        latent_vecs = get_latent_vecs(len(data_loader.dataset), config).to(config["device"])

        # Build optimizer with classification heads as third param group
        # The base get_optimizer creates groups: [latent (idx 0), model (idx 1)]
        # We add classification heads as idx 2 (uses same LR schedule as model)
        optimizer = get_optimizer(
            model, latent_vecs,
            lr_schedules=config["lr_schedules"],
            optimizer=config["optimizer"],
            weight_decay=config["weight_decay"],
        )

        if classification_heads is not None:
            optimizer.add_param_group({
                'params': classification_heads.parameters(),
                'lr': config["lr_schedules"][0].get_learning_rate(0),
            })
            # Extend lr_schedules so adjust_learning_rate handles the 3rd group
            config["lr_schedules"].append(config["lr_schedules"][0])

        # Save full config and data splits to experiment directory for reproducibility
        os.makedirs(config['experiment_directory'], exist_ok=True)

        config_save = {k: v for k, v in config.items()
                       if not isinstance(v, (torch.nn.Module, torch.optim.Optimizer))}
        config_save_path = os.path.join(config['experiment_directory'], 'config.json')
        try:
            with open(config_save_path, 'w') as f:
                json.dump(config_save, f, indent=2, default=str)
        except TypeError:
            # Fall back: skip non-serializable values
            serializable = {}
            for k, v in config_save.items():
                try:
                    json.dumps(v, default=str)
                    serializable[k] = v
                except (TypeError, ValueError):
                    serializable[k] = str(v)
            with open(config_save_path, 'w') as f:
                json.dump(serializable, f, indent=2, default=str)

        splits_path = os.path.join(config['experiment_directory'], 'data_splits.json')
        with open(splits_path, 'w') as f:
            json.dump({
                'train_paths': config.get('list_mesh_paths', []),
                'val_paths': config.get('val_paths', []),
                'test_paths': config.get('test_paths', []),
                'n_train': len(config.get('list_mesh_paths', [])),
                'n_val': len(config.get('val_paths', [])),
                'n_test': len(config.get('test_paths', [])),
            }, f, indent=2)

        if config["resume_epoch"] > 1:
            print("Loading model, optimizer, and latent states from epoch", config["resume_epoch"])
            map_loc = config["device"]
            model_ckpt = torch.load(
                os.path.join(config["experiment_directory"], "model", f'{config["resume_epoch"]}.pth'),
                map_location=map_loc,
            )
            model.load_state_dict(model_ckpt["model"])
            optimizer.load_state_dict(model_ckpt["optimizer"])

            latent_ckpt = torch.load(
                os.path.join(config["experiment_directory"], "latent_codes", f'{config["resume_epoch"]}.pth'),
                map_location=map_loc,
            )
            latent_vecs.load_state_dict(latent_ckpt["latent_codes"])

            # Resume classification heads if available
            heads_path = os.path.join(
                config["experiment_directory"], "classification_heads", f'{config["resume_epoch"]}.pth'
            )
            if classification_heads is not None and os.path.exists(heads_path):
                heads_ckpt = torch.load(heads_path, map_location=map_loc)
                classification_heads.load_state_dict(heads_ckpt["classification_heads"])
                print("Loaded classification heads from checkpoint")

        with get_profiler(config) as profiler:
            for epoch in range(config["resume_epoch"] + 1, config["n_epochs"] + 1):
                print(f'\033[92m\n\n\nEpoch: {epoch}\033[0m')

                log_dict = train_epoch_hierarchy(
                    model, data_loader, latent_vecs,
                    optimizer=optimizer, config=config, epoch=epoch,
                    taxonomy_encoder=taxonomy_encoder,
                    classification_heads=classification_heads,
                    hierarchy_contrastive=hierarchy_contrastive,
                    return_loss=True,
                    n_surfaces=config["objects_per_decoder"],
                )

                checkpoint_epoch = (
                    epoch in config["checkpoints"] or epoch % config["save_frequency"] == 0
                )

                if checkpoint_epoch:
                    print("\nCheckpoint epoch...")
                    save_model_params(config=config, list_mesh_paths=sdf_dataset.list_mesh_paths)
                    save_latent_vectors(config=config, epoch=epoch, latent_vec=latent_vecs)
                    save_model(config=config, epoch=epoch, decoder=model, optimizer=optimizer)

                    # Save classification heads
                    if classification_heads is not None:
                        heads_dir = os.path.join(config['experiment_directory'], 'classification_heads')
                        os.makedirs(heads_dir, exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'classification_heads': classification_heads.state_dict(),
                        }, os.path.join(heads_dir, f'{epoch}.pth'))

                    # Save taxonomy info
                    tax_dir = os.path.join(config['experiment_directory'], 'taxonomy')
                    os.makedirs(tax_dir, exist_ok=True)
                    with open(os.path.join(tax_dir, 'taxonomy_info.json'), 'w') as f:
                        json.dump(taxonomy_info, f, indent=2)

                if use_wandb is True:
                    import wandb
                    wandb.log(log_dict, step=epoch - 1)
                elif checkpoint_epoch:
                    write_header = not os.path.exists(log_fpath)
                    with open(log_fpath, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
                        if write_header:
                            writer.writeheader()
                        writer.writerow(log_dict)

                profiler.step()
                clear_gpu_cache(config["device"])


    def train_epoch_hierarchy(
        model, data_loader, latent_vecs, optimizer, config, epoch,
        taxonomy_encoder, classification_heads=None, hierarchy_contrastive=None,
        return_loss=True, verbose=False, n_surfaces=1,
    ):
        """
        Single training epoch with hierarchy-aware losses.
        Extends the base train_epoch from NSM/train/train_deep_sdf.py
        with contrastive and classification head losses.
        """
        start = time.time()
        model.train()
        if classification_heads is not None:
            classification_heads.train()

        if not ("schedule_free" in config["optimizer"]):
            adjust_learning_rate(config["lr_schedules"], optimizer, epoch)
        else:
            optimizer.train()

        # Accumulators
        step_losses = 0
        step_l1_loss = 0
        step_code_reg_loss = 0
        step_eikonal_loss = 0
        step_l1_losses = [0.0 for _ in range(n_surfaces)]
        step_mean_vec_length = 0
        step_std_vec_length = 0
        step_hierarchy_loss = 0
        step_cls_loss = 0
        step_cls_losses = {}

        step_mean_size = 0
        step_mean_load_time = 0
        step_mean_load_rate = 0
        step_whole_load_time = 0

        for sdf_data, indices in data_loader:
            if config["verbose"] is True:
                print("sdf index size:", indices.size())

            xyz = sdf_data["xyz"].float().to(config["device"])
            xyz = xyz.reshape(-1, 3)
            num_sdf_samples = xyz.shape[0]
            xyz.requires_grad = False

            indices = indices.to(config["device"])

            # Keep original per-object indices for hierarchy losses
            original_object_indices = indices.clone()

            # Build SDF ground truth
            sdf_gt = []
            if n_surfaces == 1:
                sdf_gt_ = sdf_data["gt_sdf"].float().reshape(-1, 1)
                if config["enforce_minmax"] is True:
                    sdf_gt_ = torch.clamp(sdf_gt_, -config["clamp_dist"], config["clamp_dist"])
                sdf_gt_.requires_grad = False
                sdf_gt.append(sdf_gt_)
            else:
                for surf_idx in range(n_surfaces):
                    sdf_gt_ = sdf_data["gt_sdf"][:, :, surf_idx].float().reshape(-1, 1)
                    if config["enforce_minmax"] is True:
                        sdf_gt_ = torch.clamp(sdf_gt_, -config["clamp_dist"], config["clamp_dist"])
                    sdf_gt_.requires_grad = False
                    sdf_gt.append(sdf_gt_)

            # Chunk for memory efficiency
            xyz = torch.chunk(xyz, config["batch_split"])
            indices_expanded = torch.chunk(
                indices.unsqueeze(-1)
                .repeat(1, config["samples_per_object_per_batch"])
                .view(-1),
                config["batch_split"],
            )
            for surf_idx in range(n_surfaces):
                sdf_gt[surf_idx] = torch.chunk(sdf_gt[surf_idx], config["batch_split"])

            batch_loss = 0.0
            batch_l1_loss = 0.0
            batch_l1_losses = [0.0 for _ in range(n_surfaces)]
            batch_code_reg_loss = 0.0
            batch_eikonal_loss = 0.0
            batch_hierarchy_loss = 0.0
            batch_cls_loss = 0.0
            batch_cls_losses = {}

            optimizer.zero_grad()

            for split_idx in range(config["batch_split"]):
                batch_vecs = latent_vecs(indices_expanded[split_idx])

                if "variational" in config and config["variational"] is True:
                    mu = batch_vecs[:, :config["latent_size"]]
                    logvar = batch_vecs[:, config["latent_size"]:]
                    std = torch.exp(0.5 * logvar)
                    err = torch.randn_like(std)
                    batch_vecs = std * err + mu

                inputs = torch.cat([batch_vecs, xyz[split_idx]], dim=1)
                pred_sdf = model(inputs, epoch=epoch)

                if n_surfaces == 1:
                    if pred_sdf.dim() == 2 and pred_sdf.shape[1] == 1:
                        pass
                    else:
                        pred_sdf = pred_sdf.unsqueeze(1)

                if config["enforce_minmax"] is True:
                    pred_sdf = torch.clamp(pred_sdf, -config["clamp_dist"], config["clamp_dist"])

                # --- L1 reconstruction loss (unchanged) ---
                l1_losses = []
                for surf_idx in range(n_surfaces):
                    l1_losses.append(
                        loss_l1(
                            pred_sdf[:, surf_idx],
                            sdf_gt[surf_idx][split_idx].squeeze(1).to(config["device"]),
                        )
                    )

                # Curriculum SDF: surface accuracy
                if config["surface_accuracy_e"] is not None:
                    weight_schedule = 1 - calc_weight(
                        epoch, config["n_epochs"],
                        config["surface_accuracy_schedule"],
                        config["surface_accuracy_cooldown"],
                    )
                    for l1_idx, l1_loss in enumerate(l1_losses):
                        l1_losses[l1_idx] = torch.maximum(
                            l1_loss - (weight_schedule * config["surface_accuracy_e"]),
                            torch.zeros_like(l1_loss),
                        )

                # Curriculum SDF: sample difficulty weighting
                if config["sample_difficulty_weight"] is not None:
                    weight_schedule = calc_weight(
                        epoch, config["n_epochs"],
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
                    weights = [w / weights_sum * weights_total for w in config["surface_weighting"]]
                else:
                    weights = [1] * n_surfaces

                for l1_idx, l1_loss_ in enumerate(l1_losses):
                    l1_loss += l1_loss_.sum() * weights[l1_idx]
                l1_loss = l1_loss / len(l1_losses)

                batch_l1_loss += l1_loss.item()
                for l1_idx, l1_loss_ in enumerate(l1_losses):
                    batch_l1_losses[l1_idx] += l1_loss_.sum().item()
                chunk_loss = l1_loss

                # --- Eikonal loss (unchanged) ---
                if config.get("eikonal_weight", 0) > 0:
                    xyz_grad = xyz[split_idx].detach().requires_grad_(True)
                    inputs_grad = torch.cat([batch_vecs, xyz_grad], dim=1)
                    pred_sdf_grad = model(inputs_grad, epoch=epoch)
                    eik_loss = eikonal_loss(pred_sdf_grad, xyz_grad, reduction="mean")
                    batch_eikonal_loss += eik_loss.item()
                    chunk_loss = chunk_loss + config["eikonal_weight"] * eik_loss

                # --- Code regularization (unchanged) ---
                if config["code_regularization"] is True:
                    if "variational" in config and config["variational"] is True:
                        kld = torch.mean(
                            -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0
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

                # ========== NEW: Hierarchy contrastive loss ==========
                if config.get('hierarchy_loss_enabled', False) and hierarchy_contrastive is not None:
                    per_object_vecs = latent_vecs(original_object_indices)
                    warmup = min(1.0, epoch / config['hierarchy_contrastive_warmup'])
                    h_loss = hierarchy_contrastive(
                        per_object_vecs,
                        original_object_indices,
                        taxonomy_encoder,
                        device=config['device'],
                    )
                    weighted_h_loss = config['hierarchy_contrastive_weight'] * warmup * h_loss
                    chunk_loss = chunk_loss + weighted_h_loss
                    batch_hierarchy_loss += weighted_h_loss.item()

                # ========== NEW: Classification head loss ==========
                if config.get('classification_heads_enabled', False) and classification_heads is not None:
                    per_object_vecs_cls = latent_vecs(original_object_indices)
                    logits = classification_heads(per_object_vecs_cls)
                    warmup = min(1.0, epoch / config['classification_head_warmup'])
                    cls_loss, cls_loss_dict = compute_classification_head_loss(
                        logits, original_object_indices, taxonomy_encoder,
                        level_weights=config['classification_level_weights'],
                        device=config['device'],
                    )
                    weighted_cls_loss = config['classification_head_weight'] * warmup * cls_loss
                    chunk_loss = chunk_loss + weighted_cls_loss
                    batch_cls_loss += weighted_cls_loss.item()
                    for k, v in cls_loss_dict.items():
                        batch_cls_losses[k] = batch_cls_losses.get(k, 0) + v

                mean_vec_length = torch.mean(torch.norm(batch_vecs, dim=1))
                std_vec_length = torch.std(torch.norm(batch_vecs, dim=1))

                chunk_loss.backward()
                batch_loss += chunk_loss.item()

            # Accumulate step losses
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

        end = time.time()
        seconds_elapsed = end - start
        n_batches = len(data_loader)

        save_loss = step_losses / n_batches
        save_l1_loss = step_l1_loss / n_batches
        save_code_reg_loss = step_code_reg_loss / n_batches
        save_eikonal_loss = step_eikonal_loss / n_batches
        save_hierarchy_loss = step_hierarchy_loss / n_batches
        save_cls_loss = step_cls_loss / n_batches
        save_l1_losses = [l / n_batches for l in step_l1_losses]
        save_mean_vec_length = step_mean_vec_length / n_batches
        save_std_vec_length = step_std_vec_length / n_batches

        print("save loss: ", save_loss)
        print("\t save l1 loss: ", save_l1_loss)
        print("\t save code loss: ", save_code_reg_loss)
        if config.get("eikonal_weight", 0) > 0:
            print(f"\t save eikonal loss: {save_eikonal_loss:.6f}")
        print(f"\t save hierarchy contrastive loss: {save_hierarchy_loss:.6f}")
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
            "hierarchy_contrastive_loss": save_hierarchy_loss,
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
            log_dict[f"l1_loss_{l1_idx}"] = l1_loss_
        for k, v in step_cls_losses.items():
            log_dict[k] = v / n_batches

        if config["log_latent"] is not None:
            vecs = latent_vecs.weight.data.cpu().numpy()
            for latent_idx in range(config["log_latent"]):
                latent_values = vecs[:, latent_idx]
                log_dict[f'latent_{latent_idx}_mean'] = float(latent_values.mean())
                log_dict[f'latent_{latent_idx}_std'] = float(latent_values.std())
                log_dict[f'latent_{latent_idx}_min'] = float(latent_values.min())
                log_dict[f'latent_{latent_idx}_max'] = float(latent_values.max())

        return log_dict


    # ======================================================================
    # Run training
    # ======================================================================
    train_deep_sdf_hierarchy(
        config=config,
        model=model,
        sdf_dataset=sdf_dataset,
        taxonomy_encoder=taxonomy_encoder,
        taxonomy_info=taxonomy_info,
        classification_heads=classification_heads,
        hierarchy_contrastive=hierarchy_contrastive,
        use_wandb=USE_WANDB,
    )


if __name__ == "__main__":
    main()
