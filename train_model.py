"""
Train a Neural Shape Model (NSM) on vertebrae VTK meshes with selectable loss modes.
Loads vertebrae meshes from a fixed folder, splits them 80/15/5 into
train/test/val, builds an SDF dataset, constructs a TriplanarDecoder model,
and launches one of:
  - standard DeepSDF training
  - contrastive DeepSDF training
  - hierarchy-aware DeepSDF training
Usage:
    python train_model.py --run_name my_experiment
    python train_model.py --run_name my_experiment --contrastive_loss
    python train_model.py --run_name my_experiment --hierarchy_loss
Arguments:
    --run_name TEXT
        Name for this training run. Controls where model checkpoints,
        latent codes, and config snapshots are saved. Default: run_v57a
    --contrastive_loss
        Enable contrastive loss training loop.
    --hierarchy_loss
        Enable hierarchy-aware loss training loop.
        NOTE: --contrastive_loss and --hierarchy_loss are mutually exclusive.
Configuration:
    Training hyperparameters (latent size, learning rate, batch size, etc.)
    are read from vertebrae_config.json in the current directory.
    Contrastive mode:
      Define "contrastive_weight" in config (recommended 0.01).
      If missing/0 while enabled, it is set to 0.01 with a warning.
    Hierarchy mode:
      Define "hierarchy_weight" in config (recommended 0.01).
      Optional config keys:
        - "hierarchy_warmup" (default 200)
        - "hierarchy_margins" (default {0:0.0, 1:1.0, 2:2.0, 3:4.0})
      If hierarchy_weight is missing/0 while enabled, it is set to 0.01 with a warning.
    After training, a copy of the resolved config is saved to
    {run_name}/model_params_config.json for downstream scripts.
Data:
    VTK meshes are loaded from ./vertebrae_meshes/*.vtk.
    SDF point samples are cached to ./nsm_sdf_cache/{run_name}/ so that
    subsequent runs with the same data skip expensive SDF computation.
    Set load_cache: true in vertebrae_config.json to reuse cached samples.
Output:
    {run_name}/model/                - Model checkpoints (.pth)
    {run_name}/latent_codes/         - Latent code tensors (.pth)
    {run_name}/model_params_config.json - Resolved config snapshot
    ./nsm_sdf_cache/{run_name}/      - SDF sample cache (NPZ files)
Notes:
    Set USE_WANDB = True and export WANDB_KEY to enable Weights & Biases logging.
    Random seed is fixed (seed from config) for reproducible train/test/val split.
    Monkey-patches pymskt signed_distance_to_mesh to enforce float64 inputs.
"""

import torch
import numpy as np
import json
import os
import random
import argparse

from NSM.datasets import SDFSamples, MultiSurfaceSDFSamples
from NSM.models import TriplanarDecoder

# --- Begin monkey-patch for type conversion ---
import pymskt.mesh.meshTools as meshTools ## check this import
old_sdf_fn = meshTools.pcu.signed_distance_to_mesh 
def new_sdf_fn(pts, points, faces):
    pts = pts.astype(np.float64)
    points = points.astype(np.float64)
    return old_sdf_fn(pts, points, faces)
meshTools.pcu.signed_distance_to_mesh = new_sdf_fn
# --- End monkey-patch for type conversion ---

path_config = 'vertebrae_config.json'
with open(path_config, 'r') as f:
    config = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='run_v57a', help='Run name used for saving model and SDF cache')
loss_mode = parser.add_mutually_exclusive_group()
loss_mode.add_argument("--contrastive_loss", action="store_true", help="Enable contrastive loss training loop.")
loss_mode.add_argument("--hierarchy_loss", action="store_true", help="Enable hierarchy-aware loss training loop.")
args = parser.parse_args()

# Set loss mode, if using
if args.hierarchy_loss:
    from NSM.train.train_deep_sdf_hierarchy import train_deep_sdf
    if config.get("hierarchy_weight", 0) == 0:
        print("WARNING: hierarchy_loss enabled but hierarchy_weight=0. Setting to 0.01")
        config["hierarchy_weight"] = 0.01
    # Config-driven params, like contrastive
    config.setdefault("hierarchy_warmup", 200)
    config.setdefault("hierarchy_margins", {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0})
    config["use_hierarchy_loss"] = True
    config["use_contrastive_loss"] = False
    print("Using hierarchy-aware DeepSDF training.")
elif args.contrastive_loss:
    from NSM.train.train_deep_sdf_contrastive import train_deep_sdf
    if config.get("contrastive_weight", 0) == 0:
        print("WARNING: contrastive_loss enabled but contrastive_weight=0. Setting to 0.01")
        config["contrastive_weight"] = 0.01
    config["use_contrastive_loss"] = True
    config["use_hierarchy_loss"] = False
    print("Using contrastive DeepSDF training.")
else:
    from NSM.train.train_deep_sdf import train_deep_sdf
    config["contrastive_weight"] = 0
    config["hierarchy_weight"] = 0
    config["use_contrastive_loss"] = False
    config["use_hierarchy_loss"] = False
    print("Using standard DeepSDF training.")

CACHE = True
USE_WANDB = False
PROJECT_NAME = 'Vertebrae' 
ENTITY_NAME = 'UF'
RUN_NAME = args.run_name
LOC_SDF_CACHE = f'./nsm_sdf_cache/{RUN_NAME}'
LOC_SAVE_NEW_MODELS = RUN_NAME

if (USE_WANDB is True) and ('WANDB_KEY' not in os.environ):
    raise ValueError('WANDB_KEY is not in the environment variables. Please set it or set USE_WANDB to False.')

if CACHE is True:
    if not os.path.exists(LOC_SDF_CACHE):
        os.makedirs(LOC_SDF_CACHE)
    LOC_SDF_CACHE = os.path.abspath(LOC_SDF_CACHE)
    os.environ['LOC_SDF_CACHE'] = LOC_SDF_CACHE

if USE_WANDB is True:
    config['project_name'] = PROJECT_NAME
    config['entity_name'] = ENTITY_NAME
    config['entity'] = ENTITY_NAME
    config['run_name'] = RUN_NAME

config['experiment_directory'] = os.path.abspath(LOC_SAVE_NEW_MODELS)

# Get vertebrae mesh paths
folder_vtk = os.path.abspath('vertebrae_meshes') # TO DO: change path
all_vtk_files = [os.path.join(folder_vtk, f) for f in os.listdir(folder_vtk) if f.lower().endswith('.vtk')]

# Calculate 80/15/5 train/test/val split
total_files = len(all_vtk_files)
N_TRAIN = int(0.8 * total_files) # TO DO
N_TEST = int(0.15 * total_files) # TO DO
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

# Set the seed value!
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

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
    #scale_all_meshes=config['scale_all_meshes'], #
    #center_all_meshes=config['center_all_meshes'], #
    #mesh_to_scale=config['mesh_to_scale'], #
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
print('len sdf_dataset', len(sdf_dataset))

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

train_deep_sdf(
    config=config,
    model=model,
    sdf_dataset=sdf_dataset,
    use_wandb=False,
)

