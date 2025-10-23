import torch
import numpy as np
import json
import os
import random

from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.train.train_deep_sdf import train_deep_sdf as train_deep_sdf

# --- Begin monkey-patch for type conversion ---
import pymskt.mesh.meshTools as meshTools ## check this import
old_sdf_fn = meshTools.pcu.signed_distance_to_mesh 
def new_sdf_fn(pts, points, faces):
    pts = pts.astype(np.float64)
    points = points.astype(np.float64)
    return old_sdf_fn(pts, points, faces)
meshTools.pcu.signed_distance_to_mesh = new_sdf_fn
# --- End monkey-patch for type conversion ---

CACHE = True
USE_WANDB = False
PROJECT_NAME = 'Vertebrae' # TO DO: change name
ENTITY_NAME = 'UF'
RUN_NAME = 'run_v32' # TO DO: update run name
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
print('len sdf_dataaset', len(sdf_dataset))

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

