# Introduction
This code uses generative deep learning models to understand the skeletal anatomy of lizards and some snakes (Squamata). This code is forked and modified from [gattia/NSM](https://github.com/gattia/NSM) following the terms of the [GNU Affero GPL 3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html). See [Original NSM Documentation](http://anthonygattiphd.com/NSM/). 

![Isomap GIF](https://github.com/aubricot/nsm/blob/main/images/isomap_4way_splitscreen_C-T-L_avg.gif)
*Figure 1: Traversing an isomap of the NSM trained model latent space using travelling salesman and k-nearest neighbors. Video animation made using [isomap_video.py](https://github.com/aubricot/nsm/blob/main/isomap_video.py)*

# Installation

```bash
# Create and activate conda environment
conda create -n NSM python=3.10
conda activate NSM

# Install pytorch and dependencies
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge -c defaults

# Install NSM
mkdir NSM
cd NSM
git clone https://github.com/aubricot/nsm.git
cd nsm
python -m pip install -r requirements.txt
pip install .
pip install mskt nibabel pandas scikit-learn scikit-image wandb

```

# Usage

## Training
Update sections of [train_model.py]() commented with # TO DO: to update PROJECT_NAME, ENTITY_NAME, RUN_NAME, folder_vtk, N_TRAIN, N_TEST, N_VAL. These variables point to where your data was collected, where it is saved, and where outputs should go. Adjust model training hyperparameters in [vertebrae_config.json](). See python script and config files for details and save before running using commands below. 
```
conda activate NSM
cd NSM/nsm
python train_model.py
```

## Model Loading

NSM provides a convenient model loader that simplifies loading pre-trained Neural Shape Models. For **trained models**, you'll have:

- `experiment_dir/model_params_config.json` - Configuration saved during training
- `experiment_dir/model/2000.pth` - Model weights at epoch 2000
- `experiment_dir/latent_codes/2000.pth` - Latent codes at epoch 2000

```python
import json, torch
from NSM.models import TriplanarDecoder

# Load config file
with open(config_path, 'r') as f:
     config = json.load(f)

# Get model weights and latent codes
latent_ckpt = torch.load(LC_PATH, map_location=device)
latent_codes = latent_ckpt['latent_codes']['weight'].detach().cpu()

# Build model
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
model_ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(model_ckpt['model'])
device = config.get("device", "cuda:0")
model.to(device)
model.eval()
```

## Create Meshes

After loading a trained model, you can generate meshes from manipulated/new latent vectors. The example below generates the mean mesh shape based on model training data.

```python
from NSM.mesh import create_mesh
import pyvista as pv

# Get the mean of the latent codes
latents_np = latent_codes.numpy()
latent_mean = np.mean(latents_np, axis=0)

# Convert the mean latent code to a pytorch tensor
new_latent = torch.tensor(new_latent_np, dtype=torch.float32).unsqueeze(0).to(device)

# Create a mesh from the latent tensor
mesh_out = create_mesh(
            decoder=model, latent_vector=new_latent, n_pts_per_axis=n_pts_per_axis,
            voxel_origin=voxel_origin, voxel_size=voxel_size, path_original_mesh=None,
            offset=offset, scale=scale, icp_transform=icp_transform,
            objects=objects, verbose=False, device=device
)

# Ensure mesh is PyVista Polydata (.vtk) 
if isinstance(mesh_out, list):
     mesh_out = mesh_out[0]

if not isinstance(mesh_out, pv.PolyData):
     mesh_pv = mesh_out.extract_geometry()
else:
     mesh_pv = mesh_out

# Write to file
mesh_pv.save(output_path)
```

# License

This code is forked and modified from [https://github.com/gattia/NSM](https://github.com/gattia/NSM) following the terms of the [GNU Affero GPL 3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html) and [NSM License](https://github.com/gattia/nsm/blob/main/LICENSE).
