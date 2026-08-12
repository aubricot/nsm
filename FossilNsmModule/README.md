# Fossil NSM Module in 3D Slicer
*Last edited 12 Aug 2026*

## Introduction
Use Fossil NSM Module in [3D Slicer](https://www.slicer.org/) to do shape completion on a fossil lizard vertebra using our trained model. 

### Directory structure
```
<model_root> (e.g., run_v44)/   
├── model_params_config.json   
├── model/   
│   └── 3000.pth   
├── latent_codes/   
│   └── 3000.pth   
├── shape_completion/   
│   ├── <results>_shape_completion.vtk   
│   └── <results>_shape_completion_unc.vtk   
└── classification/   
    └── <mesh_name>/   
        ├── <results>_classification.log   
        ├── <results>_top5.csv   
        ├── <results>_top5.json   
        ├── all_latents.npy   
        ├── fossil_latent.npy   
        └── top5_indices.npy
```

## Installation
1) Download this repository (git clone or ZIP).
2) Slicer: `View > Extension Manager > Install Extensions > PyTorch`, then restart Slicer.
3) Slicer: `Modules > PyTorch` → set Torch version to `2.5.1` → click `Install PyTorch`.
4) Slicer: `Modules > Extension Wizard > Select Extension` → choose `<path-to-repo>/FossilNsmModule`.
5) Slicer: `Modules > FossilNSM` to access Shape Completion and Classification.

> Note: If Torch fails to import, confirm the Slicer PyTorch extension is installed and the version is set

6. Use Fossil NSM Module for shape completion or classification!
   Modules -> FossilNSM -> Shape Completion -> Select your input files and click "Run Inference"
   Modules -> FossilNSM -> Classification -> Select your input files and click "Classify Input Mesh" 

## Quickstart
To identify an unknown, partial fossil using our trained model, follow the steps below.  

### A) Preprocess Data   
Preprocess your input fossils so they are smooth, clean, and aligned/scaled to the same proportions as the model training data. 
   
- Slicer: `Modules > Surface Toolkit` → smooth/simplify meshes to match training style.
- Slicer: `Modules > Segmentation Editor`  → clean and hollow the mesh so incomplete parts can be filled in during shape completion.
- Align and scale to the SSM template:
  1. Inspect the SSM and landmarks in the 3D view.
  2. Use `Modules > Markups` to place landmarks using the template.
  3. Run the alignment script on your landmarked specimen:
     - Script: [`align_model_to_ssm.py`](https://github.com/3D-fossils-Haag/nsm/blob/main/align_model_to_ssm.py)

- See detailed steps in the wiki: https://github.com/3D-fossils-Haag/nsm/wiki/Fossil-Preprocessing
    
### B) Run shape completion   
Use shape completion to fill in missing structures of your fossil.

- Slicer: `Modules > FossilNSM > Shape Completion`.
- Select `<model-root>` (folder with trained model files) and your input mesh.
- Choose optimization or enable Fast Mode (Encoder) if available.
- Click `Run Inference`. Use `Toggle Models` to flip between input and completed shape.

![Fossil NSM Shape Completion](https://github.com/3D-fossils-Haag/nsm/blob/main/images/fossilnsmmodule.png)
*Screenshot of shape completion results produced using Fossil NSM Module in 3D Slicer*

### C) Classify the mesh
Classify your shape-completed fossil to species and vertebral position.

- Slicer: `Modules > FossilNSM > Classification`.
- Select the same `<model-root>` and your input mesh.
- Click `Classify Input Mesh`.
- Results are saved as JSON in `<model-root>/shape_completion/classification/`.
- To visualize top‑5 matches:
  - In `Explore Meshes`, choose a local or Hugging Face Reference Mesh Library.
  - Filenames must match the training mesh basenames in `model_params_config.json` (subdirectories allowed).
  - Unavailable matches are reported explicitly.

- In `Explore Plots`, interactively view PCA, t‑SNE, and UMAP projections of the latent space for the encoded fossil, top‑5 matches, and training latents.
   
![Fossil NSM Classification](https://github.com/3D-fossils-Haag/nsm/blob/main/images/classification_screenshot.png)
*Screenshot of classification results produced using Fossil NSM Module in 3D Slicer*

## Additional Information

### Running Tests
To run the tests, execute the following command in your terminal:
```
<path to your slicer install>/bin/PythonSlicer -m unittest test_shape_completion.py
<path to your slicer install>/bin/PythonSlicer -m unittest test_classification.py
```
NOTE: Make sure to replace `<path to your slicer install>` with the actual path of your 3D Slicer installation.


### Fast Shape Completion (Encoder)

The Slicer shape completion module normally recovers a latent code by test-time
optimization (thousands of iterations). The `encoder/` package trains a
feed-forward PointNet encoder that infers the latent from a partial point cloud
in a single forward pass, giving near-identical reconstructions ~10,000x faster.

The encoder is distilled from a trained model, so you build one directly from an
existing `run_` folder (which holds `model_params_config.json`,
`model/<epoch>.pth`, and `latent_codes/<epoch>.pth`):

```bash
conda activate NSM
cd NSM/nsm

# 1. Build the training set: for each latent code, marching-cubes the decoder
#    into a surface and store the (surface_points -> latent) pair.
python encoder/generate_latent_dataset.py \
    --config run_v44/model_params_config.json \
    --model run_v44/model/3000.pth \
    --latent_codes run_v44/latent_codes/3000.pth \
    --out encoder/data/latent_surface_dataset.pt

# 2. Train the encoder (random cropping teaches partial -> full completion).
python encoder/train_encoder.py \
    --data encoder/data/latent_surface_dataset.pt \
    --out encoder/checkpoints/encoder.pt

# 3. (optional) Validate quality and speed vs. the optimizer.
python encoder/evaluate_shape_completion.py \
    --config run_v44/model_params_config.json \
    --model run_v44/model/3000.pth \
    --latent_codes run_v44/latent_codes/3000.pth \
    --encoder encoder/checkpoints/encoder.pt
```

Point the Slicer module's "Fast Mode (Encoder)" option at the resulting
`encoder/checkpoints/encoder.pt`. Retrain the encoder whenever the underlying
model/latent codes change, since it is specific to that decoder.

## Links
* Module folder: https://github.com/3D-fossils-Haag/nsm/tree/main/FossilNsmModule
* Slicer PyTorch extension: https://github.com/fepegar/SlicerPyTorch
* Preprocessing wiki: https://github.com/3D-fossils-Haag/nsm/wiki/Fossil-Preprocessing

## Citation
If you use this code or the trained models in your research, please cite this repository
```
Wolcott et al. 2026. “NSM: Fossil NSM Module for 3D Slicer.” GitHub repository. https://github.com/3D-fossils-Haag/nsm (accessed YYYY-MM-DD).
```
