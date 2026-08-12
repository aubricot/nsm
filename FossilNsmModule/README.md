# Fossil NSM Module in 3D Slicer
*Last edited 12 Aug 2026*

## Introduction
Use Fossil NSM Module in [3D Slicer](https://www.slicer.org/) to do shape completion on a fossil lizard vertebra using our trained model. 


## Installation
1. Open 3D Slicer
2. Download the [nsm GitHub repo](https://github.com/3D-fossils-Haag/nsm/tree/main)   
   Via git clone or download zip
3. Open 3D Slicer
4. Build the Fossil NSM Module *See Note   
   Extension Wizard -> Select Extension -> path/to/your/nsm/FossilNsmModule
5. Use Fossil NSM Module for shape completion or classification!
   Modules -> FossilNSM -> Shape Completion -> Select your input files and click "Run Inference"
   Modules -> FossilNSM -> Classification -> Select your input files and click "Classify Input Mesh"

*Note: If you run into any errors importing torch, follow the steps below to ensure 3D Slicer uses the appropriate pytorch build.    
3a. Install the PyTorch extension for 3D Slicer ([link to source code](https://github.com/fepegar/SlicerPyTorch))      
      View -> Extension Manager -> Install Extensions -> PyTorch   
3b. Restart 3D Slicer   
3c. Specify pytorch version to be compatible with NSM      
      Modules -> PyTorch     
      Torch version requirement: ==2.5.1     
      Click "Install PyTorch"   
3d. Proceed to Step 4.   

## Getting started
To identify an unknown, partial fossil using our trained model, follow the steps below.  
**A) Data Preprocessing**   
   Preprocess your input fossils so they are smooth, clean, and aligned/scaled to the same proportions as the model training data. 

**B) Shape Completion**   
   Use shape completion to fill in missing structures of your fossil.
   
**C) Classification**   
   Classify your shape-completed fossil to species and vertebral position.
   
## A) Data Preprocessing
See detailed steps for how to preprocess your fossil for inference on our [project wiki page](https://github.com/3D-fossils-Haag/nsm/wiki/Fossil-Preprocessing).  

**1) Smoothing**    
Ensure your fossil data is smooth and simplified to be comparable to our training data. This ensures that differences in shape are given priority, not differences in surface texture.
1. Open 3D Slicer
2. Modules -> Surface Toolkit
3. Set parameters following screenshot below.
   
![Surface Toolkit Parameters](https://github.com/3D-fossils-Haag/nsm/blob/main/images/surftoolkit.png)
*Screenshot of parameters used to preprocess meshes using the Surface Toolkit in 3D Slicer*

**2) Cleaning**   
Clean the mesh file so that it is hollow and cut away broken mesh regions so they can be filled in by shape completion.
1. Open 3D Slicer
2. Modules -> Segmentation Editor
3. Use a combination of the Hollow tool and Scissors tool to prepare the mesh.

![Segmentation Editor Parameters](https://github.com/3D-fossils-Haag/nsm/blob/main/images/12.png)

**3) Align and Scale**   
Align and scale your fossil to the statistical shape model (SSM) used to prepare our training data. This ensures differences in shape are given priority, not differences in position or scale.
1. Open 3D Slicer
2. Inspect SSM model and landmarks in the 3D viewer.
3. Use the Markups Module to manually landmark your specimen following the template.
4. Open your preferred code editor (e.g. VS Code)
5. Use [align_model_to_ssm.py](https://github.com/3D-fossils-Haag/nsm/blob/main/align_model_to_ssm.py) to align your landmarked fossil to the SSM used for training data preprocessing.


## Shape Completion
The Shape Completion module completes a partial input mesh by optimizing or encoding (fast mode) it into the latent space. Select the model root containing all trained model files and optionally choose encoding/optimization settings and whether to visualize uncertainty. Click `Toggle Models` to toggle between the input fossil and shape completion results in the 3D viewer.

![Fossil NSM Shape Completion](https://github.com/3D-fossils-Haag/nsm/blob/main/images/fossilnsmmodule.png)
*Screenshot of shape completion results produced using Fossil NSM Module in 3D Slicer*

## Classification

The Classification module classifies an input mesh by optimizing its latent code
and ranking the five nearest training codes by cosine distance. 

In the `Inference` tab, select the same model root used for completion, select an input mesh, then click
**Classify Input Mesh**. The result table is saved as a small JSON file in
`<model root>/shape_completion/classification/` and can be retained without
copying the model or meshes. To visualize the top5 matches in `Explore Meshes`, select a local or Hugging Face *Reference Mesh Library*. Filenames
must match the training mesh basenames stored in `model_params_config.json`.

In the `Explore Meshes` tab, visualize the top5 closest meshes.

In the `Explore Plots` tab, Interactive latent space plots of the encoded fossil, top5 matches, and training latents are shown using PCA, t-SNE, and UMAP.

![Fossil NSM Classification](https://github.com/3D-fossils-Haag/nsm/blob/main/images/classification_screenshot.png)
*Screenshot of classification results produced using Fossil NSM Module in 3D Slicer*

## Additional Information

### Running Fossil NSM Module Tests
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

