# Fossil NSM Module in 3D Slicer
*Last edited 4 May 2026*

# Introduction
Use Fossil NSM Module in [3D Slicer](https://www.slicer.org/) to do shape completion on a fossil lizard vertebra using our trained model. 

# Installation
1. Open 3D Slicer
2. Install the PyTorch extension for 3D Slicer ([link to source code](https://github.com/fepegar/SlicerPyTorch))   
   View -> Extension Manager -> Install Extensions -> PyTorch
3. Restart 3D Slicer
4. Specify pytorch version to be compatible with NSM   
   Modules -> PyTorch   
   Torch version requirement: ==2.5.1   
   Click "Install PyTorch"
6. Download the [nsm GitHub repo](https://github.com/3D-fossils-Haag/nsm/tree/main)   
   Via git clone or download zip
7. Open 3D Slicer
8. Build the Fossil NSM Module   
   Extension Wizard -> Select Extension -> path/to/your/nsm/FossilNsmModule
9. Use Fossil NSM Module for shape completion or classification!
   Modules -> FossilNSM -> Shape Completion -> Select your input files and click "Run Inference"
   Modules -> FossilNSM -> Classification -> Select your input files and click "Classify Input Mesh"

# Classification and reference-mesh packages

The Classification module classifies an input mesh by optimizing its latent code
and ranking the five nearest training codes by cosine distance. Select the same
model root used for completion, select an input mesh, then click
**Classify Input Mesh**. The result table is saved as a small JSON file in
`<model root>/shape_completion/classification/` and can be retained without
copying the model or meshes.

To visualize a result, select a local *Reference Mesh Library*. Its filenames
must match the training mesh basenames stored in `model_params_config.json`; the
library may have subdirectories. The module reports unavailable matches instead
of pretending that a checkpoint contains its reference geometry.

![Fossil NSM Module](https://github.com/aubricot/nsm/blob/main/images/fossilnsmmodule.png)
*Screenshot of shape completion results produced using Fossil NSM Module in 3D Slicer*

# Data Preprocessing
**A) Smoothing**    
Ensure your fossil data is smooth and simplified to be comparable to our training data. This ensures that differences in shape are given priority, not differences in surface texture.
1. Open 3D Slicer
2. Modules -> Surface Toolkit
3. Set parameters following screenshot below.
   
![Surface Toolkit Parameters](https://github.com/aubricot/nsm/blob/main/images/surftoolkit.png)
*Screenshot of parameters used to preprocess meshes using the Surface Toolkit in 3D Slicer*

**B) Align and Scale**   
Align and scale your fossil to the statistical shape model (SSM) used to prepare our training data. This ensures differences in shape are given priority, not differences in position or scale.
1. Open 3D Slicer
2. Inspect SSM model and landmarks in the 3D viewer.
3. Use the Markups Module to manually landmark your specimen following the template.
4. Open your preferred code editor (e.g. VS Code)
5. Use [align_model_to_ssm.py](https://github.com/3D-fossils-Haag/nsm/blob/main/align_model_to_ssm.py) to align your landmarked fossil to the SSM used for training data preprocessing.
