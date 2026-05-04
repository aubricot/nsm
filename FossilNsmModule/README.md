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
9. Use Fossil NSM Module to do shape completion!
   Modules -> Fossil NSM -> Shape Completion -> Select your input files and click "Run Inference"

   
![Fossil NSM Module](https://github.com/aubricot/nsm/blob/main/images/fossilnsmmodule.png)
*Screenshot of shape completion results produced using Fossil NSM Module in 3D Slicer*
