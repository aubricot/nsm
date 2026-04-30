cmake_minimum_required(VERSION 3.5)

project(FossilNsm)

# ---- Extension meta-data (shown in Slicer Extension Manager) ----
set(EXTENSION_HOMEPAGE         "https://github.com/3D-fossils-Haag/nsm")
set(EXTENSION_CATEGORY         "NSM")     # Pick something appropriate
set(EXTENSION_CONTRIBUTORS     "Wolcott et al.")
set(EXTENSION_DESCRIPTION      "Tool for inference and error metrics")
set(EXTENSION_ICONURL          "")              # Optional
set(EXTENSION_SCREENSHOTURLS   "")              # Optional
set(EXTENSION_DEPENDS          "PyTorch")              # Space-separated Slicer extensions your extension depends on

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

# Add your modules
add_subdirectory(FossilNsmModule)


# Packaging, testing
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})