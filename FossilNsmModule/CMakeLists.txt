project(FossilNsmModule)
set(EXTENSION_HOMEPAGE "https://github.com/3D-fossils-Haag/nsm")

set(MODULE_PKG_SHAPE_COMPLETION ShapeCompletion)
set(MODULE_PKG_CLASSIFICATION Classification)
set(MODULE_PKG_COMMON FossilNsmCommon)

slicerMacroBuildScriptedModule(
  NAME ${MODULE_PKG_SHAPE_COMPLETION}
  SCRIPTS
    ${MODULE_PKG_SHAPE_COMPLETION}.py
    ${MODULE_PKG_COMMON}.py
  RESOURCES ""
)

slicerMacroBuildScriptedModule(
  NAME ${MODULE_PKG_CLASSIFICATION}
  SCRIPTS
    ${MODULE_PKG_CLASSIFICATION}.py
    ${MODULE_PKG_COMMON}.py
    utils/reference_library.py
    utils/draco_glb.py
  RESOURCES ""
)
