project(FossilNsmModule)
set(EXTENSION_HOMEPAGE "https://github.com/3D-fossils-Haag/nsm")

set(MODULE_NAME FossilNsmModule)
set(MODULE_PKG_SHAPE_COMPLETION ShapeCompletion)

set(MODULE_PYTHON_SCRIPTS
    ${MODULE_PKG_SHAPE_COMPLETION}.py
)

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ""
)
