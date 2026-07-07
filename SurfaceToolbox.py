import os, unittest, logging, vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

## Modified using agporto's code to do batch processing of surfaces
# in 3D Slicer, open Surface Toolbox module
# Reload & Test -> Edit -> Copy-paste this script into the external window with the original SurfaceToolbox.py file

class SurfaceToolbox(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Surface Toolbox"
    self.parent.categories = ["Surface Models"]
    self.parent.dependencies = []
    self.parent.contributors = ["Luca Antiga (Orobix), Ron Kikinis (Brigham and Women's Hospital), Ben Wilson (Kitware)"]
    self.parent.helpText = """
This module supports various cleanup and optimization processes on surface models.
Select the input and output models, and then enable the stages of the pipeline by selecting the buttons.
Stages that include parameters will open up when they are enabled.
Click apply to activate the pipeline and then click the Toggle button to compare the model before and after
 the operation.
""" + self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = "This module was developed by Luca Antiga, Orobix Srl, with a little help from Steve Pieper, Isomics, Inc."

class SurfaceToolboxWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/SurfaceToolbox.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)
    uiWidget.setMRMLScene(slicer.mrmlScene)
    self.logic = SurfaceToolboxLogic()
    self.logic.updateProcessCallback = self.updateProcess
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
    self.parameterEditWidgets = [
      (self.ui.inputModelSelector, "inputModel"),
      (self.ui.outputModelSelector, "outputModel"),
      (self.ui.remeshButton, "remesh"),
      (self.ui.remeshSubdivideSlider, "remeshSubdivide"),
      (self.ui.remeshClustersSlider, "remeshClustersK"),
      (self.ui.decimationButton, "decimation"),
      (self.ui.reductionSlider, "decimationReduction"),
      (self.ui.boundaryDeletionCheckBox, "decimationBoundaryDeletion"),
      (self.ui.smoothingButton, "smoothing"),
      (self.ui.smoothingMethodComboBox, "smoothingMethod"),
      (self.ui.laplaceIterationsSlider, "smoothingLaplaceIterations"),
      (self.ui.laplaceRelaxationSlider, "smoothingLaplaceRelaxation"),
      (self.ui.taubinIterationsSlider, "smoothingTaubinIterations"),
      (self.ui.taubinPassBandSlider, "smoothingTaubinPassBand"),
      (self.ui.boundarySmoothingCheckBox, "smoothingBoundarySmoothing"),
      (self.ui.normalsButton, "normals"),
      (self.ui.autoOrientNormalsCheckBox, "normalsAutoOrient"),
      (self.ui.flipNormalsCheckBox, "normalsFlip"),
      (self.ui.splittingCheckBox, "normalsSplitting"),
      (self.ui.featureAngleSlider, "normalsFeatureAngle"),
      (self.ui.mirrorButton, "mirror"),
      (self.ui.mirrorXCheckBox, "mirrorX"),
      (self.ui.mirrorYCheckBox, "mirrorY"),
      (self.ui.mirrorZCheckBox, "mirrorZ"),
      (self.ui.cleanerButton, "cleaner"),
      (self.ui.fillHolesButton, "fillHoles"),
      (self.ui.fillHolesSizeSlider, "fillHolesSize"),
      (self.ui.connectivityButton, "connectivity"),
      (self.ui.scaleMeshButton, "scale"),
      (self.ui.scaleXSlider, "scaleX"),
      (self.ui.scaleYSlider, "scaleY"),
      (self.ui.scaleZSlider, "scaleZ"),
      (self.ui.translateMeshButton, "translate"),
      (self.ui.translateToOriginCheckBox, "translateToOrigin"),
      (self.ui.translationXSlider, "translateX"),
      (self.ui.translationYSlider, "translateY"),
      (self.ui.translationZSlider, "translateZ"),
      (self.ui.extractEdgesButton, "extractEdges"),
      (self.ui.extractEdgesBoundaryCheckBox, "extractEdgesBoundary"),
      (self.ui.extractEdgesFeatureCheckBox, "extractEdgesFeature"),
      (self.ui.extractEdgesFeatureAngleSlider, "extractEdgesFeatureAngle"),
      (self.ui.extractEdgesNonManifoldCheckBox, "extractEdgesNonManifold"),
      (self.ui.extractEdgesManifoldCheckBox, "extractEdgesManifold"),
    ]
    slicer.util.addParameterEditWidgetConnections(self.parameterEditWidgets, self.updateParameterNodeFromGUI)
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.ui.toggleModelsButton.connect('clicked()', self.onToggleModels)
    self.initializeParameterNode()
    self.updateGUIFromParameterNode()
    self.batchPanel = ctk.ctkCollapsibleButton(); self.batchPanel.text = "Batch processing"; self.layout.addWidget(self.batchPanel)
    batchLayout = qt.QFormLayout(self.batchPanel)
    self.batchInputDir = ctk.ctkPathLineEdit(); self.batchInputDir.filters = ctk.ctkPathLineEdit.Dirs; self.batchInputDir.setToolTip("Folder containing input .ply meshes"); batchLayout.addRow("Input folder:", self.batchInputDir)
    self.batchOutputDir = ctk.ctkPathLineEdit(); self.batchOutputDir.filters = ctk.ctkPathLineEdit.Dirs; self.batchOutputDir.setToolTip("Folder where processed meshes will be saved"); batchLayout.addRow("Output folder:", self.batchOutputDir)
    self.batchPattern = qt.QLineEdit("*.ply"); batchLayout.addRow("Glob pattern:", self.batchPattern)
    self.batchOverwrite = qt.QCheckBox(); self.batchOverwrite.setChecked(False); batchLayout.addRow("Overwrite existing:", self.batchOverwrite)
    self.batchRunButton = qt.QPushButton("Run Batch"); self.batchRunButton.setIcon(qt.QIcon.fromTheme("media-playback-start")); batchLayout.addRow(self.batchRunButton)
    self.batchRunButton.clicked.connect(self.onRunBatch)

  def cleanup(self): self.removeObservers()
  def enter(self): self.initializeParameterNode()
  def exit(self): self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
  def onSceneStartClose(self, caller, event): self.setParameterNode(None)
  def onSceneEndClose(self, caller, event): 
    if self.parent.isEntered: self.initializeParameterNode()

  def initializeParameterNode(self):
    self.setParameterNode(self.logic.getParameterNode())
    if not self._parameterNode.GetNodeReference("inputModel"):
      firstModelNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode")
      if firstModelNode: self._parameterNode.SetNodeReferenceID("inputModel", firstModelNode.GetID())

  def setParameterNode(self, inputParameterNode):
    if inputParameterNode: self.logic.setDefaultParameters(inputParameterNode)
    if self._parameterNode is not None: self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None: self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    if self._parameterNode is None or self._updatingGUIFromParameterNode: return
    self._updatingGUIFromParameterNode = True
    slicer.util.updateParameterEditWidgetsFromNode(self.parameterEditWidgets, self._parameterNode)
    isLaplace = (self._parameterNode.GetParameter('smoothingMethod') == "Laplace")
    self.ui.laplaceIterationsLabel.setVisible(isLaplace); self.ui.laplaceIterationsSlider.setVisible(isLaplace)
    self.ui.laplaceRelaxationLabel.setVisible(isLaplace); self.ui.laplaceRelaxationSlider.setVisible(isLaplace)
    self.ui.taubinIterationsLabel.setVisible(not isLaplace); self.ui.taubinIterationsSlider.setVisible(not isLaplace)
    self.ui.taubinPassBandLabel.setVisible(not isLaplace); self.ui.taubinPassBandSlider.setVisible(not isLaplace)
    modelsSelected = (self._parameterNode.GetNodeReference("inputModel") and self._parameterNode.GetNodeReference("outputModel"))
    self.ui.toggleModelsButton.enabled = modelsSelected; self.ui.applyButton.enabled = modelsSelected
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    if self._parameterNode is None or self._updatingGUIFromParameterNode: return
    wasModified = self._parameterNode.StartModify()
    slicer.util.updateNodeFromParameterEditWidgets(self.parameterEditWidgets, self._parameterNode)
    self._parameterNode.EndModify(wasModified)

  def updateProcess(self, value):
    self.ui.applyButton.text = value
    self.ui.applyButton.repaint()

  def onApplyButton(self):
    slicer.app.pauseRender(); qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    try:
      inputModelNode = self._parameterNode.GetNodeReference("inputModel")
      outputModelNode = self._parameterNode.GetNodeReference("outputModel")
      self.logic.applyFilters(self._parameterNode)
      slicer.app.processEvents()
      inputModelNode.GetModelDisplayNode().VisibilityOff()
      outputModelNode.GetModelDisplayNode().VisibilityOn()
      self.ui.applyButton.text = "Apply"
    except Exception as e:
      slicer.util.errorDisplay("Failed to compute output model: "+str(e))
      import traceback; traceback.print_exc()
      if 'inputModelNode' in locals(): inputModelNode.GetModelDisplayNode().VisibilityOn()
      if 'outputModelNode' in locals(): outputModelNode.GetModelDisplayNode().VisibilityOff()
    finally:
      slicer.app.resumeRender(); qt.QApplication.restoreOverrideCursor()

  def onToggleModels(self):
    inputModelNode = self._parameterNode.GetNodeReference("inputModel")
    outputModelNode = self._parameterNode.GetNodeReference("outputModel")
    if inputModelNode.GetModelDisplayNode().GetVisibility():
      inputModelNode.GetModelDisplayNode().VisibilityOff(); outputModelNode.GetModelDisplayNode().VisibilityOn(); self.ui.toggleModelsButton.text = "Toggle Models (Output)"
    else:
      inputModelNode.GetModelDisplayNode().VisibilityOn(); outputModelNode.GetModelDisplayNode().VisibilityOff(); self.ui.toggleModelsButton.text = "Toggle Models (Input)"

  def onRunBatch(self):
    inDir = self.batchInputDir.currentPath.strip()
    outDir = self.batchOutputDir.currentPath.strip()
    patt = self.batchPattern.text.strip() or "*.ply"
    overwrite = self.batchOverwrite.isChecked()
    if not inDir or not os.path.isdir(inDir):
      slicer.util.errorDisplay("Please choose a valid input folder."); return
    if not outDir:
      slicer.util.errorDisplay("Please choose an output folder."); return
    try:
      if self._parameterNode.GetParameter("remesh") == "true":
        SurfaceToolboxLogic.installRemeshPrerequisites(force=True)
    except Exception as e:
      slicer.util.errorDisplay(f"Remesh prerequisites failed to install: {e}"); return
    dlg = qt.QProgressDialog("Batch processing...", "Cancel", 0, 100, self.parent)
    dlg.windowModality = qt.Qt.WindowModal; dlg.minimumDuration = 0; dlg.setAutoReset(True); dlg.setAutoClose(False)
    def _dlg_canceled(d): 
      try:
        wc = getattr(d, "wasCanceled", None)
        return wc() if callable(wc) else bool(wc)
      except: return False
    try:
      nDone, nTotal, errors = self.logic.batchProcessFolder(
        parameterNode=self._parameterNode, inputDir=inDir, outputDir=outDir, globPattern=patt, overwrite=overwrite,
        progressCallback=lambda i,n,path:(dlg.setLabelText(f"[{i}/{n}] {os.path.basename(path)}"), dlg.setValue(int(100*i/max(1,n))), qt.QApplication.processEvents()),
        cancelCallback=lambda:_dlg_canceled(dlg))
      dlg.setValue(100)
      if errors:
        msg = f"Done. {nDone}/{nTotal} succeeded. {len(errors)} failed:\n" + "\n".join(errors[:10])
        if len(errors)>10: msg += "\n…"
        slicer.util.infoDisplay(msg, "Batch results")
      else:
        slicer.util.infoDisplay(f"Done. {nDone}/{nTotal} succeeded.", "Batch results")
    except Exception as e:
      slicer.util.errorDisplay(f"Batch failed: {e}")
    finally:
      dlg.reset()

class SurfaceToolboxLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.updateProcessCallback = None

  def setDefaultParameters(self, parameterNode):
    defaults = [
      ("remesh","false"),("remeshSubdivide","0"),("remeshClustersK","10"),
      ("decimation","false"),("decimationReduction","0.8"),("decimationBoundaryDeletion","true"),
      ("smoothing","false"),("smoothingMethod","Taubin"),("smoothingLaplaceIterations","100"),
      ("smoothingLaplaceRelaxation","0.5"),("smoothingTaubinIterations","30"),("smoothingTaubinPassBand","0.1"),
      ("smoothingBoundarySmoothing","true"),
      ("normals","false"),("normalsAutoOrient","false"),("normalsFlip","false"),("normalsSplitting","false"),("normalsFeatureAngle","30.0"),
      ("mirror","false"),("mirrorX","false"),("mirrorY","false"),("mirrorZ","false"),
      ("cleaner","false"),("fillHoles","false"),("fillHolesSize","1000.0"),
      ("connectivity","false"),
      ("scale","false"),("scaleX","0.5"),("scaleY","0.5"),("scaleZ","0.5"),
      ("translate","false"),("translateToOrigin","false"),("translateX","0.0"),("translateY","0.0"),("translateZ","0.0"),
      ("extractEdges","false"),("extractEdgesBoundary","true"),("extractEdgesFeature","true"),
      ("extractEdgesFeatureAngle","20"),("extractEdgesNonManifold","false"),("extractEdgesManifold","false"),
    ]
    for k,v in defaults:
      if not parameterNode.GetParameter(k): parameterNode.SetParameter(k,v)

  def updateProcess(self, message):
    if self.updateProcessCallback: self.updateProcessCallback(message)

  @staticmethod
  def installRemeshPrerequisites(force=False):
    try:
      import pyacvd
    except ModuleNotFoundError:
      if force or slicer.util.confirmOkCancelDisplay("This function requires 'pyacvd' Python package. Click OK to install it now."):
        slicer.util.pip_install("pyacvd==0.3.1")
      else:
        return False
    return True

  @staticmethod
  def remesh(inputModel, outputModel, subdivide=0, clusters=10000):
    if not SurfaceToolboxLogic.installRemeshPrerequisites(): return
    import pyacvd, pyvista as pv
    tri = vtk.vtkTriangleFilter(); tri.SetInputData(inputModel.GetPolyData()); tri.Update()
    inputMesh = pv.wrap(tri.GetOutput())
    clus = pyacvd.Clustering(inputMesh)
    if subdivide>-1: clus.subdivide(subdivide)
    clus.cluster(clusters)
    out = vtk.vtkPolyData(); out.DeepCopy(clus.create_mesh()); outputModel.SetAndObservePolyData(out)

  @staticmethod
  def decimate(inputModel, outputModel, reductionFactor=0.8, decimateBoundary=True, lossless=False, aggressiveness=7.0):
    params = {"inputModel": inputModel, "outputModel": outputModel, "reductionFactor": reductionFactor,
              "method": "FastQuadric" if decimateBoundary else "DecimatePro", "boundaryDeletion": decimateBoundary}
    cliNode = slicer.cli.runSync(slicer.modules.decimation, None, params); slicer.mrmlScene.RemoveNode(cliNode)

  @staticmethod
  def smooth(inputModel, outputModel, method='Taubin', iterations=30, laplaceRelaxationFactor=0.5, taubinPassBand=0.1, boundarySmoothing=True):
    if method=="Laplace":
      f = vtk.vtkSmoothPolyDataFilter(); f.SetRelaxationFactor(laplaceRelaxationFactor)
    else:
      f = vtk.vtkWindowedSincPolyDataFilter(); f.SetPassBand(taubinPassBand)
    f.SetBoundarySmoothing(boundarySmoothing); f.SetNumberOfIterations(iterations); f.SetInputData(inputModel.GetPolyData()); f.Update()
    outputModel.SetAndObservePolyData(f.GetOutput())

  @staticmethod
  def fillHoles(inputModel, outputModel, maximumHoleSize=1000.0):
    fill = vtk.vtkFillHolesFilter(); fill.SetInputData(inputModel.GetPolyData()); fill.SetHoleSize(maximumHoleSize)
    normals = vtk.vtkPolyDataNormals(); normals.SetInputConnection(fill.GetOutputPort()); normals.SetAutoOrientNormals(True); normals.Update()
    outputModel.SetAndObservePolyData(normals.GetOutput())

  @staticmethod
  def transform(inputModel, outputModel, scaleX=1.0, scaleY=1.0, scaleZ=1.0, translateX=0.0, translateY=0.0, translateZ=0.0):
    t = vtk.vtkTransform(); t.Translate(translateX, translateY, translateZ); t.Scale(scaleX, scaleY, scaleZ)
    tf = vtk.vtkTransformFilter(); tf.SetInputData(inputModel.GetPolyData()); tf.SetTransform(t)
    if t.GetMatrix().Determinant()>=0.0:
      tf.Update(); outputModel.SetAndObservePolyData(tf.GetOutput())
    else:
      rev = vtk.vtkReverseSense(); rev.SetInputConnection(tf.GetOutputPort()); rev.Update(); outputModel.SetAndObservePolyData(rev.GetOutput())

  @staticmethod
  def translateCenterToOrigin(inputModel, outputModel):
    bounds = inputModel.GetPolyData().GetBounds()
    cx,cy,cz = (bounds[1]+bounds[0])/2.0,(bounds[3]+bounds[2])/2.0,(bounds[5]+bounds[4])/2.0
    SurfaceToolboxLogic.transform(inputModel, outputModel, translateX=-cx, translateY=-cy, translateZ=-cz)

  @staticmethod
  def clean(inputModel, outputModel):
    c = vtk.vtkCleanPolyData(); c.SetInputData(inputModel.GetPolyData()); c.Update(); outputModel.SetAndObservePolyData(c.GetOutput())

  @staticmethod
  def extractBoundaryEdges(inputModel, outputModel, boundary=False, feature=False, nonManifold=False, manifold=False, featureAngle=20):
    e = vtk.vtkFeatureEdges(); e.SetInputData(inputModel.GetPolyData()); e.ExtractAllEdgeTypesOff()
    e.SetBoundaryEdges(boundary); e.SetFeatureEdges(feature); 
    if feature: e.SetFeatureAngle(featureAngle)
    e.SetNonManifoldEdges(nonManifold); e.SetManifoldEdges(manifold); e.Update()
    outputModel.SetAndObservePolyData(e.GetOutput())

  @staticmethod
  def computeNormals(inputModel, outputModel, autoOrient=False, flip=False, split=False, splitAngle=30.0):
    n = vtk.vtkPolyDataNormals(); n.SetInputData(inputModel.GetPolyData())
    n.SetAutoOrientNormals(autoOrient); n.SetFlipNormals(flip); n.SetSplitting(split)
    if split: n.SetFeatureAngle(splitAngle)
    n.Update(); outputModel.SetAndObservePolyData(n.GetOutput())

  @staticmethod
  def extractLargestConnectedComponent(inputModel, outputModel):
    con = vtk.vtkPolyDataConnectivityFilter(); con.SetInputData(inputModel.GetPolyData()); con.SetExtractionModeToLargestRegion(); con.Update()
    outputModel.SetAndObservePolyData(con.GetOutput())

  def applyFilters(self, parameterNode):
    import time
    startTime = time.time(); logging.info('Processing started')
    inputModel = parameterNode.GetNodeReference("inputModel")
    outputModel = parameterNode.GetNodeReference("outputModel")
    if outputModel != inputModel:
      if outputModel.GetPolyData() is None: outputModel.SetAndObservePolyData(vtk.vtkPolyData())
      outputModel.GetPolyData().DeepCopy(inputModel.GetPolyData())
    outputModel.CreateDefaultDisplayNodes(); outputModel.AddDefaultStorageNode()

    if parameterNode.GetParameter("cleaner") == "true":
      self.updateProcess("Clean..."); SurfaceToolboxLogic.clean(outputModel, outputModel)

    if parameterNode.GetParameter("remesh") == "true":
      self.updateProcess("Remeshing..."); SurfaceToolboxLogic.remesh(outputModel, outputModel,
        subdivide=int(float(parameterNode.GetParameter("remeshSubdivide"))),
        clusters=int(1000 * float(parameterNode.GetParameter("remeshClustersK"))))

    if parameterNode.GetParameter("decimation") == "true":
      self.updateProcess("Decimation..."); SurfaceToolboxLogic.decimate(outputModel, outputModel,
        reductionFactor=float(parameterNode.GetParameter("decimationReduction")),
        decimateBoundary=parameterNode.GetParameter("decimationBoundaryDeletion") == "true")

    if parameterNode.GetParameter("smoothing") == "true":
      self.updateProcess("Smoothing...")
      method = parameterNode.GetParameter("smoothingMethod")
      SurfaceToolboxLogic.smooth(outputModel, outputModel, method=method,
        iterations=int(float(parameterNode.GetParameter("smoothingLaplaceIterations" if method=='Laplace' else "smoothingTaubinIterations"))),
        laplaceRelaxationFactor=float(parameterNode.GetParameter("smoothingLaplaceRelaxation")),
        taubinPassBand=float(parameterNode.GetParameter("smoothingTaubinPassBand")),
        boundarySmoothing=parameterNode.GetParameter("smoothingBoundarySmoothing") == "true")

    if parameterNode.GetParameter("fillHoles") == "true":
      self.updateProcess("Fill Holes..."); SurfaceToolboxLogic.fillHoles(outputModel, outputModel,
        float(parameterNode.GetParameter("fillHolesSize")))

    if parameterNode.GetParameter("normals") == "true":
      self.updateProcess("Normals..."); SurfaceToolboxLogic.computeNormals(outputModel, outputModel,
        autoOrient = parameterNode.GetParameter("normalsAutoOrient") == "true",
        flip=parameterNode.GetParameter("normalsFlip") == "true",
        split=parameterNode.GetParameter("normalsSplitting") == "true",
        splitAngle=float(parameterNode.GetParameter("normalsFeatureAngle")))

    if parameterNode.GetParameter("mirror") == "true":
      self.updateProcess("Mirror..."); SurfaceToolboxLogic.transform(outputModel, outputModel,
        scaleX = -1.0 if parameterNode.GetParameter("mirrorX") == "true" else 1.0,
        scaleY = -1.0 if parameterNode.GetParameter("mirrorY") == "true" else 1.0,
        scaleZ = -1.0 if parameterNode.GetParameter("mirrorZ") == "true" else 1.0)

    if parameterNode.GetParameter("scale") == "true":
      self.updateProcess("Scale..."); SurfaceToolboxLogic.transform(outputModel, outputModel,
        scaleX = float(parameterNode.GetParameter("scaleX")),
        scaleY = float(parameterNode.GetParameter("scaleY")),
        scaleZ = float(parameterNode.GetParameter("scaleZ")))

    if parameterNode.GetParameter("translate") == "true":
      self.updateProcess("Translating...")
      if parameterNode.GetParameter("translateToOrigin") == "true":
        SurfaceToolboxLogic.translateCenterToOrigin(outputModel, outputModel)
      SurfaceToolboxLogic.transform(outputModel, outputModel,
        translateX = float(parameterNode.GetParameter("translateX")),
        translateY = float(parameterNode.GetParameter("translateY")),
        translateZ = float(parameterNode.GetParameter("translateZ")))

    if parameterNode.GetParameter("extractEdges") == "true":
      self.updateProcess("Extracting boundary edges..."); SurfaceToolboxLogic.extractBoundaryEdges(outputModel, outputModel,
        boundary = parameterNode.GetParameter("extractEdgesBoundary") == "true",
        feature = parameterNode.GetParameter("extractEdgesFeature") == "true",
        nonManifold = parameterNode.GetParameter("extractEdgesNonManifold") == "true",
        manifold = parameterNode.GetParameter("extractEdgesManifold") == "true",
        featureAngle = float(parameterNode.GetParameter("extractEdgesFeatureAngle")))

    if parameterNode.GetParameter("connectivity") == "true":
      self.updateProcess("Extract largest connected component..."); SurfaceToolboxLogic.extractLargestConnectedComponent(outputModel, outputModel)

    self.updateProcess("Done."); stopTime = time.time()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))

  def _ensureDir(self, d):
    if not os.path.isdir(d): os.makedirs(d, exist_ok=True)

  def batchProcessFolder(self, parameterNode, inputDir, outputDir, globPattern="*.ply", overwrite=False,
                         outputExtension=".ply", progressCallback=None, cancelCallback=None):
    import glob, time
    self._ensureDir(outputDir)
    files = sorted(glob.glob(os.path.join(inputDir, globPattern)))
    nTotal = len(files); nDone = 0; errors = []
    origIn = parameterNode.GetNodeReferenceID("inputModel")
    origOut = parameterNode.GetNodeReferenceID("outputModel")
    for i,f in enumerate(files,1):
      if cancelCallback and cancelCallback(): break
      if progressCallback: progressCallback(i, nTotal, f)
      try:
        base = os.path.splitext(os.path.basename(f))[0]
        outPath = os.path.join(outputDir, base + outputExtension)
        if (not overwrite) and os.path.exists(outPath):
          nDone += 1; continue
        inNode = slicer.util.loadModel(f)
        if inNode is None: raise RuntimeError("Failed to load model: " + f)
        outNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", base + "_processed")
        outNode.CreateDefaultDisplayNodes(); outNode.AddDefaultStorageNode()
        parameterNode.SetNodeReferenceID("inputModel", inNode.GetID())
        parameterNode.SetNodeReferenceID("outputModel", outNode.GetID())
        slicer.app.pauseRender(); qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try: self.applyFilters(parameterNode)
        finally: qt.QApplication.restoreOverrideCursor(); slicer.app.resumeRender()
        ok = slicer.util.saveNode(outNode, outPath)
        if not ok: raise RuntimeError("Failed to save: " + outPath)
        nDone += 1
      except Exception as ex:
        errors.append(f"{os.path.basename(f)}: {ex}")
      finally:
        try:
          if 'inNode' in locals() and inNode: slicer.mrmlScene.RemoveNode(inNode)
          if 'outNode' in locals() and outNode: slicer.mrmlScene.RemoveNode(outNode)
        except: pass
        slicer.app.processEvents()
    try:
      parameterNode.SetNodeReferenceID("inputModel", origIn if origIn else None)
      parameterNode.SetNodeReferenceID("outputModel", origOut if origOut else None)
    except: pass
    return nDone, nTotal, errors

class SurfaceToolboxTest(ScriptedLoadableModuleTest):
  def setUp(self): slicer.mrmlScene.Clear(0)
  def runTest(self): self.setUp(); self.test_AllProcessing()
  def test_AllProcessing(self):
    self.delayDisplay("Starting the test")
    import SampleData
    modelNode = SampleData.downloadFromURL(
      nodeNames='cow', fileNames='cow.vtp', loadFileTypes='ModelFile',
      uris='https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/d5aa4901d186902f90e17bf3b5917541cb6cb8cf223bfeea736631df4c047652',
      checksums='SHA256:d5aa4901d186902f90e17bf3b5917541cb6cb8cf223bfeea736631df4c047652')[0]
    self.delayDisplay('Finished with download and loading')
    logic = SurfaceToolboxLogic(); self.assertIsNotNone(logic)
    parameterNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode"); logic.setDefaultParameters(parameterNode)
    parameterNode.SetNodeReferenceID("inputModel", modelNode.GetID())
    outputModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "output")
    parameterNode.SetNodeReferenceID("outputModel", outputModelNode.GetID())
    parameterNode.SetParameter("remesh", "false")
    parameterNode.SetParameter("decimation", "true")
    parameterNode.SetParameter("smoothing", "true")
    parameterNode.SetParameter("normals", "true")
    parameterNode.SetParameter("mirror", "true")
    parameterNode.SetParameter("mirrorX", "true")
    parameterNode.SetParameter("cleaner", "true")
    parameterNode.SetParameter("fillHoles", "true")
    parameterNode.SetParameter("connectivity", "true")
    parameterNode.SetParameter("scale", "true")
    parameterNode.SetParameter("translate", "true")
    parameterNode.SetParameter("translateX", "5.12")
    parameterNode.SetParameter("relax", "true")
    parameterNode.SetParameter("extractEdges", "true")
    parameterNode.SetParameter("translateToOrigin", "true")
    self.delayDisplay('Module selected, input and output configured')
    logic.applyFilters(parameterNode)
    self.delayDisplay('Test passed!')