import slicer
from FossilNsmCommon import FossilNsmCommonWidget, FossilNsmLogic
FossilNsmLogic.installDependenciesIfNeeded()
import os
import qt
import ctk
import vtk
import sys
import subprocess
from slicer.ScriptedLoadableModule import *
import pyvista as pv
from utils.utils import _uniform_surface_sample, chamfer_distance, f_score, ave_sym_surface_distance

MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

class ShapeCompletion(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Shape Completion"
        self.parent.categories = ["FossilNSM"]
        self.parent.index = 20
        self.parent.contributors = ["Wolcott et al"]
        self.parent.helpText = "Encode a partial vertebrae mesh and use NSM to complete the shape."

class ShapeCompletionWidget(FossilNsmCommonWidget, ScriptedLoadableModuleWidget):

    def setup(self):
        super().setup()
        self.initializeFossilNsmState()
        FossilNsmLogic.installDependenciesIfNeeded()
        self.addFossilNsmInputSection(self.layout)

        # Fast Mode Collapsible Layout (one-shot encoder instead of optimization)
        fastCollapsible = ctk.ctkCollapsibleButton()
        fastCollapsible.text = "Fast Mode (Encoder)"
        fastCollapsible.collapsed = True
        self.layout.addWidget(fastCollapsible)
        fastLayout = qt.QFormLayout(fastCollapsible)

        self.fastModeCheckbox = qt.QCheckBox("Use encoder (skip latent optimization)")
        self.fastModeCheckbox.setChecked(False)
        fastLayout.addRow("", self.fastModeCheckbox)

        self.encoderCkptInput = qt.QLineEdit("encoder/checkpoints/encoder.pt")
        fastLayout.addRow("Encoder Checkpoint:", self.encoderCkptInput)
        self.refineItersInput = qt.QLineEdit("0")
        fastLayout.addRow("Refine Iterations:", self.refineItersInput)
        self.refineLrInput = qt.QLineEdit("1e-5")
        fastLayout.addRow("Refine Learning Rate:", self.refineLrInput)
        self.refineLambdaInput = qt.QLineEdit("1e-5")
        fastLayout.addRow("Refine Lambda Reg:", self.refineLambdaInput)

        # Optimization Settings Collapsible Layout
        optimCollapsible = ctk.ctkCollapsibleButton()
        optimCollapsible.text = "Optimization Settings"
        optimCollapsible.collapsed = True
        self.layout.addWidget(optimCollapsible)
        optimLayout = qt.QFormLayout(optimCollapsible)

        # Sample Points
        self.nSamplesOptInput = qt.QLineEdit("240")
        optimLayout.addRow("Sample Points:", self.nSamplesOptInput)

        # Phase 1
        self.phase1ItersInput = qt.QLineEdit("3000")
        optimLayout.addRow("Phase 1 Iterations:", self.phase1ItersInput)

        self.phase1LrInput = qt.QLineEdit("1e-4")
        optimLayout.addRow("Phase 1 Learning Rate:", self.phase1LrInput)

        self.phase1LambdaInput = qt.QLineEdit("1e-3")
        optimLayout.addRow("Phase 1 Lambda Reg:", self.phase1LambdaInput)

        # Phase 2
        self.phase2ItersInput = qt.QLineEdit("8000")
        optimLayout.addRow("Phase 2 Iterations:", self.phase2ItersInput)

        self.phase2LrInput = qt.QLineEdit("1e-5")
        optimLayout.addRow("Phase 2 Learning Rate:", self.phase2LrInput)

        self.phase2LambdaInput = qt.QLineEdit("1e-5")
        optimLayout.addRow("Phase 2 Lambda Reg:", self.phase2LambdaInput)

        # Resolution
        self.nPtsPerAxisInput = qt.QLineEdit("256")
        optimLayout.addRow("Resolution (pts/axis):", self.nPtsPerAxisInput)

        # Uncertainty Collapsible Layout
        uncertaintyCollapsible = ctk.ctkCollapsibleButton()
        uncertaintyCollapsible.text = "Uncertainty Settings"
        uncertaintyCollapsible.collapsed = True
        self.layout.addWidget(uncertaintyCollapsible)
        uncertaintyLayout = qt.QFormLayout(uncertaintyCollapsible)

        # Checkbox to enable uncertainty
        self.estimateUncertaintyCheckbox = qt.QCheckBox("Estimate Uncertainty")
        self.estimateUncertaintyCheckbox.setChecked(False)
        uncertaintyLayout.addRow("", self.estimateUncertaintyCheckbox)

        # Propagation Mode
        self.propagationModeCombobox = qt.QComboBox()
        self.propagationModeCombobox.addItems(["analytical", "montecarlo"])
        uncertaintyLayout.addRow("Propagation Mode:", self.propagationModeCombobox)

        # Data Std
        self.dataStdInput = qt.QLineEdit("2e-5")
        uncertaintyLayout.addRow("Data Std (sigma_Y):", self.dataStdInput)

        # Latent Std
        self.latentStdInput = qt.QLineEdit("5e-4")
        uncertaintyLayout.addRow("Latent Std (sigma_z):", self.latentStdInput)

        # Data Weight
        self.dataWeightInput = qt.QLineEdit("1.0")
        uncertaintyLayout.addRow("Data Weight:", self.dataWeightInput)

        # Latent Weight
        self.latentWeightInput = qt.QLineEdit("1.0")
        uncertaintyLayout.addRow("Latent Weight:", self.latentWeightInput)

        # Decimate Triangles
        self.nTrianglesInput = qt.QLineEdit("5000")
        uncertaintyLayout.addRow("Decimate Triangles:", self.nTrianglesInput)

        # Monte Carlo Samples
        self.nSamplesInput = qt.QLineEdit("2000")
        uncertaintyLayout.addRow("Monte Carlo Samples:", self.nSamplesInput)

        # Connect signals
        self.estimateUncertaintyCheckbox.connect("stateChanged(int)", self.onToggleUncertaintyOptions)
        self.propagationModeCombobox.connect("currentIndexChanged(int)", self.onToggleUncertaintyOptions)
        self.onToggleUncertaintyOptions()

        self.fastModeCheckbox.connect("stateChanged(int)", self.onToggleFastMode)
        self.onToggleFastMode()

        # Run Button
        self.runButton = qt.QPushButton("Run Inference")
        self.runButton.setEnabled(False)
        self.runButton.connect("clicked(bool)", self.onRunInference)
        self.layout.addWidget(self.runButton)

        # Progress Bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.setVisible(False)
        self.layout.addWidget(self.progressBar)

        # Status Log
        self.statusLog = qt.QTextEdit()
        self.statusLog.setReadOnly(True)
        self.statusLog.setFixedHeight(120)
        self.layout.addWidget(self.statusLog)

        # Similarity Metrics Form Layout
        distanceCollapsible = ctk.ctkCollapsibleButton()
        distanceCollapsible.text = "Similarity Metrics"
        self.layout.addWidget(distanceCollapsible)
        distanceLayout = qt.QFormLayout(distanceCollapsible)

        self.sampleCountLabel = qt.QLabel("Point Sampled: ")
        self.sampleCountValueInput = qt.QLineEdit(10000)
        self.similarityThresholdLabel = qt.QLabel("Similarity Threshold: ")
        self.similarityThresholdValueInput = qt.QLineEdit(0.005)
        self.sampleCountValueInput.setFixedWidth(80)
        self.similarityThresholdValueInput.setFixedWidth(80)
        self.calculateDistsButton = qt.QPushButton("Calculate")
        self.calculateDistsButton.setEnabled(False)
        self.calculateDistsButton.connect("clicked(bool)", self.onCalculateDistances)

        distanceInputRowWidget = qt.QWidget()
        distanceInputRowLayout = qt.QHBoxLayout(distanceInputRowWidget)
        distanceInputRowLayout.setContentsMargins(0, 0, 0, 0)
        distanceInputRowLayout.addWidget(self.sampleCountLabel)
        distanceInputRowLayout.addWidget(self.sampleCountValueInput)
        distanceInputRowLayout.addSpacing(40)
        distanceInputRowLayout.addWidget(self.similarityThresholdLabel)
        distanceInputRowLayout.addWidget(self.similarityThresholdValueInput)
        distanceInputRowLayout.addSpacing(40)
        distanceInputRowLayout.addWidget(self.calculateDistsButton)
        distanceLayout.addRow(distanceInputRowWidget)

        self.chamferDistance = qt.QLabel("Chamfer Distance: ")
        self.chamferDistanceValue = qt.QLabel("0.0")
        distanceLayout.addRow(self.chamferDistance, self.chamferDistanceValue)

        self.averageSymmetricSurfaceDistance = qt.QLabel("Average Symmetric Surface Distance: ")
        self.averageSymmetricSurfaceDistanceValue = qt.QLabel("0.0")
        distanceLayout.addRow(self.averageSymmetricSurfaceDistance, self.averageSymmetricSurfaceDistanceValue)

        self.fScore = qt.QLabel("F-Score: ")
        self.fScoreValue = qt.QLabel("0.0")
        distanceLayout.addRow(self.fScore, self.fScoreValue)

        self.precision = qt.QLabel("Precision: ")
        self.precisionValue = qt.QLabel("0.0")
        distanceLayout.addRow(self.precision, self.precisionValue)

        self.recall = qt.QLabel("Recall: ")
        self.recallValue = qt.QLabel("0.0")
        distanceLayout.addRow(self.recall, self.recallValue)

        self._inputModelNode = None
        self._outputModelNode = None
        self._showingCompletedModel = True

        self.toggleModelsButton = qt.QPushButton("Toggle Models")
        self.toggleModelsButton.setEnabled(False)
        self.toggleModelsButton.connect("clicked(bool)", self.onToggleModels)
        self.layout.addWidget(self.toggleModelsButton)

        self.addRefreshSceneButton(self.layout)
        self.layout.addStretch(1)
        self.updateRunButton()

    # Default view - 3D only, black bg, no labels
    def enter(self):
        self.setDefaultThreeDLayout()

    # Toggle Uncertainty Inputs
    def onToggleUncertaintyOptions(self, state=None):
        enabled = self.estimateUncertaintyCheckbox.isChecked()
        self.propagationModeCombobox.setEnabled(enabled)
        self.dataStdInput.setEnabled(enabled)
        self.latentStdInput.setEnabled(enabled)
        self.dataWeightInput.setEnabled(enabled)
        self.latentWeightInput.setEnabled(enabled)
        self.nTrianglesInput.setEnabled(enabled)
        
        is_mc = self.propagationModeCombobox.currentText == "montecarlo"
        self.nSamplesInput.setEnabled(enabled and is_mc)

    # Toggle Fast Mode Inputs
    def onToggleFastMode(self, state=None):
        fast = self.fastModeCheckbox.isChecked()
        self.encoderCkptInput.setEnabled(fast)
        self.refineItersInput.setEnabled(fast)
        # In fast mode the encoder replaces the optimization phases; grey them out.
        for w in (self.phase1ItersInput, self.phase1LrInput, self.phase1LambdaInput,
                  self.phase2ItersInput, self.phase2LrInput, self.phase2LambdaInput):
            w.setEnabled(not fast)

    def updateRunButton(self):
        self.runButton.setEnabled(self.commonInputsReady())

    def onAfterSceneCleared(self):
        self.outputPath = None
        self._resultPath = None
        self._inputModelNode = None
        self._outputModelNode = None
        self._showingCompletedModel = True
        self.toggleModelsButton.setEnabled(False)
        self.toggleModelsButton.setText("Show Original Model")
        self.calculateDistsButton.setEnabled(False)
        self.chamferDistanceValue.setText("0.0")
        self.averageSymmetricSurfaceDistanceValue.setText("0.0")
        self.fScoreValue.setText("0.0")
        self.precisionValue.setText("0.0")
        self.recallValue.setText("0.0")
        self.statusLog.clear()
        self.setDefaultThreeDLayout()

    # Inference
    def onRunInference(self):
        self.runButton.setEnabled(False)
        self.progressBar.setVisible(True)
        self.onLogMessage("Starting inference...")

        if self._inputModelNode and slicer.mrmlScene.IsNodePresent(self._inputModelNode):
            slicer.mrmlScene.RemoveNode(self._inputModelNode)
        if self._outputModelNode and slicer.mrmlScene.IsNodePresent(self._outputModelNode):
            slicer.mrmlScene.RemoveNode(self._outputModelNode)
            self._outputModelNode = None

        self._inputModelNode = slicer.util.loadModel(self.inputFilePath)
        self._inputModelNode.SetName("Original Input Mesh")
        self._inputModelNode.CreateDefaultDisplayNodes()
        self._inputModelNode.GetDisplayNode().SetColor(0.7, 0.7, 0.7)
        self._inputModelNode.GetDisplayNode().SetVisibility(False)

        base = os.path.splitext(os.path.basename(self.inputFilePath))[0]
        outDir = os.path.join(self.outputFolderPath, base) 
        os.makedirs(outDir, exist_ok=True)
        self._resultPath = os.path.join(outDir, base + ".done")
        self.onLogMessage("Result path: " + self._resultPath)

        # Path to the worker script (lives next to this module)
        workerScript = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shape_completion_slicer.py")

        # Use Slicer's own Python so the environment matches
        pythonExe = sys.executable

        # Redirect stdout to a log file instead of a pipe
        self._logFilePath = os.path.join(self.outputFolderPath, base, "shape_completion_log.txt")
        os.makedirs(os.path.dirname(self._logFilePath), exist_ok=True)
        self._logFile = open(self._logFilePath, "w")
        self._logReadPos = 0

        cmd = [
            pythonExe, workerScript,
            "--config", self.configFilePath,
            "--model", self.modelFilePath,
            "--latent_codes", self.latentCodesFilePath,
            "--input_mesh", self.inputFilePath,
            "--output_folder", outDir,
            # Optimization settings
            "--n_samples", self.nSamplesOptInput.text,
            "--phase1_iters", self.phase1ItersInput.text,
            "--phase1_lr", self.phase1LrInput.text,
            "--phase1_lambda_reg", self.phase1LambdaInput.text,
            "--phase2_iters", self.phase2ItersInput.text,
            "--phase2_lr", self.phase2LrInput.text,
            "--phase2_lambda_reg", self.phase2LambdaInput.text,
            "--n_pts_per_axis", self.nPtsPerAxisInput.text,
        ]
        if self.fastModeCheckbox.isChecked():
            cmd.append("--fast_mode")
            cmd.extend(["--refine_iters", self.refineItersInput.text])
            cmd.extend(["--refine_lr", self.refineLrInput.text])
            cmd.extend(["--refine_lambda_reg", self.refineLambdaInput.text])
            if self.encoderCkptInput.text.strip():
                cmd.extend(["--encoder_ckpt", self.encoderCkptInput.text.strip()])
                
        if self.estimateUncertaintyCheckbox.isChecked():
            cmd.append("--estimate_uncertainty")
            cmd.extend(["--propagation_mode", self.propagationModeCombobox.currentText])
            cmd.extend(["--data_std", self.dataStdInput.text])
            cmd.extend(["--latent_prior_std", self.latentStdInput.text])
            cmd.extend(["--data_weight", self.dataWeightInput.text])
            cmd.extend(["--latent_weight", self.latentWeightInput.text])
            cmd.extend(["--n_triangles", self.nTrianglesInput.text])
            cmd.extend(["--mc_samples", self.nSamplesInput.text])

        self._process = subprocess.Popen(cmd, stdout=self._logFile, stderr=subprocess.STDOUT)

        # Poll every 500 ms instead of blocking
        self._pollTimer = qt.QTimer()
        self._pollTimer.setInterval(500)
        self._pollTimer.timeout.connect(self._pollSubprocess)
        self._pollTimer.start()

    # Chamfer, ASSD, F-Score, Precision, Recall
    def onCalculateDistances(self):       
        mp = pv.read(self.outputPath).triangulate().extract_geometry()
        gt = pv.read(self.inputFilePath).triangulate().extract_geometry()
        # Sample points across surface
        sp = _uniform_surface_sample(mp, int(self.sampleCountValueInput.text))
        sg = _uniform_surface_sample(gt, int(self.sampleCountValueInput.text))

        chamfer = chamfer_distance(sg, sp)
        fscore, precision, recall = f_score(sg, sp, d=float(self.similarityThresholdValueInput.text))
        assd = ave_sym_surface_distance(sg, sp)
        self.chamferDistanceValue.setText(f"{chamfer:.6f}")
        self.averageSymmetricSurfaceDistanceValue.setText(f"{assd:.6f}")
        self.fScoreValue.setText(f"{fscore:.6f}")
        self.precisionValue.setText(f"{precision:.6f}")
        self.recallValue.setText(f"{recall:.6f}")

    def applyUncertaintyVisualization(self, modelNode):
        polyData = modelNode.GetPolyData()
        if not polyData:
            self.onLogMessage("No polydata found on output model.")
            return False
        pointData = polyData.GetPointData()
        if not pointData:
            self.onLogMessage("No point data found on output model.")
            return False

        # Prefer micro-unit scalar for readable legend.
        scalarName = "SdfUncertainty_um"
        scalarArray = pointData.GetArray(scalarName)
        # Fallback to raw scalar if old output files do not contain the _um field.
        if scalarArray is None:
            scalarName = "SdfUncertainty"
            scalarArray = pointData.GetArray(scalarName)
        if scalarArray is None:
            self.onLogMessage("\nNo SdfUncertainty scalar field found.")
            return False

        self.onLogMessage(f"Applying uncertainty visualization using {scalarName}.")

        displayNode = modelNode.GetDisplayNode()
        if not displayNode:
            modelNode.CreateDefaultDisplayNodes()
            displayNode = modelNode.GetDisplayNode()
        if not displayNode:
            self.onLogMessage("Could not create display node.")
            return False

        # Activate point scalar data
        try:
            if hasattr(displayNode, "SetActiveAttributeLocation"):
                displayNode.SetActiveAttributeLocation(vtk.vtkAssignAttribute.POINT_DATA)
        except Exception as e:
            self.onLogMessage(f"Could not set active attribute location: {str(e)}")

        displayNode.SetActiveScalarName(scalarName)
        displayNode.SetScalarVisibility(True)

        # Use manual scalar range so colorbar is stable and meaningful
        scalarRange = scalarArray.GetRange()
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
        displayNode.SetScalarRange(scalarRange[0], scalarRange[1])

        # Try to find Viridis robustly
        colorNode = None
        for node in slicer.util.getNodesByClass("vtkMRMLColorNode"):
            if "viridis" in node.GetName().lower():
                colorNode = node
                break
        if colorNode:
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())
        else:
            self.onLogMessage("Viridis color map not found. Using default scalar color map.")

        # Add color legend / colorbar
        displayNode.SetOpacity(1.0)
        displayNode.Modified()
        try:
            colorLegendDisplayNode = slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(modelNode)
            if colorLegendDisplayNode:
                colorLegendDisplayNode.SetVisibility(True)

                if scalarName == "SdfUncertainty_um":
                    colorLegendDisplayNode.SetTitleText("SDF uncertainty (µ)")
                    colorLegendDisplayNode.SetLabelFormat("%.1f")
                else:
                    colorLegendDisplayNode.SetTitleText("SDF uncertainty")
                    colorLegendDisplayNode.SetLabelFormat("%.2e")
                colorLegendDisplayNode.Modified()
        except Exception as e:
            self.onLogMessage(f"Could not show color legend: {str(e)}")
        
        return True
    
    def onToggleModels(self):
        if not self._inputModelNode or not self._outputModelNode:
            return
        inputDisplay = self._inputModelNode.GetDisplayNode()
        outputDisplay = self._outputModelNode.GetDisplayNode()
        if not inputDisplay or not outputDisplay:
            return

        self._showingCompletedModel = not self._showingCompletedModel
        if self._showingCompletedModel:
            inputDisplay.SetVisibility(False)
            outputDisplay.SetVisibility(True)
            self.toggleModelsButton.setText("Show Original Model")
        else:
            inputDisplay.SetVisibility(True)
            outputDisplay.SetVisibility(False)
            self.toggleModelsButton.setText("Show Shape Completed Model")

    def _pollSubprocess(self):
        try:
            with open(self._logFilePath, "r", errors="replace") as f:
                f.seek(self._logReadPos)
                new_text = f.read()
                if new_text:
                    self._logReadPos = f.tell()
                    for line in new_text.splitlines():
                        if line.strip():
                            self.onLogMessage(line)
        except FileNotFoundError:
            pass

        retcode = self._process.poll()
        if retcode is not None:
            self._pollTimer.stop()
            self._logFile.close()
            self.progressBar.setVisible(False)
            self.runButton.setEnabled(True)

            if retcode == 0 and os.path.exists(self._resultPath):
                with open(self._resultPath) as f:
                    self.outputPath = f.read().strip()
                if not os.path.isabs(self.outputPath):
                    self.outputPath = os.path.join(os.path.dirname(self._resultPath), self.outputPath)
                try:
                    os.remove(self._resultPath)
                except OSError:
                    pass
                self.onLogMessage(f"\nInference complete. \n\nOutput saved to: {self.outputPath}", color = "#4CAF50")
                outputNode = slicer.util.loadModel(self.outputPath)
                outputNode.SetName("Predicted Mesh")
                outputNode.CreateDefaultDisplayNodes()
                self._outputModelNode = outputNode
                displayNode = outputNode.GetDisplayNode()

                appliedUncertainty = self.applyUncertaintyVisualization(outputNode)
                if not appliedUncertainty:
                    displayNode.SetScalarVisibility(False)
                    displayNode.SetColor(0.9, 0.6, 0.2)
                    displayNode.SetAmbient(0.3)
                    displayNode.SetDiffuse(0.8)
                    displayNode.SetSpecular(0.0)
                
                if self._inputModelNode and self._inputModelNode.GetDisplayNode():
                    self._inputModelNode.GetDisplayNode().SetVisibility(False)
                if self._outputModelNode and self._outputModelNode.GetDisplayNode():
                    self._outputModelNode.GetDisplayNode().SetVisibility(True)
                self._showingCompletedModel = True
                self.toggleModelsButton.setEnabled(True)
                self.toggleModelsButton.setText("Show Original Model")

                slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
                self.calculateDistsButton.setEnabled(True)
            else:
                self.onLogMessage(f"Inference failed (exit code {retcode}, file exists: {os.path.exists(self._resultPath)} at {self._resultPath})")

class ShapeCompletionLogic(FossilNsmLogic):
    pass