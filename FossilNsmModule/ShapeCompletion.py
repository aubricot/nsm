import os
import qt
import ctk
import sys
import slicer
import subprocess
import json
import glob
from slicer.ScriptedLoadableModule import *


class ShapeCompletion(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Shape Completion"
        self.parent.categories = ["FossilNSM"]
        self.parent.contributors = ["Wolcott et all"]
        self.parent.helpText = "A shape completion module."


class ShapeCompletionWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        super().setup()

        # Initialize states
        self.modelRootPath = None
        self.inputFilePath = None
        self.configFilePath = None
        self.modelFilePath = None
        self.latentCodesFilePath = None
        self.outputFolderPath = None
        self.referenceMeshDirectory = None
        self.classificationMatches = []
        self._classificationNodes = []

        ShapeCompletionLogic.installDependenciesIfNeeded()

        # Form Layout
        inputCollapsible = ctk.ctkCollapsibleButton()
        inputCollapsible.text = "Inputs"
        self.layout.addWidget(inputCollapsible)
        inputLayout = qt.QFormLayout(inputCollapsible)

        # Model Root (ex: run_v44)
        self.modelRootButton = qt.QPushButton("Select Model Root (run_vXX)...")
        self.modelRootLabel = qt.QLabel("No model selected")
        inputLayout.addRow("Model Root:", self.modelRootButton)
        inputLayout.addRow("", self.modelRootLabel)
        self.modelRootButton.connect("clicked(bool)", self.onSelectModelRoot)

        # Validation output
        self.modelChecklist = qt.QPlainTextEdit()
        self.modelChecklist.setReadOnly(True)
        self.modelChecklist.setFixedHeight(110)
        inputLayout.addRow("Model Validation:", self.modelChecklist)

        # Derived paths (READ ONLY)
        self.configFileLabel = qt.QLabel("Not set")
        self.configFileLabel.setWordWrap(True)
        inputLayout.addRow("Config File:", self.configFileLabel)

        self.modelFileLabel = qt.QLabel("Not set")
        self.modelFileLabel.setWordWrap(True)
        inputLayout.addRow("Model File:", self.modelFileLabel)

        self.latentCodesFileLabel = qt.QLabel("Not set")
        self.latentCodesFileLabel.setWordWrap(True)
        inputLayout.addRow("Latent Codes File:", self.latentCodesFileLabel)

        self.outputFolderLabel = qt.QLabel("Not set")
        self.outputFolderLabel.setWordWrap(True)
        inputLayout.addRow("Output Folder:", self.outputFolderLabel)

        # Input Mesh File 
        self.inputFileButton = qt.QPushButton("Select Input Mesh...")
        self.inputFileButton.connect("clicked(bool)", self.onSelectInputFile)
        self.inputFileLabel = qt.QLabel("No file selected")
        self.inputFileLabel.setWordWrap(True)
        inputLayout.addRow("Input Mesh:", self.inputFileButton)
        inputLayout.addRow("", self.inputFileLabel)

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
        self.statusLog = qt.QPlainTextEdit()
        self.statusLog.setReadOnly(True)
        self.statusLog.setFixedHeight(120)
        self.layout.addWidget(self.statusLog)

        # Classification is kept separate from completion. An intact mesh can be
        # classified directly, while a completed mesh can be selected afterwards.
        classificationCollapsible = ctk.ctkCollapsibleButton()
        classificationCollapsible.text = "Classification: Top-5 closest meshes"
        self.layout.addWidget(classificationCollapsible)
        classificationLayout = qt.QFormLayout(classificationCollapsible)

        self.referenceMeshesButton = qt.QPushButton("Select Reference Mesh Library...")
        self.referenceMeshesButton.connect("clicked(bool)", self.onSelectReferenceMeshDirectory)
        self.referenceMeshesLabel = qt.QLabel("Optional: needed to visualize returned meshes")
        self.referenceMeshesLabel.setWordWrap(True)
        classificationLayout.addRow("Reference Mesh Library:", self.referenceMeshesButton)
        classificationLayout.addRow("", self.referenceMeshesLabel)

        self.classificationIterationsInput = qt.QLineEdit("1000")
        classificationLayout.addRow("Latent Optimization Iterations:", self.classificationIterationsInput)
        self.classifyButton = qt.QPushButton("Classify Input Mesh")
        self.classifyButton.setEnabled(False)
        self.classifyButton.connect("clicked(bool)", self.onClassifyInputMesh)
        classificationLayout.addRow(self.classifyButton)

        self.classificationTable = qt.QTableWidget(0, 4)
        self.classificationTable.setHorizontalHeaderLabels(["Rank", "Reference mesh", "Cosine distance", "Available"])
        self.classificationTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.classificationTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.classificationTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.classificationTable.horizontalHeader().setStretchLastSection(True)
        classificationLayout.addRow("Top-5 matches:", self.classificationTable)
        self.showSelectedMatchButton = qt.QPushButton("Show Selected Match")
        self.showSelectedMatchButton.setEnabled(False)
        self.showSelectedMatchButton.connect("clicked(bool)", self.onShowSelectedMatch)
        self.showAllMatchesButton = qt.QPushButton("Show All Available Matches")
        self.showAllMatchesButton.setEnabled(False)
        self.showAllMatchesButton.connect("clicked(bool)", self.onShowAllMatches)
        matchButtons = qt.QWidget()
        matchButtonsLayout = qt.QHBoxLayout(matchButtons)
        matchButtonsLayout.setContentsMargins(0, 0, 0, 0)
        matchButtonsLayout.addWidget(self.showSelectedMatchButton)
        matchButtonsLayout.addWidget(self.showAllMatchesButton)
        classificationLayout.addRow(matchButtons)

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

        self.layout.addStretch(1)

        # Update labels to reflect pre-filled defaults
        if self.configFilePath:
            self.configFileLabel.setText(self.configFilePath)
        if self.modelFilePath:
            self.modelFileLabel.setText(self.modelFilePath)
        if self.latentCodesFilePath:
            self.latentCodesFileLabel.setText(self.latentCodesFilePath)
        if self.outputFolderPath:
            self.outputFolderLabel.setText(self.outputFolderPath)

        self.updateRunButton()

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

    # File/Folder Selection
    def onSelectConfigFile(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Config File", "", "Config Files (*.json)"
        )
        if path:
            self.configFilePath = path
            self.configFileLabel.setText(path)
            self.updateRunButton()

    # Auto populate filepaths from model root and validate
    def validateModelRoot(self, rootDir):
        checks = []

        def check(label, path, isDir=False):
            ok = os.path.isdir(path) if isDir else os.path.isfile(path)
            checks.append((label, ok, path))
            return ok

        check("model_params_config.json", os.path.join(rootDir, "model_params_config.json"))
        check("model/ folder", os.path.join(rootDir, "model"), isDir=True)
        check("latent_codes/ folder", os.path.join(rootDir, "latent_codes"), isDir=True)
        # Optional output folder (not required)
        check("shape_completion/ (optional)", os.path.join(rootDir, "shape_completion"), isDir=True)
        return checks

    # Print validated filepath checklist in UI
    def updateChecklistUI(self, rootDir):
        checks = self.validateModelRoot(rootDir)
        lines = []
        allRequiredOk = True

        for label, ok, path in checks:
            icon = "✔" if ok else "✖"
            lines.append(f"{icon} {label}")
            if not ok and "optional" not in label:
                allRequiredOk = False

        self.modelChecklist.setPlainText("\n".join(lines))
        return allRequiredOk

    # From specified model root, resolve other default paths
    def resolveModelRoot(self, rootDir):
        config = os.path.join(rootDir, "model_params_config.json")
        modelDir = os.path.join(rootDir, "model")
        latentDir = os.path.join(rootDir, "latent_codes")
        outputDir = os.path.join(rootDir, "shape_completion")

        # Validate required structure
        missing = []
        if not os.path.isfile(config):
            missing.append("model_params_config.json")
        if not os.path.isdir(modelDir):
            missing.append("model/")
        if not os.path.isdir(latentDir):
            missing.append("latent_codes/")

        if missing:
            raise ValueError(f"Invalid model package. Missing: {missing}")

        # Try to auto-detect files inside folders
        modelFiles = sorted([f for f in os.listdir(modelDir) if f.endswith(".pth")])
        latentFiles = sorted([f for f in os.listdir(latentDir) if f.endswith(".pth")])

        if not modelFiles:
            raise ValueError("No .pth file found in model/")
        if not latentFiles:
            raise ValueError("No .pth file found in latent_codes/")

        modelPath = os.path.join(modelDir, modelFiles[-1])
        latentPath = os.path.join(latentDir, latentFiles[-1])

        return config, modelPath, latentPath, outputDir

    # Select model root dir
    def onSelectModelRoot(self):
        path = qt.QFileDialog.getExistingDirectory(None, "Select Model Root Folder")
        if not path:
            return

        # Reset paths to avoid stale paths causing problems
        self.configFilePath = None
        self.modelFilePath = None
        self.latentCodesFilePath = None
        self.outputFolderPath = None

        self.modelRootPath = path
        self.modelRootLabel.setText(path)

        ok = self.updateChecklistUI(path)
        if not ok:
            self.onLogMessage("Model root is incomplete. Fix missing files before running.")
            self.runButton.setEnabled(False)
            return

        config, model, latents, output = self.resolveModelRoot(path)

        self.configFilePath = config
        self.modelFilePath = model
        self.latentCodesFilePath = latents
        self.outputFolderPath = output

        self.configFileLabel.setText(config)
        self.modelFileLabel.setText(model)
        self.latentCodesFileLabel.setText(latents)
        self.outputFolderLabel.setText(output)

        self.updateRunButton()

    # Input mesh selector
    def onSelectInputFile(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Input Mesh", "", "Mesh Files (*.vtk *.vtp *.stl *.obj *.ply)"
        )
        if not path:
            return
        self.inputFilePath = path
        self.inputFileLabel.setText(path)
        self.updateRunButton()

    def onSelectReferenceMeshDirectory(self):
        path = qt.QFileDialog.getExistingDirectory(None, "Select Reference Mesh Library")
        if not path:
            return
        self.referenceMeshDirectory = path
        self.referenceMeshesLabel.setText(path)
        self._updateClassificationTable()

    def updateRunButton(self):
        ready = bool(self.modelRootPath and self.inputFilePath)
        self.runButton.setEnabled(ready)
        self.classifyButton.setEnabled(ready)

    def onClassifyInputMesh(self):
        try:
            iterations = int(self.classificationIterationsInput.text)
            if iterations < 1:
                raise ValueError
        except ValueError:
            slicer.util.errorDisplay("Latent optimization iterations must be a positive integer.")
            return

        resultDirectory = os.path.join(self.outputFolderPath, "classification")
        os.makedirs(resultDirectory, exist_ok=True)
        inputBase = os.path.splitext(os.path.basename(self.inputFilePath))[0]
        self._classificationResultPath = os.path.join(resultDirectory, inputBase + "_top5.json")
        logPath = os.path.join(resultDirectory, inputBase + "_classification.log")
        workerScript = os.path.join(os.path.dirname(os.path.dirname(__file__)), "classification_slicer.py")
        self._classificationLog = open(logPath, "w")
        self._classificationLogReadPos = 0
        self._classificationLogPath = logPath
        self.classifyButton.setEnabled(False)
        self.onLogMessage("Starting nearest-mesh classification (latent optimization may take several minutes)...")
        self._classificationProcess = subprocess.Popen([
            sys.executable, workerScript, "--config", self.configFilePath,
            "--model", self.modelFilePath, "--latent_codes", self.latentCodesFilePath,
            "--input_mesh", self.inputFilePath, "--output_dir", resultDirectory,
            "--result", self._classificationResultPath, "--iterations", str(iterations),
        ], stdout=self._classificationLog, stderr=subprocess.STDOUT)
        self._classificationTimer = qt.QTimer()
        self._classificationTimer.setInterval(500)
        self._classificationTimer.timeout.connect(self._pollClassification)
        self._classificationTimer.start()

    def _pollClassification(self):
        try:
            with open(self._classificationLogPath, "r", errors="replace") as stream:
                stream.seek(self._classificationLogReadPos)
                text = stream.read()
                self._classificationLogReadPos = stream.tell()
                for line in text.splitlines():
                    if line.strip():
                        self.onLogMessage(line)
        except FileNotFoundError:
            pass
        code = self._classificationProcess.poll()
        if code is None:
            return
        self._classificationTimer.stop()
        self._classificationLog.close()
        self.classifyButton.setEnabled(True)
        if code != 0 or not os.path.isfile(self._classificationResultPath):
            self.onLogMessage("Classification failed (exit code {}).".format(code))
            return
        with open(self._classificationResultPath, encoding="utf-8") as stream:
            self.classificationMatches = json.load(stream).get("matches", [])
        self._updateClassificationTable()
        self.onLogMessage("Classification complete. Top-5 matches saved to " + self._classificationResultPath)

    def _referencePath(self, match):
        if not self.referenceMeshDirectory:
            return None
        name = match.get("mesh_name", "")
        direct = os.path.join(self.referenceMeshDirectory, name)
        if os.path.isfile(direct):
            return direct
        found = glob.glob(os.path.join(self.referenceMeshDirectory, "**", name), recursive=True)
        return found[0] if found else None

    def _updateClassificationTable(self):
        self.classificationTable.setRowCount(len(self.classificationMatches))
        available = 0
        for row, match in enumerate(self.classificationMatches):
            meshPath = self._referencePath(match)
            values = [str(match["rank"]), match["mesh_name"], "{:.6f}".format(match["cosine_distance"]),
                      "Yes" if meshPath else "No — select matching library"]
            for column, value in enumerate(values):
                self.classificationTable.setItem(row, column, qt.QTableWidgetItem(value))
            if meshPath:
                available += 1
        self.showSelectedMatchButton.setEnabled(available > 0)
        self.showAllMatchesButton.setEnabled(available > 0)

    def _clearClassificationNodes(self):
        for node in self._classificationNodes:
            if node and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self._classificationNodes = []

    def _showMatch(self, match, offset=0.0):
        meshPath = self._referencePath(match)
        if not meshPath:
            self.onLogMessage("Reference mesh is not available: " + match["mesh_name"])
            return None
        node = slicer.util.loadModel(meshPath)
        node.SetName("Top {} — {} ({:.4f})".format(match["rank"], match["mesh_name"], match["cosine_distance"]))
        node.CreateDefaultDisplayNodes()
        node.GetDisplayNode().SetColor(0.2, 0.65, 0.9)
        if offset:
            transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", node.GetName() + " transform")
            transform.GetTransformToParent().Translate(offset, 0, 0)
            node.SetAndObserveTransformNodeID(transform.GetID())
            self._classificationNodes.append(transform)
        self._classificationNodes.append(node)
        return node

    def onShowSelectedMatch(self):
        rows = self.classificationTable.selectionModel().selectedRows()
        if not rows:
            slicer.util.infoDisplay("Select a result row first.")
            return
        self._clearClassificationNodes()
        self._showMatch(self.classificationMatches[rows[0].row()])

    def onShowAllMatches(self):
        self._clearClassificationNodes()
        for index, match in enumerate(self.classificationMatches):
            self._showMatch(match, offset=index * 50.0)
        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()

    # Inference
    def onRunInference(self):
        self.runButton.setEnabled(False)
        self.progressBar.setVisible(True)
        self.onLogMessage("Starting inference...")

        base = os.path.splitext(os.path.basename(self.inputFilePath))[0]
        self._resultPath = os.path.join(self.outputFolderPath, base + ".done")
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
            "--output_folder", self.outputFolderPath,
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
        if self.estimateUncertaintyCheckbox.isChecked():
            cmd.append("--estimate_uncertainty")
            cmd.extend(["--propagation_mode", self.propagationModeCombobox.currentText])
            cmd.extend(["--data_std", self.dataStdInput.text])
            cmd.extend(["--latent_prior_std", self.latentStdInput.text])
            cmd.extend(["--data_weight", self.dataWeightInput.text])
            cmd.extend(["--latent_weight", self.latentWeightInput.text])
            cmd.extend(["--n_triangles", self.nTrianglesInput.text])
            cmd.extend(["--mc_samples", self.nSamplesInput.text])

        self._process = subprocess.Popen(
            cmd,
            stdout=self._logFile,
            stderr=subprocess.STDOUT,
        )

        # Poll every 500 ms instead of blocking
        self._pollTimer = qt.QTimer()
        self._pollTimer.setInterval(500)
        self._pollTimer.timeout.connect(self._pollSubprocess)
        self._pollTimer.start()

    # Chamfer, ASSD, F-Score, Precision, Recall
    def onCalculateDistances(self):
        import pyvista as pv
        from utils.utils import _uniform_surface_sample, chamfer_distance, f_score, ave_sym_surface_distance
        
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
        import vtk

        polyData = modelNode.GetPolyData()
        if not polyData:
            self.onLogMessage("No polydata found on output model.")
            return

        pointData = polyData.GetPointData()
        if not pointData:
            self.onLogMessage("No point data found on output model.")
            return

        # Prefer micro-unit scalar for readable legend.
        scalarName = "SdfUncertainty_um"
        scalarArray = pointData.GetArray(scalarName)

        # Fallback to raw scalar if old output files do not contain the _um field.
        if scalarArray is None:
            scalarName = "SdfUncertainty"
            scalarArray = pointData.GetArray(scalarName)

        if scalarArray is None:
            self.onLogMessage("No SdfUncertainty scalar field found.")
            return

        self.onLogMessage(f"Applying uncertainty visualization using {scalarName}.")

        displayNode = modelNode.GetDisplayNode()
        if not displayNode:
            modelNode.CreateDefaultDisplayNodes()
            displayNode = modelNode.GetDisplayNode()

        if not displayNode:
            self.onLogMessage("Could not create display node.")
            return

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

        # Optional: for cleaner visualization, use percentiles instead of full min/max.
        # For exact notebook-like auto range, keep scalarRange as above.
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

        displayNode.SetOpacity(1.0)
        displayNode.Modified()

        # Add color legend / colorbar
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

    def _pollSubprocess(self):
        # Read any new lines from the log file (never blocks)
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
            # Process finished
            self._pollTimer.stop()
            self._logFile.close()
            self.progressBar.setVisible(False)
            self.runButton.setEnabled(True)

            if retcode == 0 and os.path.exists(self._resultPath):
                with open(self._resultPath) as f:
                    self.outputPath = f.read().strip()
                self.onLogMessage(f"Inference complete: {self.outputPath}")
                outputNode = slicer.util.loadModel(self.outputPath)
                outputNode.SetName("Predicted Mesh")

                # Configure visualization if uncertainty scalars are present
                self.applyUncertaintyVisualization(outputNode)

                slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
                self.calculateDistsButton.setEnabled(True)
            else:
                self.onLogMessage(f"Inference failed (exit code {retcode}, file exists: {os.path.exists(self._resultPath)} at {self._resultPath})")


    def onLogMessage(self, message):
        self.statusLog.appendPlainText(str(message))


class ShapeCompletionLogic(ScriptedLoadableModuleLogic):

    @staticmethod
    def installDependenciesIfNeeded():
        # Conditionally use tiny3d (o3d) for Slicer
        USE_TINY3D = True  # Set this to False in normal use (outside Slicer)

        if USE_TINY3D:
            try:
                import tiny3d as o3d  # Import tiny3d as o3d for Slicer logic
            except ImportError:
                slicer.util.pip_install('tiny3d')
                import tiny3d as o3d  # Try again after installing
        else:
            try:
                import open3d as o3d  # Use open3d for the rest of the project
            except ImportError:
                slicer.util.pip_install('open3d')
                import open3d as o3d  # Try again after installing

        try:
            import cv2
        except ImportError:
            slicer.util.pip_install('opencv-python')

        try:
            import nibabel
        except ImportError:
            slicer.util.pip_install('nibabel')

        try:
            import pymskt
        except ImportError:
            slicer.util.pip_install('mskt')

        try:
            import pyvista
        except ImportError:
            slicer.util.pip_install('pyvista')

        try:
            import pymeshfix
        except ImportError:
            slicer.util.pip_install('pymeshfix')

        try:
            import skimage
        except ImportError:
            slicer.util.pip_install('scikit-image')

        try:
            import sklearn
        except ImportError:
            slicer.util.pip_install('scikit-learn')

        try:
            import torch
        except ImportError:
            slicer.util.pip_install('torch')

        try:
            import vtk
        except ImportError:
            slicer.util.pip_install('vtk')
