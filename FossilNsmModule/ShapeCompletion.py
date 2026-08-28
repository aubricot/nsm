import slicer
from FossilNsmCommon import FossilNsmCommonWidget, FossilNsmLogic
FossilNsmLogic.installDependenciesIfNeeded()
import os
import csv
import json
import qt
import ctk
import vtk
import sys
import subprocess
from slicer.ScriptedLoadableModule import *
import pyvista as pv

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
        self.tabWidget = qt.QTabWidget()
        self.layout.addWidget(self.tabWidget)

        # Tab 1 — Inference
        inferenceTab = qt.QWidget()
        inferenceLayout = qt.QVBoxLayout(inferenceTab)
        inferenceLayout.setContentsMargins(0, 0, 0, 0)
        self.tabWidget.addTab(inferenceTab, "Inference")
        self.addFossilNsmInputSection(inferenceLayout)
        self.addBatchFolderInput(label="Input Folder (Batch):", hint="No folder selected (for batch completion)")

        # Fast Mode Collapsible Layout
        fastCollapsible = ctk.ctkCollapsibleButton()
        fastCollapsible.text = "Fast Mode (Encoder)"
        fastCollapsible.collapsed = True
        inferenceLayout.addWidget(fastCollapsible)
        fastLayout = qt.QFormLayout(fastCollapsible)

        self.fastModeCheckbox = qt.QCheckBox("Use encoder (skip latent optimization)")
        self.fastModeCheckbox.setChecked(True)
        fastLayout.addRow("", self.fastModeCheckbox)

        self.encoderCkptInput = qt.QLineEdit("encoder/checkpoints/encoder.pt")
        fastLayout.addRow("Encoder Checkpoint:", self.encoderCkptInput)
        self.refineItersInput = qt.QLineEdit("200")
        fastLayout.addRow("Refine Iterations:", self.refineItersInput)
        self.refineLrInput = qt.QLineEdit("1e-3")
        fastLayout.addRow("Refine Learning Rate:", self.refineLrInput)
        self.refineLambdaInput = qt.QLineEdit("1e-6")
        fastLayout.addRow("Refine Lambda Reg:", self.refineLambdaInput)

        # Optimization Settings Collapsible Layout
        optimCollapsible = ctk.ctkCollapsibleButton()
        optimCollapsible.text = "Optimization Settings"
        optimCollapsible.collapsed = True
        inferenceLayout.addWidget(optimCollapsible)
        optimLayout = qt.QFormLayout(optimCollapsible)

        self.nSamplesOptInput = qt.QLineEdit("240")
        optimLayout.addRow("Sample Points:", self.nSamplesOptInput)

        self.phase1ItersInput = qt.QLineEdit("3000")
        optimLayout.addRow("Phase 1 Iterations:", self.phase1ItersInput)
        self.phase1LrInput = qt.QLineEdit("1e-4")
        optimLayout.addRow("Phase 1 Learning Rate:", self.phase1LrInput)
        self.phase1LambdaInput = qt.QLineEdit("1e-3")
        optimLayout.addRow("Phase 1 Lambda Reg:", self.phase1LambdaInput)

        self.phase2ItersInput = qt.QLineEdit("8000")
        optimLayout.addRow("Phase 2 Iterations:", self.phase2ItersInput)
        self.phase2LrInput = qt.QLineEdit("1e-5")
        optimLayout.addRow("Phase 2 Learning Rate:", self.phase2LrInput)
        self.phase2LambdaInput = qt.QLineEdit("1e-5")
        optimLayout.addRow("Phase 2 Lambda Reg:", self.phase2LambdaInput)

        self.nPtsPerAxisInput = qt.QLineEdit("256")
        optimLayout.addRow("Resolution (pts/axis):", self.nPtsPerAxisInput)

        # Uncertainty Collapsible Layout
        uncertaintyCollapsible = ctk.ctkCollapsibleButton()
        uncertaintyCollapsible.text = "Uncertainty Settings"
        uncertaintyCollapsible.collapsed = True
        inferenceLayout.addWidget(uncertaintyCollapsible)
        uncertaintyLayout = qt.QFormLayout(uncertaintyCollapsible)

        self.estimateUncertaintyCheckbox = qt.QCheckBox("Estimate Uncertainty")
        self.estimateUncertaintyCheckbox.setChecked(False)
        uncertaintyLayout.addRow("", self.estimateUncertaintyCheckbox)

        self.propagationModeCombobox = qt.QComboBox()
        self.propagationModeCombobox.addItems(["analytical", "montecarlo"])
        uncertaintyLayout.addRow("Propagation Mode:", self.propagationModeCombobox)

        self.dataStdInput = qt.QLineEdit("2e-5")
        uncertaintyLayout.addRow("Data Std (sigma_Y):", self.dataStdInput)
        self.latentStdInput = qt.QLineEdit("5e-4")
        uncertaintyLayout.addRow("Latent Std (sigma_z):", self.latentStdInput)
        self.dataWeightInput = qt.QLineEdit("1.0")
        uncertaintyLayout.addRow("Data Weight:", self.dataWeightInput)
        self.latentWeightInput = qt.QLineEdit("1.0")
        uncertaintyLayout.addRow("Latent Weight:", self.latentWeightInput)
        self.nTrianglesInput = qt.QLineEdit("5000")
        uncertaintyLayout.addRow("Decimate Triangles:", self.nTrianglesInput)
        self.nSamplesInput = qt.QLineEdit("2000")
        uncertaintyLayout.addRow("Monte Carlo Samples:", self.nSamplesInput)

        self.estimateUncertaintyCheckbox.connect("stateChanged(int)", self.onToggleUncertaintyOptions)
        self.propagationModeCombobox.connect("currentIndexChanged(int)", self.onToggleUncertaintyOptions)
        self.onToggleUncertaintyOptions()

        self.fastModeCheckbox.connect("stateChanged(int)", self.onToggleFastMode)
        self.onToggleFastMode()

        # Run Button
        self.runButton = qt.QPushButton("Run Inference")
        self.runButton.setEnabled(False)
        self.runButton.connect("clicked(bool)", self.onRunInference)
        inferenceLayout.addWidget(self.runButton)

        # Batch Run Button
        self.runBatchButton = qt.QPushButton("Run Inference (Batch)")
        self.runBatchButton.setEnabled(False)
        self.runBatchButton.connect("clicked(bool)", self.onRunBatchInference)
        inferenceLayout.addWidget(self.runBatchButton)

        # Progress Bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.setVisible(False)
        inferenceLayout.addWidget(self.progressBar)

        # Status Log
        self.statusLog = qt.QTextEdit()
        self.statusLog.setReadOnly(True)
        self.statusLog.setFixedHeight(120)
        inferenceLayout.addWidget(self.statusLog)

        # Load previous results from file
        self.loadPreviousResultsButton = qt.QPushButton("Load Previous Shape Completion Results")
        self.loadPreviousResultsButton.connect("clicked(bool)", self.onLoadPreviousResults)
        inferenceLayout.addWidget(self.loadPreviousResultsButton)

        self._inputModelNode = None
        self._outputModelNode = None
        self._showingCompletedModel = True

        self.toggleModelsButton = qt.QPushButton("Toggle Models")
        self.toggleModelsButton.setEnabled(False)
        self.toggleModelsButton.connect("clicked(bool)", self.onToggleModels)
        inferenceLayout.addWidget(self.toggleModelsButton)

        self.addRefreshSceneButton(inferenceLayout)

        # Tab 2 — Batch Results
        batchResultsTab = qt.QWidget()
        batchResultsLayout = qt.QVBoxLayout(batchResultsTab)
        batchResultsLayout.setContentsMargins(4, 4, 4, 4)

        self.batchStatusLabel = qt.QLabel("Run 'Run Inference (Batch)' to populate batch results.")
        self.batchStatusLabel.setWordWrap(True)
        batchResultsLayout.addWidget(self.batchStatusLabel)

        self.bulkTable = qt.QTableWidget(0, 2)
        self.bulkTable.setHorizontalHeaderLabels(["Input mesh", "Output mesh"])
        self.bulkTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.bulkTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.bulkTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.bulkTable.horizontalHeader().setStretchLastSection(True)
        self.bulkTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        self.bulkTable.connect("itemSelectionChanged()", self.onBulkRowSelected)
        batchResultsLayout.addWidget(self.bulkTable, 1)

        self.exportBatchButton = qt.QPushButton("Export Batch Results (CSV)")
        self.exportBatchButton.connect("clicked(bool)", self.onExportBatchResults)
        batchResultsLayout.addWidget(self.exportBatchButton)

        self._batchTabIndex = self.tabWidget.addTab(batchResultsTab, "Batch Results")

        self._completionMode = "single"
        self._bulkResults = []

        inferenceLayout.addStretch(1)
        self.updateRunButton()

    def enter(self):
        self.setDefaultThreeDLayout()

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

    def onToggleFastMode(self, state=None):
        fast = self.fastModeCheckbox.isChecked()
        self.encoderCkptInput.setEnabled(fast)
        self.refineItersInput.setEnabled(fast)
        for w in (self.phase1ItersInput, self.phase1LrInput, self.phase1LambdaInput,
                  self.phase2ItersInput, self.phase2LrInput, self.phase2LambdaInput):
            w.setEnabled(not fast)

    def updateRunButton(self):
        self.runButton.setEnabled(self.commonInputsReady())
        self.runBatchButton.setEnabled(self.modelReady() and bool(self.inputFolderPath))

    def onSelectInputFolder(self):
        path = qt.QFileDialog.getExistingDirectory(None, "Select Folder of Meshes for Batch Completion")
        if not path:
            return
        self.inputFolderPath = path
        self.inputFolderLabel.setText(path)
        self.updateRunButton()

    def _sharedCmd(self):
        """Return the command prefix shared by single and batch runs."""
        workerScript = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shape_completion_slicer.py")
        cmd = [
            sys.executable, workerScript,
            "--config", self.configFilePath,
            "--model", self.modelFilePath,
            "--latent_codes", self.latentCodesFilePath,
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
        return cmd

    def _startProcess(self, cmd, logPath):
        os.makedirs(os.path.dirname(logPath), exist_ok=True)
        self._logFilePath = logPath
        self._logFile = open(logPath, "w")
        self._logReadPos = 0
        self.runButton.setEnabled(False)
        self.runBatchButton.setEnabled(False)
        self.progressBar.setVisible(True)
        self._process = subprocess.Popen(cmd, stdout=self._logFile, stderr=subprocess.STDOUT)
        self._pollTimer = qt.QTimer()
        self._pollTimer.setInterval(500)
        self._pollTimer.timeout.connect(self._pollSubprocess)
        self._pollTimer.start()

    def onRunInference(self):
        self._completionMode = "single"
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

        logPath = os.path.join(outDir, "shape_completion_log.txt")
        cmd = self._sharedCmd() + [
            "--input_mesh", self.inputFilePath,
            "--output_folder", outDir,
        ]
        self._startProcess(cmd, logPath)

    def onRunBatchInference(self):
        if not self.inputFolderPath or not os.path.isdir(self.inputFolderPath):
            slicer.util.errorDisplay("Select an input folder first (Input Folder (Batch)).")
            return
        self._completionMode = "bulk"
        outDir = self.outputFolderPath
        os.makedirs(outDir, exist_ok=True)
        self._resultPath = os.path.join(outDir, "bulk_summary.json")
        logPath = os.path.join(outDir, "bulk_shape_completion_log.txt")
        self.batchStatusLabel.setText("Running batch completion on: " + self.inputFolderPath)
        self.onLogMessage("Starting BATCH shape completion of folder:\n{}\n".format(self.inputFolderPath))
        cmd = self._sharedCmd() + [
            "--input_dir", self.inputFolderPath,
            "--output_folder", outDir,
        ]
        self._startProcess(cmd, logPath)

    def onBulkRowSelected(self):
        row = self.bulkTable.currentRow()
        if row < 0 or row >= len(self._bulkResults):
            return
        result = self._bulkResults[row]

        input_path = result.get("input_path")
        output_path = result.get("output_path")
        if not input_path or not output_path:
            return
        if not os.path.isfile(input_path) or not os.path.isfile(output_path):
            self.onLogMessage("Could not find mesh files for this row.", color="orange")
            return

        # Clear previous nodes
        if self._inputModelNode and slicer.mrmlScene.IsNodePresent(self._inputModelNode):
            slicer.mrmlScene.RemoveNode(self._inputModelNode)
        if self._outputModelNode and slicer.mrmlScene.IsNodePresent(self._outputModelNode):
            slicer.mrmlScene.RemoveNode(self._outputModelNode)

        # Load input
        self.inputFilePath = input_path
        self.inputFileLabel.setText(input_path)
        self._inputModelNode = slicer.util.loadModel(input_path)
        self._inputModelNode.SetName("Original Input Mesh")
        self._inputModelNode.CreateDefaultDisplayNodes()
        self._inputModelNode.GetDisplayNode().SetColor(0.7, 0.7, 0.7)
        self._inputModelNode.GetDisplayNode().SetVisibility(False)

        # Load output
        self.outputPath = output_path
        outputNode = slicer.util.loadModel(output_path)
        outputNode.SetName("Predicted Mesh")
        outputNode.CreateDefaultDisplayNodes()
        self._outputModelNode = outputNode

        appliedUncertainty = self.applyUncertaintyVisualization(outputNode)
        if not appliedUncertainty:
            displayNode = outputNode.GetDisplayNode()
            displayNode.SetScalarVisibility(False)
            displayNode.SetColor(0.9, 0.6, 0.2)
            displayNode.SetAmbient(0.3)
            displayNode.SetDiffuse(0.8)
            displayNode.SetSpecular(0.0)

        self._outputModelNode.GetDisplayNode().SetVisibility(True)
        self._showingCompletedModel = True
        self.toggleModelsButton.setEnabled(True)
        self.toggleModelsButton.setText("Show Original Model")

        self.onLogMessage("Loaded batch result: " + result.get("input_name", ""), color="#4CAF50")
        self.tabWidget.setCurrentIndex(0)  # Switch to Inference tab
        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()

    def onLoadPreviousResults(self):
        startDir = self.outputFolderPath or ""
        path = qt.QFileDialog.getOpenFileName(None, "Select Shape Completion Results", 
                                            startDir, "Results (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            slicer.util.errorDisplay(
                "Could not read results file:\n{}".format(e))
            return

        # Batch: bulk_summary.json
        if isinstance(data, dict) and "results" in data:
            self._bulkResults = data.get("results", [])
            self._completionMode = "bulk"
            self._populateBulkTable()
            self.batchStatusLabel.setText("Loaded {} previous results from {}. " 
                                        "Select a row to inspect it.".format(
                                            len(self._bulkResults), os.path.basename(path)))
            self.onLogMessage("Loaded previous batch results: " + path, color="#4CAF50")
            self.tabWidget.setCurrentIndex(self._batchTabIndex)
            return

        # Single: <mesh>_result.json
        if isinstance(data, dict) and "input_path" in data and "output_path" in data:
            input_path = data.get("input_path")
            output_path = data.get("output_path")
            if not output_path or not os.path.isfile(output_path):
                slicer.util.errorDisplay("Output mesh not found:\n{}".format(output_path))
                return
            if self._inputModelNode and slicer.mrmlScene.IsNodePresent(self._inputModelNode):
                slicer.mrmlScene.RemoveNode(self._inputModelNode)
            if self._outputModelNode and slicer.mrmlScene.IsNodePresent(self._outputModelNode):
                slicer.mrmlScene.RemoveNode(self._outputModelNode)
            if input_path and os.path.isfile(input_path):
                self.inputFilePath = input_path
                self.inputFileLabel.setText(input_path)
                self._inputModelNode = slicer.util.loadModel(input_path)
                self._inputModelNode.SetName("Original Input Mesh")
                self._inputModelNode.CreateDefaultDisplayNodes()
                self._inputModelNode.GetDisplayNode().SetColor(0.7, 0.7, 0.7)
                self._inputModelNode.GetDisplayNode().SetVisibility(False)
            else:
                self.onLogMessage("Original input mesh not found; toggle will be unavailable.", color="orange")

            self.outputPath = output_path
            outputNode = slicer.util.loadModel(output_path)
            outputNode.SetName("Predicted Mesh")
            outputNode.CreateDefaultDisplayNodes()
            self._outputModelNode = outputNode

            appliedUncertainty = self.applyUncertaintyVisualization(outputNode)
            if not appliedUncertainty:
                displayNode = outputNode.GetDisplayNode()
                displayNode.SetScalarVisibility(False)
                displayNode.SetColor(0.9, 0.6, 0.2)
                displayNode.SetAmbient(0.3)
                displayNode.SetDiffuse(0.8)
                displayNode.SetSpecular(0.0)

            self._outputModelNode.GetDisplayNode().SetVisibility(True)
            self._showingCompletedModel = True
            self.toggleModelsButton.setEnabled(bool(self._inputModelNode))
            self.toggleModelsButton.setText("Show Original Model")
            self._completionMode = "single"
            self.onLogMessage("Loaded previous result: " + output_path, color="#4CAF50")
            self.tabWidget.setCurrentIndex(0)
            slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
            return

        slicer.util.errorDisplay("Unrecognized results file. Expected bulk_summary.json "
                                "or a single shape-completion result JSON.")

    def _pollSubprocess(self):
        self._logReadPos = self.pollLogFile(self._logFilePath, self._logReadPos)
        retcode = self._process.poll()
        if retcode is not None:
            self._pollTimer.stop()
            self._logFile.close()
            self.progressBar.setVisible(False)
            self.runButton.setEnabled(self.commonInputsReady())
            self.runBatchButton.setEnabled(self.modelReady() and bool(self.inputFolderPath))
            if retcode != 0 or not os.path.exists(self._resultPath):
                self.onLogMessage("Inference failed (exit code {}).".format(retcode), color="red")
                return
            if self._completionMode == "bulk":
                self._loadBulkResults(self._resultPath)
                return

            # single mode
            with open(self._resultPath) as f:
                self.outputPath = f.read().strip()
            if not os.path.isabs(self.outputPath):
                self.outputPath = os.path.join(os.path.dirname(self._resultPath), self.outputPath)
            resultJson = {"input_name": os.path.basename(self.inputFilePath),
                          "input_path": self.inputFilePath,
                          "output_path": self.outputPath}
            jsonPath = os.path.splitext(self._resultPath)[0] + "_result.json"
            with open(jsonPath, "w", encoding="utf-8") as f:
                json.dump(resultJson, f, indent=2)
            
            self.onLogMessage("\nInference complete. \n\nOutput saved to: {}".format(self.outputPath), color="#4CAF50")
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

    def _loadBulkResults(self, summaryPath):
        with open(summaryPath, encoding="utf-8") as f:
            summary = json.load(f)
        self._bulkResults = summary.get("results", [])
        self._populateBulkTable()
        self.batchStatusLabel.setText(
            "Completed {} meshes. Results saved to: {}".format(
                len(self._bulkResults), summaryPath))
        self.onLogMessage(
            "\n\nBatch shape completion complete: {} meshes.\n\nSummary saved to {}".format(
                len(self._bulkResults), summaryPath), color="#4CAF50")
        self.tabWidget.setCurrentIndex(self._batchTabIndex)

    def _populateBulkTable(self):
        self.bulkTable.setRowCount(len(self._bulkResults))
        for row, result in enumerate(self._bulkResults):
            values = [result.get("input_name", ""), result.get("output_path", "")]
            for column, value in enumerate(values):
                self.bulkTable.setItem(row, column, qt.QTableWidgetItem(value))

    def onExportBatchResults(self):
        if not self._bulkResults:
            slicer.util.warningDisplay("No batch results to export. Run 'Run Inference (Batch)' first.")
            return
        default = os.path.join(self.outputFolderPath or "", "batch_shape_completion.csv")
        path = qt.QFileDialog.getSaveFileName(None, "Export Batch Results", default, "CSV (*.csv)")
        if not path:
            return
        headers = ["input_name", "input_path", "output_path"]
        rows = [[r.get("input_name", ""), r.get("input_path", ""), r.get("output_path", "")] for r in self._bulkResults]
        csvPath = self.writeTable(os.path.splitext(path)[0], headers, rows)
        self.onLogMessage("Exported batch results to:\n{}".format(csvPath), color="#4CAF50")

    def applyUncertaintyVisualization(self, modelNode):
        polyData = modelNode.GetPolyData()
        if not polyData:
            self.onLogMessage("No polydata found on output model.")
            return False
        pointData = polyData.GetPointData()
        if not pointData:
            self.onLogMessage("No point data found on output model.")
            return False
        scalarName = "SdfUncertainty_um"
        scalarArray = pointData.GetArray(scalarName)
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
        try:
            if hasattr(displayNode, "SetActiveAttributeLocation"):
                displayNode.SetActiveAttributeLocation(vtk.vtkAssignAttribute.POINT_DATA)
        except Exception as e:
            self.onLogMessage(f"Could not set active attribute location: {str(e)}")
        displayNode.SetActiveScalarName(scalarName)
        displayNode.SetScalarVisibility(True)
        scalarRange = scalarArray.GetRange()
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
        displayNode.SetScalarRange(scalarRange[0], scalarRange[1])
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

    def onAfterSceneCleared(self):
        self.outputPath = None
        self._resultPath = None
        self._inputModelNode = None
        self._outputModelNode = None
        self._showingCompletedModel = True
        self._bulkResults = []
        self.bulkTable.setRowCount(0)
        self.batchStatusLabel.setText("Run 'Run Inference (Batch)' to populate batch results.")
        self.toggleModelsButton.setEnabled(False)
        self.toggleModelsButton.setText("Show Original Model")
        self.statusLog.clear()
        self.setDefaultThreeDLayout()

class ShapeCompletionLogic(FossilNsmLogic):
    pass