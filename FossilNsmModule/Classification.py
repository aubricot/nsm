import slicer
from FossilNsmCommon import FossilNsmCommonWidget, FossilNsmLogic
FossilNsmLogic.installDependenciesIfNeeded()
import glob
import json
import os
import csv
import subprocess
import sys
import vtk
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import ctk
import qt
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
from slicer.ScriptedLoadableModule import *

FOSSIL_NSM_MESH_LAYOUT_ID = 702
FOSSIL_NSM_PLOT_LAYOUT_ID = 703
MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)
from FossilNsmCommon import FossilNsmCommonWidget, FossilNsmLogic, FossilNsmHuggingFaceAuthMixin

UTILS_DIR = os.path.join(MODULE_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)
from reference_library import LocalFolderBackend, HuggingFaceBackend, Resolution, STATE_MISSING


class Classification(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Classification"
        self.parent.categories = ["FossilNSM"]
        self.parent.contributors = ["Wolcott et all"]
        self.parent.helpText = "Classify an input fossil mesh by ranking the nearest latent-space meshes."


class ClassificationWidget(FossilNsmHuggingFaceAuthMixin, FossilNsmCommonWidget, ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.initializeFossilNsmState()
        self.referenceMeshDirectory = None
        self.initializeHuggingFaceState()
        self._referenceSource = "local"
        self._referenceBackend = self._makeBackend()
        self.classificationMatches = []
        self._classificationNodes = []
        self._plotsRenderedFor = None
        self._classificationMode = "single"
        self._allLatentsPath = None
        self._fossilPath = None
        self._fossilLatentPath = None
        self._top5IndicesPath = None
        self.inputFolderPath = None
        self._bulkResults = []
        self._bulkAllLatentsPath = None
        self._pcaCoords = None
        self._tsneCoords = None
        self._umapCoords = None
        self._cachedCombined = None
        self._cachedFossilIdx = None
        self._cachedTop5Indices = None
        self._lastExportDir = None
        self._previousResultsBaseName = None

        FossilNsmLogic.installDependenciesIfNeeded()

        classificationCollapsible = ctk.ctkCollapsibleButton()
        classificationCollapsible.text = "Classification: Top-5 Matches"
        classificationLayout = qt.QFormLayout(classificationCollapsible)

        self.statusLog = qt.QTextEdit()
        self.statusLog.setReadOnly(True)
        self.statusLog.setFixedHeight(120)

        self.previousLayout = slicer.app.layoutManager().layout

        self.tabWidget = qt.QTabWidget()
        self.layout.addWidget(self.tabWidget)
        self.tabWidget.connect("currentChanged(int)", self.onTabChanged)

        # Tab 1 — Inference
        inferenceTab = qt.QWidget()
        inferenceLayout = qt.QVBoxLayout(inferenceTab)
        inferenceLayout.setContentsMargins(0, 0, 0, 0)
        self.tabWidget.addTab(inferenceTab, "Inference")
        self.addFossilNsmInputSection(inferenceLayout)

        self.inputFolderButton = qt.QPushButton("Select Input Folder")
        self.inputFolderButton.connect("clicked(bool)", self.onSelectInputFolder)
        self.inputFolderLabel = qt.QLabel("No folder selected (for batch classification)")
        self.inputFolderLabel.setWordWrap(True)
        self.inputLayout.addRow("Input Folder (Batch):", self.inputFolderButton)
        self.inputLayout.addRow("", self.inputFolderLabel)

        self.loadShapeCompletionButton = qt.QPushButton("Load Shape Completion Result")
        self.loadShapeCompletionButton.connect("clicked(bool)", self.onLoadShapeCompletionResult)
        self.loadShapeCompletionLabel = qt.QLabel("Pick a completed mesh from a previous Shape Completion run to classify it.")
        self.loadShapeCompletionLabel.setWordWrap(True)
        self.inputLayout.addRow("From Shape Completion:", self.loadShapeCompletionButton)
        self.inputLayout.addRow("", self.loadShapeCompletionLabel)

        inferenceLayout.addWidget(self.statusLog)

        inferenceBottomCollapsible = ctk.ctkCollapsibleButton()
        inferenceBottomCollapsible.text = "Classification Settings"
        inferenceLayout.addWidget(inferenceBottomCollapsible)
        inferenceBottomLayout = qt.QFormLayout(inferenceBottomCollapsible)

        # Reference source selector
        self.referenceSourceCombo = qt.QComboBox()
        self.referenceSourceCombo.addItems(["Local folder", "HuggingFace (gated)"])
        self.referenceSourceCombo.connect("currentIndexChanged(int)", self.onReferenceSourceChanged)
        inferenceBottomLayout.addRow("Reference Source:", self.referenceSourceCombo)

        # Local folder controls
        self.referenceMeshesButton = qt.QPushButton("Select Reference Mesh Library")
        self.referenceMeshesButton.connect("clicked(bool)", self.onSelectReferenceMeshDirectory)
        self.referenceMeshesLabel = qt.QLabel("Optional: needed to visualize returned meshes")
        self.referenceMeshesLabel.setWordWrap(True)
        inferenceBottomLayout.addRow("Reference Mesh Library:", self.referenceMeshesButton)
        inferenceBottomLayout.addRow("", self.referenceMeshesLabel)

        # HuggingFace controls
        self.hfControls = qt.QWidget()
        hfControlsLayout = qt.QVBoxLayout(self.hfControls)
        hfControlsLayout.setContentsMargins(0, 0, 0, 0)
        self.addHuggingFaceTokenSection(hfControlsLayout)
        self.hfTokenChecklist = qt.QPlainTextEdit()
        self.hfTokenChecklist.setReadOnly(True)
        self.hfTokenChecklist.setFixedHeight(70)
        hfControlsLayout.addWidget(self.hfTokenChecklist)
        self.prefetchButton = qt.QPushButton("Prefetch entire library (175 MB)")
        self.prefetchButton.connect("clicked(bool)", self.onPrefetchLibrary)
        hfControlsLayout.addWidget(self.prefetchButton)
        hfNote = qt.QLabel("Compressed for download and slightly lossy. Use the source .vtk for precise measurements.")
        hfNote.setWordWrap(True)
        hfControlsLayout.addWidget(hfNote)
        self.hfControls.setVisible(False)
        inferenceBottomLayout.addRow(self.hfControls)

        self.classificationIterationsInput = qt.QLineEdit("1000")
        inferenceBottomLayout.addRow("Latent Optimization Iterations:", self.classificationIterationsInput)

        self.classifyButton = qt.QPushButton("Classify Input Mesh")
        self.classifyButton.setEnabled(False)
        self.classifyButton.connect("clicked(bool)", self.onClassifyInputMesh)
        inferenceBottomLayout.addRow(self.classifyButton)

        self.classifyFolderButton = qt.QPushButton("Classify Folder (Batch)")
        self.classifyFolderButton.setEnabled(False)
        self.classifyFolderButton.connect("clicked(bool)", self.onClassifyFolder)
        inferenceBottomLayout.addRow(self.classifyFolderButton)

        self.loadPreviousResultsButton = qt.QPushButton("Load Previous Classification Results")
        self.loadPreviousResultsButton.connect("clicked(bool)", self.onLoadPreviousResults)
        inferenceBottomLayout.addRow(self.loadPreviousResultsButton)
        loadPreviousHint = qt.QLabel("Load bulk_summary.json or <mesh>_top5.json to inspect a previous run "
                                     "in Explore Meshes / Explore Plots.")
        loadPreviousHint.setWordWrap(True)
        inferenceBottomLayout.addRow("", loadPreviousHint)

        self.addRefreshSceneButton(inferenceBottomLayout)

        # Tab 2 — Explore Meshes
        exploreMeshesTab = qt.QWidget()
        exploreMeshesOuterLayout = qt.QHBoxLayout(exploreMeshesTab)
        exploreMeshesOuterLayout.setContentsMargins(0, 0, 0, 0)

        meshSidePanel = qt.QWidget()
        meshSidePanel.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Expanding)
        meshSidePanelLayout = qt.QVBoxLayout(meshSidePanel)
        meshSidePanelLayout.setContentsMargins(4, 4, 4, 4)
        meshSidePanelLayout.addWidget(classificationCollapsible)
        exploreMeshesOuterLayout.addWidget(meshSidePanel)
        self.tabWidget.addTab(exploreMeshesTab, "Explore Meshes")

        # Tab 3 — Explore Plots
        explorePlotsTab = qt.QWidget()
        explorePlotsOuterLayout = qt.QHBoxLayout(explorePlotsTab)
        explorePlotsOuterLayout.setContentsMargins(0, 0, 0, 0)

        plotSidePanel = qt.QWidget()
        plotSidePanel.setFixedWidth(300)
        plotSidePanelLayout = qt.QVBoxLayout(plotSidePanel)
        plotSidePanelLayout.setContentsMargins(4, 4, 4, 4)

        plotTypeLabel = qt.QLabel("Plot type:")
        plotSidePanelLayout.addWidget(plotTypeLabel)

        self.plotTypeComboBox = qt.QComboBox()
        self.plotTypeComboBox.addItems(["PCA", "t-SNE", "UMAP"])
        self.plotTypeComboBox.connect("currentIndexChanged(int)", self.onPlotTypeChanged)
        plotSidePanelLayout.addWidget(self.plotTypeComboBox)

        # t-SNE parameters
        self.tsneParamWidget = qt.QWidget()
        tsneParamLayout = qt.QFormLayout(self.tsneParamWidget)
        tsneParamLayout.setContentsMargins(0, 4, 0, 0)
        self.tsnePerplexityInput = qt.QDoubleSpinBox()
        self.tsnePerplexityInput.setRange(5, 100)
        self.tsnePerplexityInput.setValue(30)
        self.tsnePerplexityInput.setSingleStep(5)
        tsneParamLayout.addRow("Perplexity:", self.tsnePerplexityInput)
        self.tsneLearningRateInput = qt.QDoubleSpinBox()
        self.tsneLearningRateInput.setRange(10, 1000)
        self.tsneLearningRateInput.setValue(50)
        self.tsneLearningRateInput.setSingleStep(10)
        tsneParamLayout.addRow("Learning Rate:", self.tsneLearningRateInput)
        self.tsneEarlyExaggerationInput = qt.QDoubleSpinBox()
        self.tsneEarlyExaggerationInput.setRange(1, 50)
        self.tsneEarlyExaggerationInput.setValue(12)
        self.tsneEarlyExaggerationInput.setSingleStep(1)
        tsneParamLayout.addRow("Early Exaggeration:", self.tsneEarlyExaggerationInput)
        self.tsneNoProgressInput = qt.QSpinBox()
        self.tsneNoProgressInput.setRange(100, 10000)
        self.tsneNoProgressInput.setValue(2000)
        self.tsneNoProgressInput.setSingleStep(100)
        tsneParamLayout.addRow("Iter Without Progress:", self.tsneNoProgressInput)
        self.tsneMetricCombo = qt.QComboBox()
        self.tsneMetricCombo.addItems(["cosine", "euclidean", "manhattan"])
        tsneParamLayout.addRow("Metric:", self.tsneMetricCombo)
        self.tsneApplyButton = qt.QPushButton("Recompute t-SNE")
        self.tsneApplyButton.connect("clicked(bool)", self.onRecomputeTSNE)
        tsneParamLayout.addRow(self.tsneApplyButton)
        self.tsneParamWidget.setVisible(False)
        plotSidePanelLayout.addWidget(self.tsneParamWidget)

        # UMAP parameters
        self.umapParamWidget = qt.QWidget()
        umapParamLayout = qt.QFormLayout(self.umapParamWidget)
        umapParamLayout.setContentsMargins(0, 4, 0, 0)
        self.umapNeighborsInput = qt.QSpinBox()
        self.umapNeighborsInput.setRange(2, 200)
        self.umapNeighborsInput.setValue(50)
        self.umapNeighborsInput.setSingleStep(5)
        umapParamLayout.addRow("N Neighbors:", self.umapNeighborsInput)
        self.umapMinDistInput = qt.QDoubleSpinBox()
        self.umapMinDistInput.setRange(0.0, 1.0)
        self.umapMinDistInput.setValue(0.1)
        self.umapMinDistInput.setSingleStep(0.05)
        self.umapMinDistInput.setDecimals(2)
        umapParamLayout.addRow("Min Dist:", self.umapMinDistInput)
        self.umapSpreadInput = qt.QDoubleSpinBox()
        self.umapSpreadInput.setRange(0.1, 5.0)
        self.umapSpreadInput.setValue(0.5)
        self.umapSpreadInput.setSingleStep(0.1)
        self.umapSpreadInput.setDecimals(2)
        umapParamLayout.addRow("Spread:", self.umapSpreadInput)
        self.umapEpochsInput = qt.QSpinBox()
        self.umapEpochsInput.setRange(50, 2000)
        self.umapEpochsInput.setValue(500)
        self.umapEpochsInput.setSingleStep(50)
        umapParamLayout.addRow("N Epochs:", self.umapEpochsInput)
        self.umapPcaComponentsInput = qt.QSpinBox()
        self.umapPcaComponentsInput.setRange(2, 512)
        self.umapPcaComponentsInput.setValue(50)
        self.umapPcaComponentsInput.setSingleStep(10)
        umapParamLayout.addRow("PCA Pre-reduction (n-components):", self.umapPcaComponentsInput)
        self.umapApplyButton = qt.QPushButton("Recompute UMAP")
        self.umapApplyButton.connect("clicked(bool)", self.onRecomputeUMAP)
        umapParamLayout.addRow(self.umapApplyButton)
        self.umapParamWidget.setVisible(False)
        plotSidePanelLayout.addWidget(self.umapParamWidget)

        self.plotStatusLabel = qt.QLabel("Run classification to populate plots.")
        self.plotStatusLabel.setWordWrap(True)
        plotSidePanelLayout.addWidget(self.plotStatusLabel)

        self.savePlotButton = qt.QPushButton("Save Current Plot (PNG + HTML)")
        self.savePlotButton.connect("clicked(bool)", self.onSaveCurrentPlot)
        plotSidePanelLayout.addWidget(self.savePlotButton)

        self.saveAllPlotsCheckbox = qt.QCheckBox("Save all computed plot types")
        self.saveAllPlotsCheckbox.setChecked(False)
        plotSidePanelLayout.addWidget(self.saveAllPlotsCheckbox)

        self.openSaveFolderButton = qt.QPushButton("Open Save Folder")
        self.openSaveFolderButton.setEnabled(False)
        self.openSaveFolderButton.connect("clicked(bool)", self.onOpenSaveFolder)
        plotSidePanelLayout.addWidget(self.openSaveFolderButton)

        plotSidePanelLayout.addStretch(1)
        explorePlotsOuterLayout.addWidget(plotSidePanel)
        self.tabWidget.addTab(explorePlotsTab, "Explore Plots")

        # Tab 4 — Batch Results
        batchResultsTab = qt.QWidget()
        batchResultsLayout = qt.QVBoxLayout(batchResultsTab)
        batchResultsLayout.setContentsMargins(4, 4, 4, 4)

        self.batchStatusLabel = qt.QLabel("Run 'Classify Folder' to populate batch results.")
        self.batchStatusLabel.setWordWrap(True)
        batchResultsLayout.addWidget(self.batchStatusLabel)

        self.bulkTable = qt.QTableWidget(0, 3)
        self.bulkTable.setHorizontalHeaderLabels(["Input mesh", "Top match", "Cosine distance"])
        self.bulkTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.bulkTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.bulkTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.bulkTable.horizontalHeader().setStretchLastSection(True)
        self.bulkTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        self.bulkTable.connect("itemSelectionChanged()", self.onBulkRowSelected)
        batchResultsLayout.addWidget(self.bulkTable, 1)

        batchHint = qt.QLabel("Select a row to load that mesh into the Explore Meshes and Explore Plots tabs.")
        batchHint.setWordWrap(True)
        batchResultsLayout.addWidget(batchHint)

        self.exportBatchButton = qt.QPushButton("Export Batch Results (CSV)")
        self.exportBatchButton.connect("clicked(bool)", self.onExportBatchResults)
        batchResultsLayout.addWidget(self.exportBatchButton)

        self._batchTabIndex = self.tabWidget.addTab(batchResultsTab, "Batch Results")

        self.classificationTable = qt.QTableWidget(0, 3)
        self.classificationTable.setHorizontalHeaderLabels(["Reference mesh", "Cosine distance", "Available"])
        self.classificationTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.classificationTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.classificationTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.classificationTable.horizontalHeader().setStretchLastSection(True)
        self.classificationTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        classificationLayout.addRow("Top-5 matches:", self.classificationTable)

        self.exportTopMatchesButton = qt.QPushButton("Export Top-5 (CSV)")
        self.exportTopMatchesButton.connect("clicked(bool)", self.onExportTopMatches)
        classificationLayout.addRow(self.exportTopMatchesButton)

        self.updateRunButton()
        self.layout.addStretch(1)

    # ------------------------------------------------------------------ #
    #  Inference helpers
    # ------------------------------------------------------------------ #

    def onSelectReferenceMeshDirectory(self):
        path = qt.QFileDialog.getExistingDirectory(None, "Select Reference Mesh Library")
        if not path:
            return
        self.referenceMeshDirectory = path
        self.referenceMeshesLabel.setText(path)
        self._referenceBackend = self._makeBackend()
        self._updateClassificationTable()

    def _makeBackend(self):
        if self._referenceSource == "huggingface":
            return HuggingFaceBackend(
                repo_id=self.hfRepoId, revision=self.hfRevision,
                token_provider=self.resolveHuggingFaceToken)
        return LocalFolderBackend(self.referenceMeshDirectory)

    def onHuggingFaceRepoConfigChanged(self):
        self._referenceBackend = self._makeBackend()
        self._updateClassificationTable()

    def onReferenceSourceChanged(self, index):
        self._referenceSource = "huggingface" if index == 1 else "local"
        isHf = self._referenceSource == "huggingface"
        self.referenceMeshesButton.setVisible(not isHf)
        self.referenceMeshesLabel.setVisible(not isHf)
        self.hfControls.setVisible(isHf)
        if isHf:
            self.updateHuggingFaceChecklistUI()
        self._referenceBackend = self._makeBackend()
        self._updateClassificationTable()

    def onPrefetchLibrary(self):
        if not self.hfRepoId:
            self.onLogMessage("Enter a HuggingFace dataset repo first.", color="red")
            return
        try:
            token = self.resolveHuggingFaceToken()
        except ValueError as e:
            self.onLogMessage(str(e), color="red")
            return
        self.onLogMessage("\n\n\nPrefetching reference library...", color="#4CAF50")
        slicer.app.processEvents()
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                self.hfRepoId, repo_type="dataset",
                revision=self.hfRevision, token=token,
                allow_patterns=["*.glb"])
        except Exception as e:
            self.onLogMessage("Prefetch failed: check your network. " + str(e), color="red")
            return
        self.onLogMessage("\n\n\nPrefetch complete. Meshes now render with no network.", color="#4CAF50")
        self._updateClassificationTable()

    def onLoadShapeCompletionResult(self):
        startDir = self.outputFolderPath if self.outputFolderPath and os.path.isdir(self.outputFolderPath) else ""
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Shape Completion Result Mesh", startDir,
            "Shape Completion Meshes (*_shape_completion*.vtk);;Mesh Files (*.vtk *.vtp *.stl *.obj *.ply)"
        )
        if not path:
            return
        self.inputFilePath = path
        self.inputFileLabel.setText(path)
        self.updateRunButton()
        self.onLogMessage("Loaded shape completion result as input mesh:\n" + path, color="#4CAF50")

    def _pickExistingFile(self, directory, *names):
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
        return None

    def onLoadPreviousResults(self):
        startDir = ""
        candidate = self._classificationDir()
        if candidate and os.path.isdir(candidate):
            startDir = candidate
        elif self.modelRootPath:
            startDir = self.modelRootPath
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Classification Results JSON", startDir, "Results (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
        except Exception as error:
            slicer.util.errorDisplay("Could not read results file:\n{}".format(error))
            return

        resultDir = os.path.dirname(path)
        base = os.path.splitext(os.path.basename(path))[0]
        if base.endswith("_top5"):
            base = base[: -len("_top5")]
        self._previousResultsBaseName = base

        # Bulk summary (has a "results" list) --------------------------------
        if isinstance(data, dict) and "results" in data:
            self._bulkResults = data.get("results", [])
            self._bulkAllLatentsPath = data.get("all_latents")
            if self._bulkAllLatentsPath and not os.path.isfile(self._bulkAllLatentsPath):
                sibling = os.path.join(resultDir, os.path.basename(self._bulkAllLatentsPath))
                if os.path.isfile(sibling):
                    self._bulkAllLatentsPath = sibling
            self._classificationMode = "bulk"
            self._populateBulkTable()
            self.batchStatusLabel.setText(
                "Loaded {} previous results from {}. Select a row to inspect it.".format(
                    len(self._bulkResults), os.path.basename(path)))
            self.onLogMessage("Loaded previous batch results: " + path, color="#4CAF50")
            self.tabWidget.setCurrentIndex(self._batchTabIndex)
            return

        # Single result (has a "matches" list) ------------------------------
        if isinstance(data, dict) and "matches" in data:
            self.classificationMatches = data.get("matches", [])
            self._fossilPath = data.get("fossil_path") or data.get("fossil_name")
            self._classificationMode = "single"

            self._allLatentsPath = self._pickExistingFile(resultDir, "all_latents.npy")
            self._fossilLatentPath = self._pickExistingFile(resultDir, base + "_fossil_latent.npy", "fossil_latent.npy")
            self._top5IndicesPath = self._pickExistingFile(resultDir, base + "_top5_indices.npy", "top5_indices.npy")
            self._plotsRenderedFor = None
            self._updateClassificationTable()

            missing = [n for n, p in [
                ("all_latents", self._allLatentsPath),
                ("fossil_latent", self._fossilLatentPath),
                ("top5_indices", self._top5IndicesPath)] if not p]
            self.onLogMessage("Loaded previous single result: " + path, color="#4CAF50")
            if missing:
                self.onLogMessage(
                    "Note: latent files {} not found next to the JSON; plots are unavailable "
                    "for this result.".format(", ".join(missing)), color="orange")
            return

        slicer.util.errorDisplay(
            "Unrecognized results file. Expected a bulk_summary.json or a <mesh>_top5.json.")

    def _classificationDir(self):
        root = self.modelRootPath or self.outputFolderPath
        return os.path.join(root, "classification") if root else None

    def _modelReady(self):
        return bool(
            self.modelRootPath
            and self.configFilePath
            and self.modelFilePath
            and self.latentCodesFilePath
            and self.outputFolderPath)

    def updateRunButton(self):
        self.classifyButton.setEnabled(self.commonInputsReady())
        self.classifyFolderButton.setEnabled(self._modelReady() and bool(self.inputFolderPath))

    def onSelectInputFolder(self):
        path = qt.QFileDialog.getExistingDirectory(None, "Select Folder of Meshes to Classify")
        if not path:
            return
        self.inputFolderPath = path
        self.inputFolderLabel.setText(path)
        self.updateRunButton()

    def _readIterations(self):
        try:
            iterations = int(self.classificationIterationsInput.text)
            if iterations < 1:
                raise ValueError
            return iterations
        except ValueError:
            slicer.util.errorDisplay("Latent optimization iterations must be a positive integer.")
            return None

    def onClassifyInputMesh(self):
        iterations = self._readIterations()
        if iterations is None:
            return
        inputBase = os.path.splitext(os.path.basename(self.inputFilePath))[0]
        resultDirectory = os.path.join(self._classificationDir(), inputBase)
        os.makedirs(resultDirectory, exist_ok=True)
        resultPath = os.path.join(resultDirectory, inputBase + "_top5.json")
        logPath = os.path.join(resultDirectory, inputBase + "_classification.log")
        self._classificationMode = "single"
        self._allLatentsPath = os.path.join(resultDirectory, "all_latents.npy")
        self._fossilLatentPath = os.path.join(resultDirectory, "fossil_latent.npy")
        self._top5IndicesPath = os.path.join(resultDirectory, "top5_indices.npy")
        self.onLogMessage("Starting nearest-mesh classification (latent optimization may take several minutes)...\n\n\n", color="#4CAF50")
        self._runWorker([
            "--input_mesh", self.inputFilePath, "--output_dir", resultDirectory,
            "--iterations", str(iterations),
        ], resultPath, logPath)

    def onClassifyFolder(self):
        iterations = self._readIterations()
        if iterations is None:
            return
        folderPath = self.inputFolderPath
        if not folderPath or not os.path.isdir(folderPath):
            slicer.util.errorDisplay("Select an input folder first (Input Folder (Batch)).")
            return
        resultDirectory = self._classificationDir()
        os.makedirs(resultDirectory, exist_ok=True)
        resultPath = os.path.join(resultDirectory, "bulk_summary.json")
        logPath = os.path.join(resultDirectory, "bulk_classification.log")
        self._classificationMode = "bulk"
        self.batchStatusLabel.setText("Classifying folder: " + folderPath)
        self.onLogMessage("Starting BATCH classification of folder:\n{}\n(each mesh runs a full latent optimization)\n\n\n".format(folderPath), color="#4CAF50")
        self._runWorker([
            "--input_dir", folderPath, "--output_dir", resultDirectory,
            "--iterations", str(iterations),
        ], resultPath, logPath)

    def _runWorker(self, extraArgs, resultPath, logPath):
        workerScript = os.path.join(os.path.dirname(os.path.dirname(__file__)), "classification_slicer.py")
        self._classificationResultPath = resultPath
        self._classificationLog = open(logPath, "w")
        self._classificationLogReadPos = 0
        self._classificationLogPath = logPath
        self.classifyButton.setEnabled(False)
        self.classifyFolderButton.setEnabled(False)
        command = [
            sys.executable, workerScript, "--config", self.configFilePath,
            "--model", self.modelFilePath, "--latent_codes", self.latentCodesFilePath,
            "--result", resultPath,
        ] + extraArgs
        self._classificationProcess = subprocess.Popen(
            command, stdout=self._classificationLog, stderr=subprocess.STDOUT)
        self._classificationTimer = qt.QTimer()
        self._classificationTimer.setInterval(500)
        self._classificationTimer.timeout.connect(self._pollClassification)
        self._classificationTimer.start()

    # ------------------------------------------------------------------ #
    #  Tab switching
    # ------------------------------------------------------------------ #

    def onTabChanged(self, index):
        print("[FossilNSM] onTabChanged index={}".format(index))
        if index == 0:  # Inference
            self.setDefaultThreeDLayout()
        elif index == 1:  # Explore Meshes
            self._applyMeshLayout()
            slicer.app.processEvents()
            self._loadMeshesIntoViewers()
            self._linkMeshViewers()
            self._styleViewers()
            #self._resetMeshViews() # TO DO: Check if needed; was in Sean code

        elif index == 2:  # Explore Plots
            self._applyPlotLayout()
            self._renderLatentSpacePlots()
        elif index == self._batchTabIndex:  # Batch Results
            self.setDefaultThreeDLayout()

    # ------------------------------------------------------------------ #
    #  Plot rendering
    # ------------------------------------------------------------------ #

    def _renderLatentSpacePlots(self):
        print("[FossilNSM] _renderLatentSpacePlots called")
        if not self.classificationMatches:
            self.plotStatusLabel.setText("Run classification first.")
            print("[FossilNSM] no classificationMatches, returning")
            return

        allLatentsPath   = self._allLatentsPath
        fossilLatentPath = self._fossilLatentPath
        top5IndicesPath  = self._top5IndicesPath

        if not (allLatentsPath and fossilLatentPath and top5IndicesPath
                and all(os.path.isfile(p) for p in [allLatentsPath, fossilLatentPath, top5IndicesPath])):
            self.plotStatusLabel.setText("Latent files not found. Re-run classification.")
            print("[FossilNSM] latent files missing")
            return

        cacheKey = (fossilLatentPath, os.path.getmtime(fossilLatentPath))
        if self._plotsRenderedFor != cacheKey:
            self._pcaCoords = None
            self._tsneCoords = None
            self._umapCoords = None

            allLatents   = np.load(allLatentsPath)
            fossilLatent = np.load(fossilLatentPath)
            top5Indices  = np.load(top5IndicesPath)
            combined     = np.vstack([allLatents, fossilLatent])
            fossilIdx    = len(allLatents)

            self._cachedCombined    = combined
            self._cachedFossilIdx   = fossilIdx
            self._cachedTop5Indices = top5Indices
            self._plotsRenderedFor  = cacheKey
            print("[FossilNSM] data loaded, combined shape={}".format(combined.shape))

        self._ensurePlotComputed(self.plotTypeComboBox.currentText)
        self._showActivePlot(self.plotTypeComboBox.currentText)

    def _ensurePlotComputed(self, plotType):
        print("[FossilNSM] _ensurePlotComputed plotType={}".format(plotType))
        combined    = self._cachedCombined
        fossilIdx   = self._cachedFossilIdx
        top5Indices = self._cachedTop5Indices

        if plotType == "PCA" and self._pcaCoords is None:
            self.plotStatusLabel.setText("\n\n\nComputing PCA...")
            slicer.app.processEvents()
            pca = PCA(n_components=2)
            self._pcaCoords = pca.fit_transform(combined)
            self.plotStatusLabel.setText("PCA computed.\n\n"
                                         "PC1 explained variance: {:.2f}%\n"
                                         "PC2 explained variance: {:.2f}%".format(
                                             100 * pca.explained_variance_ratio_[0], 
                                             100 * pca.explained_variance_ratio_[1]))
            self._buildPlot(self._pcaCoords, fossilIdx, top5Indices, "PCA")
            print("[FossilNSM] PCA built")

        elif plotType == "t-SNE" and self._tsneCoords is None:
            perplexity         = self.tsnePerplexityInput.value
            learning_rate      = self.tsneLearningRateInput.value
            early_exaggeration = self.tsneEarlyExaggerationInput.value
            no_progress        = self.tsneNoProgressInput.value
            metric             = self.tsneMetricCombo.currentText
            self.plotStatusLabel.setText("\n\n\nComputing t-SNE...")
            slicer.app.processEvents()
            tsne_kwargs = dict(
                n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                early_exaggeration=early_exaggeration, n_iter_without_progress=no_progress,
                metric=metric, random_state=42,)
            if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4):
                tsne_kwargs["max_iter"] = 1000
            else:
                tsne_kwargs["n_iter"] = 1000
            self._tsneCoords = TSNE(**tsne_kwargs).fit_transform(combined)
            self._buildPlot(self._tsneCoords, fossilIdx, top5Indices, "t-SNE")
            print("[FossilNSM] t-SNE built")

        elif plotType == "UMAP" and self._umapCoords is None:
            n_neighbors = self.umapNeighborsInput.value
            min_dist    = self.umapMinDistInput.value
            spread      = self.umapSpreadInput.value
            n_epochs    = self.umapEpochsInput.value
            n_pca       = self.umapPcaComponentsInput.value
            self.plotStatusLabel.setText("Computing UMAP (PCA pre-reduction to {})...".format(n_pca))
            slicer.app.processEvents()
            combined_pca = PCA(n_components=n_pca).fit_transform(combined)
            self.plotStatusLabel.setText("\n\n\nComputing UMAP...")
            slicer.app.processEvents()
            self._umapCoords = umap.UMAP(
                n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                spread=spread, n_epochs=n_epochs, random_state=42,).fit_transform(combined_pca)
            self._buildPlot(self._umapCoords, fossilIdx, top5Indices, "UMAP")
            print("[FossilNSM] UMAP built")

    def _showActivePlot(self, plotType):
        print("[FossilNSM] _showActivePlot plotType={}".format(plotType))
        chartNode = slicer.mrmlScene.GetFirstNodeByName(plotType + "_chart")
        if not chartNode:
            self.plotStatusLabel.setText("Chart not found for: " + plotType)
            print("[FossilNSM] chart node NOT found: " + plotType + "_chart")
            return
        plotViewNode = slicer.mrmlScene.GetSingletonNode("ActivePlot", "vtkMRMLPlotViewNode")
        if not plotViewNode:
            self.plotStatusLabel.setText("Plot view node not found — layout may not have applied.")
            print("[FossilNSM] ActivePlot view node NOT found")
            return
        plotViewNode.SetPlotChartNodeID(chartNode.GetID())
        if plotType != "PCA":
            self.plotStatusLabel.setText("{} rendered.".format(plotType))
        print("[FossilNSM] chart assigned to ActivePlot view")

    def onPlotTypeChanged(self, index):
        plotType = self.plotTypeComboBox.itemText(index)
        print("[FossilNSM] onPlotTypeChanged plotType={}".format(plotType))
        self.tsneParamWidget.setVisible(plotType == "t-SNE")
        self.umapParamWidget.setVisible(plotType == "UMAP")
        if self._cachedCombined is None:
            return
        self._ensurePlotComputed(plotType)
        self._showActivePlot(plotType)

    def onRecomputeTSNE(self):
        self._tsneCoords = None
        self._ensurePlotComputed("t-SNE")
        self._showActivePlot("t-SNE")

    def onRecomputeUMAP(self):
        self._umapCoords = None
        self._ensurePlotComputed("UMAP")
        self._showActivePlot("UMAP")

    def _makeTable(self, name, x, y, labels):
        t = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        t.RemoveAllColumns()
        xArr = vtk.vtkFloatArray()
        xArr.SetName("x")
        yArr = vtk.vtkFloatArray()
        yArr.SetName("y")
        labelArr = vtk.vtkStringArray()
        labelArr.SetName("label")
        for xi, yi, li in zip(x, y, labels):
            xArr.InsertNextValue(float(xi))
            yArr.InsertNextValue(float(yi))
            labelArr.InsertNextValue(str(li))
        t.AddColumn(xArr)
        t.AddColumn(yArr)
        t.AddColumn(labelArr)
        return t

    def _makeSeries(self, name, table, color, size=5):
        s = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", name)
        s.SetAndObserveTableNodeID(table.GetID())
        s.SetXColumnName("x")
        s.SetYColumnName("y")
        s.SetLabelColumnName("label")
        s.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        s.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
        s.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleSquare)
        s.SetMarkerSize(size)
        s.SetColor(*color)
        return s

    def _buildPlot(self, coords, fossilIdx, top5Indices, title):
        print("[FossilNSM] _buildPlot title={}".format(title))
        names = [title + ext for ext in ["_chart", "_bg", "_top5", "_fossil"]]
        old_nodes = [slicer.mrmlScene.GetFirstNodeByName(n) for n in names]
        if old_nodes[0]:  # If the chart exists, grab its series nodes too
            old_nodes += [old_nodes[0].GetNthPlotSeriesNode(i) for i in range(old_nodes[0].GetNumberOfPlotSeriesNodes())]
        for node in filter(None, old_nodes):
            slicer.mrmlScene.RemoveNode(node)

        train_filenames = []
        if self.configFilePath and os.path.isfile(self.configFilePath):
            try:
                import json
                with open(self.configFilePath, 'r') as f:
                    cfg = json.load(f)
                    train_filenames = [os.path.basename(p) for p in cfg.get('list_mesh_paths', [])]
            except Exception as e:
                print("[FossilNSM] Could not parse config for filenames: {}".format(e))
        bgMask = np.ones(len(coords), dtype=bool)
        bgMask[fossilIdx] = False
        valid_top5 = top5Indices[top5Indices < len(coords)]
        bgMask[valid_top5] = False

        bgLabels = []
        for i, is_bg in enumerate(bgMask):
            if is_bg:
                name = train_filenames[i] if i < len(train_filenames) else "Mesh {}".format(i)
                bgLabels.append("\n{}".format(name))
        top5Labels = []
        for i, idx in enumerate(valid_top5):
            match = self.classificationMatches[i] if i < len(self.classificationMatches) else {}
            name = match.get("mesh_name", "Mesh {}".format(idx))
            top5Labels.append("\nTop {} Match: {}".format(i + 1, name))
        fossil_name = (os.path.basename(self.inputFilePath) if self.inputFilePath
                       else os.path.basename(self._fossilPath) if self._fossilPath
                       else "Unknown")
        fossilLabels = ["\nFossil: {}".format(fossil_name)]

        bgTable     = self._makeTable(title + "_bg",     coords[bgMask, 0],      coords[bgMask, 1],      bgLabels)
        top5Table   = self._makeTable(title + "_top5",   coords[valid_top5, 0],  coords[valid_top5, 1],  top5Labels)
        fossilTable = self._makeTable(title + "_fossil", [coords[fossilIdx, 0]], [coords[fossilIdx, 1]], fossilLabels)

        bgSeries     = self._makeSeries("Train Data",    bgTable,     (0.5, 0.5, 0.5), size=4)
        top5Series   = self._makeSeries("Top-5 Matches", top5Table,   (0.2, 0.65, 0.9), size=10)
        fossilSeries = self._makeSeries("Fossil",        fossilTable, (0.9, 0.6, 0.2), size=14)

        chart = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", title + "_chart")
        chart.SetTitle(title)
        chart.AddAndObservePlotSeriesNodeID(bgSeries.GetID())
        chart.AddAndObservePlotSeriesNodeID(top5Series.GetID())
        chart.AddAndObservePlotSeriesNodeID(fossilSeries.GetID())
        
        print("[FossilNSM] chart node created: {}".format(title + "_chart"))

    # ------------------------------------------------------------------ #
    #  Layout helpers
    # ------------------------------------------------------------------ #

    def _applyPlotLayout(self):
        print("[FossilNSM] _applyPlotLayout called")
        layoutDescription = """
        <layout type="horizontal" split="false">
          <item><view class="vtkMRMLPlotViewNode" singletontag="ActivePlot">
            <property name="viewlabel" action="default">Plot</property>
          </view></item>
        </layout>
        """
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        if not layoutNode.IsLayoutDescription(FOSSIL_NSM_PLOT_LAYOUT_ID):
            layoutNode.AddLayoutDescription(FOSSIL_NSM_PLOT_LAYOUT_ID, layoutDescription)
        else:
            layoutNode.SetLayoutDescription(FOSSIL_NSM_PLOT_LAYOUT_ID, layoutDescription)
        slicer.app.layoutManager().setLayout(FOSSIL_NSM_PLOT_LAYOUT_ID)
        print("[FossilNSM] layout set to {}".format(FOSSIL_NSM_PLOT_LAYOUT_ID))

    def _applyMeshLayout(self):
        layoutDescription = """
        <layout type="vertical" split="false">
        <item>
            <layout type="horizontal">
            <item><view class="vtkMRMLViewNode" singletontag="FossilInput"><property name="viewlabel" action="default">Fossil Input</property></view></item>
            <item><view class="vtkMRMLViewNode" singletontag="Match1"><property name="viewlabel" action="default">Match 1</property></view></item>
            </layout>
        </item>
        <item>
            <layout type="horizontal">
            <item><view class="vtkMRMLViewNode" singletontag="Match2"><property name="viewlabel" action="default">Match 2</property></view></item>
            <item><view class="vtkMRMLViewNode" singletontag="Match3"><property name="viewlabel" action="default">Match 3</property></view></item>
            </layout>
        </item>
        <item>
            <layout type="horizontal">
            <item><view class="vtkMRMLViewNode" singletontag="Match4"><property name="viewlabel" action="default">Match 4</property></view></item>
            <item><view class="vtkMRMLViewNode" singletontag="Match5"><property name="viewlabel" action="default">Match 5</property></view></item>
            </layout>
        </item>
        </layout>
        """
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        if not layoutNode.IsLayoutDescription(FOSSIL_NSM_MESH_LAYOUT_ID):
            layoutNode.AddLayoutDescription(FOSSIL_NSM_MESH_LAYOUT_ID, layoutDescription)
        slicer.app.layoutManager().setLayout(FOSSIL_NSM_MESH_LAYOUT_ID)

    def _loadMeshesIntoViewers(self):
        if not self.classificationMatches:
            return
        self._clearClassificationNodes()
        for nodeName in ["Input Fossil"] + ["Match {} - {}".format(m["rank"], m["mesh_name"]) for m in self.classificationMatches]:
            old = slicer.mrmlScene.GetFirstNodeByName(nodeName)
            if old:
                slicer.mrmlScene.RemoveNode(old)
        fossilPath = self._resolveFossilPath()
        self.onLogMessage("Resolved fossil path: {}".format(fossilPath), color=None if fossilPath else "orange")
        if fossilPath and os.path.isfile(fossilPath):
            fossilNode = slicer.util.loadModel(fossilPath)
            if not fossilNode:
                self.onLogMessage("Failed to load fossil mesh: " + fossilPath, color="red")
            else:
                fossilNode.SetName("Input Fossil")
                fossilNode.CreateDefaultDisplayNodes()
                fossilNode.GetDisplayNode().SetColor(0.9, 0.6, 0.2)
                fossilNode.GetDisplayNode().SetAmbient(0.3)
                fossilNode.GetDisplayNode().SetDiffuse(0.8)
                fossilNode.GetDisplayNode().SetSpecular(0.0)
                self._assignNodeToView(fossilNode, "FossilInput")
                self._classificationNodes.append(fossilNode)
                self._labelViewer("FossilInput", os.path.basename(fossilPath))
        else:
            self.onLogMessage("Fossil file not found on disk.", color="orange")

        for match in self.classificationMatches[:5]:
            tag = "Match{}".format(match["rank"])
            res = self._resolveMatch(match)
            if res.kind != "path":
                self.onLogMessage(res.reason, color="red")
                continue
            meshPath = res.value
            node = slicer.util.loadModel(meshPath)
            node.SetName("Match {} - {}".format(match["rank"], match["mesh_name"]))
            self._normalizeForDisplay(node)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(0.2, 0.65, 0.9)
            node.GetDisplayNode().SetAmbient(0.3)
            node.GetDisplayNode().SetDiffuse(0.8)
            node.GetDisplayNode().SetSpecular(0.0)
            self._assignNodeToView(node, tag)
            self._classificationNodes.append(node)
            self._labelViewer(tag, match["mesh_name"])

    def _assignNodeToView(self, modelNode, viewTag):
        displayNode = modelNode.GetDisplayNode()
        displayNode.RemoveAllViewNodeIDs()
        viewNode = slicer.mrmlScene.GetSingletonNode(viewTag, "vtkMRMLViewNode")
        if viewNode:
            displayNode.AddViewNodeID(viewNode.GetID())
            self.onLogMessage("Assigned {} to view {}".format(modelNode.GetName(), viewTag))
        else:
            self.onLogMessage("View node not found for tag: {}".format(viewTag), color="red")

    def _normalizeForDisplay(self, modelNode):
        # Display only: center each mesh at the origin and scale to a common size so
        # the comparison panels line up. Does not touch the SDF pipeline or results.
        poly = modelNode.GetPolyData()
        if not poly or poly.GetNumberOfPoints() == 0:
            return
        bounds = [0.0] * 6
        poly.GetBounds(bounds)
        cx = (bounds[0] + bounds[1]) / 2.0
        cy = (bounds[2] + bounds[3]) / 2.0
        cz = (bounds[4] + bounds[5]) / 2.0
        radius = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) / 2.0
        if radius <= 0:
            return
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Translate(-cx, -cy, -cz)
        transform.Scale(1.0 / radius, 1.0 / radius, 1.0 / radius)
        filt = vtk.vtkTransformPolyDataFilter()
        filt.SetInputData(poly)
        filt.SetTransform(transform)
        filt.Update()
        modelNode.SetAndObservePolyData(filt.GetOutput())

    def _resetMeshViews(self):
        layoutManager = slicer.app.layoutManager()
        for i in range(layoutManager.threeDViewCount):
            view = layoutManager.threeDWidget(i).threeDView()
            view.resetFocalPoint()
            view.resetCamera()

    def _linkMeshViewers(self):
        tags = ["FossilInput", "Match1", "Match2", "Match3", "Match4", "Match5"]
        for tag in tags:
            viewNode = slicer.mrmlScene.GetSingletonNode(tag, "vtkMRMLViewNode")
            if viewNode:
                viewNode.SetLinkedControl(True)

    def _labelViewer(self, viewTag, text):
        viewNode = slicer.mrmlScene.GetSingletonNode(viewTag, "vtkMRMLViewNode")
        if not viewNode:
            return
        widget = slicer.app.layoutManager().viewWidget(viewNode)
        if not widget:
            return
        view = widget.threeDView()
        view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft, text)
        view.cornerAnnotation().GetTextProperty().SetFontSize(14)
        view.forceRender()

    def _styleViewers(self):
        tags = ["FossilInput", "Match1", "Match2", "Match3", "Match4", "Match5"]
        for tag in tags:
            viewNode = slicer.mrmlScene.GetSingletonNode(tag, "vtkMRMLViewNode")
            if not viewNode:
                continue
            viewNode.SetBackgroundColor(0, 0, 0)
            viewNode.SetBackgroundColor2(0, 0, 0)
            viewNode.SetBoxVisible(False)
            viewNode.SetAxisLabelsVisible(False)

    # ------------------------------------------------------------------ #
    #  Classification helpers
    # ------------------------------------------------------------------ #

    def onAfterSceneCleared(self):
        self.classificationMatches = []
        self._plotsRenderedFor = None
        self._pcaCoords = None
        self._tsneCoords = None
        self._umapCoords = None
        self._cachedCombined = None
        self._cachedFossilIdx = None
        self._cachedTop5Indices = None
        self._bulkResults = []
        self.bulkTable.setRowCount(0)
        self.batchStatusLabel.setText("Run 'Classify Folder' to populate batch results.")
        self.classificationTable.setRowCount(0)
        self._classificationNodes = []
        self._clearClassificationNodes()

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
        self.classifyButton.setEnabled(self.commonInputsReady())
        self.classifyFolderButton.setEnabled(self._modelReady())
        if code != 0 or not os.path.isfile(self._classificationResultPath):
            self.onLogMessage("\n\n\nClassification failed (exit code {}).".format(code), color="red")
            return
        if self._classificationMode == "bulk":
            self._loadBulkResults(self._classificationResultPath)
            return
        with open(self._classificationResultPath, encoding="utf-8") as stream:
            data = json.load(stream)
        self.classificationMatches = data.get("matches", [])
        self._fossilPath = data.get("fossil_path") or data.get("fossil_name")
        self._plotsRenderedFor = None
        self._updateClassificationTable()
        self.onLogMessage("\n\n\nClassification complete. \n\nTop-5 matches saved to " + self._classificationResultPath, color="#4CAF50")
        headers, rows = self._topMatchesTable(self.classificationMatches)
        self._autoSaveTables(os.path.splitext(self._classificationResultPath)[0], headers, rows)

    def _loadBulkResults(self, summaryPath):
        with open(summaryPath, encoding="utf-8") as stream:
            summary = json.load(stream)
        self._bulkResults = summary.get("results", [])
        self._bulkAllLatentsPath = summary.get("all_latents")
        self._populateBulkTable()
        self.batchStatusLabel.setText(
            "Classified {} meshes. Select a row to inspect it.".format(len(self._bulkResults)))
        self.onLogMessage(
            "\n\n\nBatch classification complete: {} meshes.\n\nSummary saved to {}".format(
                len(self._bulkResults), summaryPath), color="#4CAF50")
        headers, rows = self._batchTable(self._bulkResults)
        self._autoSaveTables(os.path.splitext(summaryPath)[0], headers, rows)
        self.tabWidget.setCurrentIndex(self._batchTabIndex)

    def _populateBulkTable(self):
        self.bulkTable.setRowCount(len(self._bulkResults))
        for row, result in enumerate(self._bulkResults):
            matches = result.get("matches", [])
            top = matches[0] if matches else None
            values = [
                result.get("input_name", ""),
                top["mesh_name"] if top else "-",
                "{:.6f}".format(top["cosine_distance"]) if top else "-",
            ]
            for column, value in enumerate(values):
                self.bulkTable.setItem(row, column, qt.QTableWidgetItem(value))

    def onBulkRowSelected(self):
        row = self.bulkTable.currentRow()
        if row < 0 or row >= len(self._bulkResults):
            return
        result = self._bulkResults[row]
        self.inputFilePath = result.get("input_path")
        self._fossilPath = result.get("fossil_path") or result.get("fossil_name")
        if self.inputFilePath:
            self.inputFileLabel.setText(self.inputFilePath)
        self.classificationMatches = result.get("matches", [])
        self._allLatentsPath = self._bulkAllLatentsPath
        self._fossilLatentPath = result.get("fossil_latent")
        self._top5IndicesPath = result.get("top5_indices")
        self._plotsRenderedFor = None
        self._updateClassificationTable()
        self.onLogMessage("Loaded batch result: " + result.get("input_name", ""), color="#4CAF50")

    def _resolveMatch(self, match):
        # Never let a backend exception escape a Qt slot and abort the viewer/table loop.
        meshName = match.get("mesh_name", "")
        try:
            return self._referenceBackend.resolve(meshName)
        except Exception as e:
            return Resolution("missing", None, "Could not resolve {}: {}".format(meshName, e), STATE_MISSING)
            
    def _referencePath(self, match):
        if not self.referenceMeshDirectory:
            return None
        name = match.get("mesh_name", "")
        direct = os.path.join(self.referenceMeshDirectory, name)
        if os.path.isfile(direct):
            return direct
        found = glob.glob(os.path.join(self.referenceMeshDirectory, "**", name), recursive=True)
        return found[0] if found else None

    def _resolveFossilPath(self):
        if self.inputFilePath and os.path.isfile(self.inputFilePath):
            return self.inputFilePath
        if hasattr(self, "_fossilPath") and self._fossilPath:
            if os.path.isfile(self._fossilPath):
                return self._fossilPath

        return None
        
    def _updateClassificationTable(self):
        self.classificationTable.setRowCount(len(self.classificationMatches))
        available = 0
        for row, match in enumerate(self.classificationMatches):
            meshPath = self._referencePath(match)
            values = [
                match["mesh_name"],
                "{:.6f}".format(match["cosine_distance"]),
                "Yes" if meshPath else "No - select matching library",
            ]
            for column, value in enumerate(values):
                self.classificationTable.setItem(row, column, qt.QTableWidgetItem(value))
            if meshPath:
                available += 1

    # ------------------------------------------------------------------ #
    #  Saving / exporting results (CSV, PNG, HTML)
    # ------------------------------------------------------------------ #

    def _noteExportDir(self, directory):
        self._lastExportDir = directory
        self.openSaveFolderButton.setEnabled(True)
        self.openSaveFolderButton.setText("Open Save Folder")
        self.openSaveFolderButton.setToolTip(directory)

    def onOpenSaveFolder(self):
        directory = self._lastExportDir
        if not directory or not os.path.isdir(directory):
            slicer.util.warningDisplay("No saved output folder yet — export or save a plot first.")
            return
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(directory))

    def _resultsDir(self):
        for path in (self._allLatentsPath, self._fossilLatentPath, self._top5IndicesPath):
            if path and os.path.dirname(path):
                return os.path.dirname(path)
        candidate = self._classificationDir()
        if candidate:
            os.makedirs(candidate, exist_ok=True)
            return candidate
        return os.getcwd()

    def _writeTable(self, base, headers, rows):
        rows = [[("" if value is None else str(value)) for value in row] for row in rows]
        csvPath = base + ".csv"
        with open(csvPath, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        return csvPath

    def _autoSaveTables(self, base, headers, rows):
        try:
            csvPath = self._writeTable(base, headers, rows)
            self._noteExportDir(os.path.dirname(csvPath))
            self.onLogMessage("\n{}".format(csvPath), color="#4CAF50")
        except Exception as error:
            self.onLogMessage("Could not auto-save CSV: {}".format(error), color="orange")

    def onExportTopMatches(self):
        if not self.classificationMatches:
            slicer.util.warningDisplay("No top-5 matches to export. Run or load a classification first.")
            return
        default = os.path.join(self._resultsDir(), "top5_matches.csv")
        path = qt.QFileDialog.getSaveFileName(
            None, "Export Top-5 Matches", default, "CSV (*.csv)")
        if not path:
            return
        headers, rows = self._topMatchesTable(self.classificationMatches)
        csvPath = self._writeTable(os.path.splitext(path)[0], headers, rows)
        self._noteExportDir(os.path.dirname(csvPath))
        self.onLogMessage("Exported top-5 matches to:\n{}".format(csvPath), color="#4CAF50")

    def _topMatchesTable(self, matches):
        headers = ["rank", "mesh_name", "cosine_distance", "latent_index", "training_path"]
        rows = [[
            match.get("rank", ""),
            match.get("mesh_name", ""),
            "{:.6f}".format(match["cosine_distance"]) if "cosine_distance" in match else "",
            match.get("latent_index", ""),
            match.get("training_path", ""),
        ] for match in matches]
        return headers, rows

    def _batchTable(self, results):
        headers = ["input_name"]
        for rank in range(1, 6):
            headers += ["top{}_mesh".format(rank), "top{}_cosine".format(rank)]
        rows = []
        for result in results:
            matches = result.get("matches", [])
            row = [result.get("input_name", "")]
            for i in range(5):
                if i < len(matches):
                    row += [matches[i].get("mesh_name", ""),
                            "{:.6f}".format(matches[i]["cosine_distance"])]
                else:
                    row += ["", ""]
            rows.append(row)
        return headers, rows

    def onExportBatchResults(self):
        if not self._bulkResults:
            slicer.util.warningDisplay("No batch results to export. Run or load 'Classify Folder' first.")
            return
        default = os.path.join(self._resultsDir(), "batch_results.csv")
        path = qt.QFileDialog.getSaveFileName(
            None, "Export Batch Results", default, "CSV (*.csv)")
        if not path:
            return
        headers, rows = self._batchTable(self._bulkResults)
        csvPath = self._writeTable(os.path.splitext(path)[0], headers, rows)
        self._noteExportDir(os.path.dirname(csvPath))
        self.onLogMessage("Exported batch results to:\n{}".format(csvPath), color="#4CAF50")

    def _ensurePlotLibs(self):
        global matplotlib, plt, go

        if matplotlib is None or plt is None:
            slicer.util.pip_install("matplotlib")
            import matplotlib
            import matplotlib.pyplot as plt

        if go is None:
            slicer.util.pip_install("plotly")
            import plotly.graph_objects as go

    def _coordsForPlotType(self, plotType):
        return {
            "PCA": self._pcaCoords,
            "t-SNE": self._tsneCoords,
            "UMAP": self._umapCoords,
        }.get(plotType)

    def _plotGroups(self, coords):
        fossilIdx = self._cachedFossilIdx
        top5Indices = self._cachedTop5Indices
        train_filenames = []
        if self.configFilePath and os.path.isfile(self.configFilePath):
            try:
                with open(self.configFilePath, "r") as stream:
                    cfg = json.load(stream)
                    train_filenames = [os.path.basename(p) for p in cfg.get("list_mesh_paths", [])]
            except Exception as error:
                print("[FossilNSM] Could not parse config for filenames: {}".format(error))

        bgMask = np.ones(len(coords), dtype=bool)
        bgMask[fossilIdx] = False
        valid_top5 = top5Indices[top5Indices < len(coords)]
        bgMask[valid_top5] = False

        bgLabels = [train_filenames[i] if i < len(train_filenames) else "Mesh {}".format(i)
                    for i in range(len(coords)) if bgMask[i]]
        top5Labels = []
        for i, idx in enumerate(valid_top5):
            match = self.classificationMatches[i] if i < len(self.classificationMatches) else {}
            top5Labels.append("Top {}: {}".format(i + 1, match.get("mesh_name", "Mesh {}".format(idx))))
        fossil_name = (os.path.basename(self.inputFilePath) if self.inputFilePath
                       else os.path.basename(self._fossilPath) if self._fossilPath
                       else "Fossil")
        return {
            "bg": (coords[bgMask, 0], coords[bgMask, 1], bgLabels),
            "top5": (coords[valid_top5, 0], coords[valid_top5, 1], top5Labels),
            "fossil": ([coords[fossilIdx, 0]], [coords[fossilIdx, 1]], ["Fossil: " + fossil_name]),
        }

    def _savePlotFiles(self, plotType, coords, outDir):
        groups = self._plotGroups(coords)
        if self.inputFilePath:
            meshBase = os.path.splitext(os.path.basename(self.inputFilePath))[0]
        elif self._fossilPath:
            meshBase = os.path.splitext(os.path.basename(self._fossilPath))[0]
        elif self._previousResultsBaseName:
            meshBase = self._previousResultsBaseName
        else:
            meshBase = "fossil"
        pngPath = os.path.join(outDir, "{}_classif_{}.png".format(meshBase, plotType))
        htmlPath = os.path.join(outDir, "{}_classif_{}.html".format(meshBase, plotType))
        written = []

        bx, by, _ = groups["bg"]
        tx, ty, _ = groups["top5"]
        fx, fy, _ = groups["fossil"]
        _, _, bl = groups["bg"]
        _, _, tl = groups["top5"]
        _, _, fl = groups["fossil"]

        try:
            matplotlib.use("Agg")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(bx, by, s=12, c="#808080", alpha=0.6, label="Train data")
            ax.scatter(tx, ty, s=60, c="#33a6e6", edgecolors="k", linewidths=0.3, label="Top-5 matches")
            ax.scatter(fx, fy, s=180, c="#e69933", marker="*", edgecolors="k", linewidths=0.5, label="Fossil")
            ax.set_title("{} of latent space".format(plotType))
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(pngPath, dpi=200)
            plt.close(fig)
            written.append(pngPath)
        except Exception as error:
            self.onLogMessage("Could not save PNG for {}: {}".format(plotType, error), color="red")

        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(bx), y=list(by), mode="markers", name="Train data",
                marker=dict(size=6, color="#808080"), text=bl, hoverinfo="text"))
            fig.add_trace(go.Scatter(
                x=list(tx), y=list(ty), mode="markers", name="Top-5 matches",
                marker=dict(size=12, color="#33a6e6", line=dict(width=1, color="black")),
                text=tl, hoverinfo="text"))
            fig.add_trace(go.Scatter(
                x=list(fx), y=list(fy), mode="markers", name="Fossil",
                marker=dict(size=18, color="#e69933", symbol="star", line=dict(width=1, color="black")),
                text=fl, hoverinfo="text"))
            fig.update_layout(
                title="{} of latent space".format(plotType),
                xaxis_title="Dimension 1", yaxis_title="Dimension 2")
            fig.write_html(htmlPath)
            written.append(htmlPath)
        except Exception as error:
            self.onLogMessage("Could not save interactive HTML for {}: {}".format(plotType, error), color="red")

        return written

    def onSaveCurrentPlot(self):
        if not self.classificationMatches:
            slicer.util.warningDisplay("Run or load a classification result first.")
            return
        if self._cachedCombined is None or self._cachedFossilIdx is None:
            slicer.util.warningDisplay(
                "Open the Explore Plots tab first so the latent projection is computed.")
            return
        self._ensurePlotLibs()
        outDir = self._resultsDir()
        if self.saveAllPlotsCheckbox.checked:
            plotTypes = ["PCA", "t-SNE", "UMAP"]
        else:
            plotTypes = [self.plotTypeComboBox.currentText]
        saved = []
        for plotType in plotTypes:
            coords = self._coordsForPlotType(plotType)
            if coords is None:
                self.onLogMessage(
                    "Skipping {} (not computed yet — open it in Explore Plots first).".format(plotType),
                    color="orange")
                continue
            saved.extend(self._savePlotFiles(plotType, coords, outDir))

        if saved:
            self._noteExportDir(outDir)
            self.plotStatusLabel.setText(
                "Saved to:\n{}\n\n".format(outDir)
                + "\n".join(os.path.basename(p) for p in saved)
                + "\n\nUse 'Open Save Folder' to view them.")
            self.onLogMessage("Saved plot files:\n" + "\n".join(saved), color="#4CAF50")
        else:
            self.onLogMessage("No plots were saved.", color="orange")

    def _clearClassificationNodes(self):
        prefixes = ["Input Fossil", "Match 1 - ", "Match 2 - ", "Match 3 - ", "Match 4 - ", "Match 5 - ",
                    "Top 1 - ", "Top 2 - ", "Top 3 - ", "Top 4 - ", "Top 5 - ",]
        nodesToRemove = []
        for node in self._classificationNodes:
            if node and slicer.mrmlScene.IsNodePresent(node):
                nodesToRemove.append(node)
        collection = slicer.mrmlScene.GetNodes()
        collection.InitTraversal()
        for _ in range(collection.GetNumberOfItems()):
            node = collection.GetNextItemAsObject()
            if not node:
                continue
            name = node.GetName() if hasattr(node, "GetName") else ""
            if any(name.startswith(prefix) for prefix in prefixes):
                nodesToRemove.append(node)
        seen = set()
        uniqueNodes = []
        for node in nodesToRemove:
            nodeID = node.GetID() if node else None
            if nodeID and nodeID not in seen:
                seen.add(nodeID)
                uniqueNodes.append(node)
        for node in uniqueNodes:
            if node and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self._classificationNodes = []


class ClassificationLogic(FossilNsmLogic):
    pass
