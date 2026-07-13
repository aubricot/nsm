import glob
import json
import os
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
import slicer
from slicer.ScriptedLoadableModule import *

FOSSIL_NSM_MESH_LAYOUT_ID = 702
FOSSIL_NSM_PLOT_LAYOUT_ID = 703
MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)
from FossilNsmCommon import FossilNsmCommonWidget, FossilNsmLogic


class Classification(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Classification"
        self.parent.categories = ["FossilNSM"]
        self.parent.contributors = ["Wolcott et all"]
        self.parent.helpText = "Classify an input fossil mesh by ranking the nearest latent-space meshes."


class ClassificationWidget(FossilNsmCommonWidget, ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.initializeFossilNsmState()
        self.referenceMeshDirectory = None
        self.classificationMatches = []
        self._classificationNodes = []
        self._plotsRenderedFor = None
        self._classificationMode = "single"
        self._allLatentsPath = None
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

        self.inputFolderButton = qt.QPushButton("Select Input Folder...")
        self.inputFolderButton.connect("clicked(bool)", self.onSelectInputFolder)
        self.inputFolderLabel = qt.QLabel("No folder selected (for batch classification)")
        self.inputFolderLabel.setWordWrap(True)
        self.inputLayout.addRow("Input Folder (Batch):", self.inputFolderButton)
        self.inputLayout.addRow("", self.inputFolderLabel)

        inferenceLayout.addWidget(self.statusLog)

        inferenceBottomCollapsible = ctk.ctkCollapsibleButton()
        inferenceBottomCollapsible.text = "Classification Settings"
        inferenceLayout.addWidget(inferenceBottomCollapsible)
        inferenceBottomLayout = qt.QFormLayout(inferenceBottomCollapsible)

        self.referenceMeshesButton = qt.QPushButton("Select Reference Mesh Library...")
        self.referenceMeshesButton.connect("clicked(bool)", self.onSelectReferenceMeshDirectory)
        self.referenceMeshesLabel = qt.QLabel("Optional: needed to visualize returned meshes")
        self.referenceMeshesLabel.setWordWrap(True)
        inferenceBottomLayout.addRow("Reference Mesh Library:", self.referenceMeshesButton)
        inferenceBottomLayout.addRow("", self.referenceMeshesLabel)

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

        self.refreshButton = qt.QPushButton("Refresh (Clear Scene)")
        self.refreshButton.connect("clicked(bool)", self.onRefreshScene)
        inferenceBottomLayout.addRow(self.refreshButton)

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
        self._batchTabIndex = self.tabWidget.addTab(batchResultsTab, "Batch Results")

        self.classificationTable = qt.QTableWidget(0, 3)
        self.classificationTable.setHorizontalHeaderLabels(["Reference mesh", "Cosine distance", "Available"])
        self.classificationTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.classificationTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.classificationTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.classificationTable.horizontalHeader().setStretchLastSection(True)
        self.classificationTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        classificationLayout.addRow("Top-5 matches:", self.classificationTable)

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
        self._updateClassificationTable()

    def _modelReady(self):
        return bool(
            self.modelRootPath
            and self.configFilePath
            and self.modelFilePath
            and self.latentCodesFilePath
            and self.outputFolderPath
        )

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
        resultDirectory = os.path.join(self.outputFolderPath, "classification")
        os.makedirs(resultDirectory, exist_ok=True)
        inputBase = os.path.splitext(os.path.basename(self.inputFilePath))[0]
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
        resultDirectory = os.path.join(self.outputFolderPath, "classification")
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
        if index == 1:
            self._applyMeshLayout()
            self._loadMeshesIntoViewers()
            self._linkMeshViewers()
            self._styleViewers()
        elif index == 2:
            self._applyPlotLayout()
            self._renderLatentSpacePlots()
        else:
            slicer.app.layoutManager().setLayout(self.previousLayout)

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
            self._pcaCoords = PCA(n_components=2).fit_transform(combined)
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
                metric=metric, random_state=42,
            )
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
                spread=spread, n_epochs=n_epochs, random_state=42,
            ).fit_transform(combined_pca)
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

    def _buildPlot(self, coords, fossilIdx, top5Indices, title):
        print("[FossilNSM] _buildPlot title={}".format(title))

        # Remove old chart and its series nodes if they exist
        oldChart = slicer.mrmlScene.GetFirstNodeByName(title + "_chart")
        if oldChart:
            for i in range(oldChart.GetNumberOfPlotSeriesNodes()):
                s = oldChart.GetNthPlotSeriesNode(i)
                if s:
                    slicer.mrmlScene.RemoveNode(s)
            slicer.mrmlScene.RemoveNode(oldChart)

        # Remove old table nodes
        for suffix in ["_bg", "_top5", "_fossil"]:
            old = slicer.mrmlScene.GetFirstNodeByName(title + suffix)
            if old:
                slicer.mrmlScene.RemoveNode(old)

        def makeTable(name, x, y):
            t = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
            t.RemoveAllColumns()
            xArr = vtk.vtkFloatArray()
            xArr.SetName("x")
            yArr = vtk.vtkFloatArray()
            yArr.SetName("y")
            for xi, yi in zip(x, y):
                xArr.InsertNextValue(float(xi))
                yArr.InsertNextValue(float(yi))
            t.AddColumn(xArr)
            t.AddColumn(yArr)
            return t

        bgMask = np.ones(len(coords), dtype=bool)
        bgMask[fossilIdx] = False
        bgMask[top5Indices] = False

        bgTable     = makeTable(title + "_bg",     coords[bgMask, 0],      coords[bgMask, 1])
        top5Table   = makeTable(title + "_top5",   coords[top5Indices, 0], coords[top5Indices, 1])
        fossilTable = makeTable(title + "_fossil", [coords[fossilIdx, 0]], [coords[fossilIdx, 1]])

        def makeSeries(label, table, color, size=5):
            s = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", label)
            s.SetAndObserveTableNodeID(table.GetID())
            s.SetXColumnName("x")
            s.SetYColumnName("y")
            s.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
            s.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
            s.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleSquare)
            s.SetMarkerSize(size)
            s.SetColor(*color)
            return s

        bgSeries     = makeSeries("Train Data",    bgTable,     (0.5, 0.5, 0.5), size=4)
        top5Series   = makeSeries("Top-5 Matches", top5Table,   (0.2, 0.65, 0.9), size=10)
        fossilSeries = makeSeries("Fossil",        fossilTable, (0.9, 0.6, 0.2), size=14)

        chart = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", title + "_chart")
        chart.SetTitle(title)
        chart.AddAndObservePlotSeriesNodeID(fossilSeries.GetID())
        chart.AddAndObservePlotSeriesNodeID(top5Series.GetID())
        chart.AddAndObservePlotSeriesNodeID(bgSeries.GetID())
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

        # Clear by tracking list first
        self._clearClassificationNodes()

        # Also sweep scene for any leftover nodes by known name patterns
        for nodeName in ["Input Fossil"] + ["Match {} - {}".format(m["rank"], m["mesh_name"]) for m in self.classificationMatches]:
            old = slicer.mrmlScene.GetFirstNodeByName(nodeName)
            if old:
                slicer.mrmlScene.RemoveNode(old)

        if self.inputFilePath and os.path.isfile(self.inputFilePath):
            fossilNode = slicer.util.loadModel(self.inputFilePath)
            fossilNode.SetName("Input Fossil")
            fossilNode.CreateDefaultDisplayNodes()
            fossilNode.GetDisplayNode().SetColor(0.9, 0.6, 0.2)
            fossilNode.GetDisplayNode().SetAmbient(0.3)
            fossilNode.GetDisplayNode().SetDiffuse(0.8)
            fossilNode.GetDisplayNode().SetSpecular(0.0)
            self._assignNodeToView(fossilNode, "FossilInput")
            self._classificationNodes.append(fossilNode)
            self._labelViewer("FossilInput", os.path.basename(self.inputFilePath))

        for match in self.classificationMatches[:5]:
            tag = "Match{}".format(match["rank"])
            meshPath = self._referencePath(match)
            if not meshPath:
                continue
            node = slicer.util.loadModel(meshPath)
            node.SetName("Match {} - {}".format(match["rank"], match["mesh_name"]))
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(0.2, 0.65, 0.9)
            node.GetDisplayNode().SetAmbient(0.3)
            node.GetDisplayNode().SetDiffuse(0.8)
            node.GetDisplayNode().SetSpecular(0.0)
            self._assignNodeToView(node, tag)
            self._classificationNodes.append(node)
            self._labelViewer(tag, match["mesh_name"])

    def _assignNodeToView(self, modelNode, viewTag):
        # Remove all existing view restrictions first, then pin to exactly one view
        displayNode = modelNode.GetDisplayNode()
        displayNode.RemoveAllViewNodeIDs()
        viewNode = slicer.mrmlScene.GetSingletonNode(viewTag, "vtkMRMLViewNode")
        if viewNode:
            displayNode.AddViewNodeID(viewNode.GetID())

    def _linkMeshViewers(self):
        tags = ["FossilInput", "Match1", "Match2", "Match3", "Match4", "Match5"]
        # All views must share the same LinkedControl group for synced rotation
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

    def onRefreshScene(self):
        self._clearClassificationNodes()
        slicer.mrmlScene.Clear(0)
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
        self.updateRunButton()
        self.onLogMessage("Scene cleared.", color="#4CAF50")

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
            self.classificationMatches = json.load(stream).get("matches", [])
        self._plotsRenderedFor = None
        self._updateClassificationTable()
        self.onLogMessage("\n\n\nClassification complete. \nTop-5 matches saved to " + self._classificationResultPath, color="#4CAF50")

    def _loadBulkResults(self, summaryPath):
        with open(summaryPath, encoding="utf-8") as stream:
            summary = json.load(stream)
        self._bulkResults = summary.get("results", [])
        self._bulkAllLatentsPath = summary.get("all_latents")
        self._populateBulkTable()
        self.batchStatusLabel.setText(
            "Classified {} meshes. Select a row to inspect it.".format(len(self._bulkResults)))
        self.onLogMessage(
            "\n\n\nBatch classification complete: {} meshes.\nSummary saved to {}".format(
                len(self._bulkResults), summaryPath), color="#4CAF50")
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
        if self.inputFilePath:
            self.inputFileLabel.setText(self.inputFilePath)
        self.classificationMatches = result.get("matches", [])
        self._allLatentsPath = self._bulkAllLatentsPath
        self._fossilLatentPath = result.get("fossil_latent")
        self._top5IndicesPath = result.get("top5_indices")
        self._plotsRenderedFor = None
        self._updateClassificationTable()
        self.onLogMessage("Loaded batch result: " + result.get("input_name", ""), color="#4CAF50")

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
            values = [
                match["mesh_name"],
                "{:.6f}".format(match["cosine_distance"]),
                "Yes" if meshPath else "No - select matching library",
            ]
            for column, value in enumerate(values):
                self.classificationTable.setItem(row, column, qt.QTableWidgetItem(value))
            if meshPath:
                available += 1

    def _clearClassificationNodes(self):
        for node in self._classificationNodes:
            if node and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self._classificationNodes = []

    def _showMatch(self, match, offset=0.0):
        meshPath = self._referencePath(match)
        if not meshPath:
            self.onLogMessage("Reference mesh is not available: " + match["mesh_name"], color="red")
            return None
        node = slicer.util.loadModel(meshPath)
        node.SetName("Top {} - {} ({:.4f})".format(match["rank"], match["mesh_name"], match["cosine_distance"]))
        node.CreateDefaultDisplayNodes()
        node.GetDisplayNode().SetColor(0.2, 0.65, 0.9)
        if offset:
            transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", node.GetName() + " transform")
            transform.GetTransformToParent().Translate(offset, 0, 0)
            node.SetAndObserveTransformNodeID(transform.GetID())
            self._classificationNodes.append(transform)
        self._classificationNodes.append(node)
        return node


class ClassificationLogic(FossilNsmLogic):
    pass