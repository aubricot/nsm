import glob
import json
import os
import subprocess
import sys
import vtk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
        inferenceLayout.addWidget(self.statusLog)

        # Reference library + iterations at bottom of Inference tab
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

        self.refreshButton = qt.QPushButton("Refresh (Clear Scene)")
        self.refreshButton.connect("clicked(bool)", self.onRefreshScene)
        inferenceBottomLayout.addRow(self.refreshButton)          

        # Tab 2 — Explore Meshes
        exploreMeshesTab = qt.QWidget()
        exploreMeshesOuterLayout = qt.QHBoxLayout(exploreMeshesTab)
        exploreMeshesOuterLayout.setContentsMargins(0, 0, 0, 0)

        # Left side panel: table + buttons
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

        # Reference label so user knows what they're looking at
        self.plotStatusLabel = qt.QLabel("Run classification to populate plots.")
        self.plotStatusLabel.setWordWrap(True)
        plotSidePanelLayout.addWidget(self.plotStatusLabel)
        plotSidePanelLayout.addStretch(1)
        explorePlotsOuterLayout.addWidget(plotSidePanel)

        self.tabWidget.addTab(explorePlotsTab, "Explore Plots")

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

    def onSelectReferenceMeshDirectory(self):
        path = qt.QFileDialog.getExistingDirectory(None, "Select Reference Mesh Library")
        if not path:
            return
        self.referenceMeshDirectory = path
        self.referenceMeshesLabel.setText(path)
        self._updateClassificationTable()

    def updateRunButton(self):
        self.classifyButton.setEnabled(self.commonInputsReady())

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
        self.onLogMessage("Starting nearest-mesh classification (latent optimization may take several minutes)...\n\n\n", color="#4CAF50")
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

    def onTabChanged(self, index):
        if index == 0:
            slicer.app.layoutManager().setLayout(self.previousLayout)
        elif index == 1:
            self._applyMeshLayout()
            self._loadMeshesIntoViewers()  
            self._linkMeshViewers()
            self._styleViewers()
        elif index == 2:
            self._applyPlotLayout()
            self._renderLatentSpacePlots()

    def _renderLatentSpacePlots(self):
        if not self.classificationMatches:
            self.plotStatusLabel.setText("Run classification first.")
            return
        resultDirectory = os.path.join(self.outputFolderPath, "classification")
        allLatentsPath = os.path.join(resultDirectory, "all_latents.npy")
        fossilLatentPath = os.path.join(resultDirectory, "fossil_latent.npy")
        top5IndicesPath = os.path.join(resultDirectory, "top5_indices.npy")

        if not all(os.path.isfile(p) for p in [allLatentsPath, fossilLatentPath, top5IndicesPath]):
            self.plotStatusLabel.setText("Latent files not found. Re-run classification.")
            return

        # Only (re)compute the embeddings when the classification result changed.
        # Switching tabs re-applies the layout but reuses the existing plot nodes.
        cacheKey = os.path.getmtime(fossilLatentPath)
        if self._plotsRenderedFor == cacheKey and slicer.mrmlScene.GetFirstNodeByName("PCA_chart"):
            self.plotStatusLabel.setText("Plots rendered (cached).")
            return

        self.plotStatusLabel.setText("Computing PCA / t-SNE / UMAP...")
        slicer.app.processEvents()

        allLatents = np.load(allLatentsPath)
        fossilLatent = np.load(fossilLatentPath)
        top5Indices = np.load(top5IndicesPath)

        combined = np.vstack([allLatents, fossilLatent])
        fossilIdx = len(allLatents)

        pcaCoords = PCA(n_components=2).fit_transform(combined)
        tsneCoords = TSNE(n_components=2, random_state=42).fit_transform(combined)

        import umap
        umapCoords = umap.UMAP(n_components=2, random_state=42).fit_transform(combined)

        self._buildPlot(pcaCoords, fossilIdx, top5Indices, "PCAPlot", "PCA")
        self._buildPlot(tsneCoords, fossilIdx, top5Indices, "TSNEPlot", "t-SNE")
        self._buildPlot(umapCoords, fossilIdx, top5Indices, "UMAPPlot", "UMAP")

        self._plotsRenderedFor = cacheKey
        self.plotStatusLabel.setText("Plots rendered.")

    def _buildPlot(self, coords, fossilIdx, top5Indices, viewTag, title):
        for suffix in ["_bg", "_top5", "_fossil", "_bg_series", "_top5_series", "_fossil_series", "_chart"]:
            old = slicer.mrmlScene.GetFirstNodeByName(title + suffix)
            if old:
                slicer.mrmlScene.RemoveNode(old)

        def makeTable(name, x, y):
            import vtk
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

        bgTable = makeTable(title + "_bg", coords[bgMask, 0], coords[bgMask, 1])
        top5Table = makeTable(title + "_top5", coords[top5Indices, 0], coords[top5Indices, 1])
        fossilTable = makeTable(title + "_fossil", [coords[fossilIdx, 0]], [coords[fossilIdx, 1]])

        def makeSeries(name, table, color, size=5):
            s = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", name)
            s.SetAndObserveTableNodeID(table.GetID())
            s.SetXColumnName("x")
            s.SetYColumnName("y")
            s.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
            s.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
            s.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleSquare)  # ← match GPA exactly
            s.SetMarkerSize(size)
            s.SetColor(*color)
            return s

        bgSeries = makeSeries("Training Meshes", bgTable, (0.5, 0.5, 0.5), size=4)
        top5Series = makeSeries("Top 5 Matches", top5Table, (0.2, 0.65, 0.9), size=10)
        fossilSeries = makeSeries("Unknown Fossil", fossilTable, (0.9, 0.6, 0.2), size=14)

        chart = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", title + "_chart")
        chart.SetTitle(title)
        chart.AddAndObservePlotSeriesNodeID(bgSeries.GetID())
        chart.AddAndObservePlotSeriesNodeID(top5Series.GetID())
        chart.AddAndObservePlotSeriesNodeID(fossilSeries.GetID())

        plotViewNode = slicer.mrmlScene.GetSingletonNode(viewTag, "vtkMRMLPlotViewNode")
        if plotViewNode:
            plotViewNode.SetPlotChartNodeID(chart.GetID())

    def onRefreshScene(self):
        self._clearClassificationNodes()
        slicer.mrmlScene.Clear(0)
        self.classificationMatches = []
        self._plotsRenderedFor = None
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
        self.classifyButton.setEnabled(True)
        if code != 0 or not os.path.isfile(self._classificationResultPath):
            self.onLogMessage("\n\n\nClassification failed (exit code {}).".format(code), color="red")
            return
        with open(self._classificationResultPath, encoding="utf-8") as stream:
            self.classificationMatches = json.load(stream).get("matches", [])
        self._plotsRenderedFor = None
        self._updateClassificationTable()
        self.onLogMessage("\n\n\nClassification complete. \nTop-5 matches saved to " + self._classificationResultPath, color="#4CAF50")

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

        # Load fossil into FossilInput view
        if self.inputFilePath and os.path.isfile(self.inputFilePath):
            fossilNode = slicer.util.loadModel(self.inputFilePath)
            fossilNode.SetName("Input Fossil")
            fossilNode.CreateDefaultDisplayNodes()
            fossilNode.GetDisplayNode().SetColor(0.9, 0.6, 0.2)
            self._assignNodeToView(fossilNode, "FossilInput")
            self._classificationNodes.append(fossilNode)
            self._labelViewer("FossilInput", os.path.basename(self.inputFilePath))

        # Load matches into Match1–Match5
        for match in self.classificationMatches[:5]:
            tag = "Match{}".format(match["rank"])
            meshPath = self._referencePath(match)
            if not meshPath:
                continue
            node = slicer.util.loadModel(meshPath)
            node.SetName("Match {} - {}".format(match["rank"], match["mesh_name"]))
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(0.2, 0.65, 0.9)
            self._assignNodeToView(node, tag)
            self._classificationNodes.append(node)
            self._labelViewer(tag, match["mesh_name"])

    def _assignNodeToView(self, modelNode, viewTag):
        viewNode = slicer.mrmlScene.GetSingletonNode(viewTag, "vtkMRMLViewNode")
        if viewNode:
            modelNode.GetDisplayNode().AddViewNodeID(viewNode.GetID())

    def _linkMeshViewers(self):
        tags = ["FossilInput", "Match1", "Match2", "Match3", "Match4", "Match5"]
        for tag in tags:
            viewNode = slicer.mrmlScene.GetSingletonNode(tag, "vtkMRMLViewNode")
            if viewNode:
                viewNode.LinkedControlOn()

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
            viewNode.SetBackgroundColor2(0, 0, 0)  # gradient second color
            viewNode.SetBoxVisible(False)
            viewNode.SetAxisLabelsVisible(False)

    def _applyPlotLayout(self):
        layoutDescription = """
        <layout type="horizontal" split="false">
        <item><view class="vtkMRMLPlotViewNode" singletontag="PCAPlot"><property name="viewlabel" action="default">PCA</property></view></item>
        <item><view class="vtkMRMLPlotViewNode" singletontag="TSNEPlot"><property name="viewlabel" action="default">t-SNE</property></view></item>
        <item><view class="vtkMRMLPlotViewNode" singletontag="UMAPPlot"><property name="viewlabel" action="default">UMAP</property></view></item>
        </layout>
        """
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        if not layoutNode.IsLayoutDescription(FOSSIL_NSM_PLOT_LAYOUT_ID):
            layoutNode.AddLayoutDescription(FOSSIL_NSM_PLOT_LAYOUT_ID, layoutDescription)
        slicer.app.layoutManager().setLayout(FOSSIL_NSM_PLOT_LAYOUT_ID)


class ClassificationLogic(FossilNsmLogic):
    pass
