import glob
import json
import os
import subprocess
import sys

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import *

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

        FossilNsmLogic.installDependenciesIfNeeded()

        self.addFossilNsmInputSection(self.layout)

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

        self.statusLog = qt.QPlainTextEdit()
        self.statusLog.setReadOnly(True)
        self.statusLog.setFixedHeight(120)
        self.layout.addWidget(self.statusLog)

        self.layout.addStretch(1)
        self.updateRunButton()

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
            values = [
                str(match["rank"]),
                match["mesh_name"],
                "{:.6f}".format(match["cosine_distance"]),
                "Yes" if meshPath else "No - select matching library",
            ]
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


class ClassificationLogic(FossilNsmLogic):
    pass
