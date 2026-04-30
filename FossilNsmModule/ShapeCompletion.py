import os
import qt
import ctk
import sys
import slicer
import subprocess
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

        ShapeCompletionLogic.installDependenciesIfNeeded()

        # Form Layout
        inputCollapsible = ctk.ctkCollapsibleButton()
        inputCollapsible.text = "Inputs"
        self.layout.addWidget(inputCollapsible)
        inputLayout = qt.QFormLayout(inputCollapsible)

        # Config File
        self.configFileButton = qt.QPushButton("Select Config File...")
        self.configFileButton.connect("clicked(bool)", self.onSelectConfigFile)
        self.configFileLabel = qt.QLabel("No file selected")
        self.configFileLabel.setWordWrap(True)
        inputLayout.addRow("Config File:", self.configFileButton)
        inputLayout.addRow("", self.configFileLabel)

        # Model File
        self.modelFileButton = qt.QPushButton("Select Model File...")
        self.modelFileButton.connect("clicked(bool)", self.onSelectModelFile)
        self.modelFileLabel = qt.QLabel("No file selected")
        self.modelFileLabel.setWordWrap(True)
        inputLayout.addRow("Model File:", self.modelFileButton)
        inputLayout.addRow("", self.modelFileLabel)

        # Latent Codes File
        self.latentCodesFileButton = qt.QPushButton("Select Latent Codes File...")
        self.latentCodesFileButton.connect("clicked(bool)", self.onSelectLatentCodesFile)
        self.latentCodesFileLabel = qt.QLabel("No file selected")
        self.latentCodesFileLabel.setWordWrap(True)
        inputLayout.addRow("Latent Codes File:", self.latentCodesFileButton)
        inputLayout.addRow("", self.latentCodesFileLabel)

        # Input Mesh File
        self.inputFileButton = qt.QPushButton("Select Input Mesh...")
        self.inputFileButton.connect("clicked(bool)", self.onSelectInputFile)
        self.inputFileLabel = qt.QLabel("No file selected")
        self.inputFileLabel.setWordWrap(True)
        inputLayout.addRow("Input Mesh:", self.inputFileButton)
        inputLayout.addRow("", self.inputFileLabel)

        # Output Folder
        self.outputFolderButton = qt.QPushButton("Select Output Folder...")
        self.outputFolderButton.connect("clicked(bool)", self.onSelectOutputFolder)
        self.outputFolderLabel = qt.QLabel("No folder selected")
        self.outputFolderLabel.setWordWrap(True)
        inputLayout.addRow("Output Folder:", self.outputFolderButton)
        inputLayout.addRow("", self.outputFolderLabel)

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

        # Internal state
        self.configFilePath = None
        self.modelFilePath = None
        self.latentCodesFilePath = None
        self.inputFilePath = None
        self.outputFolderPath = None

    # File/Folder Selection
    def onSelectConfigFile(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Config File", "", "Config Files (*.json)"
        )
        if path:
            self.configFilePath = path
            self.configFileLabel.setText(path)
            self.updateRunButton()

    def onSelectModelFile(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Model File", "", "Model Files (*.pth)"
        )
        if path:
            self.modelFilePath = path
            self.modelFileLabel.setText(path)
            self.updateRunButton()
    
    def onSelectLatentCodesFile(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Latent Codes File", "", "Latent Codes Files (*.pth)"
        )
        if path:
            self.latentCodesFilePath = path
            self.latentCodesFileLabel.setText(path)
            self.updateRunButton()

    def onSelectInputFile(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Input Mesh", "", "Mesh Files (*.ply *.vtk)"
        )
        if path:
            self.inputFilePath = path
            self.inputFileLabel.setText(path)
            self.onLogMessage("Loading mesh into scene...")
            modelNode = slicer.util.loadModel(self.inputFilePath)
            modelNode.SetName("Input Mesh")
            slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
            self.updateRunButton()

    def onSelectOutputFolder(self):
        path = qt.QFileDialog.getExistingDirectory(None, "Select Output Folder")
        if path:
            self.outputFolderPath = path
            self.outputFolderLabel.setText(path)
            self.updateRunButton()

    def updateRunButton(self):
        ready = all([self.configFilePath, self.modelFilePath, self.latentCodesFilePath, self.inputFilePath, self.outputFolderPath])
        self.runButton.setEnabled(ready)

    # Inference
    def onRunInference(self):
        self.runButton.setEnabled(False)
        self.progressBar.setVisible(True)
        self.onLogMessage("Starting inference...")

        base = os.path.splitext(os.path.basename(self.inputFilePath))[0]
        self._resultPath = self.outputFolderPath + '/' + base + ".done"
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

        self._process = subprocess.Popen(
            [pythonExe, workerScript,
            self.configFilePath, self.modelFilePath,
            self.latentCodesFilePath, self.inputFilePath,
            self.outputFolderPath],
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
                slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
                self.calculateDistsButton.setEnabled(True)
            else:
                self.onLogMessage(f"Inference failed (exit code {retcode}, file exists: {os.path.exists(self._resultPath)} at {self._resultPath})")

    def onLogMessage(self, message):
        self.statusLog.appendPlainText(message)


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
