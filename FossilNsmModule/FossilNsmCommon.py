import os
import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleLogic

FOSSIL_NSM_MESH_LAYOUT_ID = 702
FOSSIL_NSM_PLOT_LAYOUT_ID = 703

class FossilNsmCommon(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "FossilNsmCommon"
        self.parent.categories = ["FossilNSM"]
        self.parent.contributors = ["Wolcott et al"]
        self.parent.helpText = "Shared logic for FossilNSM modules."
        self.parent.hidden = True 

class FossilNsmCommonWidget:
    def initializeFossilNsmState(self):
        self.modelRootPath = None
        self.inputFilePath = None
        self.configFilePath = None
        self.modelFilePath = None
        self.latentCodesFilePath = None
        self.outputFolderPath = None

    def addFossilNsmInputSection(self, parentLayout):
        inputCollapsible = ctk.ctkCollapsibleButton()
        inputCollapsible.text = "Inputs"
        parentLayout.addWidget(inputCollapsible)
        inputLayout = qt.QFormLayout(inputCollapsible)
        self.inputLayout = inputLayout

        self.modelRootButton = qt.QPushButton("Select Model Root (run_vXX)...")
        self.modelRootLabel = qt.QLabel("No model selected")
        inputLayout.addRow("Model Root:", self.modelRootButton)
        inputLayout.addRow("", self.modelRootLabel)
        self.modelRootButton.connect("clicked(bool)", self.onSelectModelRoot)

        self.modelChecklist = qt.QTextEdit()
        self.modelChecklist.setReadOnly(True)
        self.modelChecklist.setFixedHeight(110)
        inputLayout.addRow("Model Validation:", self.modelChecklist)

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

        self.inputFileButton = qt.QPushButton("Select Input Mesh...")
        self.inputFileButton.connect("clicked(bool)", self.onSelectInputFile)
        self.inputFileLabel = qt.QLabel("No file selected")
        self.inputFileLabel.setWordWrap(True)
        inputLayout.addRow("Input Mesh:", self.inputFileButton)
        inputLayout.addRow("", self.inputFileLabel)

    def onSelectConfigFile(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Config File", "", "Config Files (*.json)"
        )
        if path:
            self.configFilePath = path
            self.configFileLabel.setText(path)
            self.updateRunButton()

    def validateModelRoot(self, rootDir):
        checks = []

        def check(label, path, isDir=False):
            ok = os.path.isdir(path) if isDir else os.path.isfile(path)
            checks.append((label, ok, path))
            return ok

        check("model_params_config.json", os.path.join(rootDir, "model_params_config.json"))
        check("model/ folder", os.path.join(rootDir, "model"), isDir=True)
        check("latent_codes/ folder", os.path.join(rootDir, "latent_codes"), isDir=True)
        check("shape_completion/ (optional)", os.path.join(rootDir, "shape_completion"), isDir=True)
        return checks

    def updateChecklistUI(self, rootDir):
        checks = self.validateModelRoot(rootDir)
        lines = []
        allRequiredOk = True

        for label, ok, path in checks:
            icon = "OK" if ok else "MISSING"
            lines.append("{} {}".format(icon, label))
            if not ok and "optional" not in label:
                allRequiredOk = False

        self.modelChecklist.setPlainText("\n".join(lines))
        return allRequiredOk

    def resolveModelRoot(self, rootDir):
        config = os.path.join(rootDir, "model_params_config.json")
        modelDir = os.path.join(rootDir, "model")
        latentDir = os.path.join(rootDir, "latent_codes")
        outputDir = os.path.join(rootDir, "shape_completion")

        missing = []
        if not os.path.isfile(config):
            missing.append("model_params_config.json")
        if not os.path.isdir(modelDir):
            missing.append("model/")
        if not os.path.isdir(latentDir):
            missing.append("latent_codes/")

        if missing:
            raise ValueError("Invalid model package. Missing: {}".format(missing))

        modelFiles = sorted([f for f in os.listdir(modelDir) if f.endswith(".pth")])
        latentFiles = sorted([f for f in os.listdir(latentDir) if f.endswith(".pth")])

        if not modelFiles:
            raise ValueError("No .pth file found in model/")
        if not latentFiles:
            raise ValueError("No .pth file found in latent_codes/")

        modelPath = os.path.join(modelDir, modelFiles[-1])
        latentPath = os.path.join(latentDir, latentFiles[-1])

        return config, modelPath, latentPath, outputDir

    def onSelectModelRoot(self):
        path = qt.QFileDialog.getExistingDirectory(None, "Select Model Root Folder")
        if not path:
            return

        self.configFilePath = None
        self.modelFilePath = None
        self.latentCodesFilePath = None
        self.outputFolderPath = None

        self.modelRootPath = path
        self.modelRootLabel.setText(path)

        ok = self.updateChecklistUI(path)
        if not ok:
            self.onLogMessage("Model root is incomplete. Fix missing files before running.", color="red")
            self.updateRunButton()
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

        encoderCkpt = os.path.join(path, "encoder", "checkpoints", "encoder.pt")
        if hasattr(self, "encoderCkptInput"):
            self.encoderCkptInput.setText(encoderCkpt if os.path.isfile(encoderCkpt) else "")

        self.updateRunButton()

    def onSelectInputFile(self):
        path = qt.QFileDialog.getOpenFileName(
            None, "Select Input Mesh", "", "Mesh Files (*.vtk *.vtp *.stl *.obj *.ply)"
        )
        if not path:
            return
        self.inputFilePath = path
        self.inputFileLabel.setText(path)
        self.updateRunButton()

    def commonInputsReady(self):
        return bool(
            self.modelRootPath
            and self.inputFilePath
            and self.configFilePath
            and self.modelFilePath
            and self.latentCodesFilePath
            and self.outputFolderPath
        )

    def onLogMessage(self, message, color=None):
        html_message = str(message).replace("\n", "<br>")
        if color:
            self.statusLog.append('<span style="color:{};">{}</span>'.format(color, html_message))
        else:
            self.statusLog.append(html_message)

    def addRefreshSceneButton(self, parentLayout):
        self.refreshButton = qt.QPushButton("Refresh (Clear Scene)")
        self.refreshButton.connect("clicked(bool)", self.onRefreshScene)
        parentLayout.addRow(self.refreshButton) if hasattr(parentLayout, "addRow") else parentLayout.addWidget(self.refreshButton)

    def onRefreshScene(self):
        slicer.mrmlScene.Clear(0)
        self.onAfterSceneCleared()
        self.updateRunButton()
        self.onLogMessage("\nScene cleared.\n")

    def onAfterSceneCleared(self):
        pass

    def registerMeshLayout(self):
        layoutDescription = """
        <layout type="vertical" split="false">
        <item>
            <layout type="horizontal">
            <item>
                <view class="vtkMRMLViewNode" singletontag="FossilInput">
                <property name="viewlabel" action="default">Input</property>
                </view>
            </item>
            <item>
                <view class="vtkMRMLViewNode" singletontag="Match1">
                <property name="viewlabel" action="default">Match 1</property>
                </view>
            </item>
            <item>
                <view class="vtkMRMLViewNode" singletontag="Match2">
                <property name="viewlabel" action="default">Match 2</property>
                </view>
            </item>
            </layout>
        </item>
        <item>
            <layout type="horizontal">
            <item>
                <view class="vtkMRMLViewNode" singletontag="Match3">
                <property name="viewlabel" action="default">Match 3</property>
                </view>
            </item>
            <item>
                <view class="vtkMRMLViewNode" singletontag="Match4">
                <property name="viewlabel" action="default">Match 4</property>
                </view>
            </item>
            <item>
                <view class="vtkMRMLViewNode" singletontag="Match5">
                <property name="viewlabel" action="default">Match 5</property>
                </view>
            </item>
            </layout>
        </item>
        </layout>
        """
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        if not layoutNode.IsLayoutDescription(FOSSIL_NSM_MESH_LAYOUT_ID):
            layoutNode.AddLayoutDescription(FOSSIL_NSM_MESH_LAYOUT_ID, layoutDescription)

    def registerPlotLayout(self):
        layoutDescription = """
        <layout type="horizontal" split="false">
        <item>
            <view class="vtkMRMLPlotViewNode" singletontag="PCAPlot">
            <property name="viewlabel" action="default">PCA</property>
            </view>
        </item>
        <item>
            <view class="vtkMRMLPlotViewNode" singletontag="TSNEPlot">
            <property name="viewlabel" action="default">t-SNE</property>
            </view>
        </item>
        </layout>
        """
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        if not layoutNode.IsLayoutDescription(FOSSIL_NSM_PLOT_LAYOUT_ID):
            layoutNode.AddLayoutDescription(FOSSIL_NSM_PLOT_LAYOUT_ID, layoutDescription)

    def getViewNode(self, tag):
        return slicer.mrmlScene.GetSingletonNode(tag, "vtkMRMLViewNode")
    
    def setDefaultThreeDLayout(self):
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        threeDWidget = layoutManager.threeDWidget(0)
        viewNode = threeDWidget.mrmlViewNode()
        viewNode.SetBackgroundColor(0, 0, 0)
        viewNode.SetBackgroundColor2(0, 0, 0)
        viewNode.SetBoxVisible(False)
        viewNode.SetAxisLabelsVisible(False)
        threeDWidget.threeDView().resetFocalPoint()

class FossilNsmLogic(ScriptedLoadableModuleLogic):
    @staticmethod
    def installDependenciesIfNeeded():
        USE_TINY3D = True

        if USE_TINY3D:
            try:
                import tiny3d as o3d
            except ImportError:
                slicer.util.pip_install("tiny3d")
                import tiny3d as o3d
        else:
            try:
                import open3d as o3d
            except ImportError:
                slicer.util.pip_install("open3d")
                import open3d as o3d

        try:
            import cv2
        except ImportError:
            slicer.util.pip_install("opencv-python")

        try:
            import pandas
        except ImportError:
            slicer.util.pip_install("pandas")

        try:
            import nibabel
        except ImportError:
            slicer.util.pip_install("nibabel")

        try:
            import pymskt
        except ImportError:
            slicer.util.pip_install("mskt")

        try:
            import pyvista
        except ImportError:
            slicer.util.pip_install("pyvista")

        try:
            import pymeshfix
        except ImportError:
            slicer.util.pip_install("pymeshfix")

        try:
            import skimage
        except ImportError:
            slicer.util.pip_install("scikit-image")

        try:
            import sklearn
        except ImportError:
            slicer.util.pip_install("scikit-learn")

        try:
            import umap
        except ImportError:
            slicer.util.pip_install("umap-learn")

        try:
            import torch
        except ImportError:
            slicer.util.pip_install("torch")

        try:
            import vtk
        except ImportError:
            slicer.util.pip_install("vtk")
