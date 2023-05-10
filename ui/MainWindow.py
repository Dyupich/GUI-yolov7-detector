import datetime
import os
import subprocess
import sys

import pylab as p
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLineEdit, QFileDialog, QTextEdit
from PyQt5.QtGui import QIcon, QRegExpValidator
from PyQt5.QtCore import QRegExp, QProcess, QByteArray


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        ui_path = "ui/MainWindow.ui"
        uic.loadUi(ui_path, self)
        self.setWindowTitle("YOLO Object Detector")
        self.setWindowIcon(QIcon("icons/logo.png"))
        self.setFixedSize(self.size())


        # Set updating Ui for yolo messages
        self.yolo_process = QProcess()

        self.yolo_process.started.connect(self.detection_output_clear)
        self.yolo_process.readyRead.connect(self.detection_output_update)

        # Set variables from GUI
        self.source_path_button: QPushButton = self.findChild(QPushButton, "SourcePathButton")
        self.weight_button: QPushButton = self.findChild(QPushButton, "WeightPathPushButton")
        self.save_labels_button: QPushButton = self.findChild(QPushButton, "SaveLabelsButton")
        self.detect_objects_button: QPushButton = self.findChild(QPushButton, "DetectObjectsButton")
        self.source_path_edit: QLineEdit = self.findChild(QLineEdit, "SourcePathLineEdit")
        self.weight_edit: QLineEdit = self.findChild(QLineEdit, "WeightPathLineEdit")
        self.threshold_edit: QLineEdit = self.findChild(QLineEdit, "ThresholdLineEdit")
        self.detection_output_edit: QTextEdit = self.findChild(QTextEdit, "DetectionOutputTextEdit")

        # Update UI with user pathes
        self.set_line_edit_init_path(self.source_path_edit, "data/example.jpg")
        self.set_line_edit_init_path(self.weight_edit, "weights/yolov7_x_640_sgd_best.pt")

        # set limits for line edits

        validator = QRegExpValidator(QRegExp(r'[0-9].+'))
        self.threshold_edit.setValidator(validator)
        # Connect listeners to buttons
        self.weight_button.clicked.connect(self.weight_button_clicked)
        self.source_path_button.clicked.connect(self.source_button_clicked)
        self.detect_objects_button.clicked.connect(self.detect_objects)

    def set_line_edit_init_path(self, line_edit: QLineEdit, base_path: str):
        base_weight_path = os.path.join(os.getcwd(), base_path).replace('\\', '/')
        if os.path.isfile(base_weight_path):
            line_edit.setText(base_weight_path)
            return
        line_edit.setText("Project files damaged. Choose path manually.")

    def weight_button_clicked(self):
        file_path = str(
            QFileDialog.getOpenFileName(self, "Select weight file", os.getcwd(), "PyTorch weights (*.pt)")[0]
        )
        if not os.path.exists(file_path):
            return
        self.weight_edit.setText(file_path)

    def source_button_clicked(self):
        file_path = str(
            QFileDialog.getOpenFileName(self, "Select source file", os.getcwd(),
                                        "Source file (*.jpg *.jpeg *.png *.avi *mp4)")[0]
        )
        if not os.path.exists(file_path):
            return
        self.source_path_edit.setText(file_path)

    def detect_objects(self):
        file_path = QFileDialog.getExistingDirectory(self, "Выберите каталог для сохранения результатов детектирования",
                                                     os.getcwd())
        if not file_path:
            return

        # variables initalization for next usage
        weight_path = self.weight_edit.text()
        source_path = self.source_path_edit.text()
        project_path = weight_path[weight_path.rfind('/') + 1: weight_path.rfind('.pt')]
        current_time = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

        # Call yolo by using subprocess
        cmd = ['./yolov7/detect.py', "--weights", weight_path, "--conf",
               self.threshold_edit.text(),
               "--img-size", "640", '--source', source_path, "--no-trace", "--save-txt",
               "--project",
               f"{file_path}/{project_path}_detections",
               "--name", f"detection_{current_time}"]

        self.yolo_process.start(sys.executable, cmd)  # sys.executable -> for using venv dependencies

    def detection_output_update(self):
        out: QByteArray = self.yolo_process.readAll()
        old_txt: str = self.detection_output_edit.toPlainText()
        new_txt: str = old_txt + bytes(out).decode()
        self.detection_output_edit.setText(new_txt)
        self.detection_output_edit.verticalScrollBar().setValue(
            self.detection_output_edit.verticalScrollBar().maximum()
        )

    def detection_output_clear(self):
        self.detection_output_edit.setText("")
