import datetime
import os
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QPushButton, QLineEdit, QFileDialog, QTextEdit
from PyQt5.QtGui import QIcon, QRegExpValidator, QTextCursor
from PyQt5.QtCore import QRegExp, QProcess, QByteArray


class MainWindowModel:
    def __init__(self):
        self.window_title = "GUI YOLOv7 Object Detector"
        self.ui_path = "ui/MainWindow.ui"
        self.window_icon_path = "icons/logo.png"
        self.source_base_path = "data/example.jpg"
        self.weight_base_path = "weights/yolov7_x_640_sgd_best.pt"
        self.REG_FLOATS_ONLY = r'[0-9].+'
        self.weight_path_dialog_txt = "Выберите файл нейросетевых весов"
        self.weight_extensions_dialog_txt = "PyTorch weights (*.pt)"
        self.source_path_dialog_txt = "Выберите файл с входными данными"
        self.source_extenstions_dialog_txt = "Source file (*.jpg *.jpeg *.png *.avi *mp4)"
        self.save_path_dialog_txt = "Выберите каталог для сохранения результатов детектирования"
        self.edit_path_base_txt = "Выберите путь, нажав клавишу 'Обзор...'"

        self.detection_start_msg = "Процесс детектирования начат. Ожидайте результатов!\n"

        # Error messages
        self.unicode_err_msg = "Произошла ошибка! Проверьте правильность входных данных"


class MainWindowController:
    def __init__(self, main_window: QMainWindow, main_model: MainWindowModel):
        self.main_window = main_window
        self.main_model = main_model
        self.source_path_button: QPushButton = None
        self.weight_button: QPushButton = None
        self.save_labels_button: QPushButton = None
        self.detect_objects_button: QPushButton = None
        self.source_path_edit: QLineEdit = None
        self.weight_edit: QLineEdit = None
        self.threshold_edit: QLineEdit = None
        self.detection_output_edit: QTextEdit = None

        # Set updating Ui for yolo messages
        self.yolo_process = QProcess()
        self.yolo_process.started.connect(self.detection_output_clear)
        self.yolo_process.readyRead.connect(self.detection_output_update)

    def get_window_attrs(self):
        self.source_path_button: QPushButton = self.main_window.findChild(QPushButton, "SourcePathButton")
        self.weight_button: QPushButton = self.main_window.findChild(QPushButton, "WeightPathPushButton")
        self.save_labels_button: QPushButton = self.main_window.findChild(QPushButton, "SaveLabelsButton")
        self.detect_objects_button: QPushButton = self.main_window.findChild(QPushButton, "DetectObjectsButton")
        self.source_path_edit: QLineEdit = self.main_window.findChild(QLineEdit, "SourcePathLineEdit")
        self.weight_edit: QLineEdit = self.main_window.findChild(QLineEdit, "WeightPathLineEdit")
        self.threshold_edit: QLineEdit = self.main_window.findChild(QLineEdit, "ThresholdLineEdit")
        self.detection_output_edit: QTextEdit = self.main_window.findChild(QTextEdit, "DetectionOutputTextEdit")

    def init_gui_pathes(self, gui_paths: dict):
        for edit, base_path in gui_paths.items():
            path = os.path.join(os.getcwd(), base_path).replace('\\', '/')
            if os.path.isfile(path):
                edit.setText(path)
                continue
            edit.setText(self.main_model.edit_path_base_txt)

    def weight_button_clicked(self):
        file_path = str(
            QFileDialog.getOpenFileName(self.main_window,
                                        self.main_model.weight_path_dialog_txt,
                                        os.getcwd(),
                                        self.main_model.weight_extensions_dialog_txt)[0]
        )
        if not os.path.exists(file_path):
            return
        self.weight_edit.setText(file_path)

    def source_button_clicked(self):
        file_path = str(
            QFileDialog.getOpenFileName(self.main_window,
                                        self.main_model.source_path_dialog_txt,
                                        os.getcwd(),
                                        self.main_model.source_extenstions_dialog_txt)[0]
        )
        if not os.path.exists(file_path):
            return
        self.source_path_edit.setText(file_path)

    def detect_objects(self):
        file_path = QFileDialog.getExistingDirectory(self.main_window,
                                                     self.main_model.save_path_dialog_txt,
                                                     os.getcwd()
                                                     )
        if not file_path:
            return

        # variables initialization for next usage
        weight_path = self.weight_edit.text()
        source_path = self.source_path_edit.text()
        project_path = weight_path[weight_path.rfind('/') + 1: weight_path.rfind('.pt')]
        current_time = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

        # Call yolo by using QProcess
        cmd = ['./yolov7/detect.py', "--weights", weight_path, "--conf",
               self.threshold_edit.text(),
               "--img-size", "640", '--source', source_path, "--no-trace", "--save-txt",
               "--project",
               f"{file_path}/{project_path}_detections",
               "--name", f"detection_{current_time}"]

        self.yolo_process.start(sys.executable, cmd)  # sys.executable -> for using venv dependencies

    def detection_output_update(self):
        if self.detection_output_edit is None:
            return

        try:
            output_bytes: QByteArray = self.yolo_process.readAll()
            output_string: str = bytes(output_bytes).decode()
        except UnicodeDecodeError:
            self.detection_output_edit.setText(self.main_model.unicode_err_msg)
            return
        cursor: QTextCursor = QTextCursor(self.detection_output_edit.document())
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(output_string)
        self.detection_output_edit.verticalScrollBar().setValue(
            self.detection_output_edit.verticalScrollBar().maximum()
        )

    def detection_output_clear(self):
        # Clean old text and set new start text
        self.detection_output_edit.setText(self.main_model.detection_start_msg)


class MainWindowView(QMainWindow):

    def __init__(self):
        super(MainWindowView, self).__init__()
        self.model = MainWindowModel()
        self.controller = MainWindowController(main_window=self, main_model=self.model)
        uic.loadUi(self.model.ui_path, self)
        self.setWindowTitle(self.model.window_title)
        self.setWindowIcon(QIcon(self.model.window_icon_path))
        self.setFixedSize(self.size())

        # Set attrs from GUI
        self.source_path_button: QPushButton = self.findChild(QPushButton, "SourcePathButton")
        self.weight_button: QPushButton = self.findChild(QPushButton, "WeightPathPushButton")
        self.save_labels_button: QPushButton = self.findChild(QPushButton, "SaveLabelsButton")
        self.detect_objects_button: QPushButton = self.findChild(QPushButton, "DetectObjectsButton")
        self.source_path_edit: QLineEdit = self.findChild(QLineEdit, "SourcePathLineEdit")
        self.weight_edit: QLineEdit = self.findChild(QLineEdit, "WeightPathLineEdit")
        self.threshold_edit: QLineEdit = self.findChild(QLineEdit, "ThresholdLineEdit")
        self.detection_output_edit: QTextEdit = self.findChild(QTextEdit, "DetectionOutputTextEdit")

        # Give info about attrs to controller
        self.controller.get_window_attrs()

        # Update UI with base  pathes
        self.controller.init_gui_pathes(
            {
                self.source_path_edit: self.model.source_base_path,
                self.weight_edit: self.model.weight_base_path
            }
        )

        # set limits for line edits
        validator = QRegExpValidator(QRegExp(self.model.REG_FLOATS_ONLY))
        self.threshold_edit.setValidator(validator)

        # Connect listeners to buttons
        self.weight_button.clicked.connect(self.controller.weight_button_clicked)
        self.source_path_button.clicked.connect(self.controller.source_button_clicked)
        self.detect_objects_button.clicked.connect(self.controller.detect_objects)
