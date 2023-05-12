import sys

from PyQt5.QtWidgets import QApplication
from ui.MainWindow import MainWindowView


class App(QApplication):
    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        self.main_window = MainWindowView()
        self.main_window.show()


if __name__ == '__main__':
    app = App(sys.argv)
    sys.exit(app.exec())
