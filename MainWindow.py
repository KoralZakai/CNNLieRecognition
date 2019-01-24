import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
class App(QWidget):


    def __init__(self):
        super().__init__()
        self.title = 'Lie Detection'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        form = QtWidgets.QFormLayout()
        self.setLayout(form)
        lbl = QtWidgets.QLabel("Lie Detection", self)
        lbl.setAlignment(Qt.AlignCenter)
        form.addRow(lbl)
        lbl = QtWidgets.QLabel("Pick wav record", self)
        lbl.setAlignment(Qt.AlignCenter)
        txt = QtWidgets.QLineEdit()
        lbl.setAlignment(Qt.AlignCenter)
        btn = QtWidgets.QPushButton("Browse", self
        #horizontal = QtWidgets.Ho
        form.addRow(lbl,txt,btn)
        #lbl = QtWidgets.QLabel("Pick audio file:",self)
        #grid.addWidget(lbl, 3, 3)

        #lbl = QtWidgets.QTextEdit(self)
        #grid.addWidget(lbl, 0, 0)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())