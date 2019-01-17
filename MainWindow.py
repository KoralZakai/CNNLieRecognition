import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets, uic

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
        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)
        btn = QtWidgets.QPushButton("1 1",self)
        grid.addWidget(btn, 0, 0)
        btn = QtWidgets.QPushButton("1 2",self)
        grid.addWidget(btn,1,2)
        btn = QtWidgets.QPushButton("1 3", self)
        grid.addWidget(btn, 1, 3)
        btn = QtWidgets.QPushButton("3 2", self)
        grid.addWidget(btn, 3, 2)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())