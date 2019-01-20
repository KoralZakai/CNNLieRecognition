from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets, uic
import sys
from Gui_User import Window
from Gui_Admin import App


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        # init the initial parameters of this GUI
        self.title = 'Lie Detection'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()



    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        #Creating main container-frame, parent it to QWindow
        self.main_CF = QtWidgets.QFrame(self)
        self.main_CF.setStyleSheet('background-color: rgba(150, 0, 0, 1);')

        #the first sub window
        self.main_CL = QtWidgets.QVBoxLayout(self.main_CF)
        asset_CGF = QtWidgets.QFrame(self.main_CF)
        self.main_CL.addWidget(asset_CGF)
        asset_CGF.setStyleSheet('background-color: rgba(0, 150, 0, 1);')
        asset_CGL = QtWidgets.QHBoxLayout(asset_CGF)
        fileBrowserBtn = QtWidgets.QPushButton("file Browse", self)
        fileBrowserBtn.clicked.connect(lambda: Window())
        asset_CGL.addWidget(fileBrowserBtn)


        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
