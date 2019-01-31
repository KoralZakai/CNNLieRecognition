from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets, uic
import sys
from PyQt5.QtCore import Qt
from ModelTrainingUtils.CNN import *
import ctypes
from Gui_Admin import Gui_Admin
from Gui_User import Gui_User


class Main_Window(QWidget):
    def __init__(self, parent=None):
        super(Main_Window, self).__init__(parent)
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Lie Detection'
        self.width = w
        self.height = h
        self.initUI()



    def initUI(self):
        #self.setStyleSheet(open('StyleSheet.css').read())
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(os.getcwd()+'\pictures\logo.png'))
        self.setGeometry(0, 0, self.width, self.height-60)
        #Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        self.main_frame.setFixedWidth(self.width)
        self.main_frame.setFixedHeight(self.height)

        #the first sub window
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)
        self.firstsub_Frame = QtWidgets.QFrame(self.main_frame)
        #self.firstsub_Frame.setObjectName("FormFrame")
        self.firstsub_Frame.setFixedWidth(self.width)
        self.firstsub_Frame.setFixedHeight(400)
        self.main_layout.addWidget(self.firstsub_Frame)
        self.firstsub_Layout = QtWidgets.QHBoxLayout(self.firstsub_Frame)
        self.firstsub_Layout.setAlignment(Qt.AlignCenter)


        # The second sub window
        self.secondsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.secondsub_Frame)
        self.secondsub_Layout = QtWidgets.QHBoxLayout(self.secondsub_Frame)
        self.secondsub_Frame.setFixedWidth(self.width)
        self.secondsub_Layout.setAlignment(Qt.AlignTop|Qt.AlignCenter)
        #self.secondsub_Frame.setObjectName("FormFrame")

        #Setting up the fields

        logo = QtWidgets.QLabel('',self)
        pixmap = QPixmap(os.getcwd()+'\Pictures\logo.png')
        logo.setPixmap(pixmap)
        self.firstsub_Layout.addWidget(logo)
        logo.setAlignment(Qt.AlignCenter)

        # Admin button
        adminBtn = QtWidgets.QPushButton("Admin Console", self)
        adminBtn.setObjectName("Buttons")
        adminBtn.setFixedWidth(300)
        adminBtn.setFixedHeight(300)
        adminBtn.clicked.connect(self.openAdminGui)

        # User button
        userBtn = QtWidgets.QPushButton("User Console", self)
        userBtn.setObjectName("Buttons")
        userBtn.setFixedWidth(300)
        userBtn.setFixedHeight(300)
        userBtn.clicked.connect(self.openUserGui)

        self.secondsub_Layout.addWidget(adminBtn)
        self.secondsub_Layout.addWidget(userBtn)


        #show the window
        self.show()

    def openAdminGui(self):
        adminGui = Gui_Admin()
        adminGui.show()


    def openUserGui(self):
        userGui = Gui_User()
        userGui.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main_Window()
    sys.exit(app.exec_())