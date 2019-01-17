import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt

class App(QWidget):


    def __init__(self):
        super().__init__()
        self.title = 'Lie Detection - Admin'
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

        arrTxt = []
        arrLbl = []
        arrLblNames = ['Batch Size','Train Percintage','Epoch Number','Column Number']
        for i in range(4):
            arrTxt.append(QtWidgets.QTextEdit("", self))
            arrTxt[i].setFixedWidth(100)
            arrTxt[i].setFixedHeight(25)
            arrLbl.append(QtWidgets.QLabel(arrLblNames[i], self))
            form.addRow(arrLbl[i],arrTxt[i])
        comboBox = QtWidgets.QComboBox(self)
        comboBox.addItem("sgd")
        comboBox.addItem("adam")
        comboBox.addItem("rmsprop")
        comboBox.setFixedWidth(100)
        comboBoxLbl=QtWidgets.QLabel('Optimizer')
        form.addRow(comboBoxLbl,comboBox)
        learnRateScale = QtWidgets.QSlider(Qt.Horizontal)
        learnRateScale.setFixedWidth(100)
        form.addRow(learnRateScale)


        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())