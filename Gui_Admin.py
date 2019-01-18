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
        #Setting up the form fields
        form = QtWidgets.QFormLayout()
        self.setLayout(form)
        formTitleLbl = QtWidgets.QLabel('Admin Management Settings')
        formTitleLbl.setContentsMargins(Qt.AlignCenter,0,0,50)
        myFont = QtGui.QFont()
        myFont.setBold(True)
        myFont.setPixelSize(25)
        formTitleLbl.setFont(myFont)
        form.addRow(formTitleLbl)

        arrTxt = []
        arrLbl = []

        arrLblNames = ['Batch Size','Train Percintage','Epoch Number','Column Number']
        for i in range(4):
            arrTxt.append(QtWidgets.QTextEdit("", self))
            arrTxt[i].setFixedWidth(150)
            arrTxt[i].setFixedHeight(25)
            arrTxt[i].setContentsMargins(20,200,200,200)
            arrTxt[i].setAlignment(Qt.AlignCenter)
            arrLbl.append(QtWidgets.QLabel(arrLblNames[i], self))
            arrLbl[i].setFixedWidth(100)
            form.addRow(arrLbl[i],arrTxt[i])


        comboBox = QtWidgets.QComboBox(self)
        comboBox.addItem("sgd")
        comboBox.addItem("adam")
        comboBox.addItem("rmsprop")
        comboBox.setFixedWidth(150)
        comboBoxLbl=QtWidgets.QLabel('Optimizer')
        form.addRow(comboBoxLbl,comboBox)
        learnRateLbl=QtWidgets.QLabel('Learn Rate =  0.0  ')
        #the values are between 0-100 ( 0 - 1 , but the slide is working only with int , so the value is multiplied with 100,
        learnRateScale = QtWidgets.QSlider(Qt.Horizontal)
        learnRateScale.setFixedWidth(150)
        learnRateScale.setMinimum(0)
        learnRateScale.setMaximum(100)
        learnRateScale.setTickInterval(1)
        learnRateScale.setValue(0)
        learnRateScale.valueChanged.connect(lambda: self.updateSlideValue(learnRateScale,learnRateLbl))
        form.addRow(learnRateLbl,learnRateScale)

        self.show()

    def updateSlideValue(self, learnRateScale,learnRateLbl):
        learnRateLbl.setText('Learn Rate =  '+str(learnRateScale.value()/100)+'  ')
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())