import sys
from enum import Enum
from select import select

import keras
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from CNN import  CNN as CNN


class Feature():

    BATCH_SIZE  = 0
    TRAIN_PERC  = 1
    EPOCH_NBR   = 2
    FEATURE_NBR = 3

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Lie Detection - Admin'
        self.defaultDict = {'Batch size':10,'Train Percent': 0.01,'Epoch Number': 5,'Column Number':32}
        self.comboText = 'sgd'
        self.learn_rate = 0
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

        self.arrTxt = []
        arrLbl = []
        i=0
        for key, value in self.defaultDict.items():
            self.arrTxt.append(QtWidgets.QTextEdit(""+str(value), self))
            self.arrTxt[i].setFixedWidth(150)
            self.arrTxt[i].setFixedHeight(25)
            self.arrTxt[i].setContentsMargins(20,200,200,200)
            self.arrTxt[i].setAlignment(Qt.AlignCenter)
            arrLbl.append(QtWidgets.QLabel(key  , self))
            arrLbl[i].setFixedWidth(100)
            form.addRow(arrLbl[i],self.arrTxt[i])
            i += 1


        comboBox = QtWidgets.QComboBox(self)
        comboBox.addItem("sgd")
        comboBox.addItem("adam")
        comboBox.addItem("rmsprop")
        comboBox.setFixedWidth(150)
        comboBox.activated[str].connect(self.onActivated)
        comboBoxLbl=QtWidgets.QLabel('Optimizer')
        form.addRow(comboBoxLbl,comboBox)
        learnRateLbl=QtWidgets.QLabel('Learn Rate =  0.0  ')
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

        btnBack = QtWidgets.QPushButton("Back")
        btnStartLearnPhase = QtWidgets.QPushButton("Start")
        btnStartLearnPhase.setFixedWidth(150)
        form.addRow(btnBack,btnStartLearnPhase)
        btnStartLearnPhase.clicked.connect(lambda: self.startTrain())

        self.show()

    def onActivated(self, text):
        self.comboText = text

    def updateSlideValue(self, learnRateScale,learnRateLbl):
        self.learn_rate = learnRateScale.value()/100
        learnRateLbl.setText('Learn Rate =  '+str(learnRateScale.value()/100)+'  ')

    def startTrain(self):
        batch_size= int(self.arrTxt[Feature.BATCH_SIZE].toPlainText())
        train_perc = float(self.arrTxt[Feature.TRAIN_PERC].toPlainText())
        epoch_nbr = int(self.arrTxt[Feature.EPOCH_NBR].toPlainText())
        feature_nbr = int(self.arrTxt[Feature.FEATURE_NBR].toPlainText())
        #learn_rate =
        self.CNN = CNN(batch_size,train_perc,epoch_nbr=epoch_nbr,column_nbr=feature_nbr, optimizer=self.comboText,learn_rate=self.learn_rate )
        print("-----------------creating dataset-----------------")
        self.CNN.createDataSet()
        print("-----------------load local data-----------------")
        self.CNN.load_data()
        print("-----------------train model----------------")
        self.CNN.trainModel()

#define functionality inside class
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
