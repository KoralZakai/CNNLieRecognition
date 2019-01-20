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
    LEARN_RATE  = 1
    EPOCH_NBR   = 2
    FEATURE_NBR = 3

class App(QWidget):


    def __init__(self):
        super().__init__()
        self.CNN = None
        self.title = 'Lie Detection - Admin'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self._initModelDefaultParams()
        self._initUI()

    def _initModelDefaultParams(self):
        self.defaultDict = {'Batch size': 200, 'Learning Rate': 0.01,'Epoch Number': 1,'Column Number':32}
        self.comboText = 'sgd'
        self.train_percent = 0.5

    def _initUI(self):
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
        i = 0
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
        lblComboBox, comboBox = self._initCombobox()
        form.addRow(lblComboBox, comboBox)
        train_percentLbl =QtWidgets.QLabel('Training percent =  50%')
        learnRateScale = QtWidgets.QSlider(Qt.Horizontal)
        learnRateScale.setFixedWidth(150)
        learnRateScale.setMinimum(0)
        learnRateScale.setMaximum(100)
        learnRateScale.setTickInterval(1)
        learnRateScale.setValue(50)
        learnRateScale.valueChanged.connect(lambda: self.updateSlideValue(learnRateScale,train_percentLbl))
        form.addRow(train_percentLbl, learnRateScale)

        btnBack = QtWidgets.QPushButton("Back")
        btnStartLearnPhase = QtWidgets.QPushButton("Start")
        btnStartLearnPhase.setFixedWidth(150)
        form.addRow(btnBack,btnStartLearnPhase)
        btnStartLearnPhase.clicked.connect(lambda: self.startTrain())

        self.show()

    def _initCombobox(self):
        comboBox = QtWidgets.QComboBox(self)
        comboBox.addItem("sgd")
        comboBox.addItem("adam")
        comboBox.addItem("rmsprop")
        comboBox.setFixedWidth(150)
        comboBox.activated[str].connect(self.onActivated)
        comboBoxLbl = QtWidgets.QLabel('Optimizer')
        return comboBoxLbl, comboBox

    def onActivated(self, text):
        self.comboText = text

    def updateSlideValue(self, learnRateScale,train_percentLbl):
        self.train_percent = learnRateScale.value()/100
        train_percentLbl.setText('Training percent =  '+str(learnRateScale.value())+'%  ')

    def startTrain(self):
        batch_size= int(self.arrTxt[Feature.BATCH_SIZE].toPlainText())
        learning_rate = float(self.arrTxt[Feature.LEARN_RATE].toPlainText())
        epoch_nbr = int(self.arrTxt[Feature.EPOCH_NBR].toPlainText())
        feature_nbr = int(self.arrTxt[Feature.FEATURE_NBR].toPlainText())
        self.CNN = CNN(calbackFunc=AccuracyHistory,
                       batch_size=batch_size,
                       train_perc=self.train_percent,
                       epoch_nbr=epoch_nbr,
                       column_nbr=feature_nbr,
                       optimizer=self.comboText,
                       learn_rate=learning_rate)
        self.CNN.createNewVGG16Model()
        print("-----------------creating dataset-----------------")
        self.CNN.createDataSet()
        print("-----------------train model----------------")
        self.CNN.trainModel()
        print(self.CNN.predict(None))


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
