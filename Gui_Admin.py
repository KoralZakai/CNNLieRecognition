import sys
import threading
from enum import Enum
from select import select

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from CNN import  CNN as CNN
from keras.callbacks import Callback
from multiprocessing.pool import ThreadPool

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
        self.train_percent = 0.9

    def _initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        #Setting up the form fields
        form = QtWidgets.QFormLayout()
        self.setLayout(form)
        formTitleLbl = QtWidgets.QLabel('Admin Management Settings')
        formTitleLbl.setContentsMargins(Qt.AlignCenter, 0, 0, 50)
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
        train_percentLbl,train_percentScale = self._initSlider()
        form.addRow(train_percentLbl, train_percentScale)

        self.btnBack = QtWidgets.QPushButton("Back")
        self.btnStartLearnPhase = QtWidgets.QPushButton("Start")
        self.btnStartLearnPhase.setFixedWidth(150)
        form.addRow(self.btnBack,self.btnStartLearnPhase)
        self.btnStartLearnPhase.clicked.connect(lambda: self.learnPhase())

        self.show()

    def _initSlider(self):
        train_percentLbl = QtWidgets.QLabel('Training percent =  50%')
        train_percentScale = QtWidgets.QSlider(Qt.Horizontal)
        train_percentScale.setFixedWidth(150)
        train_percentScale.setMinimum(0)
        train_percentScale.setMaximum(100)
        train_percentScale.setTickInterval(1)
        train_percentScale.setValue(80)
        train_percentScale.valueChanged.connect(lambda: self.updateSlideValue(train_percentScale, train_percentLbl))
        return train_percentLbl,train_percentScale

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



    def learnPhase(self):
        if self.btnStartLearnPhase.text()=="Start":
            self.btnStartLearnPhase.setText("Cancel")
            batch_size = int(self.arrTxt[Feature.BATCH_SIZE].toPlainText())
            learning_rate = float(self.arrTxt[Feature.LEARN_RATE].toPlainText())
            epoch_nbr = int(self.arrTxt[Feature.EPOCH_NBR].toPlainText())
            feature_nbr = int(self.arrTxt[Feature.FEATURE_NBR].toPlainText())
            self.CNN_model = CNN(calbackFunc=AccuracyHistory,
                           batch_size=batch_size,
                           train_perc=self.train_percent,
                           epoch_nbr=epoch_nbr,
                           column_nbr=feature_nbr,
                           optimizer=self.comboText,
                           learn_rate=learning_rate)

            self.pool = ThreadPool(processes=1)
            self.pool.apply_async(task1)
        else:
            #self.AsyncTask.stop()
            #self.AsyncTask = None
            self.pool.terminate()
            self.pool.join()
            self.btnStartLearnPhase.setText("Start")

def task1(self):
    print("-----------------creating dataset-----------------")
    self.CNN_model.createDataSet()
    print("-----------------train model----------------")
    self.CNN_model.trainModel()
    print(self.CNN.predict(None))


    class AsyncTrainModel(QtCore.QThread):
        def __init__(self,model):
            super().__init__()
            self.model = model

        def run(self):
            print("-----------------creating dataset-----------------")
            self.model.createDataSet()
            print("-----------------train model----------------")
            self.model.trainModel()
            print(self.CNN.predict(None))

        def stop(self):
            self.wait()

#define functionality inside class
class AccuracyHistory(Callback):
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
