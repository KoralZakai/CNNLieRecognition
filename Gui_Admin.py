import sys
import ctypes
from threading import Thread
from tkinter import *
from tkinter import filedialog

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread
from PyQt5 import QtCore
from CNN import CNN as CNN
from keras.callbacks import Callback
from multiprocessing.pool import ThreadPool
import pyqtgraph as pg
import numpy as np

class Graph():
    ACC_EPOCH  = 0
    LOSS_EPOCH = 1
    ACC_BATCH  = 2
    LOSS_BATCH = 3

class Feature():
    BATCH_SIZE = 0
    LEARN_RATE = 1
    EPOCH_NBR = 2
    FEATURE_NBR = 3


class Gui_Admin(QWidget):
    showMessageBox = QtCore.pyqtSignal(str)
    def __init__(self):
        super().__init__()
        super(Gui_Admin, self).__init__()
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        # init the initial parameters of this GUI
        self.title = 'Lie Detection'
        self.width = w
        self.height = h
        self.CNN = None
        self.title = 'Lie Detection - Admin'
        self.left = 0
        self.top = 0
        self._initModelDefaultParams()
        self._initUI()
        self.showMessageBox.connect(self.on_show_message_box)

    def _initModelDefaultParams(self):
        self.defaultDict = {'Batch size': 20, 'Learning Rate': 0.01, 'Epoch Number': 1, 'Column Number': 32}
        self.comboText = 'sgd'
        self.train_percent = 0.9

    def _initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.width, self.height-60)
        # Setting up the form fields
        self.setStyleSheet("margin:0 auto")
        main_frame = QtWidgets.QFrame(self)
        main_frame.setStyleSheet("background-color:green;margin:0 auto")
        main_layout = QtWidgets.QVBoxLayout(main_frame)
        main_frame.setFixedSize(self.width, self.height-100)
        main_layout.setAlignment(Qt.AlignTop)

        form_frame = QtWidgets.QFrame(main_frame)
        form_frame.setStyleSheet("background-color:red;")
        form_layout = QtWidgets.QGridLayout(form_frame)
        form_frame.setFixedHeight(self.height/2)
        main_layout.addWidget(form_frame)

        self.graph_frame = QtWidgets.QFrame(main_frame)
        self.graph_frame.setStyleSheet("background-color:blue")
        self.graph_frame.setVisible(False)
        self.graph_layout = QtWidgets.QHBoxLayout(self.graph_frame)
        main_layout.addWidget(self.graph_frame)


        self.graph_arr = []
        for i in range(4):
            self.graph_arr.append(pg.PlotWidget())
            self.graph_arr[i].showGrid(x=True,y=True)
            self.graph_arr[i].getAxis('bottom').enableAutoSIPrefix(False)
            self.graph_arr[i].getAxis('left').enableAutoSIPrefix(False)
            self.graph_layout.addWidget(self.graph_arr[i])
            if i in [Graph.ACC_BATCH, Graph.ACC_EPOCH]:
                self.graph_arr[i].setYRange(0,1)
            else:
                self.graph_arr[i].setYRange(0, 5)

        formTitleLbl = QtWidgets.QLabel('Admin Management Settings')
        form_layout.addWidget(formTitleLbl, 1, 1)

        myFont = QtGui.QFont()
        myFont.setBold(True)
        myFont.setPixelSize(25)
        formTitleLbl.setFont(myFont)
        self.arrTxt = []
        arrLbl = []
        i = 0
        for key, value in self.defaultDict.items():
            self.arrTxt.append(QtWidgets.QTextEdit("" + str(value), self))
            self.arrTxt[i].setFixedWidth(150)
            self.arrTxt[i].setFixedHeight(25)
            self.arrTxt[i].setContentsMargins(20, 200, 200, 200)
            self.arrTxt[i].setAlignment(Qt.AlignCenter)
            arrLbl.append(QtWidgets.QLabel(key, self))
            arrLbl[i].setFixedWidth(100)
            form_layout.addWidget(arrLbl[i], i + 3, 1)
            form_layout.addWidget(self.arrTxt[i], i + 3, 2)
            i += 1
        lblComboBox, comboBox = self._initCombobox()
        form_layout.addWidget(lblComboBox, 6, 1)
        form_layout.addWidget(comboBox, 6, 2)
        train_percentLbl, train_percentScale = self._initSlider()
        form_layout.addWidget(train_percentLbl, 7, 1)
        form_layout.addWidget(train_percentScale, 7, 2)

        self.btnBack = QtWidgets.QPushButton("Back")
        self.btnStartLearnPhase = QtWidgets.QPushButton("Start")
        self.btnStartLearnPhase.setFixedWidth(150)
        form_layout.addWidget(self.btnBack, 8, 1)
        form_layout.addWidget(self.btnStartLearnPhase, 8, 2)
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
        return train_percentLbl, train_percentScale

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

    def updateSlideValue(self, learnRateScale, train_percentLbl):
        self.train_percent = learnRateScale.value() / 100
        train_percentLbl.setText('Training percent =  ' + str(learnRateScale.value()) + '%  ')

    def learnPhase(self):
        try:
            if self.btnStartLearnPhase.text() == "Start":
                self.btnStartLearnPhase.setText("Cancel")
                batch_size = int(self.arrTxt[Feature.BATCH_SIZE].toPlainText())
                learning_rate = float(self.arrTxt[Feature.LEARN_RATE].toPlainText())
                epoch_nbr = int(self.arrTxt[Feature.EPOCH_NBR].toPlainText())
                feature_nbr = int(self.arrTxt[Feature.FEATURE_NBR].toPlainText())
                displayGraph = AccuracyHistory(self.graph_arr,self.graph_frame)
                self.CNN_model = CNN(calbackFunc=displayGraph,
                                     batch_size=batch_size,
                                     train_perc=self.train_percent,
                                     epoch_nbr=epoch_nbr,
                                     column_nbr=feature_nbr,
                                     optimizer=self.comboText,
                                     learn_rate=learning_rate)

                self.CNNThread = CNNThreadWork(self,self.CNN_model)
                self.CNNThread.run()
            else:
                self.pool.terminate()
                self.pool.join()
                self.btnStartLearnPhase.setText("Start")
        except Exception as e:
            print(e)

    def on_show_message_box(self, res):
        if res=='Finished':
            buttonReply = QMessageBox.question(self, 'System message', "Do you want to save model?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            self.file_save()

    def file_save(self):
        """get a filename and save the text in the editor widget"""
        path, _ = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        self.CNN_model.saveModel(path)
    '''
    def task1(*args):
        self.app.showMessageBox.emit(thread_name,
                                     'finished',
                                     'Hello',
                                     'Thread {} sent this message.'.format(thread_name))
        CNN_model=args[0]
        print("-----------------creating dataset-----------------")
        CNN_model.createDataSet()
        print("-----------------train model----------------")
        CNN_model.trainModel()
    
    
        print(CNN.predict(None))
    '''

class CNNThreadWork(Thread):
    def __init__(self,app,CNN):
        super(CNNThreadWork).__init__()
        self.app = app
        self.CNN_model = CNN

    def run(self):

        print("-----------------creating dataset-----------------")
        self.CNN_model.createDataSet()
        print("-----------------train model----------------")
        self.CNN_model.trainModel()
        self.app.showMessageBox.emit('Finished')





# define functionality inside class
class AccuracyHistory(Callback):

    def __init__(self, graph, frame):
        self.graph_arr = graph
        frame.setVisible(True)
        self.index_on_epoch = self.index_on_batch = 0
        self.index_log_on_batch = []
        self.index_log_on_epoch = []
        for lbl in [Graph.ACC_EPOCH,Graph.ACC_BATCH]:
            self.graph_arr[lbl].setLabel('bottom', 'Epoch number', units='times')
            self.graph_arr[lbl].setLabel('left', 'Accuracy', units='%')
        for lbl in [Graph.LOSS_BATCH,Graph.LOSS_EPOCH]:
            self.graph_arr[lbl].setLabel('left', 'Loss value', units='%')
            self.graph_arr[lbl].setLabel('bottom', 'Epoch number', units='times')


    def on_train_begin(self, logs={}):
        self.logs = [[], [], [], []]


    def on_epoch_begin(self, epoch, logs=None):
        self.index_log_on_batch = []
        self.index_on_batch = 1
        self.logs[Graph.ACC_BATCH] = []
        self.logs[Graph.LOSS_BATCH] = []

    def on_epoch_end(self, batch, logs={}):
        self.index_on_epoch +=1
        self.index_log_on_epoch.append(self.index_on_epoch)
        self.logs[Graph.ACC_EPOCH].append(logs.get('acc'))
        self.logs[Graph.LOSS_EPOCH].append(logs.get('loss'))
        thread_acc = PlotLogs(self.graph_arr[Graph.ACC_EPOCH],self.logs[Graph.ACC_EPOCH],self.index_log_on_epoch)
        thread_loss = PlotLogs(self.graph_arr[Graph.LOSS_EPOCH], self.logs[Graph.LOSS_EPOCH],self.index_log_on_epoch)
        thread_acc.start()
        thread_loss.start()

    def on_batch_end(self, batch, logs=None):
        self.index_on_batch += 1
        self.index_log_on_batch.append(self.index_on_batch)
        self.logs[Graph.ACC_BATCH].append(logs.get('acc'))
        self.logs[Graph.LOSS_BATCH].append(logs.get('loss'))
        thread_acc = PlotLogs(self.graph_arr[Graph.ACC_BATCH], self.logs[Graph.ACC_BATCH], self.index_log_on_batch)
        thread_loss = PlotLogs(self.graph_arr[Graph.LOSS_BATCH], self.logs[Graph.LOSS_BATCH], self.index_log_on_batch)
        thread_acc.start()
        thread_loss.start()

    def plot(self, data):
        try:
            self.graph_acc.plot(self.loss)

        except Exception as e: print(e)

class PlotLogs(Thread):
    def __init__(self, graph, data,index):
        super().__init__()
        self.graph = graph
        self.data = data
        self.index = index

    def run(self):
        self.graph.plot(self.index, self.data, pen='r', symbolBrush=0.3, name='blue')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Gui_Admin()
    sys.exit(app.exec_())
