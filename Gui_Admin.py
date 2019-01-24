import sys
import ctypes
from PyQt5.QtWidgets import QApplication, QWidget
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
            self.graph_layout.addWidget(self.graph_arr[i])

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

                self.pool = ThreadPool(processes=1)
                self.pool.apply_async(task1, args=(self.CNN_model,))
            else:
                self.pool.terminate()
                self.pool.join()
                self.btnStartLearnPhase.setText("Start")
        except Exception as e:
            print(e)


def task1(*args):
    CNN_model=args[0]
    print("-----------------creating dataset-----------------")
    CNN_model.createDataSet()
    print("-----------------train model----------------")
    CNN_model.trainModel()
    print(CNN.predict(None))


# define functionality inside class
class AccuracyHistory(Callback):

    def __init__(self, graph, frame):
        self.graph_arr = graph
        frame.setVisible(True)


    def on_train_begin(self, logs={}):
        self.logs = [[], [], [], []]

    def on_epoch_end(self, batch, logs={}):
        self.logs[Graph.ACC_EPOCH].append(logs.get('acc'))
        self.logs[Graph.LOSS_EPOCH].append(logs.get('loss'))
        thread_acc = PlotLogs(self.graph_arr[Graph.ACC_EPOCH],self.logs[Graph.ACC_EPOCH])
        thread_loss = PlotLogs(self.graph_arr[Graph.LOSS_EPOCH], self.logs[Graph.LOSS_EPOCH])
        thread_acc.start()
        thread_loss.start()

    def on_batch_end(self, batch, logs=None):
        self.logs[Graph.ACC_BATCH].append(logs.get('acc'))
        self.logs[Graph.LOSS_BATCH].append(logs.get('loss'))
        thread_acc = PlotLogs(self.graph_arr[Graph.ACC_BATCH], self.logs[Graph.ACC_BATCH])
        thread_loss = PlotLogs(self.graph_arr[Graph.LOSS_BATCH], self.logs[Graph.LOSS_BATCH])
        thread_acc.start()
        thread_loss.start()

    def plot(self, data):
        try:
            self.graph_acc.plot(self.loss)

        except Exception as e: print(e)

    def clearGraph(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

class PlotLogs(QThread):
    def __init__(self, graph, data):
        super().__init__()
        self.graph = graph
        self.data = data

    def run(self):
        self.graph.plot(self.data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Gui_Admin()
    sys.exit(app.exec_())
