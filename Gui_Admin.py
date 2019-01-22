import sys
import threading
from enum import Enum
from select import select
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5 import QtCore
from CNN import CNN as CNN
from keras.callbacks import Callback
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import pyqtgraph as pg
import numpy as np

class Feature():
    BATCH_SIZE = 0
    LEARN_RATE = 1
    EPOCH_NBR = 2
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
        self.defaultDict = {'Batch size': 20, 'Learning Rate': 0.01, 'Epoch Number': 1, 'Column Number': 32}
        self.comboText = 'sgd'
        self.train_percent = 0.9

    def _initUI(self):
        x = np.arange(1000)
        y = np.random.normal(size=(3, 1000))
        plotWidget = pg.plot(title="Three plot curves")
        for i in range(3):
            plotWidget.plot(x, y[i], pen=(i, 3))  ## setting pen=(i,3) automaticaly creates three different-colored pens
            plotWidget.c
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Setting up the form fields
        self.setStyleSheet("margin:0 auto")
        main_frame = QtWidgets.QFrame(self)
        main_frame.setStyleSheet("background-color:green;margin:0 auto")
        main_layout = QtWidgets.QVBoxLayout(main_frame)

        form_frame = QtWidgets.QFrame(main_frame)
        form_frame.setStyleSheet("background-color:red;")
        form_layout = QtWidgets.QGridLayout(form_frame)
        main_layout.addWidget(form_frame)

        graph_frame = QtWidgets.QFrame(main_frame)
        graph_frame.setStyleSheet("background-color:blue")
        self.graph_layout = QtWidgets.QHBoxLayout(graph_frame)
        main_layout.addWidget(graph_frame)

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
                displayGraph = AccuracyHistory(self.graph_layout)
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
                # self.AsyncTask.stop()
                # self.AsyncTask = None
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

    def __init__(self,graph_layout):
        self.layout = graph_layout


    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.plot(self.acc)

    def plot(self, data):
        try:

            x = [1,5,9,2,3,1]
            y = [1,2,3,4,5,6]
            win2 = pg.GraphicsWindow()
            win2.resize(1000, 600)
            win2.setWindowTitle('pyqtgraph example: Plotting')
            p2 = win2.addPlot(title="Updating plot")
            curve = p2.plot(pen='y')
            # clear the previues graphs
            self.clearGraph()
            # Sound figure
            # a figure instance to plot on
            #self.figure = plt.figure()

            # this is the Canvas Widget that displays the `figure`
            # it takes the `figure` instance as a parameter to __init__
            #self.canvas = FigureCanvas(self.figure)

            # this is the Navigation widget
            # it takes the Canvas widget and a parent
            #self.toolbar = NavigationToolbar(self.canvas, self)
            #self.layout.addWidget(self.toolbar)
            #self.layout.addWidget(self.figure.canvas)

            # create an axis
            #ax = self.figure.add_subplot(111)
            # plot data
           #ax.plot(data, '*-')
            ## refresh canvas
           # self.canvas.draw()
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
