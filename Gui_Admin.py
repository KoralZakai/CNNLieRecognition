import ctypes
from tkinter import *

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from ModelTrainingUtils.CNN import CNN as CNN
import pyqtgraph as pg

from ModelTrainingUtils.AccuracyHistory import AccuracyHistory, Graph
from ModelTrainingUtils.CNNThreadWork import CNNThreadWork


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
                self.init_graph_by_params(epoch_nbr)
                displayGraph = AccuracyHistory(self.graph_arr,self.graph_frame,epoch_nbr)
                self.CNN_model = CNN(calbackFunc=displayGraph,
                                     batch_size=batch_size,
                                     train_perc=self.train_percent,
                                     epoch_nbr=epoch_nbr,
                                     column_nbr=feature_nbr,
                                     optimizer=self.comboText,
                                     learn_rate=learning_rate)

                self.CNNThread = CNNThreadWork(self,self.CNN_model)
                self.CNNThread.start()
            else:
                self.CNNThread.terminate()
                self.btnStartLearnPhase.setText("Start")
        except Exception as e:
            print(e)

    def init_graph_by_params(self,epoch_nbr):
        self.graph_arr[Graph.LOSS_EPOCH].setXRange(1,epoch_nbr)
        self.graph_arr[Graph.ACC_BATCH].setXRange(1,epoch_nbr)
        for lbl in [Graph.ACC_EPOCH, Graph.ACC_BATCH]:
            self.graph_arr[lbl].setLabel('bottom', 'Epoch number', units='times')
            self.graph_arr[lbl].setLabel('left', 'Accuracy', units='%')
        for lbl in [Graph.LOSS_BATCH, Graph.LOSS_EPOCH]:
            self.graph_arr[lbl].setLabel('left', 'Loss value', units='%')
            self.graph_arr[lbl].setLabel('bottom', 'Epoch number', units='times')

    def on_show_message_box(self, res):
        if res == 'Finished':
            buttonReply = QMessageBox.question(self, 'System message', "Do you want to save model?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            path = self.file_save()
            QMessageBox.information(self, "Success", "The file has been saved to:\r\n {}.h5".format(path))
            return
        QMessageBox.information(self, "Cancel", "The file was not saved")

    def file_save(self):
        path, _ = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        self.CNN_model.saveModel(path)
        return path


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Gui_Admin()
    sys.exit(app.exec_())
