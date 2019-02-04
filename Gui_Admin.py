import ctypes
import os
from functools import reduce
from tkinter import *
from datetime import datetime
from PyQt5.QtGui import QIcon, QColor, QTextCursor, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QDialog, QSizePolicy
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QFile, QTextStream
from PyQt5 import QtCore
from ModelTrainingUtils.CNN import CNN as CNN
import pyqtgraph as pg
import multiprocessing as mp
from ModelTrainingUtils.AccuracyHistory import AccuracyHistory, Graph
from ModelTrainingUtils.CNNThreadWork import CNNThreadWork
from scipy.signal import savgol_filter
import numpy as np

class Feature():
    BATCH_SIZE = 0
    LEARN_RATE = 1
    EPOCH_NBR = 2
    FEATURE_NBR = 3


class Gui_Admin(QWidget):
    logText = QtCore.pyqtSignal(str)
    showMessageBox = QtCore.pyqtSignal(str)
    draw_plot = QtCore.pyqtSignal(pg.PlotWidget, list, list)

    def __init__(self,parent=None):
        super(Gui_Admin, self).__init__(parent)
        self.show
        self.queue = mp.Queue()
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
        self.draw_plot.connect(self.draw_graph)
        self.logText.connect(self._show_log)
        self.CNNThread = None


    def draw_graph(self,graph, y, x):
        graph.clear()
        graph.plot(x, y, pen='r', name='blue')
        QApplication.processEvents()

    def _show_log(self,log_text):
        self.text_edit.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+" : " +log_text)

    def _initModelDefaultParams(self):
        self.defaultDict = {'Batch size': 10, 'Learning Rate': 0.0001, 'Epoch Number': 30, 'Column Number': 32}
        self.comboText = 'adam'
        self.train_percent = 0.8

    def _initUI(self):
        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)


        self.setWindowIcon(QIcon(':Pictures/logo.png'))
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.width, self.height-60)

        main_frame = QtWidgets.QFrame(self)
        main_layout = QtWidgets.QGridLayout(main_frame)
        main_frame.setFixedSize(self.width, self.height-100)
        main_frame.setObjectName("MainFrame")

        #Return to main window button
        returnBtn = QtWidgets.QPushButton("", self)
        returnBtn.setStyleSheet("QPushButton {background: url(:Pictures/backimg.png) no-repeat transparent;} ")
        returnBtn.setFixedWidth(110)
        returnBtn.setFixedHeight(110)
        returnBtn.clicked.connect(self.closeThisWindow)

        title_frame = QtWidgets.QFrame()
        title_layout = QtWidgets.QHBoxLayout(title_frame)
        title_layout.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_frame,0,1,1,1)

        formTitleLbl = QtWidgets.QLabel('Admin Management Settings')
        formTitleLbl.setObjectName("LableHeader")
        title_layout.addWidget(formTitleLbl)

        first_sub_frame = QtWidgets.QFrame(main_frame)
        first_sub_layout = QtWidgets.QVBoxLayout(first_sub_frame)
        first_sub_frame.setFixedWidth(self.width/3)
        main_layout.addWidget(first_sub_frame,1,0,1,2)

        form_frame = QtWidgets.QFrame(first_sub_frame)
        form_frame.setObjectName("FormFrame")
        form_layout = QtWidgets.QFormLayout(form_frame)
        lbl = QtWidgets.QLabel("Parameters")
        lbl.setStyleSheet("font-size:30px bold")
        lbl.setAlignment(Qt.AlignCenter)
        first_sub_layout.addWidget(lbl)
        first_sub_layout.addWidget(form_frame)



        self.graph_frame = QtWidgets.QFrame(main_frame)
        self.graph_frame.setFixedWidth(self.width*2/3)
        self.graph_layout = QtWidgets.QGridLayout(self.graph_frame)

        logo = QtWidgets.QLabel('', self)
        pixmap = QPixmap(':Pictures/logo.png')
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)
        self.graph_layout.addWidget(logo)

        main_layout.addWidget(self.graph_frame,1,1,2,2)

        graphNames = ["Accuracy Epoch",
                      "Loss Epoch",
                      "Accuracy Batch",
                      "Loss Batch"]
        self.graph_arr = []
        #self.graph_frame.setVisible(False)
        """j=0
        for i in range(4):
            self.graph_arr.append(pg.PlotWidget(title=graphNames[i]))
            self.graph_arr[i].showGrid(x=True,y=True)
            self.graph_arr[i].getAxis('bottom').enableAutoSIPrefix(False)
            self.graph_arr[i].getAxis('left').enableAutoSIPrefix(False)
            self.graph_layout.addWidget(self.graph_arr[i], j, i % 2, 1, 1)
            if i == 1:
                j +=1
            if i in [Graph.ACC_BATCH, Graph.ACC_EPOCH]:
                self.graph_arr[i].setYRange(0,1)
            else:
                self.graph_arr[i].setYRange(0, 5)"""

        myFont = QtGui.QFont()
        myFont.setBold(True)
        myFont.setPixelSize(25)
        formTitleLbl.setFont(myFont)
        self.arrTxt = []
        arrLbl = []
        i = 0
        for key, value in self.defaultDict.items():
            self.arrTxt.append(QtWidgets.QTextEdit("" + str(value), self))
            self.arrTxt[i].setFixedWidth(50)
            self.arrTxt[i].setFixedHeight(25)
            self.arrTxt[i].setAlignment(Qt.AlignCenter)
            arrLbl.append(QtWidgets.QLabel(key, self))
            form_layout.addRow(arrLbl[i], self.arrTxt[i])
            i += 1

        lblComboBox, self.comboBox = self._initCombobox()
        form_layout.addRow(lblComboBox,self.comboBox)
        train_percentLbl, self.train_percentScale = self._initSlider()
        form_layout.addRow(train_percentLbl)
        form_layout.addRow(self.train_percentScale)

        btnLayout = QtWidgets.QHBoxLayout()
        self.btnStartLearnPhase = QtWidgets.QPushButton("Start")
        self.btnStartLearnPhase.setFixedWidth(150)
        self.btnStartLearnPhase.setObjectName("Buttons")
        btnLayout.addWidget(self.btnStartLearnPhase)
        first_sub_layout.addLayout(btnLayout)
        self.btnStartLearnPhase.clicked.connect(lambda: self.learnPhase())

        myFont = QtGui.QFont()
        myFont.setPixelSize(16)
        self.text_edit = QtWidgets.QTextEdit("")
        self.text_edit.setStyleSheet("color:black")
        self.text_edit.setFont(myFont)
        self.text_edit.setReadOnly(True)
        lbl = QtWidgets.QLabel("Log:")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size:32px bold")
        first_sub_layout.addWidget(lbl)
        first_sub_layout.addWidget(self.text_edit)
        self.show()

    def _initSlider(self):
        train_percentLbl = QtWidgets.QLabel('Training percent =  {}%\t'.format(int(self.train_percent*100)))
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
        comboBox.addItem("adam")
        comboBox.addItem("sgd")
        comboBox.addItem("rmsprop")
        comboBox.setFixedWidth(150)
        comboBox.activated[str].connect(self.onActivated)
        comboBoxLbl = QtWidgets.QLabel('Optimizer')
        return comboBoxLbl, comboBox

    def onActivated(self, text):
        self.comboText = text

    def updateSlideValue(self, learnRateScale, train_percentLbl):
        self.train_percent = learnRateScale.value() / 100
        train_percentLbl.setText('Training percent =  ' + str(learnRateScale.value()) + '%')

    def learnPhase(self):
        #Batch size  =  0 < x < infinty (int)
        #Learning rate =  0 < x < 1
        #Epoch number = 0 < x < infinty (int)
        #feature number = Filter numbers =  31 < x < 226

        try:
            if self.btnStartLearnPhase.text() == "Start":
                graphNames = ["Accuracy Epoch",
                              "Loss Epoch",
                              "Accuracy Batch",
                              "Loss Batch"]
                j = 0
                pg.setConfigOption("antialias", True)
                for i in range(4):
                    # Building the graphs
                    self.graph_arr.append(pg.PlotWidget(title=graphNames[i]))
                    self.graph_arr[i].showGrid(x=True, y=True)
                    self.graph_arr[i].getAxis('bottom').enableAutoSIPrefix(False)
                    self.graph_arr[i].getAxis('left').enableAutoSIPrefix(False)
                    self.graph_arr[i].setEnabled(False)
                    self.graph_layout.addWidget(self.graph_arr[i], j, i % 2, 1, 1)
                    if i == 1:
                        j += 1
                    if i in [Graph.LOSS_EPOCH, Graph.LOSS_BATCH]:
                        self.graph_arr[i].setYRange(0, 5)

                batch_size = int(self.arrTxt[Feature.BATCH_SIZE].toPlainText())
                learning_rate = float(self.arrTxt[Feature.LEARN_RATE].toPlainText())
                epoch_nbr = int(self.arrTxt[Feature.EPOCH_NBR].toPlainText())
                feature_nbr = int(self.arrTxt[Feature.FEATURE_NBR].toPlainText())
                exceptionMsg = ""
                if batch_size <= 0:
                    exceptionMsg = exceptionMsg + "Batch size must be positive number.\n"
                if learning_rate >= 1 or learning_rate <= 0:
                    exceptionMsg = exceptionMsg + "Learn rate must be between 0 and 1.\n"
                if epoch_nbr <= 0:
                    exceptionMsg = exceptionMsg + "Epoch number must be positive number.\n"
                if feature_nbr > 225 or feature_nbr < 32:
                    exceptionMsg = exceptionMsg + "Filter number must be between 32 and 225.\n"
                if len(exceptionMsg) > 0:
                    raise Exception(exceptionMsg)
                # training model
                self.btnStartLearnPhase.setText("Cancel")
                self.init_graph_by_params()
                displayGraph = AccuracyHistory(self.graph_arr,self.graph_frame,self.logText, self.draw_plot,epoch_nbr)
                self.CNN_model = CNN(output=self.logText,
                                     calback_func =displayGraph,
                                     batch_size=batch_size,
                                     train_perc=self.train_percent,
                                     epoch_nbr=epoch_nbr,
                                     column_nbr=feature_nbr,
                                     optimizer=self.comboText,
                                     learn_rate=learning_rate)
                self.CNNThread = CNNThreadWork(self,self.CNN_model)
                self.CNNThread.daemon = True
                self.CNNThread.start()
                self.text_edit.setText("")
                self.changeDisable(True)
            else:
                self.CNNThread.stopThread()
                self.CNNThread.join()
                self.btnStartLearnPhase.setText("Start")
                self.changeDisable(False)

        except Exception as e:
            QMessageBox.information(self, "Warning", e)

    def changeDisable(self,status):
        for txt in self.arrTxt:
            txt.setDisabled(status)
        self.comboBox.setDisabled(status)
        self.train_percentScale.setDisabled(status)
        self.graph_frame.setVisible(status)

    def init_graph_by_params(self):
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
        else:
            QMessageBox.information(self, "Cancel", "The file was not saved")
        self.btnStartLearnPhase.setText("Start")
        self.graph_frame.setVisible(False)

    def file_save(self):
        path, _ = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        self.CNN_model.saveModel(path)
        return path

    def closeThisWindow(self):
        self.parent().show()
        self.parent().main_frame.setVisible(True)
        self.close()
