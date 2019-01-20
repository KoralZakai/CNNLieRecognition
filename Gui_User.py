import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets, uic
import time
import wave
import threading
import sys
import pyaudio
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import matplotlib.pyplot as plt
import numpy as np
import os

import random

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        # init the initial parameters of this GUI
        self.title = 'Lie Detection'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.startRec = True
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.frames = None
        self.pyrecorded = None
        self.stream = None
        self.recThread = None
        self.movie = None
        self.browseFilePat = None
        self.initUI()




    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]

        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        # ax.hold(False) # deprecated, see above

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        #Creating main container-frame, parent it to QWindow
        self.main_CF = QtWidgets.QFrame(self)
        self.main_CF.setStyleSheet('background-color: rgba(150, 0, 0, 1);')

        #the first sub window
        self.main_CL = QtWidgets.QVBoxLayout(self.main_CF)
        asset_CGF = QtWidgets.QFrame(self.main_CF)
        self.main_CL.addWidget(asset_CGF)
        asset_CGF.setStyleSheet('background-color: rgba(0, 150, 0, 1);')
        asset_CGL = QtWidgets.QHBoxLayout(asset_CGF)


        #Setting up the form fields
        form = QtWidgets.QFormLayout()
        self.setLayout(form)
        formTitleLbl = QtWidgets.QLabel('Lie Detection')
        formTitleLbl.setContentsMargins(self.width/3,0,0,50)
        myFont = QtGui.QFont()
        myFont.setBold(True)
        myFont.setPixelSize(25)
        formTitleLbl.setFont(myFont)
        form.addRow(formTitleLbl)
        fileBrowseHBoxLayout = QtWidgets.QHBoxLayout()
        fileBrowserTxt=QtWidgets.QTextEdit("", self)
        fileBrowserLbl=QtWidgets.QLabel('Pick Wav File', self)
        fileBrowserTxt.setFixedWidth(500)
        fileBrowserTxt.setFixedHeight(25)
        fileBrowserLbl.setFixedWidth(150)
        fileBrowserLbl.setFixedHeight(25)
        fileBrowserBtn = QtWidgets.QPushButton("file Browse", self)
        fileBrowserBtn.clicked.connect(lambda: self.openFile(form, fileBrowserTxt))
        fileBrowseHBoxLayout.addWidget(fileBrowserLbl)
        fileBrowseHBoxLayout.addWidget(fileBrowserTxt)
        fileBrowseHBoxLayout.addWidget(fileBrowserBtn)
        form.addRow(fileBrowseHBoxLayout)
        recordHBoxLayout = QtWidgets.QHBoxLayout()
        startRecordBtn = QtWidgets.QPushButton("Start Record", self)
        startRecordBtn.setFixedWidth(75)
        startRecordBtn.setFixedHeight(25)
        recordingLbl = QtWidgets.QLabel('Recording', self)
        recordingLbl.hide()
        recordingLbl.setFixedWidth(100)
        recordingLbl.setFixedHeight(25)
        loadingLbl = QtWidgets.QLabel('', self)
        loadingLbl.setFixedWidth(200)
        loadingLbl.setFixedHeight(25)
        stopRecordBtn = QtWidgets.QPushButton("Stop Record", self)
        stopRecordBtn.hide()
        stopRecordBtn.setFixedWidth(75)
        stopRecordBtn.setFixedHeight(25)
        recordHBoxLayout.addWidget(startRecordBtn)
        recordHBoxLayout.addWidget(recordingLbl)
        recordHBoxLayout.addWidget(loadingLbl)
        recordHBoxLayout.addWidget(stopRecordBtn)
        form.addRow(recordHBoxLayout)

        #Sound figure
        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        resultsHBoxLayout = QVBoxLayout()
        resultsHBoxLayout.addWidget(self.toolbar)
        resultsHBoxLayout.addWidget(self.canvas)

        form.addRow(resultsHBoxLayout)

        startRecordBtn.clicked.connect(
        lambda: self.startRecord(loadingLbl, recordingLbl, startRecordBtn, stopRecordBtn))
        stopRecordBtn.clicked.connect(
        lambda: self.stopRecord(loadingLbl, recordingLbl, startRecordBtn, stopRecordBtn, fileBrowserTxt))

        self.show()

    #Opening file browser to import the Wav file.
    def openFile(self, form, fileBrowserTxt):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "File Browser", "", "Wav Files (*.wav)", options=options)
        word = fileName.split('/')
        word = word[len(word) - 1]
        if len(word) != 0:
            if word.endswith('.wav'):
                fileBrowserTxt.setText(''+word)
                self.browseFilePat = fileName
                self.showWavPlot(fileName)
            else:
                QMessageBox.about(form, "Error", "Wrong File Type , Please Use Only Wav Files")



    #Recording voice using microphone
    def startRecord(self, loadingLbl,recordingLbl,startRecordBtn,stopRecordBtn):
        self.startRec = True
        self.pyrecorded = pyaudio.PyAudio()
        self.stream = self.pyrecorded.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        print("* recording")
        self.movie = QtGui.QMovie("Pictures/loading2.gif")
        loadingLbl.setMovie(self.movie)
        self.movie.start()
        loadingLbl.show()
        recordingLbl.show()
        startRecordBtn.setEnabled(False)
        stopRecordBtn.show()
        self.frames = []
        self.recThread = threading.Thread(target = self.inputData)
        self.recThread.start()
    #getting stream of data from the microphone
    def inputData(self):
        while (self.startRec):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
        sys.exit()

    #Stopp recording voice using microphone
    def stopRecord(self,loadingLbl,recordingLbl,startRecordBtn,stopRecordBtn,fileBrowserTxt):
        self.startRec = False
        loadingLbl.hide()
        stopRecordBtn.hide()
        startRecordBtn.setEnabled(True)
        recordingLbl.hide()
        print("* done recording")
        self.stream.stop_stream()
        self.stream.close()
        self.pyrecorded.terminate()
        WAVE_OUTPUT_FILENAME = time.strftime("%Y%m%d-%H%M%S")
        WAVE_OUTPUT_FILENAME = WAVE_OUTPUT_FILENAME + ".wav"
        path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(path+"\\Records"):
            os.mkdir("Records")
        wf = wave.open("Records\\"+WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.pyrecorded.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        fileBrowserTxt.setText(WAVE_OUTPUT_FILENAME)

        self.showWavPlot(os.path.dirname(os.path.realpath(__file__)) + "\\Records\\" + WAVE_OUTPUT_FILENAME)

    def showWavPlot(self, WAVE_OUTPUT_FILENAME ):
        # create an axis
        ax = self.figure.add_subplot(111)
        # plot data
        spf = wave.open(WAVE_OUTPUT_FILENAME, 'r')
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')
        ax.plot(signal, '*-')
        # refresh canvas
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())