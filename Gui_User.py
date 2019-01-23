import ctypes

from matplotlib import cm
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
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
from PyQt5.QtCore import Qt
import random

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        # init the initial parameters of this GUI
        self.title = 'Lie Detection'
        self.width = w
        self.height = h
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

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.width, self.height-60)
        #Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)

        #the first sub window
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)
        self.firstsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.firstsub_Frame)
        self.firstsub_Layout = QtWidgets.QFormLayout(self.firstsub_Frame)

        #Setting up the form fields
        self.formTitleLbl = QtWidgets.QLabel('Lie Detection')
        self.formTitleLbl.setAlignment(Qt.AlignCenter)
        self.formTitleLbl.setContentsMargins(0,0,0,20)
        myFont = QtGui.QFont()
        myFont.setBold(True)
        myFont.setPixelSize(25)
        self. formTitleLbl.setFont(myFont)
        self.firstsub_Layout.addRow(self.formTitleLbl)
        fileBrowseHBoxLayout = QtWidgets.QGridLayout()
        self.fileBrowserTxt=QtWidgets.QTextEdit("", self)
        self.fileBrowserLbl=QtWidgets.QLabel('Pick Wav File', self)
        myFont.setPixelSize(18)
        self.fileBrowserLbl.setFont(myFont)
        self.fileBrowserTxt.setFixedWidth(500)
        self.fileBrowserTxt.setFixedHeight(25)
        self.fileBrowserLbl.setFixedWidth(150)
        self.fileBrowserLbl.setFixedHeight(25)
        self.fileBrowserBtn = QtWidgets.QPushButton("", self)
        self.fileBrowserBtn.setMaximumHeight(100)
        self.fileBrowserBtn.setMaximumWidth(100)
        self.fileBrowserBtn.setFixedHeight(27)
        self.fileBrowserBtn.setFixedWidth(27)
        self.fileBrowserBtn.setStyleSheet("QPushButton {background: url(Pictures/filebrowse.png) no-repeat transparent;} ")
        self.fileBrowserBtn.clicked.connect(lambda: self.openFile(self.firstsub_Layout))
        fileBrowseHBoxLayout.addWidget(self.fileBrowserLbl,1,0)
        fileBrowseHBoxLayout.addWidget(self.fileBrowserTxt,1,1)
        fileBrowseHBoxLayout.addWidget(self.fileBrowserBtn,1,2)
        fileBrowseHBoxLayout.setAlignment(Qt.AlignCenter)
        self.firstsub_Layout.addRow(fileBrowseHBoxLayout)

        recordHBoxLayout = QtWidgets.QGridLayout()
        self.startRecordBtn = QtWidgets.QPushButton("", self)
        self.startRecordBtn.setFixedHeight(25)
        self.startRecordBtn.setFixedWidth(25)
        self.startRecordBtn.setStyleSheet("QPushButton {background: url(Pictures/microphone1.png) no-repeat transparent;} ")
        self.recordingLbl = QtWidgets.QLabel('Recording', self)
        self.recordingLbl.setContentsMargins(self.height/2,self.width/2,50,50)
        self.recordingLbl.setVisible(False)
        self.recordingLbl.setFixedWidth(100)
        self.recordingLbl.setFixedHeight(25)
        self.loadingLbl = QtWidgets.QLabel('', self)
        self.loadingLbl.setFixedWidth(200)
        self.loadingLbl.setFixedHeight(25)

        self.stopRecordBtn = QtWidgets.QPushButton("", self)
        self.stopRecordBtn.setStyleSheet("QPushButton {background: url(Pictures/microphone2.png) no-repeat transparent;} ")
        self.stopRecordBtn.setVisible(False)
        self.stopRecordBtn.setFixedWidth(25)
        self.stopRecordBtn.setFixedHeight(25)
        fileBrowseHBoxLayout.addWidget(self.startRecordBtn,1,4)
        fileBrowseHBoxLayout.addWidget(self.stopRecordBtn,1,4)
        recordHBoxLayout.setAlignment(Qt.AlignCenter)
        self.firstsub_Layout.addRow(recordHBoxLayout)

        # the between first and second  sub window
        self.betweenfirstsecondsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.betweenfirstsecondsub_Frame)
        self.betweenfirstsecondsub_Layout = QtWidgets.QFormLayout(self.betweenfirstsecondsub_Frame)
        self.betweenfirstsecondsub_Frame.setFixedWidth(self.width)
        self.betweenfirstsecondsub_Frame.setFixedHeight(30)
        self.betweenfirstsecondsub_Layout.addRow(self.recordingLbl,self.loadingLbl)
        self.betweenfirstsecondsub_Layout.setContentsMargins(self.width/2-self.recordingLbl.width(),0,0,0)
        #self.betweenfirstsecondsub_Layout.addWidget(self.loadingLbl,1,1)

        # the second sub window
        self.secondsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.secondsub_Frame)
        self.secondsub_Layout = QtWidgets.QFormLayout(self.secondsub_Frame)
        self.secondsub_Frame.setFixedWidth(self.width)
        self.secondsub_Frame.setFixedHeight(self.height/3)

        # the third sub window
        self.thirdsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.thirdsub_Frame)
        self.thirdsub_Layout = QtWidgets.QFormLayout(self.thirdsub_Frame)
        self.thirdsub_Frame.setFixedWidth(self.width)
        self.thirdsub_Frame.setFixedHeight(self.height / 3)


        self.startRecordBtn.clicked.connect(lambda: self.startRecord())
        self.stopRecordBtn.clicked.connect(lambda: self.stopRecord())

        self.show()

    #Opening file browser to import the Wav file.
    def openFile(self,form ):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "File Browser", "", "Wav Files (*.wav)", options=options)
        word = fileName.split('/')
        word = word[len(word) - 1]
        if len(word) != 0:
            if word.endswith('.wav'):
                self.fileBrowserTxt.setText(''+word)
                self.browseFilePat = fileName
                self.showWavPlot(fileName)
            else:
                QMessageBox.about(form, "Error", "Wrong File Type , Please Use Only Wav Files")



    #Recording voice using microphone
    def startRecord(self):
        self.startRec = True
        self.pyrecorded = pyaudio.PyAudio()
        self.stream = self.pyrecorded.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        print("* recording")
        self.movie = QtGui.QMovie("Pictures/loading2.gif")
        self.loadingLbl.setMovie(self.movie)
        self.movie.start()
        self.loadingLbl.setVisible(True)
        self.recordingLbl.setVisible(True)
        self.startRecordBtn.setVisible(False)
        self.stopRecordBtn.setVisible(True)
        self.frames = []
        self.recThread = threading.Thread(target = self.inputData)
        self.recThread.start()


    #getting stream of data from the microphone
    def inputData(self):
        while (self.startRec):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
        sys.exit()

    #launching the waiting gif
    def startWaitingGif(self):
        self.movieGraphWait = QtGui.QMovie("Pictures/loading2.gif")
        loadingGraphLbl = QtWidgets.QLabel('', self)
        loadingGraphLbl.setMaximumHeight(100)
        loadingGraphLbl.setMaximumWidth(100)
        loadingGraphLbl.setMovie(self.movieGraphWait)
        self.firstsub_Layout.addWidget(loadingGraphLbl)
        self.movieGraphWait.start()

    def stopWaitingGif(self):
    #Stopp recording voice using microphone
        print()

    def stopRecord(self):
        self.startRec = False
        self.loadingLbl.setVisible(False)
        self.stopRecordBtn.setVisible(False)
        self.startRecordBtn.setVisible(True)
        self.recordingLbl.setVisible(False)
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
        self.fileBrowserTxt.setText(WAVE_OUTPUT_FILENAME)


        self.showWavPlot(os.path.dirname(os.path.realpath(__file__)) + "\\Records\\" + WAVE_OUTPUT_FILENAME)

    def clearGraph(self):
        while self.secondsub_Layout.count():
            child = self.secondsub_Layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        while self.thirdsub_Layout.count():
            child = self.thirdsub_Layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def showWavPlot(self, WAVE_OUTPUT_PATH ):

        #self.startWaitingGifThread = threading.Thread(target=self.startWaitingGif())
        #self.startWaitingGifThread.start()
        #clear the previues graphs
        self.clearGraph()
        # Sound figure
        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.secondsub_Layout.addWidget(self.toolbar)
        self.secondsub_Layout.addWidget(self.canvas)

        # create an axis
        ax = self.figure.add_subplot(111)
        # plot data
        spf = wave.open(WAVE_OUTPUT_PATH, 'r')
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')
        ax.plot(signal, '*-')
        # refresh canvas
        self.canvas.draw()

        #drawing mfcc graph
        (rate, sig) = wav.read(WAVE_OUTPUT_PATH)
        mfcc_feat = mfcc(sig, rate)
        mfcc_data = np.swapaxes(mfcc_feat, 0, 1)

        # Sound figure
        # a figure instance to plot on
        self.mfccfigure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.mfcccanvas = FigureCanvas(self.mfccfigure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.mfcctoolbar = NavigationToolbar(self.mfcccanvas, self)
        self.thirdsub_Layout.addWidget(self.mfcctoolbar)
        self.thirdsub_Layout.addWidget(self.mfcccanvas)

        # create an axis
        ax = self.mfccfigure.add_subplot(111)
        word = WAVE_OUTPUT_PATH.split('/')
        if len(word) == 1:
            word = WAVE_OUTPUT_PATH.split('\\')

        word = word[len(word) - 1]
        ax.set_title('MFCC - '+word)
        #ax.plot(mfcc_feat,'*-')
        ax.imshow(mfcc_data, interpolation='nearest', origin='lower', aspect='auto')

        self.mfcccanvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())