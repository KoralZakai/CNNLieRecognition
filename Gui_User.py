import ctypes
from multiprocessing.pool import ThreadPool

from matplotlib import cm
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
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
import pyqtgraph
class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Lie Detection'
        self.width = w
        self.height = h
        self.startRec = True
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.NUMCEP = 13
        self.RATE = 44100
        self.frames = None
        self.pyrecorded = None
        self.stream = None
        self.recThread = None
        self.movie = None
        self.figureSoundWav = None
        self.figureMFCC = None
        self.WAVE_OUTPUT_FILENAME = None
        self.WAVE_OUTPUT_FILEPATH = None

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
        #form title init
        self.formTitleLbl = QtWidgets.QLabel('Lie Detection')
        self.formTitleLbl.setAlignment(Qt.AlignCenter)
        self.formTitleLbl.setContentsMargins(0,0,0,20)
        myFont = QtGui.QFont()
        myFont.setBold(True)
        myFont.setPixelSize(25)
        self. formTitleLbl.setFont(myFont)
        self.firstsub_Layout.addRow(self.formTitleLbl)
        #init the browse file fields - lable , textfield, file browse button , start/stop record buttons
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

        # The second sub window - loading gif window
        self.secondsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.secondsub_Frame)
        self.secondsub_Layout = QtWidgets.QFormLayout(self.secondsub_Frame)
        self.secondsub_Frame.setFixedWidth(self.width)
        self.secondsub_Frame.setFixedHeight(30)
        self.secondsub_Layout.addRow(self.recordingLbl,self.loadingLbl)
        self.secondsub_Frame.setContentsMargins(self.width/2-self.recordingLbl.width(),0,0,0)

        #Settings Layout
        self.settings_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.settings_Frame)
        self.settings_Layout = QtWidgets.QGridLayout(self.settings_Frame)
        self.settings_Frame.setFixedWidth(self.width)
        self.settings_Frame.setFixedHeight(35)

        # the third sub window
        self.thirdsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.thirdsub_Frame)
        self.thirdsub_Layout = QtWidgets.QGridLayout(self.thirdsub_Frame)
        self.thirdsub_Frame.setFixedWidth(self.width-25)
        self.thirdsub_Frame.setFixedHeight(self.height/1.8)

        # the 4rth sub window
        self.fourthsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.fourthsub_Frame)
        self.fourthsub_Layout = QtWidgets.QFormLayout(self.fourthsub_Frame)
        self.fourthsub_Frame.setFixedWidth(self.width)
        self.fourthsub_Frame.setFixedHeight(self.height / 1.8)

        # assign functions to buttons
        self.startRecordBtn.clicked.connect(lambda: self.startRecord())
        self.stopRecordBtn.clicked.connect(lambda: self.stopRecord())

        self.comboBoxCoef = QtWidgets.QComboBox(self)
        for i in range(12,226):
            self.comboBoxCoef.addItem(str(i))

        self.comboBoxCoef.setFixedWidth(150)
        self.comboBoxCoef.activated[str].connect(self.onActivated)

        self.comboBoxCoefLbl = QtWidgets.QLabel('Coefficients')
        self.comboBoxCoefLbl.setFixedWidth(75)
        self.comboBoxCoefLbl.setFixedHeight(25)
        self.comboBoxCoef.setFixedWidth(50)
        self.comboBoxCoef.setFixedHeight(25)
        self.settings_Layout.addWidget(self.comboBoxCoefLbl,1,1)
        self.settings_Layout.addWidget(self.comboBoxCoef,1,2)
        self.settings_Frame.setContentsMargins(self.width,0,0,0)
        self.settings_Frame.setVisible(False)
        self.processGraphsBtn = QtWidgets.QPushButton("Process", self)
        self.processGraphsBtn.setFixedWidth(75)
        self.processGraphsBtn.setFixedHeight(25)
        self.processGraphsBtn.clicked.connect(lambda: self.dataProcessing())
        self.settings_Layout.setAlignment(Qt.AlignCenter)
        self.settings_Layout.addWidget(self.processGraphsBtn,1,3)



        #show the window
        self.show()

    def onActivated(self, text):
        self.NUMCEP = int(text)

    def initSettings(self):
        self.clearGraph(3)
        self.clearGraph(4)
        self.settings_Frame.setVisible(False)
        self.NUMCEP = 12

    #Opening file browser to import the Wav file.
    def openFile(self,form ):
        self.initSettings()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "File Browser", "", "Wav Files (*.wav)", options=options)
        word = fileName.split('/')
        word = word[len(word) - 1]
        if len(word) != 0:
            if word.endswith('.wav'):
                self.fileBrowserTxt.setText(''+word)
                self.WAVE_OUTPUT_FILENAME = word
                self.WAVE_OUTPUT_FILEPATH = fileName
                self.settings_Frame.setVisible(True)
            else:
                QMessageBox.about(form, "Error", "Wrong File Type , Please Use Only Wav Files")

    #Recording voice using microphone
    def startRecord(self):
        self.initSettings()
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
        self.WAVE_OUTPUT_FILENAME = time.strftime("%Y%m%d-%H%M%S")
        self.WAVE_OUTPUT_FILENAME = self.WAVE_OUTPUT_FILENAME + ".wav"
        path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(path+"\\Records"):
            os.mkdir("Records")
        wf = wave.open("Records\\"+self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.pyrecorded.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.fileBrowserTxt.setText(self.WAVE_OUTPUT_FILENAME)
        self.WAVE_OUTPUT_FILEPATH=os.path.dirname(os.path.realpath(__file__)) + "\\Records\\" + self.WAVE_OUTPUT_FILENAME
        self.settings_Frame.setVisible(True)

    def dataProcessing(self):

        #getting coefficients number
        self.showSoundWav()
        self.showMfcc()
       #self.settings_Frame.setVisible(False)


    #clearing all the layouts fields
    def clearGraph(self,layoutnum):
        if layoutnum == 3:
            while self.thirdsub_Layout.count():
                child = self.thirdsub_Layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        if layoutnum == 4:
            while self.fourthsub_Layout.count():
                child = self.fourthsub_Layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()


    #plotting sound wav
    def showSoundWav(self ):

        #clear the graph
        self.clearGraph(3)
        # Sound figure
        # plot data
        spf = wave.open(self.WAVE_OUTPUT_FILEPATH, 'r')
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')
        # a figure instance to plot on
        self.figureSoundWav = pyqtgraph.PlotWidget()
        self.thirdsub_Layout.addWidget(self.figureSoundWav,2,1)
        self.figureSoundWav.setYRange(-35000,35000)
        self.figureSoundWav.setTitle('Wav - '+self.WAVE_OUTPUT_FILENAME)
        self.figureSoundWav.setLabel('left','Amplitude (db)')
        self.figureSoundWav.setLabel('bottom', 'Frame')
        self.figureSoundWav.plot(signal)

    def showMfcc(self):
        # clear the graph
        self.clearGraph(4)
        #mfcc graph
        (rate, sig) = wav.read(self.WAVE_OUTPUT_FILEPATH)
        mfcc_feat = mfcc(sig, rate,winstep=0.0025,numcep=self.NUMCEP,nfilt=self.NUMCEP)

        # Sound figure
        # a figure instance to plot on
        self.mfccfigure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.mfcccanvas = FigureCanvas(self.mfccfigure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.thirdsub_Layout.addWidget(self.mfcccanvas,2,2)

        # create an axis
        ax = self.mfccfigure.add_subplot(111)
        ax.set_ylabel('MFCC values')
        ax.set_xlabel('MFC Coefficients')


        ax.set_title('MFCC - '+self.WAVE_OUTPUT_FILENAME)

        ax.imshow(mfcc_feat, interpolation='nearest', origin='lower', aspect='auto')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())