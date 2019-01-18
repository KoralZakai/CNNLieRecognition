import sys
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

class App(QWidget):


    def __init__(self):
        super().__init__()
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
        self.initUI()



    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
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
        resultsHBoxLayout = QtWidgets.QHBoxLayout()
        form.addRow(resultsHBoxLayout)
        startRecordBtn.clicked.connect(lambda: self.startRecord(loadingLbl,recordingLbl,startRecordBtn,stopRecordBtn))
        stopRecordBtn.clicked.connect(lambda: self.stopRecord(loadingLbl,recordingLbl,startRecordBtn,stopRecordBtn,fileBrowserTxt,resultsHBoxLayout))




        self.show()


    def openFile(self, form, fileBrowserTxt):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "File Browser", "", "Wav Files (*.wav)", options=options)
        word = fileName.split('/')
        word = word[len(word) - 1]
        if len(word) != 0:
            if word.endswith('.wav'):
                fileBrowserTxt.setText(''+word)
                #self.showWavPlot( form, fileName
            else:
                QMessageBox.about(form, "Error", "Wrong File Type , Please Use Only Wav Files")

    def showWavPlot(self, form , filepath):
        spf = wave.open(filepath, 'r')
        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')
        plt.figure(1)
        plt.title('Signal Wave')
        plt.plot(signal)

        #plt.ioff()
        plt.savefig('Wave.png')

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

    def inputData(self):
        while (self.startRec):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
        sys.exit()

    def stopRecord(self,loadingLbl,recordingLbl,startRecordBtn,stopRecordBtn,fileBrowserTxt,resultsHBoxLayout):
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
        self.showWavPlot(os.path.dirname(os.path.realpath(__file__))+"\\Records\\"+WAVE_OUTPUT_FILENAME,resultsHBoxLayout)

    def showWavPlot(self, filepath, resultsHBoxLayout):
        spf = wave.open(filepath, 'r')
        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')
        plt.figure(1)
        plt.title('Signal Wave')
        plt.plot(signal)
        #resultsHBoxLayout.addWidget(plt.plot)
        #plt.ioff()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())