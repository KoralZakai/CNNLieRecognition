from PyQt5 import QtGui, QtWidgets, uic
import time
import wave
import threading
import sys
import pyaudio
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib import figure


class Functions:
    def __init__(self):
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

    def inputData(self):
        while(self.startRec):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
        sys.exit()

    def startRecord(self, dlg):
        self.startRec = True
        self.pyrecorded = pyaudio.PyAudio()
        self.stream = self.pyrecorded.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        print("* recording")
        self.movie = QtGui.QMovie("Pictures/loading2.gif")
        dlg.loadingLbl.setMovie(self.movie)
        self.movie.start()
        dlg.loadingLbl.show()
        dlg.recordingLbl.show()
        dlg.startRecordBtn.setEnabled(False)
        dlg.stopRecordBtn.show()
        self.frames = []
        self.recThread = threading.Thread(target = self.inputData)
        self.recThread.start()

    def stopRecord(self, dlg):
        self.startRec = False
        dlg.loadingLbl.hide()
        dlg.stopRecordBtn.hide()
        dlg.startRecordBtn.setEnabled(True)
        dlg.recordingLbl.hide()

        print("* done recording")
        self.stream.stop_stream()
        self.stream.close()
        self.pyrecorded.terminate()
        WAVE_OUTPUT_FILENAME = time.strftime("%Y%m%d-%H%M%S")
        WAVE_OUTPUT_FILENAME = WAVE_OUTPUT_FILENAME + ".wav"
        wf = wave.open("Records\\"+WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.pyrecorded.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def openFile(self, dlg):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "File Browser", "","Wav Files (*.wav)", options=options)
        word = fileName.split('/')
        word = word[len(word) - 1]
        if len(word) != 0:
            if word.endswith('.wav'):
                dlg.wavPathTxt.setText(word)
            else:
                QMessageBox.about(dlg, "Error", "Wrong File Type , Please Use Only Wav Files")
        Functions.showWavPlot(self, dlg, fileName)

    def showWavPlot(self, dlg , filepath):
        spf = wave.open(filepath, 'r')
        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, 'Int16')
        plt.figure(2)
        plt.title('Signal Wave...')
        plt.plot(signal)
        plt.show()
        #plt.ioff()
        plt.savefig('Wave.png')


