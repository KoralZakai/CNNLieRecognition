from PyQt5 import QtGui, QtWidgets, uic
import pyaudio
import time
import wave
import threading
import sys

from PyQt5.QtWidgets import QFileDialog


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
        fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.wav)", options=options)
        if fileName:
                dlg.wavPathTxt.hide()

