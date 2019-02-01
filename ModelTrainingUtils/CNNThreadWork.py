import threading
from threading import Thread

from PyQt5 import QtCore
from PyQt5.QtCore import QThread


class CNNThreadWork(Thread):
    def __init__(self,app, CNN):
        super().__init__()
        self.app = app
        self.CNN_model = CNN
        self.isRun=False

    def stopThread(self):
        self.isRun = False
        self.CNN_model.set_running_status(self.isRun)

    def run(self):
        self.isRun = True
        print("creating dataset")
        self.CNN_model.set_running_status(self.isRun)
        if self.isRun:
            self.app.logText.emit("creating dataset")
            self.CNN_model.createDataSet()
        if self.isRun:
            self.app.logText.emit("train model")
            self.CNN_model.trainModel()
            self.app.logText.emit("Finished")
        if self.isRun:
            self.app.showMessageBox.emit('Finished')