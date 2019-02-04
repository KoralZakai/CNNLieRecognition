import threading
from threading import Thread

from PyQt5 import QtCore
from PyQt5.QtCore import QThread


class CNNThreadWork(Thread):
    def __init__(self, app, CNN):
        super(CNNThreadWork, self).__init__()
        self.app = app
        self.CNN_model = CNN
        self.is_run = False

    def stopThread(self):
        self.is_run = False
        self.CNN_model.set_running_status(self.is_run)

    def run(self):
        self.is_run = True
        self.CNN_model.set_running_status(self.is_run)
        self.app.logText.emit("creating dataset...")
        if not self.is_run:
            return
        #self.CNN_model.createDataSet()
        self.app.logText.emit("loading csv file to variables...")
        if not self.is_run:
            return
        self.CNN_model.load_data()
        self.app.logText.emit("starting to train model...")
        if not self.is_run:
            return
        self.CNN_model.trainModel()
        self.app.logText.emit("Finished training model")
        if not self.is_run:
            return
        self.app.showMessageBox.emit('Finished')
