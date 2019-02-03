import threading
from threading import Thread

from PyQt5 import QtCore
from PyQt5.QtCore import QThread


class CNNThreadWork(Thread):
    def __init__(self,app, CNN):
        super().__init__()
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
        self.CNN_model.createDataSet()
        self.app.logText.emit("loading csv file to variables...")
        self.CNN_model.load_data()
        self.app.logText.emit("statring to train model...")
        self.CNN_model.trainModel()
        self.app.logText.emit("Finished training model")
        self.app.showMessageBox.emit('Finished')
