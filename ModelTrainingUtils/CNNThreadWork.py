import threading
from threading import Thread

from PyQt5.QtCore import QThread


class CNNThreadWork(Thread):
    def __init__(self,app, CNN):
        super().__init__()
        # The shutdown_flag is a threading.Event object that
        self.app = app
        self.CNN_model = CNN

    def run(self):
        #self.CNN_model.createNewVGG16Model()
        print("-----------------creating dataset-----------------")
        self.CNN_model.createDataSet()
        print("-----------------train model----------------")
        self.CNN_model.trainModel()
        self.app.showMessageBox.emit('Finished')

    def __exit__(self, exc_type, exc_val, exc_tb):
        exit()