from PyQt5.QtCore import QThread


class CNNThreadWork(QThread):
    def __init__(self,app,CNN):
        super().__init__()
        self.app = app
        self.CNN_model = CNN

    def run(self):
        #self.CNN_model.createNewVGG16Model()
        print("-----------------creating dataset-----------------")
        self.CNN_model.createDataSet()
        print("-----------------train model----------------")
        self.CNN_model.trainModel()
        self.app.showMessageBox.emit('Finished')