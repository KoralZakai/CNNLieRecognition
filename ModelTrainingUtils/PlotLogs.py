from threading import Thread

from PyQt5.QtWidgets import QApplication
from scipy.signal import savgol_filter

class PlotLogs(Thread):
    def __init__(self, graph, data,index):
        super().__init__()
        self.graph = graph
        self.data = data
        self.index = index

    def run(self):
        #self.graph.clear()
        #if(len(self.data)>=5):
        #    data = savgol_filter(self.data, 5, 3)
        #else:
        #    data = self.data
        self.graph.plot(self.index, self.data, pen='r', name='blue' )
        QApplication.processEvents()
