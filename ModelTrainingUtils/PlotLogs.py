from threading import Thread


class PlotLogs(Thread):
    def __init__(self, graph, data,index):
        super().__init__()
        self.graph = graph
        self.data = data
        self.index = index

    def run(self):
        self.graph.plot(self.index, self.data, pen='r', name='blue' )
