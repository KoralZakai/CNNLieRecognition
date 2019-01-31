
# define functionality inside class
from keras.callbacks import Callback

from ModelTrainingUtils.PlotLogs import PlotLogs


class Graph():
    ACC_EPOCH  = 0
    LOSS_EPOCH = 1
    ACC_BATCH  = 2
    LOSS_BATCH = 3

class AccuracyHistory(Callback):

    def __init__(self, graph, frame,logPrint):
        self.graph_arr = graph
        frame.setVisible(True)
        self.index_on_epoch = self.index_on_batch = 0
        self.index_log_on_batch = []
        self.index_log_on_epoch = []
        self.log_print = logPrint

    def on_train_begin(self, logs={}):
        self.logs = [[], [], [], []]


    def on_epoch_begin(self, epoch, logs=None):
        self.index_log_on_batch = []
        self.index_on_batch = 1
        self.logs[Graph.ACC_BATCH] = []
        self.logs[Graph.LOSS_BATCH] = []
        self.graph_arr[Graph.ACC_BATCH].clear()
        self.graph_arr[Graph.LOSS_BATCH].clear()

    def on_epoch_end(self, batch, logs={}):
        self.index_on_epoch +=1
        self.index_log_on_epoch.append(self.index_on_epoch)
        self.logs[Graph.ACC_EPOCH].append(logs.get('acc'))
        self.logs[Graph.LOSS_EPOCH].append(logs.get('loss'))
        thread_acc = PlotLogs(self.graph_arr[Graph.ACC_EPOCH],self.logs[Graph.ACC_EPOCH],self.index_log_on_epoch)
        thread_loss = PlotLogs(self.graph_arr[Graph.LOSS_EPOCH], self.logs[Graph.LOSS_EPOCH],self.index_log_on_epoch)
        thread_acc.start()
        thread_loss.start()

    def on_batch_end(self, batch, logs=None):
        self.index_on_batch += 1
        self.index_log_on_batch.append(self.index_on_batch)
        self.logs[Graph.ACC_BATCH].append(logs.get('acc'))
        self.logs[Graph.LOSS_BATCH].append(logs.get('loss'))
        thread_acc = PlotLogs(self.graph_arr[Graph.ACC_BATCH], self.logs[Graph.ACC_BATCH], self.index_log_on_batch)
        thread_loss = PlotLogs(self.graph_arr[Graph.LOSS_BATCH], self.logs[Graph.LOSS_BATCH], self.index_log_on_batch)
        self.log_print.emit("acc: {} loss:{}".format(logs.get('acc'),logs.get('loss')))
        thread_acc.start()
        thread_loss.start()