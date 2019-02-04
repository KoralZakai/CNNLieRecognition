
# define functionality inside class
from keras.callbacks import Callback



class Graph():
    ACC_EPOCH  = 0
    LOSS_EPOCH = 1
    ACC_BATCH  = 2
    LOSS_BATCH = 3

class AccuracyHistory(Callback):

    def __init__(self, graph, frame, log_print, graph_draw,epoch):
        super(AccuracyHistory, self).__init__()
        self.graph_arr = graph
        frame.setVisible(True)
        self.index_on_epoch = self.index_on_batch = 0
        self.index_log_on_batch = [0]
        self.index_log_on_epoch = [0]
        self.log_print = log_print
        self.draw = graph_draw
        self.epoch = epoch
        self.logs = [[0], [0], [0], [0]]

    def on_epoch_begin(self, epoch, logs=None):
        self.index_log_on_batch = [0]
        self.index_on_batch = 0
        self.logs[Graph.ACC_BATCH] = [self.logs[Graph.ACC_BATCH][-1]]
        self.logs[Graph.LOSS_BATCH] = [self.logs[Graph.LOSS_BATCH][-1]]
        self.graph_arr[Graph.ACC_BATCH].clear()
        self.graph_arr[Graph.LOSS_BATCH].clear()
        self.log_print.emit("Epoch number {} of {}".format(self.index_on_epoch, self.epoch))

    def on_epoch_end(self, batch, logs={}):
        self.index_on_epoch +=1
        self.index_log_on_epoch.append(self.index_on_epoch)
        self.logs[Graph.ACC_EPOCH].append(logs.get('acc'))
        self.logs[Graph.LOSS_EPOCH].append(logs.get('loss'))
        self.draw.emit(self.graph_arr[Graph.ACC_EPOCH], self.logs[Graph.ACC_EPOCH],self.index_log_on_epoch)
        self.draw.emit(self.graph_arr[Graph.LOSS_EPOCH], self.logs[Graph.LOSS_EPOCH],self.index_log_on_epoch)
        self.log_print.emit("Epoch training result: acc: {0:.4f} loss:{0:.4f}".format(logs.get('acc'), logs.get('loss')))
        self.log_print.emit("Epoch validation result: acc: {0:.4f} loss:{0:.4f}\n\n".format(logs.get('val_acc'), logs.get('val_loss')))

    def on_batch_end(self, batch, logs=None):
        self.index_on_batch += 1
        self.index_log_on_batch.append(self.index_on_batch)
        # calculate new accumulative average
        avg_acc = ((self.index_on_batch-1)*self.logs[Graph.ACC_BATCH][-1]+logs.get('acc'))/self.index_on_batch
        avg_loss =((self.index_on_batch-1)*self.logs[Graph.ACC_BATCH][-1]+logs.get('loss'))/self.index_on_batch
        self.logs[Graph.ACC_BATCH].append(avg_acc)
        self.logs[Graph.LOSS_BATCH].append(avg_loss)
        self.draw.emit(self.graph_arr[Graph.ACC_BATCH], self.logs[Graph.ACC_BATCH], self.index_log_on_batch)
        self.draw.emit(self.graph_arr[Graph.LOSS_BATCH], self.logs[Graph.LOSS_BATCH], self.index_log_on_batch)
        self.log_print.emit("acc: {0:.4f} loss:{1:.4f}".format(avg_acc, avg_loss))
