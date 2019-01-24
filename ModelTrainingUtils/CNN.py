from __future__ import print_function
import keras
import tensorflow as tf
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Activation
from keras.models import Model
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from PyQt5 import QtCore
CLASSES_NBR = 2
VERBOSE = 1
HIDDEN_NBR = 1000
INPUTFILES_NBR = 535
TESTFILES_NBR = 30

class CNN():
    # constructor to initialize parameters for data and model
    # batch_size - int value number of training examples utilized in one iteration
    # train_perc - the percent of data which usig to train the model
    # epoch_nbr - number of iteration over the training data
    # learn_rate - controls how much we are adjusting the weights of our network
    # optimizer - optimizer for model could be one of ('sgd','adam','rmsprop')
    # column_nbr - number of columns in the input data minimum 32
    def __init__(self, calbackFunc, batch_size=10, train_perc=0.8, epoch_nbr=10, learn_rate=0.001, optimizer='adam', column_nbr=32):
        super().__init__()

        self.historyCallBackFunction = calbackFunc
        self._setOptimizer(optimizer, learn_rate)
        self.epoch_nbr = epoch_nbr
        self.batch_size = batch_size
        self.train_percent = train_perc
        self.dictionary = {"W": "Anger", "L": "Boredom", "E": "Disgust", "A": "Fear", "F": "Happiness", "T": "Sadness",
                           "N": "Neutral"}
        self.column_nbr = column_nbr
        self.line_nbr = 225

        self.session = tf.Session()
        K.set_session(self.session)
        self.session.run(tf.global_variables_initializer())

        self.createNewVGG16Model()


        self.default_graph = tf.get_default_graph()

        #self.default_graph.finalize()  # avoid modifications





    # initialize the optimizer of the model
    def _setOptimizer(self, optimizer, learn_rate):
        if optimizer == 'adam':
            self.opt = keras.optimizers.Adam(lr=learn_rate)
        elif optimizer == 'sgd':
            self.opt = keras.optimizers.SGD(lr=learn_rate)
        else:
            self.opt = keras.optimizers.RMSprop(lr=learn_rate)

    # init data set of csv files ta
    def createDataSet(self):
        # loading all wav files names
        winstep = 0.005
        filenames = os.listdir("db\\wav")
        self.data = np.zeros((len(filenames), 3, self.line_nbr,self.column_nbr),dtype=float)
        self.label = np.zeros((len(filenames), 1), dtype=int)
        if not os.path.exists("db\MFCC"):
            os.makedirs("db\MFCC")
        for i in range(len(filenames)):
            (rate, sig) = wav.read("db\\wav\\{0}".format(filenames[i]))
            temp = mfcc(sig, rate, winstep=winstep, numcep=self.column_nbr, nfilt=self.column_nbr)
            np.savetxt("db\\MFCC\\{0}.csv".format(filenames[i]), temp[0:self.line_nbr, :], delimiter=",")
            # run over the data and label each one
            for j in range(3):
                self.data[i][j] = temp[self.line_nbr, :]
            if self.dictionary[filenames[i][5]] == "Fear":
                self.label[i] = True
            else:
                self.label[i] = False
    # load csv files into arrays

    def load_data(self):
        # shuffle the data
        filenames = os.listdir("db\\MFCC\\")
        # init the data for model
        self.label = np.zeros((len(filenames), 1), dtype=int)
        self.data = np.zeros((len(filenames), 3, self.line_nbr, self.column_nbr), dtype=float)
        # run over the data and label each one
        for i in range(len(filenames)):
            for j in range(3):
                self.train_data[i][j] = np.loadtxt(open("db\\MFCC\\{0}".format(filenames[i]), "rb"), delimiter=",")
            if self.dictionary[filenames[i][5]] == "Fear":
                self.train_label[i] = True
            else:
                self.train_label[i] = False

    # create VGG16 model
    def createNewVGG16Model(self):
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        sess = tf.Session(config=config)
        K.set_session(sess)  # K is keras backend
        # getting convolution layers with weights
        VGG16_conv2D = VGG16(weights='imagenet', include_top=False, input_shape=(225, self.column_nbr, 3))
        # don't change training weights
        for layer in VGG16_conv2D.layers:
            layer.trainable = False
        # adding our layers to convolution layer
        last = VGG16_conv2D.output
        nextLayer = Flatten()(last)
        nextLayer = Dense(1000, kernel_regularizer=regularizers.l2(0.001))(nextLayer)
        nextLayer = Activation("relu")(nextLayer)
        nextLayer = Dense(CLASSES_NBR)(nextLayer)
        lastLayer = Activation("softmax", dtype="DT_FLOAT")(nextLayer)
        self.model = Model(VGG16_conv2D.input, lastLayer)
        # compile the model
        self.model.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        self.model.summary()

    def saveModel(self, fileName):
        self.model.save('{}.h5'.format(fileName))  # creates a HDF5 file 'my_model.h5'

    def loadModel(self, fileName):
        self.model = load_model('{}.h5'.format(fileName))

    def predict(self, input):
        input = np.zeros((1,self.line_nbr, self.column_nbr, 3), dtype=float)
        input[0] =self.data[0]
        res =self.model.predict(input, verbose=1)
        return float(res[0][0]), float(res[0][1])

    def trainModel(self):
        # normalize training data
        max = np.max(self.data)
        min = np.min(self.data)

        self.data = (self.data - min) / (max - min)
        self.label = keras.utils.to_categorical(self.label, CLASSES_NBR)
        self.data = self.data.reshape(self.data.shape[0], self.line_nbr, self.column_nbr, 3)
        # normalize validating data
        with self.default_graph.as_default():
            history = self.model.fit(self.data,
                                     self.label,
                                     batch_size=self.batch_size,
                                     epochs=self.epoch_nbr,
                                     validation_split=1-self.train_percent,
                                     shuffle=True,
                                     verbose=VERBOSE,
                                     callbacks=self.getCallBacks())
        return history

    def validateModel(self):
        scores = self.model.evaluate(self.test_data, self.test_label)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def getCallBacks(self):
        earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=.01,
                                                  patience=5,
                                                  verbose=1,
                                                  mode='auto',
                                                  baseline=None,
                                                  restore_best_weights=True)
        tensorBoard = keras.callbacks.TensorBoard(log_dir='./logs/',
                                                  histogram_freq=10,
                                                  batch_size=32,
                                                  write_graph=True,
                                                  write_grads=True,
                                                  write_images=True,
                                                  embeddings_freq=0,
                                                  embeddings_layer_names=None,
                                                  embeddings_metadata=None,
                                                  embeddings_data=None,
                                                  update_freq='epoch')
        return [tensorBoard, earlyStop, self.historyCallBackFunction]



