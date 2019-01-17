from __future__ import print_function
import keras
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Activation
from keras.models import Model
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import random as rnd
import math
import os
from keras import backend as K
from keras.models import load_model

CLASSES_NBR = 2
VERBOSE = 1
HIDDEN_NBR = 1000
INPUTFILES_NBR = 535
TESTFILES_NBR = 30


class CNN:

    # constructor to initialize parameters for data and model
    # batch_size - int value number of training examples utilized in one iteration
    # train_perc - the percent of data which usig to train the model
    # epoch_nbr - number of iteration over the training data
    # learn_rate - controls how much we are adjusting the weights of our network
    # optimizer - optimizer for model could be one of ('sgd','adam','rmsprop')
    # column_nbr - number of columns in the input data minimum 32
    def __init__(self, batch_size=10, train_perc=0.8, epoch_nbr=10, learn_rate=0.001, optimizer='adam', column_nbr=32):
        K.clear_session()
        self._setOptimizer(optimizer, learn_rate)
        self.epoch_nbr = epoch_nbr
        self.batch_size = batch_size
        self.train_percent = train_perc
        self.dictionary = {"W": "Anger", "L": "Boredom", "E": "Disgust", "A": "Fear", "F": "Happiness", "T": "Sadness",
                           "N": "Neutral"}
        self.column_nbr = column_nbr
        self.line_nbr = 225

    # initialize the optimizer of the model
    def _setOptimizer(self, optimizer, learn_rate):
        if optimizer == 'adam':
            self.opt = keras.optimizers.Adam(lr=learn_rate)
        elif optimizer == 'SGD':
            self.opt = keras.optimizers.SGD(lr=learn_rate)
        else:
            self.opt = keras.optimizers.RMSprop(lr=learn_rate)

    # init data set of csv files ta
    def createDataSet(self, winstep=0.005):
        # loading all wav files names
        filenames = os.listdir("db\\wav")
        for i in range(len(filenames)):
            (rate, sig) = wav.read("db\\wav\\{0}".format(filenames[i]))
            temp = mfcc(sig, rate, winstep=winstep, numcep=self.column_nbr, nfilt=self.column_nbr)
            np.savetxt("db\\MFCC\\{0}.csv".format(filenames[i]), temp[0:self.line_nbr, :], delimiter=",")

    # load csv files into arrays
    def load_data(self):
        # shuffle the data
        filenames = os.listdir("db\\MFCC\\")
        rnd.shuffle(filenames)
        split = math.floor(self.train_percent * len(filenames))
        train_files = filenames[0:split]
        test_files = filenames[split:]
        # init the data for model
        self.train_label = np.zeros((len(train_files), 1), dtype=int)
        self.test_label = np.zeros((len(test_files), 1), dtype=int)
        self.train_data = np.zeros((len(train_files), 3, self.line_nbr, self.column_nbr), dtype=float)
        self.test_data = np.zeros((len(test_files), 3, self.line_nbr, self.column_nbr), dtype=float)
        # run over the data and label each one
        for i in range(len(train_files)):
            for j in range(3):
                self.train_data[i][j] = np.loadtxt(open("db\\MFCC\\{0}".format(filenames[i]), "rb"), delimiter=",")
            if self.dictionary[filenames[i][5]] == "Fear":
                self.train_label[i] = True
            else:
                self.train_label[i] = False
        for i in range(len(test_files)):
            for j in range(3):
                self.test_data[i][j] = np.loadtxt(open("db\\MFCC\\{0}".format(filenames[i]), "rb"), delimiter=",")
            if self.dictionary[filenames[i][5]] == "Fear":
                self.test_label[i] = True
            else:
                self.test_label[i] = False

    # create VGG16 model
    def createNewVGG16Model(self):

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
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def saveModel(self, fileName):
        self.model.save('{}.h5'.format(fileName))  # creates a HDF5 file 'my_model.h5'

    def loadModel(self, fileName):
        self.model = load_model('{}.h5'.format(fileName))

    def trainModel(self):
        # normalize training data
        max = np.max(self.train_data)
        min = np.min(self.train_data)

        self.train_data = (self.train_data - min) / (max - min)
        self.train_label = keras.utils.to_categorical(self.train_label, CLASSES_NBR)
        self.train_data = self.train_data.reshape(self.train_data.shape[0], self.line_nbr, self.column_nbr, 3)
        # normalize validating data
        max = np.max(self.test_data)
        min = np.min(self.test_data)
        self.test_data = (self.test_data - min) / (max - min)
        self.test_label = keras.utils.to_categorical(self.test_label, CLASSES_NBR)
        self.test_data = self.test_data.reshape(self.test_data.shape[0], self.line_nbr, self.column_nbr, 3)

        # fit the model with training data
        history = self.model.fit(self.train_data,
                                 self.train_label,
                                 batch_size=self.batch_size,
                                 epochs=self.epoch_nbr,
                                 validation_data=(self.test_data, self.test_label),
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
        history = AccuracyHistory()
        return [tensorBoard, earlyStop, history]

#define functionality inside class
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))

