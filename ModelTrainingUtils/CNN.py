import math
import keras
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.models import Model
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from datetime import datetime
import random as rnd
CLASSES_NBR = 2
HIDDEN_NBR = 1000


class CNN():
    """
    This class give the ability to create CNN model with different parameters.
    It also have the ability to store and load model.
    """
    def __init__(self, output=None, model=None, calback_func=None, batch_size=10, train_perc=0.8, epoch_nbr=10,
                 learn_rate=0.001, optimizer='adam', column_nbr=32, name=None):
        """
        constructor to initialize parameters for data and model
        :param output:
        :param model:
        :param calback_func:
        :param batch_size: int value number of training examples utilized in one iteration
        :param train_perc: the percent of data which usig to train the model
        :param epoch_nbr: number of iteration over the training data
        :param learn_rate:  controls how much we are adjusting the weights of our network
        :param optimizer:  optimizer for model could be one of ('sgd','adam','rmsprop')
        :param column_nbr: number of columns in the input data minimum 32
        :param name:
        """
        super(CNN, self).__init__()
        if name is None:
            self.name = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        else:
            self.name = name
        self.isRun = False
        self.AccuracyCallback = calback_func
        self.output = output
        self.line_nbr = 225
        # check if file with model parameters was passed
        if model != None:
            self.loadModel(model)
            return
        self._setOptimizer(optimizer, learn_rate)
        self.epoch_nbr = epoch_nbr
        self.batch_size = batch_size
        self.train_percent = train_perc
        self.dictionary = {"W": "Anger", "L": "Boredom", "E": "Disgust", "A": "Fear", "F": "Happiness", "T": "Sadness",
                           "N": "Neutral"}
        self.column_nbr = column_nbr
        self.session = tf.Session()
        self.createNewVGG16Model()
        self.default_graph = tf.get_default_graph()

    def set_running_status(self, isRun):
        """
        function to control learning process
        :param isRun: change status of thread
        :return:
        """
        self.isRun = isRun

    def _setOptimizer(self, optimizer, learn_rate):
        """
        init the optimizer
        """
        if optimizer == 'adam':
            self.opt = keras.optimizers.Adam(lr=learn_rate)
        elif optimizer == 'sgd':
            self.opt = keras.optimizers.SGD(lr=learn_rate)
        else:
            self.opt = keras.optimizers.RMSprop(lr=learn_rate)
        if self.output:
            self.output[str].emit("Setup optimizer")

    def createDataSet(self):
        """
        parsing all files in db/wav files with MFCC function and store it as
        csv file in db/MFCC folder
        """
        winstep = 0.005
        filenames = os.listdir("db\\wav")
        # create store folder if it not exists
        if not os.path.exists("db\MFCC"):
            os.makedirs("db\MFCC")
        # run over wav files
        for i in range(len(filenames)):
            if not self.isRun:
                break
            (rate, sig) = wav.read("db\\wav\\{0}".format(filenames[i]))
            temp = mfcc(sig, rate, winstep=winstep, numcep=self.column_nbr, nfilt=self.column_nbr)
            np.savetxt("db\\MFCC\\{0}.csv".format(filenames[i]), temp[0:self.line_nbr, :], delimiter=",")
            # print to log
            if self.dictionary[filenames[i][5]] == "Fear":
                toPrint = "True"
            else:
                toPrint = "False"
            if self.output:
                self.output[str].emit("{} has been parsed with value {}".format(filenames[i], toPrint))

    def load_data(self):
        """
        load csv files into variables of the class
        """
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
        # run over the train data and label each one
        for i in range(len(train_files)):
            if not self.isRun:
                break
            for j in range(3):
                self.train_data[i][j] = np.loadtxt(open("db\\MFCC\\{0}".format(filenames[i]), "rb"), delimiter=",")
            if self.dictionary[filenames[i][5]] == "Fear":
                self.train_label[i] = False
            else:
                self.train_label[i] = True
        # run over the test data and label each one
        for i in range(len(test_files)):
            if not self.isRun:
                break
            for j in range(3):
                self.test_data[i][j] = np.loadtxt(open("db\\MFCC\\{0}".format(filenames[i]), "rb"), delimiter=",")
            if self.dictionary[filenames[i][5]] == "Fear":
                self.test_label[i] = False
            else:
                self.test_label[i] = True

    def createNewVGG16Model(self):
        """
        create VGG16 model with contains VGG16 covolution layers and our fully connected layers
        """
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        sess = tf.Session(config=config)
        K.set_session(sess)  # K is keras backend
        self.session.run(tf.global_variables_initializer())
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
        nextLayer = Dropout(0.5)(nextLayer)
        lastLayer = Dense(CLASSES_NBR,activation="softmax")(nextLayer)
        #lastLayer = Activation("softmax", dtype="float32")(nextLayer)
        self.model = Model(VGG16_conv2D.input, lastLayer)
        # compile the model
        self.model.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        self.model.summary()
        if self.output:
            self.output[str].emit(self.model.summary())

    def saveModel(self, fileName):
        """
        saving model as h5 file
        :param fileName: store file name
        """
        self.model.save('{}.h5'.format(fileName))  # creates a HDF5 file 'my_model.h5'

    def loadModel(self, fileName):
        """
        loading model from file system
        :param fileName:
        """
        self.model = load_model('{}'.format(fileName))
        # getting the model filter numbers
        thirdDimension = self.model.input.shape[2]
        self.column_nbr = thirdDimension.__int__()

    def predict(self, input):
        """
        prediction function using input and choosed model
        :param input:
        :return:
        """
        # reshaping input
        input_model = np.zeros((1, 3, self.line_nbr, self.column_nbr), dtype=float)
        input_model[0][0] = input_model[0][1] = input_model[0][2] = input[0:225, :]
        input_model[0][0] = input_model[0][1] = input_model[0][2] = input[0:225, :]
        input_model = input_model.reshape(input_model.shape[0], self.line_nbr, self.column_nbr, 3)
        res = self.model.predict(input_model, verbose=1)

        return float(res[0][0]), float(res[0][1])

    def trainModel(self):
        """
        traing model using train and test data
        """
        # normalize train data
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

        if self.isRun:
            #starting to train model
            with self.default_graph.as_default():
                history = self.model.fit(self.train_data,
                                         self.train_label,
                                         batch_size=self.batch_size,
                                         epochs=self.epoch_nbr,
                                         validation_data=(self.test_data, self.test_label),
                                         shuffle=True,
                                         verbose=1,
                                         callbacks=self.getCallBacks())
                return history

    def validateModel(self):
        """
        function to validate if model which we loaded is the same as we stored
        for testing plan
        """
        scores = self.model.evaluate(self.test_data, self.test_label)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def getCallBacks(self):
        """
        callbacks to use when training model.
        - Early stopping to stop training if it;s going to be overfitting and restore best weights.
        - TensorBoard to get option to view logs in dashboard
        - AccuracyCallBack our Callback to print log and redrawing graphs
        """
        earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=.01,
                                                  patience=5,
                                                  verbose=1,
                                                  mode='auto',
                                                  baseline=None,
                                                  restore_best_weights=True)
        tensorBoard = keras.callbacks.TensorBoard(log_dir='./logs/{}/'.format(self.name),
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
        return [tensorBoard, earlyStop, self.AccuracyCallback]
