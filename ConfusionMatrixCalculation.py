import  os
import wave
import keras
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

PATH = "db\\wav"
from ModelTrainingUtils.CNNCreator import CNNCreator
def confusion():
    for modelFile in ["TheBestOfTheBest.h5"]:#:os.listdir("Model"):
        print(modelFile)
        filenames = os.listdir("{}\\".format(PATH))
        # create store folder if it not exists
        model = CNNCreator(modelName="Model/{}".format(modelFile))
        model.set_running_status(True)
        true_pos = false_neg = true_neg = false_pos = 0
        model.createDataSet()
        #
        data = model.data.reshape(model.data.shape[0], 225, 32, 3)
        max = np.max(data)
        min = np.min(data)
        data = (data - min) / (max - min)
        data = (data * 255)
        predictions = model.model.predict(data, batch_size=1, verbose=0)
        label = model.label.ravel()

        print(confusion_matrix(label, np.argmax(predictions, axis=1)))
        print(classification_report(label, np.argmax(predictions, axis=1)))
        for i in range(len(label)):
            if label[i] == 0 and predictions[i][0] != 1:
                true_pos += 1
            elif label[i] == 0 and predictions[i][0] == 1:
                true_neg += 1
            elif label[i] == 1 and predictions[i][0] == 1:
                false_neg += 1
            else:
                false_pos += 1
        print("True positive:{} True negative:{}\n False positive{} False negative{}".format(true_pos, true_neg,
                                                                                             false_pos, false_neg))

        recall = true_pos / (true_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        accuracy = (true_pos + false_neg) / (true_pos + false_neg + true_neg + false_pos)

        print("Recall:{} Precision:{} accuracy:{}".format(recall, precision, accuracy))

def changeFile():
    Lie = 0
    NotLie = 0
    for file in os.listdir(PATH):
        #if file[5] == "A":
        if file[7] == "6":
            Lie += 1
            os.rename("{}\\{}".format(PATH,file), "{}\\Lie_{}.wav".format(PATH, Lie))
        else:
            NotLie += 1
            os.rename("{}\\{}".format(PATH, file), "{}\\NotLie_{}.wav".format(PATH, NotLie))

#changeFile()
confusion()
