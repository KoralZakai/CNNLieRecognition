import  os
import wave
import keras
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from ModelTrainingUtils.CNN import CNN



for modelFile in os.listdir("Model"):
    print(modelFile)
    winstep = 0.005
    filenames = os.listdir("db\\wav")
    dictionary = {"W": "Anger", "L": "Boredom", "E": "Disgust", "A": "Fear", "F": "Happiness", "T": "Sadness",
                               "N": "Neutral"}
    # create store folder if it not exists
    model = CNN(model="Model/{}".format(modelFile))
    true_pos = false_neg = true_neg = false_pos = 0
    data = np.zeros((len(filenames), 3, 225, 32), dtype=float)
    label = np.zeros((len(filenames), 1), dtype=int)
    for i in range(len(filenames)):
        (rate, sig) = wav.read("db\\wav\\{0}".format(filenames[i]))
        temp = mfcc(sig, rate, winstep=winstep, numcep=32, nfilt=32)
        for j in range(3):
            data[i][j] = temp[0:225, :]
        # print to log
        if dictionary[filenames[i][5]] == "Fear":
            label[i] = 1
        else:
            label[i] = 0
    '''
    predictions = model.model.predict(data)
    val = np.argmax(predictions, axis=1)
    for i in label:
        if 
    if label[i] == "True" and val == 1:
        true_pos += 1
    elif label[i] == "False" and val != 1:
        false_neg +=1
    elif label[i] == "True" and val != 1:
        true_neg +=1
    else:
        false_pos+=1
    #print("True positive:{} True negative:{}\n False positive{} False negative{}".format(true_pos, true_neg,
                                                                                            false_pos, false_neg))
    '''
    #
    data = data.reshape(data.shape[0], 225, 32, 3)
    predictions = model.model.predict(data, batch_size=1, verbose=0)
    label = label.ravel()

    print(confusion_matrix(label, np.argmax(predictions, axis=1)))
    print(classification_report(label, np.argmax(predictions,axis=1)))
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

    recall = true_pos/(true_pos+false_neg)
    precision = true_pos/(true_pos+false_pos)
    accuracy = (true_pos+false_neg)/(true_pos+false_neg+true_neg+false_pos)

    print("Recall:{} Precision:{} accuracy:{}".format(recall, precision,accuracy))
