from CNN import CNN
import matplotlib.pyplot as plt
def displayLearning(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

 


for lr in [0.001]:
    for epoch in [20]:
        print("GG16 LR={} Epoch={}".format(lr, epoch))
        model = CNN(learn_rate=lr, epoch_nbr=epoch)
        model.createNewVGG16Model()
        #model.createDataSet()
        model.load_data()
        history = model.trainModel()
        displayLearning(history)

        model.validateModel()
        model.saveModel("my")
        model.loadModel("my")
        model.validateModel()


