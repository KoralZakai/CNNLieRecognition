from matplotlib.figure import Figure
from PyQt5.QtGui import QPixmap, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtGui, QtWidgets
import time
import wave
import threading
import pyaudio
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QFile, QTextStream
import pyqtgraph
from ModelTrainingUtils.CNN import *
import ctypes

class Gui_User(QWidget):
    def __init__(self, parent=None):
        super(Gui_User, self).__init__(parent)
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Lie Detection'
        self.width = w
        self.height = h
        self.startRec = True
        #stream params
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.NUMCEP = 32
        self.RATE = 8000
        self.frames = None
        self.pyrecorded = None
        self.stream = None
        self.recThread = None
        self.movie = None
        self.figureSoundWav = None
        self.mfccResult = None
        self.WAVE_OUTPUT_FILENAME = None
        self.WAVE_OUTPUT_FILEPATH = None
        self.pickedModelPath = None
        self.checkEnv = True
        self.checkEnvErr = None
        self.checkEnvironment()
        if self.checkEnv:
            self.initUI()
        else:
            QMessageBox.about(self, "Error", self.checkEnvErr)

    def initUI(self):
        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)
        self.setObjectName("Windowimg")
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(':Pictures/logo.png'))
        self.setGeometry(0, 0, self.width, self.height-60)
        #Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        #Return to main window button
        returnBtn = QtWidgets.QPushButton("", self)
        returnBtn.setStyleSheet("QPushButton {background: url(:Pictures/backimg.png) no-repeat transparent;} ")
        returnBtn.setFixedWidth(110)
        returnBtn.setFixedHeight(110)
        returnBtn.clicked.connect(self.closeThisWindow)



        #the first sub window
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)
        self.firstsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.firstsub_Frame)
        self.firstsub_Layout = QtWidgets.QFormLayout(self.firstsub_Frame)



        #Setting up the form fields
        #form title init
        self.formTitleLbl = QtWidgets.QLabel('Lie Detector')
        myFont = QtGui.QFont()
        myFont.setBold(True)
        myFont.setPixelSize(25)
        self.formTitleLbl.setFont(myFont)
        self.formTitleLbl.setAlignment(Qt.AlignCenter)
        self.formTitleLbl.setContentsMargins(0,0,50,20)
        self.formTitleLbl.setObjectName("LableHeader")

        self.firstsub_Layout.addRow(self.formTitleLbl)
        #init the browse file fields - lable , textfield, file browse button , start/stop record buttons
        fileBrowseHBoxLayout = QtWidgets.QGridLayout()
        self.fileBrowserTxt=QtWidgets.QTextEdit("", self)
        self.fileBrowserTxt.setReadOnly(True)
        self.fileBrowserLbl=QtWidgets.QLabel('Pick Wav File', self)
        myFont.setPixelSize(18)
        self.fileBrowserLbl.setFont(myFont)
        self.fileBrowserTxt.setFixedWidth(500)
        self.fileBrowserTxt.setFixedHeight(25)
        self.fileBrowserLbl.setFixedWidth(150)
        self.fileBrowserLbl.setFixedHeight(25)
        self.fileBrowserBtn = QtWidgets.QPushButton("", self)
        self.fileBrowserBtn.setMaximumHeight(100)
        self.fileBrowserBtn.setMaximumWidth(100)
        self.fileBrowserBtn.setFixedHeight(27)
        self.fileBrowserBtn.setFixedWidth(27)
        self.fileBrowserBtn.setStyleSheet("QPushButton {background: url(:Pictures/filebrowse.png) no-repeat transparent;} ")
        self.fileBrowserBtn.clicked.connect(lambda: self.openFile(self.firstsub_Layout))
        fileBrowseHBoxLayout.addWidget(self.fileBrowserLbl,1,0)
        fileBrowseHBoxLayout.addWidget(self.fileBrowserTxt,1,1)
        fileBrowseHBoxLayout.addWidget(self.fileBrowserBtn,1,2)
        fileBrowseHBoxLayout.setAlignment(Qt.AlignCenter)
        self.firstsub_Layout.addRow(fileBrowseHBoxLayout)
        recordHBoxLayout = QtWidgets.QGridLayout()
        self.startRecordBtn = QtWidgets.QPushButton("", self)
        self.startRecordBtn.setFixedHeight(25)
        self.startRecordBtn.setFixedWidth(25)
        self.startRecordBtn.setStyleSheet("QPushButton {background: url(:Pictures/microphone1.png) no-repeat transparent;} ")
        self.recordingLbl = QtWidgets.QLabel('Recording', self)
        self.recordingLbl.setContentsMargins(self.height/2,self.width/2,50,50)
        self.recordingLbl.setVisible(False)
        self.recordingLbl.setFixedWidth(100)
        self.recordingLbl.setFixedHeight(40)
        self.loadingLbl = QtWidgets.QLabel('', self)
        self.loadingLbl.setFixedWidth(200)
        self.loadingLbl.setFixedHeight(25)
        self.stopRecordBtn = QtWidgets.QPushButton("", self)
        self.stopRecordBtn.setStyleSheet("QPushButton {background: url(:Pictures/microphone2.png) no-repeat transparent;} ")
        self.stopRecordBtn.setVisible(False)
        self.stopRecordBtn.setFixedWidth(25)
        self.stopRecordBtn.setFixedHeight(25)
        fileBrowseHBoxLayout.addWidget(self.startRecordBtn,1,4)
        fileBrowseHBoxLayout.addWidget(self.stopRecordBtn,1,4)
        recordHBoxLayout.setAlignment(Qt.AlignCenter)
        self.firstsub_Layout.addRow(recordHBoxLayout)

        # The second sub window - loading gif window
        self.secondsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.secondsub_Frame)
        self.secondsub_Layout = QtWidgets.QFormLayout(self.secondsub_Frame)
        self.secondsub_Frame.setFixedWidth(self.width)
        self.secondsub_Frame.setFixedHeight(50)
        self.secondsub_Layout.addRow(self.recordingLbl,self.loadingLbl)
        self.secondsub_Frame.setContentsMargins(self.width/2-self.recordingLbl.width(),0,0,0)

        # Settings Layout
        self.settings_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.settings_Frame)
        self.settings_Layout = QtWidgets.QFormLayout(self.settings_Frame)
        self.settings_Frame.setFixedWidth(self.width)
        self.settings_Frame.setFixedHeight(self.height/8)
        self.settings_Frame.setContentsMargins(self.width, 0, 0, 0)
        self.settings_Layout.setFormAlignment(Qt.AlignCenter)
        self.settings_Frame.setVisible(False)

        # the third sub window
        self.thirdsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.thirdsub_Frame)
        self.thirdsub_Layout = QtWidgets.QGridLayout(self.thirdsub_Frame)
        self.thirdsub_Frame.setFixedWidth(self.width-25)
        self.thirdsub_Frame.setFixedHeight(self.height/1.8)

        logo = QtWidgets.QLabel('', self)
        pixmap = QPixmap(':Pictures/logo.png')
        logo.setPixmap(pixmap)
        self.thirdsub_Layout.addWidget(logo)
        logo.setAlignment(Qt.AlignCenter)

        # the 4rth sub window
        self.fourthsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.fourthsub_Frame)
        self.fourthsub_Layout = QtWidgets.QFormLayout(self.fourthsub_Frame)
        self.fourthsub_Frame.setFixedWidth(self.width)
        self.fourthsub_Frame.setFixedHeight(self.height / 1.8)

        # assign functions to buttons
        self.startRecordBtn.clicked.connect(lambda: self.startRecord())
        self.stopRecordBtn.clicked.connect(lambda: self.stopRecord())

        # building the Coefficients number comboBox
        self.buildCoefComboBox()

        # building the Model comboBox
        self.buildModelComboBox()

        #Predict button
        self.processGraphsBtn = QtWidgets.QPushButton("Predict", self)
        self.processGraphsBtn.setObjectName("Buttons")
        self.processGraphsBtn.setFixedWidth(131)
        self.processGraphsBtn.setFixedHeight(25)
        self.processGraphsBtn.clicked.connect(lambda: self.dataProcessingmfcc())
        self.settings_Layout.addRow(self.processGraphsBtn)

        # Predict button
        self.mfccGraphsBtn = QtWidgets.QPushButton("MFCC", self)
        self.mfccGraphsBtn.setObjectName("Buttons")
        self.mfccGraphsBtn.setFixedWidth(131)
        self.mfccGraphsBtn.setFixedHeight(25)
        self.mfccGraphsBtn.clicked.connect(lambda: self.showMfcc())
        self.settings_Layout.addRow(self.mfccGraphsBtn,self.processGraphsBtn)



        #show the window
        self.show()

    # Validate that the working environment is safe to work .
    def checkEnvironment(self):
        winmm = ctypes.windll.winmm
        if winmm.waveInGetNumDevs() != 1:
            self.checkEnv = False
            self.checkEnvErr = "Microphone is missing, please plugin you'r microphone"

        # Checking existing models
        modelPath = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\Model\\"
        modelDir = os.listdir(modelPath)
        if len(modelDir) == 0:
            self.checkEnv = False
            self.checkEnvErr = "There is no Models to work with"

    def buildCoefComboBox(self):
        self.comboBoxCoef = QtWidgets.QComboBox(self)
        for i in range(32, 226):
            self.comboBoxCoef.addItem(str(i))
        self.comboBoxCoef.activated[str].connect(self.onActivatedComboBoxCoef)
        self.comboBoxCoefLbl = QtWidgets.QLabel('Coefficients')
        self.comboBoxCoefLbl.setFixedWidth(125)
        self.comboBoxCoefLbl.setFixedHeight(25)
        self.comboBoxCoef.setFixedWidth(130)
        self.comboBoxCoef.setFixedHeight(25)
        self.settings_Layout.addRow(self.comboBoxCoefLbl,self.comboBoxCoef)

    def buildModelComboBox(self):
        self.comboboxModel = QtWidgets.QComboBox(self)
        self.comboboxModel.setFixedWidth(130)
        self.comboboxModel.setFixedHeight(25)
        self.comboboxModel.activated[str].connect(self.onActivatedComboBoxModel)

        modelPath = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\Model\\"
        first = True
        for modelname in os.listdir(modelPath):
            if modelname.endswith('.h5'):
                self.comboboxModel.addItem(modelname.split('.')[0])
                if first:
                    self.pickedModelPath =modelPath +modelname
                    first = False


        self.comboBoxModelLbl = QtWidgets.QLabel('Model')
        self.comboBoxModelLbl.setFixedWidth(75)
        self.comboBoxModelLbl.setFixedHeight(25)
        self.settings_Layout.addRow(self.comboBoxModelLbl,self.comboboxModel)


    def onActivatedComboBoxCoef(self, text):
        self.NUMCEP = int(text)


    def onActivatedComboBoxModel(self, text):
        self.pickedModelPath = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\Model\\"+text+'.h5'

    def initSettings(self):
        self.clearGraph(3)
        self.clearGraph(4)
        self.settings_Frame.setVisible(False)
        self.NUMCEP = 32

    #Opening file browser to import the Wav file.
    def openFile(self,form ):
        self.initSettings()
        print()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "File Browser", "", "Wav Files (*.wav)", options=options)
        word = fileName.split('/')
        word = word[len(word) - 1]
        if len(word) != 0:
            if word.endswith('.wav'):
                self.fileBrowserTxt.setText(''+word)
                self.WAVE_OUTPUT_FILENAME = word
                self.WAVE_OUTPUT_FILEPATH = fileName
                self.dataProcessing()
            else:
                QMessageBox.about(form, "Error", "Wrong file type , please use only wav files")

    #Recording voice using microphone
    def startRecord(self):
        self.initSettings()
        self.startRec = True
        self.pyrecorded = pyaudio.PyAudio()
        self.stream = self.pyrecorded.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        print("* recording")
        self.movie = QtGui.QMovie(":Pictures/loading2.gif")
        self.loadingLbl.setMovie(self.movie)
        self.movie.start()
        self.loadingLbl.setVisible(True)
        self.recordingLbl.setVisible(True)
        self.startRecordBtn.setVisible(False)
        self.stopRecordBtn.setVisible(True)
        self.frames = []
        self.recThread = threading.Thread(target = self.inputData)
        self.recThread.start()


    # Input stream of data from the microphone
    def inputData(self):
        while (self.startRec):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
        sys.exit()

    # Playing the waiting GIF
    def startWaitingGif(self):
        self.movieGraphWait = QtGui.QMovie(":Pictures/loading2.gif")
        loadingGraphLbl = QtWidgets.QLabel('', self)
        loadingGraphLbl.setMaximumHeight(100)
        loadingGraphLbl.setMaximumWidth(100)
        loadingGraphLbl.setMovie(self.movieGraphWait)
        self.firstsub_Layout.addWidget(loadingGraphLbl)
        self.movieGraphWait.start()

    # Stop record and save the stream of wav frames into wav file.
    def stopRecord(self):
        # Stopping the recording thread
        self.startRec = False
        # Handling all the fields visibility.
        self.loadingLbl.setVisible(False)
        self.stopRecordBtn.setVisible(False)
        self.startRecordBtn.setVisible(True)
        self.recordingLbl.setVisible(False)
        self.stream.stop_stream()
        self.stream.close()
        self.pyrecorded.terminate()
        # Saving the wav file.
        self.WAVE_OUTPUT_FILENAME = time.strftime("%Y%m%d-%H%M%S")
        self.WAVE_OUTPUT_FILENAME = self.WAVE_OUTPUT_FILENAME + ".wav"
        path = os.path.dirname(os.path.realpath(sys.argv[0]))+"\\db\\"
        if not os.path.exists(path+"\\Records"):
            os.mkdir(path+"\\Records")
        wf = wave.open("db\\Records\\"+self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.pyrecorded.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        # Adding the file name to the file browser text field.
        self.fileBrowserTxt.setText(self.WAVE_OUTPUT_FILENAME)
        self.WAVE_OUTPUT_FILEPATH=os.path.dirname(os.path.realpath(sys.argv[0])) + "\\db\\Records\\" + self.WAVE_OUTPUT_FILENAME

        # Processing the input file.
        self.dataProcessing()

    # Handiling the data processing.
    def dataProcessing(self):
        # Showing te graph's frame.
        self.settings_Frame.setVisible(True)
        # Drawing the sound graph / mfcc graph.
        self.showSoundWav()
        self.showMfcc()

    # Processing the wav file / recorded file , drawing mfcc.
    def dataProcessingmfcc(self):
        # Drawing mfcc for the input file.
        self.showMfcc()
        # Prediction using the picked model .
        newCNN = CNN(model=self.pickedModelPath)
        if newCNN.column_nbr != self.NUMCEP:
            QMessageBox.about(self, "Error", "The Coefficients Number Is Not Match ,Dont Worry, I Will Fix It For You ")
            self.comboBoxCoef.setCurrentIndex(newCNN.column_nbr-32)
            self.NUMCEP=int(self.comboBoxCoef.currentText())
            self.showMfcc()
        cnnResult = newCNN.predict(self.mfccResult)
        QMessageBox.information(self, "Results", "Result : "+str(cnnResult[0]))

    # Clearing graphs
    # layoutnum - the layout number that includes the wanted graph to clear.
    def clearGraph(self,layoutnum):
        # layoutnum = 3 -> sound wave graph.
        # layoutnum = 4 -> mfcc grap.
        if layoutnum == 3:
            while self.thirdsub_Layout.count():
                child = self.thirdsub_Layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        if layoutnum == 4:
            while self.fourthsub_Layout.count():
                child = self.fourthsub_Layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()


    # Drawing sound wave graph.
    def showSoundWav(self ):
        # Clear the sound wave graph.
        self.clearGraph(3)
        # Reading wave file frames.
        spf = wave.open(self.WAVE_OUTPUT_FILEPATH, 'r')
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, np.int16)
        # A figure instance to plot on.

        fs = spf.getframerate()
        Time = np.linspace(0, len(signal) / fs, num=len(signal))

        self.figureSoundWav = pyqtgraph.PlotWidget()
        self.thirdsub_Layout.addWidget(self.figureSoundWav,2,1)
        self.figureSoundWav.setTitle('Wav - '+self.WAVE_OUTPUT_FILENAME)
        self.figureSoundWav.setLabel('left','AmpsetEnabledlitude (db)')
        self.figureSoundWav.setLabel('bottom', 'Time (sec)')
        self.figureSoundWav.plot(Time,signal)
        self.figureSoundWav.setEnabled(False)
        self.figureSoundWav.setYRange(-32000,32000)
        self.figureSoundWav.getAxis('bottom').enableAutoSIPrefix(False)
        self.figureSoundWav.getAxis('left').enableAutoSIPrefix(False)


    # Drawing the MFCC graph..
    def showMfcc(self):
        # Clear the graph.
        self.clearGraph(4)
        # Reading the wav file.
        (rate, sig) = wav.read(self.WAVE_OUTPUT_FILEPATH)

        # Execute mfcc function on the wav file.
        # winstep - mfcc window step.
        # numcep - coefficients number.
        # nflit - filters number.
        self.mfccResult = mfcc(sig, rate,winstep=0.005,numcep=self.NUMCEP,nfilt=self.NUMCEP)

        # Sound figure.
        self.mfccfigure = Figure()

        # This is the Canvas Widget that displays the `figure`.
        # It takes the `figure` instance as a parameter to __init__.
        self.mfcccanvas = FigureCanvas(self.mfccfigure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.thirdsub_Layout.addWidget(self.mfcccanvas,2,2)

        # create an axis
        # 111 - 1x1 grid, first subplot
        ax = self.mfccfigure.add_subplot(111)
        ax.set_ylabel('MFCC values')
        ax.set_xlabel('MFC Coefficients')
        ax.set_title('MFCC - '+self.WAVE_OUTPUT_FILENAME)
        ax.imshow(self.mfccResult, interpolation='nearest', origin='lower', aspect='auto')

    def closeThisWindow(self):
        self.parent().show()
        self.parent().main_frame.setVisible(True)
        self.close()