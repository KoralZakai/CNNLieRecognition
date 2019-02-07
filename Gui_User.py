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
from Help_Window import Help_Window
import sys
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
        self.checkEnvErr = ""
        self.initUI()



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
        self.main_frame.setFixedSize(self.width, self.height)


        #the first sub window
        main_layout = QtWidgets.QVBoxLayout(self.main_frame)
        self.firstsub_Frame = QtWidgets.QFrame(self.main_frame)

        main_layout.addWidget(self.firstsub_Frame)
        self.firstsub_Layout = QtWidgets.QFormLayout(self.firstsub_Frame)
        self.firstsub_Frame.setFixedHeight(self.height/5.2)

        # Return to main window button
        returnBtn = QtWidgets.QPushButton("")
        returnBtn.setStyleSheet("QPushButton {background: url(:Pictures/backimg.png) no-repeat transparent;} ")
        returnBtn.setFixedWidth(110)
        returnBtn.setFixedHeight(110)
        returnBtn.clicked.connect(self.closeThisWindow)

        # help button
        helpBtn = QtWidgets.QPushButton("")
        helpBtn.setStyleSheet("QPushButton {background: url(:Pictures/help.png) no-repeat transparent;} ")
        helpBtn.setFixedWidth(110)
        helpBtn.setFixedHeight(110)
        helpBtn.clicked.connect(self.showHelp)
        buttonsform = QtWidgets.QFormLayout(self)

        buttonsform.addRow(returnBtn, helpBtn)
        #Setting up the form fields
        #form title init
        formTitleLbl = QtWidgets.QLabel('Lie Detector')
        formTitleLbl.setAlignment(Qt.AlignCenter)
        formTitleLbl.setContentsMargins(0,0,50,50)
        formTitleLbl.setObjectName("LableHeader")
        self.firstsub_Layout.addRow(formTitleLbl)

        #init the browse file fields - lable , textfield, file browse button , start/stop record buttons
        fileBrowseHBoxLayout = QtWidgets.QGridLayout()
        self.fileBrowserTxt=QtWidgets.QTextEdit("", self)
        self.fileBrowserTxt.setReadOnly(True)
        self.fileBrowserLbl=QtWidgets.QLabel('Pick Wav File', self)
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
        fileBrowseHBoxLayout.addWidget(self.startRecordBtn, 1, 4)
        fileBrowseHBoxLayout.addWidget(self.stopRecordBtn, 1, 4)
        recordHBoxLayout.setAlignment(Qt.AlignCenter)
        self.firstsub_Layout.addRow(recordHBoxLayout)

        # The second sub window - loading gif window
        self.secondsub_Frame = QtWidgets.QFrame(self.main_frame)
        main_layout.addWidget(self.secondsub_Frame)
        self.secondsub_Layout = QtWidgets.QFormLayout(self.secondsub_Frame)
        self.secondsub_Frame.setFixedWidth(self.width)
        self.secondsub_Frame.setFixedHeight(self.height/8)
        self.secondsub_Layout.addRow(self.recordingLbl,self.loadingLbl)
        self.secondsub_Frame.setContentsMargins(self.width/2-self.recordingLbl.width(),0,0,0)
        # Settings Layout
        self.settings_Frame = QtWidgets.QFrame(self.main_frame)
        main_layout.addWidget(self.settings_Frame)
        self.settings_Layout = QtWidgets.QFormLayout(self.settings_Frame)
        self.settings_Frame.setFixedWidth(self.width)
        self.settings_Frame.setFixedHeight(self.height/8)
        self.settings_Frame.setContentsMargins(self.width, 0, 0, 0)
        self.settings_Layout.setFormAlignment(Qt.AlignCenter)
        self.settings_Frame.setVisible(False)
        # the third sub window
        self.thirdsub_Frame = QtWidgets.QFrame(self.main_frame)
        main_layout.addWidget(self.thirdsub_Frame)
        self.thirdsub_Layout = QtWidgets.QGridLayout(self.thirdsub_Frame)
        self.thirdsub_Frame.setFixedWidth(self.width-25)
        self.thirdsub_Frame.setFixedHeight(self.height/2.2)
        logo = QtWidgets.QLabel('', self)
        pixmap = QPixmap(':Pictures/logo.png')
        logo.setPixmap(pixmap)
        self.thirdsub_Layout.addWidget(logo)

        logo.setAlignment(Qt.AlignCenter|Qt.AlignTop)

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
        self.processGraphsBtn.setFixedHeight(30)
        self.processGraphsBtn.clicked.connect(lambda: self.dataProcessingmfcc())
        self.settings_Layout.addRow(self.processGraphsBtn)

        # Predict button
        self.mfccGraphsBtn = QtWidgets.QPushButton("MFCC", self)
        self.mfccGraphsBtn.setObjectName("Buttons")
        self.mfccGraphsBtn.setFixedWidth(131)
        self.mfccGraphsBtn.setFixedHeight(30)
        self.mfccGraphsBtn.clicked.connect(lambda: self.showMfcc())
        self.settings_Layout.addRow(self.mfccGraphsBtn,self.processGraphsBtn)

        #show the window
        self.show()


    def checkEnvironment(self,type):
        """
        Validate that the working environment is safe to work .
        :param type: The check environment type , type = 1 -> check microphone if plugged in .

        """
        checkEnv = True
        self.checkEnvErr = ""
        winmm = ctypes.windll.winmm
        if type == 1:#check microphone
            if winmm.waveInGetNumDevs() != 1:
                checkEnv = False
                self.checkEnvErr = "Microphone is missing, please plugin you'r microphone.\n"

        # Checking existing models
        modelPath = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\Model\\"
        modelDir = os.listdir(modelPath)
        if len(modelDir) == 0:
            checkEnv = False
            self.checkEnvErr = self.checkEnvErr + "There is no Models to work with."

        return checkEnv


    def buildCoefComboBox(self):
        """
        Building the Coefficients numbers combobox
        """
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
        """
        Building the Model's combobox
        """
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
        """
        Getting the Coefficients number once the user click on the Coefficients combobox
        :param text: The text that the user clicked on in the combobox

        """
        self.NUMCEP = int(text)


    def onActivatedComboBoxModel(self, text):
        """
        Getting the Model once the user click on the Coefficients combobox
        :param text: The text that the user clicked on in the combobox
        """
        self.pickedModelPath = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\Model\\"+text+'.h5'


    def initSettings(self):
        """
        Initialize the settings before displaying graphs
        """
        self.clearGraph()

        self.settings_Frame.setVisible(False)
        self.NUMCEP = 32

    def openFile(self,form ):
        """
        Opening file browser to import the Wav file.
        :param form: The current layout to display the message box error .
        """
        if self.checkEnvironment(2):
            self.initSettings()
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
        else:
            QMessageBox.about(self, "Error", self.checkEnvErr)

    def startRecord(self):
        """
        Recording voice using microphone
        """
        if self.checkEnvironment(1):
            self.initSettings()
            self.startRec = True
            self.pyrecorded = pyaudio.PyAudio()
            self.stream = self.pyrecorded.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK)
            self.movie = QtGui.QMovie(":Pictures/loading2.gif")
            self.loadingLbl.setMovie(self.movie)
            self.movie.start()
            self.loadingLbl.setVisible(True)
            self.recordingLbl.setVisible(True)
            self.startRecordBtn.setVisible(False)
            self.stopRecordBtn.setVisible(True)
            self.secondsub_Frame.setVisible(True)
            self.fileBrowserBtn.setDisabled(True)

            self.frames = []
            self.recThread = threading.Thread(target = self.inputData)
            self.recThread.start()
        else:
            QMessageBox.about(self, "Error", self.checkEnvErr)

    def inputData(self):
        """
        Input stream of data from the microphone
        """
        while (self.startRec):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
        sys.exit()

    def startWaitingGif(self):
        """
        Playing the waiting GIF
        """
        self.movieGraphWait = QtGui.QMovie(":Pictures/loading2.gif")
        loadingGraphLbl = QtWidgets.QLabel('', self)
        loadingGraphLbl.setMaximumHeight(100)
        loadingGraphLbl.setMaximumWidth(100)
        loadingGraphLbl.setMovie(self.movieGraphWait)
        self.firstsub_Layout.addWidget(loadingGraphLbl)
        self.movieGraphWait.start()

    def stopRecord(self):
        """
        Stop record and save the stream of wav frames into wav file.
        """
        # Stopping the recording thread
        self.startRec = False
        # Handling all the fields visibility.
        self.loadingLbl.setVisible(False)
        self.stopRecordBtn.setVisible(False)
        self.startRecordBtn.setVisible(True)
        self.recordingLbl.setVisible(False)
        self.secondsub_Frame.setVisible(False)
        self.fileBrowserBtn.setDisabled(False)
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

    def dataProcessing(self):
        """
        Handiling the data processing.
        """
        # Showing te graph's frame.
        self.settings_Frame.setVisible(True)
        # Drawing the sound graph / mfcc graph.
        self.showSoundWav()
        self.showMfcc()

    def dataProcessingmfcc(self):
        """
        Processing the wav file / recorded file , drawing mfcc.
        """
        # Drawing mfcc for the input file.
        self.showMfcc()
        # Prediction using the picked model .
        newCNN = CNN(model=self.pickedModelPath)
        if newCNN.column_nbr != self.NUMCEP:
            QMessageBox.about(self, "Error", "The Coefficients number is not match the model Properties ,dont worry, I will fix it for you ")
            self.comboBoxCoef.setCurrentIndex(newCNN.column_nbr-32)
            self.NUMCEP=int(self.comboBoxCoef.currentText())
            self.showMfcc()
        cnnResult = newCNN.predict(self.mfccResult)
        QMessageBox.information(self, "Results", "Result : "+str(cnnResult[0]))


    def clearGraph(self):
        """
        Clearing graphs
        :param layoutnum: the layout number that includes the wanted graph to clear.
        layoutnum = 3 -> sound wave graph.
        """

        while self.thirdsub_Layout.count():
            child = self.thirdsub_Layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


    def showSoundWav(self ):
        """
        Drawing sound wave graph.
        """
        # Clear the sound wave graph.
        self.clearGraph()
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
        self.figureSoundWav.setLabel('left','Amplitude (db)')
        self.figureSoundWav.setLabel('bottom', 'Time (sec)')
        self.figureSoundWav.plot(Time,signal)
        self.figureSoundWav.setEnabled(False)
        self.figureSoundWav.setYRange(-32000,32000)
        self.figureSoundWav.getAxis('bottom').enableAutoSIPrefix(False)
        self.figureSoundWav.getAxis('left').enableAutoSIPrefix(False)

    def showMfcc(self):
        """
        Drawing the MFCC graph.
        """
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


    def showHelp(self):
        """
        Opens help window.
        """
        helpWindow = Help_Window(':Pictures/helpuser3.png')


    def closeThisWindow(self):
        """
        Close the current window and open the main window.
        """
        self.parent().show()
        self.parent().main_frame.setVisible(True)
        self.close()