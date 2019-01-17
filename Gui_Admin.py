from PyQt5 import QtWidgets, uic

class App(QtWidgets):

    def __init__(self):
        super().__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Lie Detection")