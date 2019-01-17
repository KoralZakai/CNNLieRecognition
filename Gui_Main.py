from PyQt5 import QtWidgets, uic
from Functions import Functions as Functions


def main():
    func = Functions()
    app = QtWidgets.QApplication([])
    #Loading the Gui_Main widgets
    dlg = uic.loadUi("Gui_Main.ui")
    dlg.startRecordBtn.show()
    dlg.stopRecordBtn.hide()
    dlg.recordingLbl.hide()
    dlg.loadingLbl.hide()
    dlg.fileBrowseBtn.clicked.connect(lambda: func.openFile(dlg))
    dlg.startRecordBtn.clicked.connect(lambda: func.startRecord(dlg))
    dlg.stopRecordBtn.clicked.connect(lambda: func.stopRecord(dlg))
    dlg.show()
    app.exec()


if __name__ == "__main__":
    main()
