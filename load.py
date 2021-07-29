from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtGui import *


class Load_Ui(object):
    def setupUi(self, MainWindow):
        self.MainWindow = MainWindow
        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(500, 500)
        self.MainWindow.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.MainWindow.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.centralwidget.setStyleSheet("background-color : rgb(224,244,253)")

        # create label
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(25, 25, 500, 500))
        self.label.setMinimumSize(QtCore.QSize(500, 500))
        self.label.setMaximumSize(QtCore.QSize(500, 500))
        self.label.setObjectName("label")

        # add label to main window
        # MainWindow.setCentralWidget(self.centralwidget)

        # set qmovie as label
        self.movie = QMovie("./image/test.gif")
        self.label.setMovie(self.movie)
        self.movie.start()

    def close(self):
        self.MainWindow.setWindowOpacity(0)

    def open(self):
        self.MainWindow.setWindowOpacity(1)