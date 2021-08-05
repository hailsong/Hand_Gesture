from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import sys


class Load_Ui2(QtWidgets.QMainWindow):
    def __init__(self, img_path='./image/loading.png', xy=[560, 340], size=1.0, on_top=False):
        super(Load_Ui2, self).__init__()
        self.timer = QtCore.QTimer(self)
        self.img_path = img_path
        self.xy = xy
        self.from_xy = xy
        self.from_xy_diff = [0, 0]
        self.to_xy = xy
        self.to_xy_diff = [0, 0]
        self.speed = 60
        self.direction = [0, 0]  # x: 0(left), 1(right), y: 0(up), 1(down)
        self.size = size
        self.on_top = on_top
        self.localPos = None

        self.setupUi()
        self.show()

        # 0 : cam on, 1 : cam off
        self.status = 0

    # # 마우스 놓았을 때
    # def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
    #     if self.to_xy_diff == [0, 0] and self.from_xy_diff == [0, 0]:
    #         pass
    #     else:
    #         self.walk_diff(self.from_xy_diff, self.to_xy_diff, self.speed, restart=True)
    #
    # # 마우스 눌렀을 때
    # def mousePressEvent(self, a0: QtGui.QMouseEvent):
    #     self.localPos = a0.localPos()
    #
    # # 드래그 할 때
    # def mouseMoveEvent(self, a0: QtGui.QMouseEvent):
    #     self.timer.stop()
    #     self.xy = [(a0.globalX() - self.localPos.x()), (a0.globalY() - self.localPos.y())]
    #     self.move(*self.xy)

    def setupUi(self):
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        flags = QtCore.Qt.WindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowOpacity(0.8)

        label = QtWidgets.QLabel(centralWidget)
        movie = QMovie(self.img_path)
        label.setMovie(movie)
        movie.start()
        movie.stop()

        w = int(movie.frameRect().size().width() * self.size)
        h = int(movie.frameRect().size().height() * self.size)
        movie.setScaledSize(QtCore.QSize(w, h))
        movie.start()

        self.setGeometry(self.xy[0], self.xy[1], w, h)

    def close(self):
        self.setWindowOpacity(0)

    def open(self):
        self.setWindowOpacity(0.7)