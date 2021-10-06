import sys
from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication
import os

class MyApp(QDialog):

    def __init__(self):
        super().__init__()
        self.setupUI(QDialog)
        # self.setGeometry(0, 0, 1366, 768)

        self.setWindowIcon((QtGui.QIcon('icon1.png')))
        self.setStyleSheet("background-color : rgb(248, 249, 251);")

    def setupUI(self, Dialog):
        exit_btn = QtWidgets.QPushButton(' ', self)
        exit_btn.resize(exit_btn.sizeHint())

        exit_btn.setGeometry(QtCore.QRect(1194, 50, 36, 36))
        exit_btn.setStyleSheet("border-radius : 20;")
        exit_btn.setStyleSheet(
            '''
            QPushButton{image:url(./image/icon/exit.png); border:0px;}
            QPushButton:hover{image:url(./image/icon/exit_hover.png); border:0px;}
            ''')
        exit_btn.setObjectName("pushButton_10")
        exit_btn.clicked.connect(Dialog.reject)

        # frame set
        self.frame = QtWidgets.QFrame(self)
        self.frame.setGeometry(QtCore.QRect(100, 100, 1080, 520))
        self.frame.setAutoFillBackground(False)
        self.frame.setStyleSheet("background-color : rgba(0, 0, 0, 0%); border-radius: 30px;")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        # Button 1 : Lite Version
        self.lite_Button = QtWidgets.QPushButton(self.frame)
        self.lite_Button.setGeometry(QtCore.QRect(0, 0, 510, 520))
        self.lite_Button.setStyleSheet(
            '''
            QPushButton{
                color: white;
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.857143, y2:0.857955,
                stop:0 rgba(0, 160, 182, 255),
                stop:1 rgba(144, 61, 167, 255));\
                border-radius: 30px;
                image:url(./Image/KOR/guide_open.png);
            }
            QPushButton:hover {
                background-color: rgb(20, 180, 202); border-radius: 30px;
            }
            ''')
        self.lite_Button.setObjectName("lite_Button")
        self.lite_Button.clicked.connect(self.lite)

        # Button 2 : Full Version
        self.full_Button = QtWidgets.QPushButton(self.frame)
        self.full_Button.setGeometry(QtCore.QRect(570, 0, 510, 520))
        self.full_Button.setStyleSheet(
            '''
            QPushButton{
                color: white;
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.857143, y2:0.857955,
                stop:0 rgba(226, 0, 46, 255),
                stop:1 rgba(144, 61, 167, 255));
                border-radius: 30px;
                image:url(./Image/KOR/guide_open.png);
            }
            QPushButton:hover {
                background-color: rgb(246, 20, 66); border-radius: 30px;
            }
            ''')
        self.full_Button.setObjectName("full_Button")
        self.full_Button.clicked.connect(self.full)




        self.setWindowTitle('Motion Presentation Intro')

        self.resize(1280, 720)
        self.show()

    def lite(self):
        print('lite')

    def full(self):
        print('full')

        os.system('''gesture_detection.bat''')


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())