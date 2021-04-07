from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QObject, QRect, pyqtSlot, pyqtSignal
import numpy as np
import time
import datetime
import sys
import cv2

class opcv(QThread):

    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()

    @pyqtSlot(bool)
    def send_img(self, bool_state):  # p를 보는 emit 함수
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while bool_state and cv2.waitKey(33) < 0: # status가 True일 동안
            ret, cv_img = self.capture.read()
            if ret:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                self.change_pixmap_signal.emit(cv_img)
        self.capture.release()
        #self.wait()

class Ui_MainWindow(QObject):

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(870, 550)
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QRect(660, 20, 81, 301))
        self.groupBox.setStyleSheet("color: rgb(255, 255, 255);")
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")

        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QRect(10, 20, 61, 61))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.pushButton.setCheckable(True)
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QRect(10, 90, 61, 61))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.pushButton_2.setCheckable(True)
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QRect(10, 160, 61, 61))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.pushButton_3.setCheckable(True)
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QRect(10, 230, 61, 61))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.pushButton_4.setCheckable(True)
        self.pushButton_4.setObjectName("pushButton_4")

        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QRect(750, 20, 111, 301))
        self.groupBox_2.setStyleSheet("color: rgb(255, 255, 255);")
        self.groupBox_2.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_2.setObjectName("groupBox_2")

        self.checkBox = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox.setGeometry(QRect(10, 40, 95, 16))
        self.checkBox.setObjectName("checkBox")

        self.checkBox_2 = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_2.setGeometry(QRect(10, 110, 81, 16))
        self.checkBox_2.setObjectName("checkBox_2")

        self.checkBox_3 = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_3.setGeometry(QRect(10, 180, 81, 16))
        self.checkBox_3.setObjectName("checkBox_3")

        self.checkBox_4 = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_4.setGeometry(QRect(10, 250, 81, 16))
        self.checkBox_4.setObjectName("checkBox_4")


        self.checkBox.setEnabled(False)
        self.checkBox_2.setEnabled(False)
        self.checkBox_3.setEnabled(False)
        self.checkBox_4.setEnabled(False)

        self.checkBox.setStyleSheet('''QCheckBox::indicator:checked { background-color: rgb(0,255,0) }''')
        self.checkBox_2.setStyleSheet('''QCheckBox::indicator:checked { background-color: rgb(0,255,0) }''')
        self.checkBox_3.setStyleSheet('''QCheckBox::indicator:checked { background-color: rgb(0,255,0) }''')
        self.checkBox_4.setStyleSheet('''QCheckBox::indicator:checked { background-color: rgb(0,255,0) }''')

        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QRect(660, 330, 201, 80))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_5.setGeometry(QRect(110, 10, 60, 60))
        self.pushButton_5.setStyleSheet("border-radius : 30; border : 2px solid white")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_5.sizePolicy().hasHeightForWidth())
        self.pushButton_5.setSizePolicy(sizePolicy)
        self.pushButton_5.setStyleSheet(
            '''
            QPushButton{image:url(./image/screenshots.png); border:0px;}
            QPushButton:hover{image:url(./image/screenshotshover.png); border:0px;}
            #QPushButton:checked{image:url(./image/screenshotsing.png); border:0px;}
            ''')
        self.pushButton_5.setCheckable(False)
        self.pushButton_5.setObjectName("pushButton_5")

        self.pushButton_6 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_6.setGeometry(QRect(20, 10, 60, 60))
        self.pushButton_6.setStyleSheet("border-radius : 30; border : 2px solid white")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_6.sizePolicy().hasHeightForWidth())
        self.pushButton_6.setSizePolicy(sizePolicy)
        self.pushButton_6.setStyleSheet(
            '''
            QPushButton{image:url(./image/power.png); border:0px;}
            QPushButton:hover{image:url(./image/powerhover.png); border:0px;}
            QPushButton:checked{image:url(./image/powering.png); border:0px;}
            ''')
        self.pushButton_6.setCheckable(True)
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.raise_()

        self.pushButton_5.clicked.connect(self.screenshot)
        self.pushButton_5.raise_()

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QRect(660, 430, 200, 60))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("./image/인바디.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QRect(10, 20, 640, 480))
        self.label_2.setPixmap(QtGui.QPixmap("./image/default.jpg")) ## <-------------- 비디오 프레임이 들어가야함
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 870, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.pushButton.toggled.connect(lambda : self.togglebutton(MainWindow, integer=0))
        self.pushButton_2.toggled.connect(lambda : self.togglebutton(MainWindow, integer=1))
        self.pushButton_3.toggled.connect(lambda : self.togglebutton(MainWindow, integer=2))
        self.pushButton_4.toggled.connect(lambda : self.togglebutton(MainWindow, integer=3))

        self.thread = opcv()

        self.pushButton_6.toggled.connect(lambda: self.checked(MainWindow))
        self.button6_checked.connect(self.thread.send_img)

        self.thread.change_pixmap_signal.connect(self.update_img)

        self.thread.start()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Handtracking"))
        self.groupBox.setTitle(_translate("MainWindow", "모드선택"))
        self.pushButton.setText(_translate("MainWindow", "Mode 1"))
        self.pushButton_2.setText(_translate("MainWindow", "Mode 2"))
        self.pushButton_3.setText(_translate("MainWindow", "Mode 3"))
        self.pushButton_4.setText(_translate("MainWindow", "Mode 4"))
        self.groupBox_2.setTitle(_translate("MainWindow", "활성 기능"))
        self.checkBox.setText(_translate("MainWindow", "마우스 움직임"))
        self.checkBox_2.setText(_translate("MainWindow", "마우스 클릭"))
        self.checkBox_3.setText(_translate("MainWindow", "드래그"))
        self.checkBox_4.setText(_translate("MainWindow", "스크롤"))


    def togglebutton(self, MainWindow, integer):
        btn1 = self.pushButton
        btn2 = self.pushButton_2
        btn3 = self.pushButton_3
        btn4 = self.pushButton_4
        Button_list = [btn1, btn2, btn3, btn4]
        self.checkBox.setEnabled(True)
        self.checkBox_2.setEnabled(True)
        self.checkBox_3.setEnabled(True)
        self.checkBox_4.setEnabled(True)
        if Button_list[integer].isChecked():
            Button_list.pop(integer)
            for button in Button_list:
                if button.isChecked():
                    button.toggle()
            if integer == 0:
                self.checkBox.setChecked(False)
                self.checkBox_2.setChecked(False)
                self.checkBox_3.setChecked(False)
                self.checkBox_4.setChecked(False)
            elif integer == 1:
                self.checkBox.setChecked(True)
                self.checkBox_2.setChecked(True)
                self.checkBox_3.setChecked(False)
                self.checkBox_4.setChecked(False)
            elif integer == 2:
                self.checkBox.setChecked(True)
                self.checkBox_2.setChecked(False)
                self.checkBox_3.setChecked(True)
                self.checkBox_4.setChecked(False)
            elif integer == 3:
                self.checkBox.setChecked(True)
                self.checkBox_2.setChecked(True)
                self.checkBox_3.setChecked(True)
                self.checkBox_4.setChecked(True)
            else:
                pass
        else:
            self.checkBox.setChecked(False)
            self.checkBox_2.setChecked(False)
            self.checkBox_3.setChecked(False)
            self.checkBox_4.setChecked(False)
        self.checkBox.setEnabled(False)
        self.checkBox_2.setEnabled(False)
        self.checkBox_3.setEnabled(False)
        self.checkBox_4.setEnabled(False)

    def screenshot(self):
        print('clicked')
        now = datetime.datetime.now().strftime("%d_%H-%M-%S")
        filename = './screenshot/' + str(now) + ".jpg"
        print(filename)
        image = self.label_2.pixmap()
        image.save(filename, 'jpg')

    button6_checked = pyqtSignal(bool)

    def cvt_qt(self, img):
        # rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv 이미지 파일 rgb 색계열로 바꿔주기
        h, w, ch = img.shape  # image 쉐입 알기
        bytes_per_line = ch * w  # 차원?
        convert_to_Qt_format = QtGui.QImage(img.data, w, h, bytes_per_line,
                                            QtGui.QImage.Format_RGB888)  # qt 포맷으로 바꾸기
        p = convert_to_Qt_format.scaled(640, 480, QtCore.Qt.KeepAspectRatio)  # 디스클레이 크기로 바꿔주기.

        return QtGui.QPixmap.fromImage(p)  # 진정한 qt 이미지 생성

    @pyqtSlot(np.ndarray)
    def update_img(self, img):
        qt_img = self.cvt_qt(img)
        self.label_2.setPixmap(qt_img)

    def checked(self, MainWindow):
        if self.pushButton_6.isChecked():
            print('checked')
            self.button6_checked.emit(True)
        else:
            self.button6_checked.emit(False)
            self.label_2.setPixmap(QtGui.QPixmap("./image/default.jpg"))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
