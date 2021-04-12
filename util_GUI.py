import cv2
import mediapipe as mp
import pyautogui
import math
import win32api, win32con, time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from tensorflow import keras

import tensorflow as tf
from mediapipe.framework.formats import location_data_pb2

#from GUI import opcv, Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, QObject, QRect, pyqtSlot, pyqtSignal
import datetime
import sys

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    #print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# For webcam input:
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

x_size, y_size = pyautogui.size().width, pyautogui.size().height

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

x_size, y_size = pyautogui.size().width, pyautogui.size().height
nowclick = False
nowclick2 = False

gesture_int = 0

MOUSE_USE = False
CLICK_USE = False
WHEEL_USE = False
DRAG_USE = False
USE_TENSORFLOW = True

MODEL = keras.models.load_model(
    'keras_util/model_save/my_model_21.h5'
)

'''
mark_pixel : 각각의 랜드마크
finger_open : 손 하나가 갖고있는 랜드마크들
Gesture : 손의 제스처를 판단하기 위한 랜드마크들의 Queue
'''

#TODO 손가락 굽힘 판단, 손바닥 상태, 오른손 왼손 확인
class Handmark():
    def __init__(self, mark_p):
        self._p_list = mark_p
        self.finger_state = [0 for _ in range(5)]

    @property
    def p_list(self):
        return self._p_list

    @p_list.setter
    def p_list(self, new_p):
        self._p_list = new_p

    def return_flatten_p_list(self):
        output = []
        for local_mark_p in self._p_list[:-1]:
            #print('type', type(local_mark_p))
            output.extend(local_mark_p.to_list())
        return output


    #엄지 제외
    def get_finger_angle(self, finger): #finger는 self에서 정의된 손가락들, 4크기 배열
        l1 = finger[0] - finger[1]
        l2 = finger[3] - finger[1]
        l1_ = np.array([l1[0], l1[1], l1[2]])
        l2_ = np.array([l2[0], l2[1], l2[2]])
        return np.arccos(np.dot(l1_, l2_) / (norm(l1) * norm(l2)))

    def get_angle(self, l1, l2):
        l1_ = np.array([l1[0], l1[1], l1[2]])
        l2_ = np.array([l2[0], l2[1], l2[2]])
        return np.arccos(np.dot(l1_, l2_) / (norm(l1) * norm(l2)))

    def get_finger_angle_thumb(self, finger):
        l1 = finger[0] - finger[1]
        l2 = finger[1] - finger[2]
        return self.get_angle(l1, l2)

    def get_palm_vector(self):
        l1 = self._p_list[17] - self._p_list[0]
        l2 = self._p_list[5] - self._p_list[0]
        l1_ = np.array([l1[0], l1[1], l1[2]])
        l2_ = np.array([l2[0], l2[1], l2[2]])

        self.palm_vector = np.cross(l1_, l2_)
        self.palm_vector = self.palm_vector / vector_magnitude(self.palm_vector)
        #print(vector_magnitude((self.palm_vector)))
        return self.palm_vector

    def get_finger_vector(self):
        l0 = self._p_list[5] - self._p_list[0]
        self.finger_vector = np.array(l0)
        self.finger_vector = self.finger_vector / vector_magnitude(self.finger_vector)
        #print(vector_magnitude((self.finger_vector)))
        return self.finger_vector


    #True 펴짐 False 내림
    def return_finger_state(self, experiment_mode = False):
        self.thumb = [self._p_list[i] for i in range(1, 5)]
        self.index = [self._p_list[i] for i in range(5, 9)]
        self.middle = [self._p_list[i] for i in range(9, 13)]
        self.ring = [self._p_list[i] for i in range(13, 17)]
        self.pinky = [self._p_list[i] for i in range(17, 21)]

        #TODO 각 손가락 각도 근거로 손가락 굽힘 판단
        self.finger_angle_list = np.array([self.get_finger_angle(self.thumb),
               self.get_finger_angle(self.index),
               self.get_finger_angle(self.middle),
               self.get_finger_angle(self.ring),
               self.get_finger_angle(self.pinky)])
        finger_angle_threshold = np.array([2.8, 1.7, 2.2, 2.2, 2.4])
        self.finger_state_angle = np.array(self.finger_angle_list > finger_angle_threshold, dtype=int)

        #TODO 각 손가락 거리정보 근거로 손가락 굽힘 판단
        self.finger_distance_list = np.array([get_distance(self.thumb[3], self.pinky[0]) / get_distance(self.index[0], self.pinky[0]),
                                     get_distance(self.index[3], self.index[0]) / get_distance(self.index[0], self.index[1]),
                                     get_distance(self.middle[3], self.middle[0]) / get_distance(self.middle[0], self.middle[1]),
                                     get_distance(self.ring[3], self.ring[0]) / get_distance(self.ring[0], self.ring[1]),
                                     get_distance(self.pinky[3], self.pinky[0]) / get_distance(self.pinky[0], self.pinky[1])])
        #print(self.finger_distance_list)
        finger_distance_threshold = np.array([1.5, 2, 2, 2, 2])
        self.finger_state_distance = np.array(self.finger_distance_list > finger_distance_threshold, dtype=int)

        # TODO 손가락과 손바닥 이용해 손가락 굽힘 판단
        self.hand_angle_list = np.array([self.get_angle(self.thumb[1] - self._p_list[0], self.thumb[3] - self.thumb[1]),
                                self.get_angle(self.index[0] - self._p_list[0], self.index[3] - self.index[0]),
                                self.get_angle(self.middle[0] - self._p_list[0], self.middle[3] - self.middle[0]),
                                self.get_angle(self.ring[0] - self._p_list[0], self.ring[3] - self.ring[0]),
                                self.get_angle(self.pinky[0] - self._p_list[0], self.pinky[3] - self.pinky[0])])
        #print(self.hand_angle_list)
        hand_angle_threshold = np.array([0.7, 1.5, 1.5, 1.5, 1.3])
        self.hand_state_angle = np.array(self.hand_angle_list < hand_angle_threshold, dtype=int)
        #print(self.finger_angle_list, self.finger_distance_list, self.hand_angle_list)
        self.input = np.concatenate((self.finger_angle_list, self.finger_distance_list, self.hand_angle_list))

        #print(predict_static(self.input))
        #print(np.round(self.finger_angle_list, 3), np.round(self.finger_distance_list, 3), np.round(self.hand_angle_list, 3))
        #print(self.finger_state_angle, self.finger_state_distance, self.hand_state_angle)

        self.result = self.finger_state_angle + self.finger_state_distance + self.hand_state_angle > 1
        #print(self.result)
        if experiment_mode == False:
            return self.result
        else:
            return np.round(self.finger_angle_list, 3), np.round(self.finger_distance_list, 3), np.round(self.hand_angle_list, 3)

    def return_finger_info(self):
        self.thumb = [self._p_list[i] for i in range(1, 5)]
        self.index = [self._p_list[i] for i in range(5, 9)]
        self.middle = [self._p_list[i] for i in range(9, 13)]
        self.ring = [self._p_list[i] for i in range(13, 17)]
        self.pinky = [self._p_list[i] for i in range(17, 21)]

        # TODO 각 손가락 각도 근거로 손가락 굽힘 판단
        self.finger_angle_list = np.array([self.get_finger_angle(self.thumb),
                                           self.get_finger_angle(self.index),
                                           self.get_finger_angle(self.middle),
                                           self.get_finger_angle(self.ring),
                                           self.get_finger_angle(self.pinky)])

        # TODO 각 손가락 거리정보 근거로 손가락 굽힘 판단
        self.finger_distance_list = np.array([get_distance(self.thumb[3], self.pinky[0]) / get_distance(self.index[0], self.pinky[0]),
                                              get_distance(self.index[3], self.index[0]) / get_distance(self.index[0], self.index[1]),
                                              get_distance(self.middle[3], self.middle[0]) / get_distance(self.middle[0], self.middle[1]),
                                              get_distance(self.ring[3], self.ring[0]) / get_distance(self.ring[0], self.ring[1]),
                                              get_distance(self.pinky[3], self.pinky[0]) / get_distance(self.pinky[0], self.pinky[1])])
        # print(self.finger_distance_list)

        # TODO 손가락과 손바닥 이용해 손가락 굽힘 판단
        self.hand_angle_list = np.array([self.get_angle(self.thumb[1] - self._p_list[0], self.thumb[3] - self.thumb[1]),
                                         self.get_angle(self.index[0] - self._p_list[0], self.index[3] - self.index[0]),
                                         self.get_angle(self.middle[0] - self._p_list[0], self.middle[3] - self.middle[0]),
                                         self.get_angle(self.ring[0] - self._p_list[0], self.ring[3] - self.ring[0]),
                                         self.get_angle(self.pinky[0] - self._p_list[0], self.pinky[3] - self.pinky[0])])
        # print(self.hand_angle_list)
        # print(self.finger_angle_list, self.finger_distance_list, self.hand_angle_list)
        self.input = np.concatenate((self.finger_angle_list, self.finger_distance_list, self.hand_angle_list))
        return self.input

    def return_18_info(self):
        output = self.return_finger_info()
        output = np.concatenate((output, self.palm_vector))
        return output

    def return_21_info(self):
        output = self.return_finger_info()
        output = np.concatenate((output, self.palm_vector, self.finger_vector))
        return output

    # def predict_static(self):
    #     self.input = self.input[np.newaxis]
    #     # print(input.shape)
    #     # print(input)
    #     prediction = model.predict(self.input)
    #     if np.max(prediction[0]) > 0.75:
    #         return np.argmax(prediction[0])
    #     else:
    #         return 0

#TODO Gesture 판단, 일단은 15프레임 (0.5초)의 Queue로?
class Gesture():
    Gesture_Array_size = 15

    def __init__(self):
        self.palm_data = [np.array([0, 0, 0]) for _ in range(Gesture.Gesture_Array_size)]
        self.d_palm_data = [np.array([0, 0, 0]) for _ in range(Gesture.Gesture_Array_size)] #palm_data의 차이를 기록할 list

        self.location_data = [np.array([0, 0, 0]) for _ in range(Gesture.Gesture_Array_size)]
        self.finger_data  = [np.array([0, 0, 0, 0, 0]) for _ in range(Gesture.Gesture_Array_size)]

    def update(self, handmark):
        self.palm_data.insert(0, handmark.palm_vector)
        self.d_palm_data.insert(0, (self.palm_data[1] - handmark.palm_vector) * 1000)
        self.location_data.insert(0, handmark._p_list)
        self.finger_data.insert(0, handmark.finger_state)
        #print(self.palm_data)
        #print(self.location_data)
        #print(self.finger_data)
        self.palm_data.pop()
        self.d_palm_data.pop()
        self.location_data.pop()
        self.finger_data.pop()
        self.fv = handmark.finger_vector

        #print(handmark.palm_vector * 1000)

    # handmark지닌 10개의 프레임이 들어온다...
    def gesture_detect(self): #이 최근꺼
        hand_open_frame = 0
        Z_rotate = 0
        Z_rotate_inv = 0
        x_diff = 0
        global gesture_int
        global gesture_time
        global image

        #print(self.d_palm_data[0], self.finger_data[0], self.location_data[0])

        for i in range(Gesture.Gesture_Array_size - 1):
            if sum(self.finger_data[i]) > 4:
                hand_open_frame += 1
            if self.d_palm_data[i][2] > 1.5:
                Z_rotate += 1
            if self.d_palm_data[i][2] < -1.5:
                Z_rotate_inv += 1
            #if self.location_data[i+1][1] - self.location_data[i][1]
            try:
                if self.location_data[i+1][5].x - self.location_data[i][5].x > 0.005:
                    x_diff += 1
                elif self.location_data[i+1][5].x - self.location_data[i][5].x < -0.005:
                    x_diff -= 1

            except:
                pass
            if gesture_int == 0 and abs(self.fv[1]) < 100:
                if Z_rotate > 2 and hand_open_frame > 6 and x_diff < -3:
                    print('To Right Sign!!')
                    #image = cv2.putText(image, 'To right', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                    win32api.keybd_event(0x27, 0, 0, 0)
                    gesture_int += 1
                    gesture_time = time.time()

                elif Z_rotate_inv > 2 and hand_open_frame > 6 and x_diff > 3:
                    print('To Left Sign!!')
                    #image = cv2.putText(image, 'To left', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                    win32api.keybd_event(0x25, 0, 0, 0)
                    gesture_int += 1
                    gesture_time = time.time()
                    break


        #print(Z_rotate, Z_rotate_inv, hand_open_frame, x_diff, self.fv[1])
        # print(np.array(self.d_palm_data)[:][2])

class Gesture_mode():
    QUEUE_SIZE = 10
    def __init__(self):
        self.left = [0] * self.QUEUE_SIZE
        self.right = [0] * self.QUEUE_SIZE
        self.left_palm_vector = [[0.] * 3] * self.QUEUE_SIZE
        self.right_palm_vector = [[0.] * 3] * self.QUEUE_SIZE
        self.left_finger_vector = [[0.] * 3] * self.QUEUE_SIZE
        self.right_finger_vector = [[0.] * 3] * self.QUEUE_SIZE
        self.right_finger_vector = [[0.] * 3] * self.QUEUE_SIZE
    def __str__(self):
        return 'left : {}, right : {}, lpv : {}, lfv : {}, rpv : {}, rfv : {}'.format(
            self.left[-1], self.right[-1],
            self.left_palm_vector[-1], self.left_finger_vector[-1], self.right_palm_vector[-1], self.right_finger_vector[-1])
    def update_left(self, left, palm_vector, finger_vector):
        self.left.append(left)
        self.left_palm_vector.append(palm_vector)
        self.left_finger_vector.append(finger_vector)
        self.left.pop(0)
        self.left_palm_vector.pop(0)
        self.left_finger_vector.pop(0)
    def update_right(self, right, palm_vector, finger_vector):
        self.right.append(right)
        self.right_palm_vector.append(palm_vector)
        self.right_finger_vector.append(finger_vector)
        self.right.pop(0)
        self.right_palm_vector.pop(0)
        self.right_finger_vector.pop(0)
    def select_mode(self):
        mode = 0
        lpv_mode_1 = [-0.44, 0.115, -0.89]
        lfv_mode_1 = [-0.28, -0.95, 0.]
        rpv_mode_1 = [-0.55, -0.2, 0.8]
        rfv_mode_1 = [0.28, -0.96, -0.05]
        mode_result = 0
        for lpv in self.left_palm_vector:
            mode_result = mode_result + get_angle(lpv, lpv_mode_1)
        for lfv in self.left_finger_vector:
            mode_result = mode_result + get_angle(lfv, lfv_mode_1)
        for rpv in self.right_palm_vector:
            mode_result = mode_result + get_angle(rpv, rpv_mode_1)
        for rfv in self.right_finger_vector:
            mode_result = mode_result + get_angle(rfv, rfv_mode_1)

        # 손바닥 펴서 앞에 보여주기
        left_idx_1 = 0
        for left in self.left:
            if left == 6:
                left_idx_1 += 1
        right_idx_1 = 0
        for right in self.right:
            if right == 6:
                right_idx_1 += 1
        if mode_result < 10 and left_idx_1 == 10 and right_idx_1 == 10:
            mode = 1

        # 탈모빔 자세
        left_idx_1 = 0
        for left in self.left:
            if left == 3:
                left_idx_1 += 1
        right_idx_1 = 0
        for right in self.right:
            if right == 3:
                right_idx_1 += 1
        if mode_result < 17 and left_idx_1 == 10 and right_idx_1 == 10:
            mode = 2

        # 손모양 삼 자세
        left_idx_1 = 0
        for left in self.left:
            if left == 4:
                left_idx_1 += 1
        right_idx_1 = 0
        for right in self.right:
            if right == 4:
                right_idx_1 += 1
        if mode_result < 17 and left_idx_1 == 10 and right_idx_1 == 10:
            mode = 3

        # 손모양 사 자세
        left_idx_1 = 0
        for left in self.left:
            if left == 7:
                left_idx_1 += 1
        right_idx_1 = 0
        for right in self.right:
            if right == 7:
                right_idx_1 += 1
        if mode_result < 17 and left_idx_1 == 10 and right_idx_1 == 10:
            mode = 4

        return mode

def get_angle(l1, l2):
    l1_ = np.array([l1[0], l1[1], l1[2]])
    l2_ = np.array([l2[0], l2[1], l2[2]])
    if (norm(l1) * norm(l2)) == 0:
        return 0
    else:
        return np.arccos(np.dot(l1_, l2_) / (norm(l1) * norm(l2)))
    #return np.arccos(np.dot(l1_, l2_) / (norm(l1) * norm(l2)))

def vector_magnitude(one_D_array):
    return math.sqrt(np.sum(one_D_array * one_D_array))

def norm(p1):
    return math.sqrt((p1[0])**2 + (p1[1])**2 + (p1[2])**2)

def convert_offset(x, y):
    return x*4/3 - x_size/8, y*4/3 - y_size/8

def inv_convert_off(x, y):
    return (x + x_size/8)*3/4, (y + y_size/8)*3/4

def get_distance(p1, p2):
    try:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    except:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

#TODO 프로세스 함수들
def process_static_gesture(array_for_static, value_for_static):
    while(True):
        input_ = np.copy(array_for_static[:])
        #print(input_)
        input_ = input_[np.newaxis]
        try:
            prediction = MODEL.predict(input_)
            if np.max(prediction[0]) > 0.4:
                value_for_static.value = np.argmax(prediction[0])
            else:
                value_for_static.value = 0
        except:
            pass

def initialize(array_for_static_l, value_for_static_l, array_for_static_r, value_for_static_r, test_np_array = []):

    global image
    global MOUSE_USE
    global CLICK_USE
    global WHEEL_USE
    global DRAG_USE


    #GUI Part
    class opcv(QThread):

        change_pixmap_signal = pyqtSignal(np.ndarray)
        mode_signal = pyqtSignal(int)

        def __init__(self):
            super().__init__()

        @pyqtSlot(int, int)
        def mode_setting(self, mode, mode_before): #1
            global MOUSE_USE, CLICK_USE, DRAG_USE, WHEEL_USE
            if mode != mode_before:
                self.mode_signal.emit(int(mode - 1)) #2 / #2-4
                if mode == 1:
                    MOUSE_USE = False
                    CLICK_USE = False
                    DRAG_USE = False
                    WHEEL_USE = False
                    print('MODE 1, 대기 모드')
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 100, 100, 0, 0)

                if mode == 2:
                    MOUSE_USE = True
                    CLICK_USE = True
                    DRAG_USE = False
                    WHEEL_USE = False
                    print('MODE 2, 기본 발표 모드')

                if mode == 3:
                    MOUSE_USE = True
                    CLICK_USE = False
                    DRAG_USE = True
                    WHEEL_USE = False
                    win32api.keybd_event(0xa2, 0, 0, 0)  # LEFT CTRL 누르기.
                    win32api.keybd_event(0x50, 0, 0, 0)  # P 누르기.
                    time.sleep(0.1)
                    win32api.keybd_event(0xa2, 0, win32con.KEYEVENTF_KEYUP, 0)
                    win32api.keybd_event(0x50, 0, win32con.KEYEVENTF_KEYUP, 0)
                    print('MODE 3, 필기 발표 모드') #3, #2-5

                if mode == 4:
                    MOUSE_USE = True
                    CLICK_USE = True
                    DRAG_USE = True
                    WHEEL_USE = True
                    print('MODE 4, 웹서핑 발표 모드')

        @pyqtSlot(bool)
        def send_img(self, bool_state):  # p를 보는 emit 함수
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap = self.capture
            # For webcam input:
            hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

            global width, height

            # TODO landmark를 대응 인스턴스로 저장
            class Mark_pixel():
                def __init__(self, x, y, z=0, LR=0):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.LR = LR

                def __str__(self):
                    return str(self.x) + '   ' + str(self.y) + '   ' + str(self.z)

                def to_list(self):
                    return [self.x, self.y, self.z]

                def to_pixel(self):
                    global x_size
                    global y_size
                    return Mark_2d(self.x * x_size, self.y * y_size)

                def __sub__(self, other):
                    return self.x - other.x, self.y - other.y, self.z - other.z

            class Mark_2d():
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def __str__(self):
                    return tuple(self.x, self.y)

                def mousemove(self):
                    if nowclick == True:
                        cv2.circle(image, (int(self.x / 3), int(self.y / 2.25)), 5, (0, 0, 255), -1)
                    else:
                        cv2.circle(image, (int(self.x / 3), int(self.y / 2.25)), 5, (255, 255, 0), -1)
                    # self.x, self.y = convert_offset(self.x, self.y)
                    win32api.SetCursorPos((int(self.x), int(self.y)))

                def wheel_up(self):
                    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 200, 200, 30, 1)

                def wheel_down(self):
                    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 200, 200, -30, 1)

                def wheel(self, before):
                    if self.y > before.y:
                        self.wheel_up()
                        print('wheel_up')
                    if self.y < before.y:
                        self.wheel_down()
                        print('wheel_down')

            # def get_center(p1, p2):
            #     return Mark_pixel((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)

            def hand_drag(landmark, pixel):
                x = pixel.x
                y = pixel.y
                # print(x, y)
                global nowclick
                if get_distance(landmark[4], landmark[8]) < get_distance(landmark[4], landmark[3]) and nowclick == False:
                    print('drag on')
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, int(x), int(y), 0, 0)
                    # ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
                    nowclick = True

                elif get_distance(landmark[4], landmark[8]) > 1.5 * get_distance(landmark[4], landmark[3]) and nowclick == True:
                    print('drag off')
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, int(x), int(y), 0, 0)
                    # ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                    nowclick = False

            def hand_click(landmark, pixel):
                x = pixel.x
                y = pixel.y
                global nowclick2

                if get_distance(landmark[4], landmark[10]) < get_distance(landmark[7], landmark[8]) and nowclick2 == False:
                    print('click')
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, int(x), int(y), 0, 0)
                    print('click off')
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, int(x), int(y), 0, 0)
                    nowclick2 = True
                    return -1
                if get_distance(landmark[4], landmark[10]) > get_distance(landmark[7], landmark[8]) and nowclick2 == True:
                    nowclick2 = False

                return 0

            hand_shape_dateset = [[0, 0, 0, 0, 0],  # 바위
                                  [1, 0, 0, 0, 0],  # 따봉
                                  [1, 1, 0, 0, 0],  # 총
                                  [1, 1, 1, 0, 0],  # 3-1
                                  [1, 1, 1, 1, 0],  # 4-1
                                  [1, 1, 1, 1, 1],  # 보
                                  [0, 1, 1, 1, 1],  # 4-2
                                  [0, 0, 1, 1, 1],  # 3-2
                                  [0, 0, 0, 1, 1],  # 2
                                  [0, 0, 0, 0, 1],  # 1-2
                                  [0, 1, 1, 0, 0],  # 가위
                                  [1, 1, 0, 0, 1],  # 스파이더맨
                                  [0, 1, 0, 0, 0],  # 1-1
                                  [0, 1, 1, 1, 0],  # 3-3
                                  [0, 0, 1, 0, 0],  # Fuck you
                                  ]
            hand_shape_name = ['바위', '따봉', '총', '3-1', '4-1', '보', '4-2', '3-2', '2', '1-2', '가위', '스파이더맨', '1-1', '3-3', 'Fuck you']

            def static_gesture_detect(finger_open_, LR_index):
                hsd = hand_shape_dateset.copy()
                for i in range(len(hsd)):
                    hsd[i] = list(map(bool, hsd[i]))  # boolean 원소로 지닌 list로 변환

                global image
                if finger_open_ in hsd:
                    # print(hand_shape_name[hand_shape_dateset.index(finger_open_)])

                    pill_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pill_image)
                    # print(LR_index, type(LR_index))
                    if len(LR_index) == 4:
                        x1, y1 = 30, 30
                    # LR_index == 'left'
                    elif len(LR_index) == 5:
                        x1, y1 = 550, 30
                    else:
                        x1, y1 = 30, 30

                    # x1, y1 = 30, 30

                    text = hand_shape_name[hsd.index(finger_open_)]
                    draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 36), fill=(255, 0, 0))
                    image = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)  # 맥
                    try:
                        return hsd.index(finger_open_)
                    except:
                        pass
                else:
                    return -1

            def static_gesture_drawing(gesture_num, LR_index):
                global image
                if gesture_num != 0:
                    pill_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pill_image)

                    if LR_index[0] == 'L':
                        x1, y1 = 30, 30
                    # LR_index == 'left'
                    elif LR_index[0] == 'R':
                        x1, y1 = 550, 30
                    else:
                        x1, y1 = 30, 30

                    # x1, y1 = 30, 30

                    text = hand_shape_name[gesture_num - 1]
                    draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 36), fill=(255, 0, 0))
                    image = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)  # 맥

            def BlurFunction(src):
                with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:  # with 문, mp_face_detection.FaceDetection 클래스를 face_detection으로서 사용
                    image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # image 파일의 BGR 색상 베이스를 RGB 베이스로 바꾸기
                    results = face_detection.process(image)  # 튜플 형태
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image_rows, image_cols, _ = image.shape
                    c_mask: ndarray = np.zeros((image_rows, image_cols), np.uint8)
                    if results.detections:
                        for detection in results.detections:
                            if not detection.location_data:
                                break
                            if image.shape[2] != 3:
                                raise ValueError('Input image must contain three channel rgb data.')
                            location = detection.location_data
                            if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
                                raise ValueError('LocationData must be relative for this drawing funtion to work.')
                            # Draws bounding box if exists.
                            if not location.HasField('relative_bounding_box'):
                                break
                            relative_bounding_box = location.relative_bounding_box
                            rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
                                relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                                image_rows)
                            rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                                relative_bounding_box.xmin + relative_bounding_box.width,
                                relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
                                image_rows)
                            try:
                                x1 = int((rect_start_point[0] + rect_end_point[0]) / 2)
                                y1 = int((rect_start_point[1] + rect_end_point[1]) / 2)
                                a = int(rect_end_point[0] - rect_start_point[0])
                                b = int(rect_end_point[1] - rect_start_point[1])
                                radius = int(math.sqrt(a * a + b * b) / 2 * 0.7)
                                # 원 스펙 설정
                                cv2.circle(c_mask, (x1, y1), radius, (255, 255, 255), -1)
                            except:
                                pass
                        img_all_blurred = cv2.blur(image, (17, 17))
                        c_mask = cv2.cvtColor(c_mask, cv2.COLOR_GRAY2BGR)
                        # print(c_mask.shape)
                        image = np.where(c_mask > 0, img_all_blurred, image)
                return image


            gesture_int = 0

            before_c = Mark_pixel(0, 0, 0)
            pixel_c = Mark_pixel(0, 0, 0)
            hm_idx = False
            finger_open_ = [False for _ in range(5)]
            gesture_time = time.time()
            gesture = Gesture()
            gesture_mode = Gesture_mode()

            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
            before_time = time.time()

            click_tr = 0
            mode_before = 0
            p_key_ready = False
            mode = 0

            while bool_state and cap.isOpened():
                #print('cam')
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image = BlurFunction(image)

                # x_size, y_size, channel = image.shape
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    mark_p_list = []
                    for hand_landmarks in results.multi_hand_landmarks:  # hand_landmarks는 감지된 손의 갯수만큼의 원소 수를 가진 list 자료구조
                        mark_p = []
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        for i in range(21):
                            mark_p.append(Mark_pixel(hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z))
                        mark_p_list.append(mark_p)

                    # TODO 지금 API에서 사용하는 자료형때문에 살짝 꼬였는데 mark_p(list)의 마지막 원소를 lR_idx(left or right)로 표현해놨음.
                    for i in range(len(mark_p_list)):  # for문 한 번 도는게 한 손에 대한 것임
                        LR_idx = results.multi_handedness[i].classification[0].label
                        #LR_idx = LR_idx[:-1]
                        # print(LR_idx)
                        image = cv2.putText(image, LR_idx[:], (int(mark_p_list[i][17].x * image.shape[1]), int(mark_p_list[i][17].y * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                        mark_p_list[i].append(LR_idx)

                        mark_p = mark_p_list[i]
                        # Handmark 정보 입력

                        if len(mark_p) == 22 and hm_idx == False:
                            HM = Handmark(mark_p)
                            hm_idx = True

                        # palm_vector 저장
                        palm_vector = HM.get_palm_vector()
                        finger_vector = HM.get_finger_vector()

                        # mark_p 입력
                        if hm_idx == True:
                            HM.p_list = mark_p
                            # mark_p[-1] = mark_p[-1][:-1]
                            if USE_TENSORFLOW == True:

                                if len(mark_p[-1]) == 4:
                                    f_p_list = HM.return_18_info()
                                    array_for_static_l[:] = f_p_list
                                    # print(array_for_static)
                                    static_gesture_num = value_for_static_l.value
                                if len(mark_p[-1]) == 5:
                                    f_p_list = HM.return_18_info()
                                    array_for_static_r[:] = f_p_list
                                    # print(array_for_static)
                                    static_gesture_num = value_for_static_r.value
                                # try:
                                #     static_gesture_drawing(static_gesture_num, mark_p[-1])
                                # except:
                                #     print('static_drawing error')
                                # print(static_gesture_num)
                            else:
                                finger_open_for_ml = np.ndarray.tolist(HM.return_finger_state())
                                # 정지 제스쳐 확인
                                #static_gesture_detect(finger_open_for_ml, mark_p[-1])
                            finger_open_ = HM.return_finger_state()

                        mark_p0 = mark_p[0].to_pixel()
                        mark_p5 = mark_p[5].to_pixel()

                        # pixel_c = mark_c.to_pixel()
                        if len(mark_p[-1]) == 5:
                            pixel_c = mark_p5
                            # gesture updating
                            gesture.update(HM)
                            if len(mark_p) == 22:
                                gesture.gesture_detect()
                                pass

                        if len(mark_p[-1]) == 4:
                            gesture_mode.update_left(static_gesture_num, palm_vector, finger_vector)
                        if len(mark_p[-1]) == 5:
                            gesture_mode.update_right(static_gesture_num, palm_vector, finger_vector)

                        pixel_c_mod = pixel_c

                        # 마우스 움직임, 드래그
                        if (get_distance(pixel_c, before_c) < get_distance(mark_p0, mark_p5)) and \
                                sum(finger_open_[3:]) == 0 and \
                                finger_open_[1] == 1 and \
                                len(mark_p[-1]) == 5 and \
                                MOUSE_USE == True:
                            pixel_c.mousemove()

                            if finger_open_[2] != 1 and click_tr > -1 and DRAG_USE == True:
                                hand_drag(hand_landmarks.landmark, pixel_c)

                            if finger_open_[2] != 1 and CLICK_USE == True:
                                if nowclick != True:
                                    click_tr = hand_click(hand_landmarks.landmark, pixel_c)
                            # else:
                            #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, int(pixel_c.x), int(pixel_c.y), 0, 0)

                            # 마우스 휠
                            if finger_open_[2] == 1 and WHEEL_USE == True:
                                pixel_c.wheel(before_c)

                        if finger_open_[2] != 1 and \
                                CLICK_USE == True and \
                                finger_open_[1] == 1 and \
                                len(mark_p[-1]) == 5 and \
                                sum(finger_open_[2:4]) == 0 and \
                                static_gesture_num == 12 and \
                                p_key_ready == False:
                            p_key_ready = True

                        if finger_open_[2] != 1 and \
                                CLICK_USE == True and \
                                finger_open_[1] == 1 and \
                                len(mark_p[-1]) == 5 and \
                                sum(finger_open_[2:4]) == 0 and \
                                static_gesture_num != 12 and \
                                p_key_ready == True:
                            p_key_ready = False

                            win32api.keybd_event(0x50, 0, 0, 0)  # P 누르기.
                            win32api.keybd_event(0x50, 0, win32con.KEYEVENTF_KEYUP, 0)
                            time.sleep(0.1)

                        before_c = pixel_c

                    mode = gesture_mode.select_mode()
                    self.mode_setting(mode, mode_before)
                    mode_before = mode
                    # mode2 = self.inv_push_button()
                    # if mode2 != None :
                    #     mode_setting(mode2, mode_before)
                    #     mode_before = mode2

                FPS = round(1 / (time.time() - before_time), 2)
                # print(FPS)
                before_time = time.time()
                image = cv2.putText(image, str(FPS), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                # test_np_array = np.array(image).tolist()
                # print(image.shape)
                # print(np.array(image).shape)
                # print(len(test_np_array[0]))
                # print(len(test_np_array))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # cv2.imshow('Gesture_Detection_Hail Song', image)
                self.change_pixmap_signal.emit(image)
                if cv2.waitKey(5) & 0xFF == 27:
                    print('exitcode : 100')
                    exit()
                    break

            hands.close()
            self.capture.release()

    class Ui_MainWindow(QObject):

        click_mode = pyqtSignal(int, int)
        button6_checked = pyqtSignal(bool)

        def setupUi(self):
            self.MainWindow = MyWindow()
            MainWindow = self.MainWindow
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(870, 550)
            self.From_button = False

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
            self.pushButton.setStyleSheet("border-radius : 30; border : 2px solid white")
            self.pushButton.setStyleSheet(
                '''
                QPushButton{image:url(./image/Mode1.png); border:0px;}
                QPushButton:hover{image:url(./image/Mode1hover.png); border:0px;}
                QPushButton:checked{image:url(./image/Mode1ing.png); border:0px;}
                ''')
            self.pushButton.setCheckable(True)
            self.pushButton.setObjectName("pushButton")

            self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
            self.pushButton_2.setGeometry(QRect(10, 90, 61, 61))
            font = QtGui.QFont()
            font.setBold(True)
            font.setWeight(75)
            self.pushButton_2.setFont(font)
            self.pushButton_2.setStyleSheet("border-radius : 30; border : 2px solid white")
            self.pushButton_2.setStyleSheet(
                '''
                QPushButton{image:url(./image/Mode2.png); border:0px;}
                QPushButton:hover{image:url(./image/Mode2hover.png); border:0px;}
                QPushButton:checked{image:url(./image/Mode2ing.png); border:0px;}
                ''')
            self.pushButton_2.setCheckable(True)
            self.pushButton_2.setObjectName("pushButton_2")

            self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
            self.pushButton_3.setGeometry(QRect(10, 160, 61, 61))
            font = QtGui.QFont()
            font.setBold(True)
            font.setWeight(75)
            self.pushButton_3.setFont(font)
            self.pushButton_3.setStyleSheet("border-radius : 30; border : 2px solid white")
            self.pushButton_3.setStyleSheet(
                '''
                QPushButton{image:url(./image/Mode3.png); border:0px;}
                QPushButton:hover{image:url(./image/Mode3hover.png); border:0px;}
                QPushButton:checked{image:url(./image/Mode3ing.png); border:0px;}
                ''')
            self.pushButton_3.setCheckable(True)
            self.pushButton_3.setObjectName("pushButton_3")

            self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
            self.pushButton_4.setGeometry(QRect(10, 230, 61, 61))
            font = QtGui.QFont()
            font.setBold(True)
            font.setWeight(75)
            self.pushButton_4.setFont(font)
            self.pushButton_4.setStyleSheet("border-radius : 30; border : 2px solid white")
            self.pushButton_4.setStyleSheet(
                '''
                QPushButton{image:url(./image/Mode4.png); border:0px;}
                QPushButton:hover{image:url(./image/Mode4hover.png); border:0px;}
                QPushButton:checked{image:url(./image/Mode4ing.png); border:0px;}
                ''')
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

            self.pushButton.setEnabled(False)
            self.pushButton_2.setEnabled(False)
            self.pushButton_3.setEnabled(False)
            self.pushButton_4.setEnabled(False)
            self.pushButton_5.setEnabled(False)

            self.label = QtWidgets.QLabel(self.centralwidget)
            self.label.setGeometry(QRect(660, 430, 200, 60))
            self.label.setText("")
            self.label.setPixmap(QtGui.QPixmap("./image/인바디.png"))
            self.label.setScaledContents(True)
            self.label.setObjectName("label")

            self.label_2 = QtWidgets.QLabel(self.centralwidget)
            self.label_2.setGeometry(QRect(10, 20, 640, 480))
            self.label_2.setPixmap(QtGui.QPixmap("./image/default.jpg"))  ## <-------------- 비디오 프레임이 들어가야함
            self.label_2.setScaledContents(True)
            self.label_2.setObjectName("label_2")

            MainWindow.setCentralWidget(self.centralwidget)

            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QRect(0, 0, 870, 21))
            self.menubar.setObjectName("menubar")
            self.MainWindow.setMenuBar(self.menubar)

            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)

            self.pushButton.toggled.connect(lambda: self.togglebutton(MainWindow, integer=0))
            self.pushButton_2.toggled.connect(lambda: self.togglebutton(MainWindow, integer=1))
            self.pushButton_3.toggled.connect(lambda: self.togglebutton(MainWindow, integer=2))
            self.pushButton_4.toggled.connect(lambda: self.togglebutton(MainWindow, integer=3))

            self.thread = opcv()

            self.pushButton_6.toggled.connect(lambda: self.checked(MainWindow))
            self.click_mode.connect(self.thread.mode_setting)
            self.button6_checked.connect(self.thread.send_img)
            MainWindow.power_off_signal.connect(self.thread.send_img)
            self.thread.change_pixmap_signal.connect(self.update_img)
            self.thread.mode_signal.connect(self.push_button)


            self.thread.start()
            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "Handtracking"))
            self.groupBox.setTitle(_translate("MainWindow", "모드선택"))
            #M
            self.groupBox_2.setTitle(_translate("MainWindow", "활성 기능"))
            self.checkBox.setText(_translate("MainWindow", "마우스 움직임"))
            self.checkBox_2.setText(_translate("MainWindow", "마우스 클릭"))
            self.checkBox_3.setText(_translate("MainWindow", "드래그"))
            self.checkBox_4.setText(_translate("MainWindow", "스크롤"))

        @pyqtSlot(int)
        def push_button(self, integer): #2-1
            if integer != -1:
                B_list = [self.pushButton, self.pushButton_2,
                               self.pushButton_3, self.pushButton_4]
                if not B_list[integer].isChecked():
                    self.From_button = True
                    B_list[integer].toggle() # #2-2
            else :
                self.From_button = False
                pass

        def togglebutton(self, MainWindow, integer):
            Button_list = [self.pushButton, self.pushButton_2,
                           self.pushButton_3, self.pushButton_4]
            Before_mode_list = []
            self.checkBox.setEnabled(True)
            self.checkBox_2.setEnabled(True)
            self.checkBox_3.setEnabled(True)
            self.checkBox_4.setEnabled(True)
            if Button_list[integer].isChecked(): #2-3
                Button_list.pop(integer)
                for button in Button_list:
                    if button.isChecked():
                        button.toggle()
                        Before_mode_list.append(button)
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

                if len(Before_mode_list) != 0:
                    if self.From_button == False:
                        if Before_mode_list[0] == self.pushButton:
                            self.click_mode.emit(integer+1, 1)
                        elif Before_mode_list[0] == self.pushButton_2:
                            self.click_mode.emit(integer + 1, 2)
                        elif Before_mode_list[0] == self.pushButton_3:
                            self.click_mode.emit(integer + 1, 3)
                        elif Before_mode_list[0] == self.pushButton_4:
                            self.click_mode.emit(integer + 1, 4)
                    else:
                        self.click_mode.emit(integer + 1, integer + 1)
                else:
                    if self.From_button == False:
                        self.click_mode.emit(integer + 1, 0)
                    else:
                        self.click_mode.emit(integer + 1, integer + 1)
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
                self.pushButton.setEnabled(True)
                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(True)
                self.pushButton_4.setEnabled(True)
                self.pushButton_5.setEnabled(True)
                self.button6_checked.emit(True)
            else:
                self.pushButton.setEnabled(False)
                self.pushButton_2.setEnabled(False)
                self.pushButton_3.setEnabled(False)
                self.pushButton_4.setEnabled(False)
                self.pushButton_5.setEnabled(False)
                self.button6_checked.emit(False)
                Button_list = [self.pushButton, self.pushButton_2,
                               self.pushButton_3, self.pushButton_4]
                for button in Button_list:
                    if button.isChecked():
                        button.toggle()
                self.button6_checked.emit(False)
                self.label_2.setPixmap(QtGui.QPixmap("./image/default.jpg"))

    class MyWindow(QtWidgets.QMainWindow):

        power_off_signal = pyqtSignal(bool)

        def __init__(self):
            super().__init__()
            # self.setStyleSheet('''QMainWindow{background-color : rgb(0, 255, 0);}''')
            self.setStyleSheet('''QMessageBox{background-color: rgb(225, 225, 225);}''')
            self.setStyleSheet('''QMainWindow{background-color : rgb(0, 0, 0);}''')
            self.msg = QMessageBox()
        def closeEvent(self, event):
            result = self.msg.question(self,
                                 "Confirm Exit...",
                                 "Are you sure you want to exit ?",
                                 self.msg.Yes | self.msg.No)
            if result == self.msg.Yes:
                self.power_off_signal.emit(False)
                event.accept()

            else :
                event.ignore()


    # class Gui(QMainWindow):
    #     d
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setupUi()
    ui.MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    print("This is util set program, it works well... maybe... XD")

    print('Running main_18input_GUI.py...')
    from os import system
    system('python main_18input_GUI.py')