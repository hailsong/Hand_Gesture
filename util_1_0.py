import cv2
import mediapipe as mp
import pyautogui
import math
import win32api
import win32con
import time
import numpy as np
from PIL import Image
from tensorflow import keras

import tensorflow as tf

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, QObject, QRect, pyqtSlot, pyqtSignal
import datetime
import sys

'''
키 코드 링크 : https://lab.cliel.com/entry/%EA%B0%80%EC%83%81-Key-Code%ED%91%9C
'''

tf.config.experimental.set_visible_devices([], 'GPU')
# physical_devices = tf.config.list_physical_devices('GPU')
# # print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# For webcam input:
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

x_size, y_size = pyautogui.size().width, pyautogui.size().height

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose

now_click = False
now_click2 = False
straight_line = False
rectangular = False
circle = False

gesture_int = 0

MOUSE_USE = False
CLICK_USE = False
WHEEL_USE = False
DRAG_USE = False
USE_TENSORFLOW = True
USE_DYNAMIC = False
# 왼손잡이 모드 개발 중
REVERSE_MODE = False
THEME = 'b'
BOARD_COLOR = 'w'

VISUALIZE_GRAPH = False

MODEL_STATIC = keras.models.load_model(
    'keras_util/model_save/my_model_21.h5'
)

gesture_check = False
mode_global = 0
pen_color = ''
laser_state = False
laser_num = 0

'''
mark_pixel : 각각의 랜드마크
finger_open : 손 하나가 갖고있는 랜드마크들
Gesture : 손의 제스처를 판단하기 위한 랜드마크들의 Queue
'''


# TODO 손가락 굽힘 판단, 손바닥 상태, 오른손 왼손 확인
class Handmark:
    def __init__(self, mark_p):
        self._p_list = mark_p
        self.finger_state = [0 for _ in range(5)]
        self.palm_vector = np.array([0., 0., 0.])
        self.finger_vector = np.array([0., 0., 0.])
        # self.thumb, self.index, self.middle, self.ring, self.pinky = np.array()
        # self.finger_angle_list = np.array()

    @property
    def p_list(self):
        return self._p_list

    @p_list.setter
    def p_list(self, new_p):
        self._p_list = new_p

    def return_flatten_p_list(self):
        """
        :return: flatten p_list information
        """
        output = []
        for local_mark_p in self._p_list:
            # print('type', type(local_mark_p))
            output.extend(local_mark_p.to_list())
        return output

    # 엄지 제외
    @staticmethod
    def get_finger_angle(finger):
        l1 = finger[0] - finger[1]
        l2 = finger[3] - finger[1]
        l1_ = np.array([l1[0], l1[1], l1[2]])
        l2_ = np.array([l2[0], l2[1], l2[2]])
        return np.arccos(np.dot(l1_, l2_) / (norm(l1) * norm(l2)))

    @staticmethod
    def get_angle(l1, l2):
        """
        :param l1: numpy_vector 1
        :param l2: numpy_vector 2
        :return: angle between l1 and l2
        """
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
        # print(vector_magnitude((self.palm_vector)))
        return self.palm_vector

    def get_finger_vector(self):
        l0 = self._p_list[5] - self._p_list[0]
        self.finger_vector = np.array(l0)
        self.finger_vector = self.finger_vector / vector_magnitude(self.finger_vector)
        # print(vector_magnitude((self.finger_vector)))
        return self.finger_vector

    # True 펴짐 False 내림
    def return_finger_state(self, experiment_mode=False):
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
        finger_angle_threshold = np.array([2.8, 1.5, 2.2, 2.2, 2.4])
        self.finger_state_angle = np.array(self.finger_angle_list > finger_angle_threshold, dtype=int)

        # TODO 각 손가락 거리정보 근거로 손가락 굽힘 판단
        self.finger_distance_list = np.array(
            [get_distance(self.thumb[3], self.pinky[0]) / get_distance(self.index[0], self.pinky[0]),
             get_distance(self.index[3], self.index[0]) / get_distance(self.index[0], self.index[1]),
             get_distance(self.middle[3], self.middle[0]) / get_distance(self.middle[0], self.middle[1]),
             get_distance(self.ring[3], self.ring[0]) / get_distance(self.ring[0], self.ring[1]),
             get_distance(self.pinky[3], self.pinky[0]) / get_distance(self.pinky[0], self.pinky[1])])
        # print(self.finger_distance_list)
        finger_distance_threshold = np.array([1.5, 1.8, 2, 2, 2])
        self.finger_state_distance = np.array(self.finger_distance_list > finger_distance_threshold, dtype=int)

        # TODO 손가락과 손바닥 이용해 손가락 굽힘 판단
        self.hand_angle_list = np.array([self.get_angle(self.thumb[1] - self._p_list[0], self.thumb[3] - self.thumb[1]),
                                         self.get_angle(self.index[0] - self._p_list[0], self.index[3] - self.index[0]),
                                         self.get_angle(self.middle[0] - self._p_list[0],
                                                        self.middle[3] - self.middle[0]),
                                         self.get_angle(self.ring[0] - self._p_list[0], self.ring[3] - self.ring[0]),
                                         self.get_angle(self.pinky[0] - self._p_list[0],
                                                        self.pinky[3] - self.pinky[0])])
        # print(self.hand_angle_list)
        hand_angle_threshold = np.array([0.7, 1.7, 1.5, 1.5, 1.3])
        self.hand_state_angle = np.array(self.hand_angle_list < hand_angle_threshold, dtype=int)
        # print(self.finger_angle_list, self.finger_distance_list, self.hand_angle_list)
        self.input = np.concatenate((self.finger_angle_list, self.finger_distance_list, self.hand_angle_list))

        # print(predict_static(self.input))
        # print(np.round(self.finger_angle_list, 3), np.round(self.finger_distance_list, 3), np.round(self.hand_angle_list, 3))
        # print(self.finger_state_angle, self.finger_state_distance, self.hand_state_angle)

        self.result = self.finger_state_angle + self.finger_state_distance + self.hand_state_angle > 1
        # print(self.result)
        if experiment_mode == False:
            return self.result
        else:
            return np.round(self.finger_angle_list, 3), np.round(self.finger_distance_list, 3), np.round(
                self.hand_angle_list, 3)

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
        self.finger_distance_list = np.array(
            [get_distance(self.thumb[3], self.pinky[0]) / get_distance(self.index[0], self.pinky[0]),
             get_distance(self.index[3], self.index[0]) / get_distance(self.index[0], self.index[1]),
             get_distance(self.middle[3], self.middle[0]) / get_distance(self.middle[0], self.middle[1]),
             get_distance(self.ring[3], self.ring[0]) / get_distance(self.ring[0], self.ring[1]),
             get_distance(self.pinky[3], self.pinky[0]) / get_distance(self.pinky[0], self.pinky[1])])
        # print(self.finger_distance_list)

        # TODO 손가락과 손바닥 이용해 손가락 굽힘 판단
        self.hand_angle_list = np.array([self.get_angle(self.thumb[1] - self._p_list[0], self.thumb[3] - self.thumb[1]),
                                         self.get_angle(self.index[0] - self._p_list[0], self.index[3] - self.index[0]),
                                         self.get_angle(self.middle[0] - self._p_list[0],
                                                        self.middle[3] - self.middle[0]),
                                         self.get_angle(self.ring[0] - self._p_list[0], self.ring[3] - self.ring[0]),
                                         self.get_angle(self.pinky[0] - self._p_list[0],
                                                        self.pinky[3] - self.pinky[0])])
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


# TODO Gesture 판단, 일단은 15프레임 (0.5초)의 Queue로?
class Gesture:
    GESTURE_ARRAY_SIZE = 7
    GESTURE_STATIC_SIZE = 30

    def __init__(self):
        self.palm_data = [np.array([0, 0, 0]) for _ in range(Gesture.GESTURE_ARRAY_SIZE)]
        self.d_palm_data = [np.array([0, 0, 0]) for _ in range(Gesture.GESTURE_ARRAY_SIZE)]  # palm_data의 차이를 기록할 list

        self.location_data = [[0.1, 0.1, 0.1] for _ in range(Gesture.GESTURE_ARRAY_SIZE)]
        self.finger_data = [np.array([0, 0, 0, 0, 0]) for _ in range(Gesture.GESTURE_ARRAY_SIZE)]
        self.gesture_data = [0] * Gesture.GESTURE_STATIC_SIZE

    @staticmethod
    def get_location(p):  # p는 프레임 수 * 좌표 세개
        x_mean, y_mean, z_mean = 0, 0, 0
        for i in range(len(p) - 1):
            x_mean += p[i].x
            y_mean += p[i].y
            z_mean += p[i].z
        x_mean, y_mean, z_mean = x_mean / (len(p) - 1), y_mean / (len(p) - 1), z_mean / (len(p) - 1)
        return [x_mean, y_mean, z_mean]

    @staticmethod
    def remove_outlier(target):  # 1D numpy array 최대/최소 제거:
        for i in range(target.shape[0]):
            if target[i] == np.max(target):
                max_i = i
            if target[i] == np.min(target):
                min_i = i
        output = np.delete(target, (min_i, max_i))
        return output

    def update(self, handmark, gesture_num):
        # print(self.get_location(handmark._p_list))
        self.palm_data.insert(0, handmark.palm_vector)
        self.d_palm_data.insert(0, (self.palm_data[1] - handmark.palm_vector) * 1000)
        self.location_data.insert(0, Gesture.get_location(handmark._p_list))  # location data는 (프레임 수) * 22 * Mark_p 객체
        self.finger_data.insert(0, handmark.finger_vector)
        self.gesture_data.insert(0, gesture_num)
        # print(gesture_num)
        # print(handmark.palm_vector)

        self.palm_data.pop()
        self.d_palm_data.pop()
        self.location_data.pop()
        self.finger_data.pop()
        self.fv = handmark.finger_vector
        self.gesture_data.pop()

        # print(self.palm_data[0], self.finger_data[0], self.location_data[0])
        # print(handmark.palm_vector * 1000)

    # handmark 지닌 10개의 프레임이 들어온다...
    def detect_gesture(self):  # 이 최근꺼
        global gesture_check
        # print(self.gesture_data.count(6))
        # print(gesture_check)
        if (gesture_check == True) or (self.gesture_data.count(6) < 15):
            # print(self.gesture_data)
            return -1

        # i가 작을수록 더 최신 것
        ld_window = self.location_data[2:]
        x_classifier = np.array(ld_window[:-1])[:, 0] - np.array(ld_window[1:])[:, 0]
        y_classifier = np.array(ld_window[:-1])[:, 1] - np.array(ld_window[1:])[:, 1]
        x_classifier = self.remove_outlier(x_classifier)
        y_classifier = self.remove_outlier(y_classifier)

        # print(np.mean(x_classifier), np.mean(y_classifier))
        # if np.mean(x_classifier) >

        # x_classfication = self.d_location_data[:][0]
        # y_classfication = self.d_location_data[:][1]
        # print(x_classfication, y_classfication)

        # 왼쪽 X감소 오른쪽 X증가
        # 위 y감소 아래 y증가

        x_mean, y_mean = np.mean(x_classifier), np.mean(y_classifier)
        # print(x_mean, y_mean)

        # 동적 제스처 - LEFT
        if y_mean != 0:
            if x_mean / abs(y_mean) < -1.3:
                condition1 = condition2 = condition3 = condition4 = 0

                angle_threshold = [-1., 0., 0.]
                angle_min = 3

                for i in range(Gesture.GESTURE_ARRAY_SIZE - 1):
                    angle = get_angle(self.palm_data[-1], angle_threshold)
                    if angle < angle_min:
                        angle_min = angle
                    if self.palm_data[i][2] - self.palm_data[i + 1][2] > 0.05:
                        condition1 += 1.2
                    if self.finger_data[i][0] - self.finger_data[i + 1][0] < -0.04:
                        condition2 += 1
                    if self.finger_data[i][2] - self.finger_data[i + 1][2] < -0.05:
                        condition3 += 1
                    if self.location_data[i][0] - self.location_data[i + 1][0] < -0.04:
                        condition4 += 1
                condition_sum = condition1 + condition2 + condition3 + condition4
                # print(condition1, condition2, condition3, condition4)
                # print(get_angle(self.palm_data[-1], angle_threshold))
                if condition_sum > 8 and angle_min < 0.5:
                    print("LEFT")
                    win32api.keybd_event(0x25, 0, 0, 0)
                    return -1

            # 동적 제스처 - RIGHT
            if x_mean / abs(y_mean) > 1.5:
                condition1 = condition2 = condition3 = condition4 = 0

                angle_threshold = [-1., 0., 0.]
                angle_min = 3
                for i in range(Gesture.GESTURE_ARRAY_SIZE - 1):
                    angle = get_angle(self.palm_data[-1], angle_threshold)
                    if angle < angle_min:
                        angle_min = angle
                    if self.palm_data[i][2] - self.palm_data[i + 1][2] < -0.06:
                        condition1 += 1
                    if self.finger_data[i][0] - self.finger_data[i + 1][0] > 0.05:
                        condition2 += 1
                    if self.finger_data[i][2] - self.finger_data[i + 1][2] > 0.05:
                        condition3 += 1
                    if self.location_data[i][0] - self.location_data[i + 1][0] > 0.04:
                        condition4 += 1
                condition_sum = condition1 + condition2 + condition3 + condition4
                angle_threshold = [-1., 0., 0.]
                # print(get_angle(self.palm_data[-1], angle_threshold))
                if condition_sum > 10 and angle_min < 0.8:
                    print("RIGHT")
                    win32api.keybd_event(0x27, 0, 0, 0)
                    return -1

            # 동적 제스처 - UP
            if y_mean / abs(x_mean) < -1.5:
                condition1 = condition2 = condition3 = condition4 = 0

                # i가 작을수록 더 최신 것
                angle_threshold = [0., -1., 0.]
                angle_min = 3
                for i in range(Gesture.GESTURE_ARRAY_SIZE - 1):
                    angle = get_angle(self.palm_data[-1], angle_threshold)
                    if angle < angle_min:
                        angle_min = angle
                    if self.palm_data[i][2] - self.palm_data[i + 1][2] > 0.05:
                        condition1 += 1
                    if self.finger_data[i][1] - self.finger_data[i + 1][1] < -0.05:
                        condition2 += 1
                    if self.location_data[i][1] - self.location_data[i + 1][1] < -0.07:
                        condition3 += 1
                condition_sum = condition1 + condition2 + condition3 + condition4

                # print(get_angle(self.palm_data[-1], angle_threshold))
                if condition_sum > 6 and angle_min < 1:
                    print("UP")
                    win32api.keybd_event(0x26, 0, 0, 0)
                    return -1

            # 동적 제스처 - DOWN
            if y_mean / abs(x_mean) > 1.5:
                condition1 = condition2 = condition3 = condition4 = 0

                # i가 작을수록 더 최신 것
                angle_threshold = [0., 1., 0.]
                angle_min = 3
                for i in range(Gesture.GESTURE_ARRAY_SIZE - 1):
                    angle = get_angle(self.palm_data[-1], angle_threshold)
                    if angle < angle_min:
                        angle_min = angle
                    if self.palm_data[i][2] - self.palm_data[i + 1][2] > 0.04:
                        condition1 += 1
                    if self.finger_data[i][1] - self.finger_data[i + 1][1] > 0.04:
                        condition2 += 1
                    if self.location_data[i][1] - self.location_data[i + 1][1] > 0.05:
                        condition3 += 1
                # print(condition1, condition2, condition3, condition4)
                condition_sum = condition1 + condition2 + condition3 + condition4
                angle_threshold = [0., 1., 0.]
                # print(get_angle(self.palm_data[-1], angle_threshold))
                if condition_sum > 7 and angle_min < 1.5:
                    print("DOWN")
                    win32api.keybd_event(0x28, 0, 0, 0)
                    return -1

        # gesture_check = True

    def gesture_LRUD(self):  # 상하좌우 변화량 판단
        LR_trigger, UD_trigger = 0, 0
        for i in range(5):
            if abs(self.location_data[i][0] - self.location_data[i + 1][0]) > (
                    self.location_data[i][1] - self.location_data[i + 1][1]):
                LR_trigger += 1
            else:
                UD_trigger += 1
        output = LR_trigger > UD_trigger
        return output


class Gesture_mode:
    """
    전체 MODE 결정하기 위한 Class
    """
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
        """
        :return: Monitoring to "string"
        """
        return 'left : {}, right : {}, lpv : {}, lfv : {}, rpv : {}, rfv : {}'.format(
            self.left[-1], self.right[-1],
            self.left_palm_vector[-1], self.left_finger_vector[-1], self.right_palm_vector[-1],
            self.right_finger_vector[-1])

    def update_left(self, left, palm_vector, finger_vector):
        # print(left, 'left')
        self.left.append(left)
        self.left_palm_vector.append(palm_vector)
        self.left_finger_vector.append(finger_vector)
        self.left.pop(0)
        self.left_palm_vector.pop(0)
        self.left_finger_vector.pop(0)

    def update_right(self, right, palm_vector, finger_vector):
        # print(right, 'right')
        self.right.append(right)
        self.right_palm_vector.append(palm_vector)
        self.right_finger_vector.append(finger_vector)
        self.right.pop(0)
        self.right_palm_vector.pop(0)
        self.right_finger_vector.pop(0)

    def select_mode(self, pixel):
        mode = 0
        lpv_mode_1 = [-0.39, 0.144, -0.90]
        lfv_mode_1 = [-0.33, -0.94, 0.]
        rpv_mode_1 = [-0.40, -0.14, -0.9]
        rfv_mode_1 = [-0.33, -0.94, 0.]
        mode_result = 0
        # print(self.left_palm_vector[0], self.left_finger_vector[0],
        # self.right_palm_vector[0], self.right_finger_vector[0])
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
        if mode_result < 20 and left_idx_1 == 10 and right_idx_1 == 10:
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
        if mode_result < 23 and left_idx_1 == 10 and right_idx_1 == 10:
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
        if mode_result < 23 and left_idx_1 == 10 and right_idx_1 == 10:
            pixel.mousemove()
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
        if mode_result < 23 and left_idx_1 == 10 and right_idx_1 == 10:
            mode = 4
        return mode


def get_angle(l1, l2):
    """
    :param l1:
    :param l2:
    :return: 두 벡터 l1, l2 사이의 값 반환 (RADIAN)
    """
    l1_ = np.array([l1[0], l1[1], l1[2]])
    l2_ = np.array([l2[0], l2[1], l2[2]])
    if (norm(l1) * norm(l2)) == 0:
        return 0
    else:
        return np.arccos(np.dot(l1_, l2_) / (norm(l1) * norm(l2)))
    # return np.arccos(np.dot(l1_, l2_) / (norm(l1) * norm(l2)))


def vector_magnitude(one_d_array):
    """
    :param one_d_array: 1D Array
    :return: 크기 반환
    """
    return math.sqrt(np.sum(one_d_array * one_d_array))


def norm(p1):
    """
    :param p1: 점 3차원 정보 list
    :return: 벡터의 크기 반환
    """
    return math.sqrt((p1[0]) ** 2 + (p1[1]) ** 2 + (p1[2]) ** 2)


def convert_offset(x, y):
    """
    :param x: offset input X
    :param y: offset input Y
    :return: Modified x, y coordinates
    """
    return x * 4 / 3 - x_size / 8, y * 4 / 3 - y_size / 8


def inv_convert_off(x, y):
    """
    :param x: offset input X
    :param y: offset input Y
    :return: Inversed x, y coordinates (to unconverted coord)
    """
    return (x + x_size / 8) * 3 / 4, (y + y_size / 8) * 3 / 4


def get_distance(p1, p2, mode='3d'):
    """
    :param p1: Mark_pixel() 객체
    :param p2: Mark_pixel() 객체
    :return: p1과 p2 사이의 거리, 3차원/2차원 mark pixel 모두 거리 반환
    """
    if mode == '3d':
        try:
            return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
        except:
            return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    elif mode == '2d':
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# TODO 프로세스 함수들
def process_static_gesture(array_for_static, value_for_static):
    """
    :param array_for_static: shared array between main process and static gesture detection process
    :param value_for_static: shared value between main process and static gesture detection process
    :return: NO RETURN BUT IT MODIFY SHARED ARR AND VAL
    """
    while True:
        input_ = np.copy(array_for_static[:])
        # print(input_)
        input_ = input_[np.newaxis]
        try:
            prediction = MODEL_STATIC.predict(input_)
            if np.max(prediction[0]) > 0.9:
                value_for_static.value = np.argmax(prediction[0])
            else:
                value_for_static.value = 0
        except:
            pass


def process_static_gesture_mod(array_for_static, value_for_static, array_for_static2, value_for_static2):
    while True:
        input_ = np.copy(array_for_static[:])
        # print(input_)
        input_ = input_[np.newaxis]
        # time.sleep(0.033)
        try:
            prediction = MODEL_STATIC.predict(input_)
            if np.max(prediction[0]) > 0.8:
                value_for_static.value = np.argmax(prediction[0])
            else:
                value_for_static.value = 0
        except:
            pass

        input_2 = np.copy(array_for_static2[:])
        # print(input_)
        input_2 = input_2[np.newaxis]
        try:
            prediction2 = MODEL_STATIC.predict(input_2)
            if np.max(prediction2[0]) > 0.8:
                value_for_static2.value = np.argmax(prediction2[0])
            else:
                value_for_static2.value = 0
        except:
            pass


def initialize(array_for_static_l, value_for_static_l, array_for_static_r, value_for_static_r):
    '''
    :param array_for_static_l: static gesture 판별하는 process 와 공유할 왼손 input data
    :param value_for_static_l: static gesture 판별하는 process 와 공유할 왼손 output data
    :param array_for_static_r: static gesture 판별하는 process 와 공유할 오른손 input data
    :param value_for_static_r: static gesture 판별하는 process 와 공유할 오른손 output data
    :return:
    '''
    global image
    global MOUSE_USE
    global CLICK_USE
    global WHEEL_USE
    global DRAG_USE
    global pen_color

    # GUI Part
    def mode_2_pre(palm, finger, left, p_check):
        palm_th = np.array([-0.41607399, -0.20192736, 0.88662719])
        finger_th = np.array([-0.08736683, -0.96164491, -0.26001175])
        # print(ctrl_z_check, left)
        parameter = get_angle(palm, palm_th) + get_angle(finger, finger_th)
        if p_check == 0 and left == 3 and parameter < 1:
            print('이전 페이지 (Left Arrow)')
            win32api.keybd_event(0x25, 0, 0, 0)  # Left Arrow 누르기.
            win32api.keybd_event(0x25, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(0.1)
            return 15
        elif p_check > 0:
            return p_check - 1
        else:
            return 0

    def mode_2_laser(state, num, right):
        LASER_CHANGE_TIME = 6
        # print(state, num, right)
        if right:
            num = max(num + 1, 0)
        else:
            num = min(num - 1, LASER_CHANGE_TIME)

        if not state and num > LASER_CHANGE_TIME and right:
            # state = True
            win32api.keybd_event(0xa2, 0, 0, 0)  # LEFT CTRL 누르기.
            win32api.keybd_event(0x4C, 0, 0, 0)  # L 누르기.
            win32api.keybd_event(0xa2, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x4C, 0, win32con.KEYEVENTF_KEYUP, 0)
            state = True
            return state, num
        elif state and num < - 2 and not right:
            # state = False
            win32api.keybd_event(0xa2, 0, 0, 0)  # LEFT CTRL 누르기.
            win32api.keybd_event(0x4C, 0, 0, 0)  # L 누르기.
            win32api.keybd_event(0xa2, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x4C, 0, win32con.KEYEVENTF_KEYUP, 0)
            state = False
            return state, num
        return state, num

    def mode_3_interrupt(mode_global):
        if mode_global == 3:
            # win32api.keybd_event(0xa2, 0, 0, 0)  # LEFT CTRL 누르기.
            # win32api.keybd_event(0x31, 0, 0, 0)  # 1 누르기.
            # time.sleep(0.1)
            # win32api.keybd_event(0xa2, 0, win32con.KEYEVENTF_KEYUP, 0)
            # win32api.keybd_event(0x31, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x1B, 0, 0, 0)  # ESC DOWN
            win32api.keybd_event(0x1B, 0, win32con.KEYEVENTF_KEYUP, 0)  # ESC UP

    def mode_2_off(mode_before, laser_state):
        """
        :param mode_before:
        :param laser_state:
        :return: mode 2 끝날때 레이저 켜져있으면 꺼주기
        """
        if mode_before == 2 and laser_state:
            win32api.keybd_event(0xa2, 0, 0, 0)  # LEFT CTRL 누르기.
            win32api.keybd_event(0x4C, 0, 0, 0)  # L 누르기.
            win32api.keybd_event(0xa2, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x4C, 0, win32con.KEYEVENTF_KEYUP, 0)
            return False

    def mode_3_ctrl_z(palm, finger, left, ctrl_z_check):
        palm_th = np.array([-0.41607399, -0.20192736, 0.88662719])
        finger_th = np.array([-0.08736683, -0.96164491, -0.26001175])
        # print(ctrl_z_check, left)
        parameter = get_angle(palm, palm_th) + get_angle(finger, finger_th)
        if ctrl_z_check == 0 and left == 3 and parameter < 1:
            print('되돌리기 (CTRL + Z)')
            win32api.keybd_event(0xa2, 0, 0, 0)  # LEFT CTRL 누르기.
            win32api.keybd_event(0x5a, 0, 0, 0)  # Z 누르기.
            time.sleep(0.1)
            win32api.keybd_event(0xa2, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x5a, 0, win32con.KEYEVENTF_KEYUP, 0)
            # 최소 N 프레임마다 Control + Z 상황과 가까운지 확인
            return 15
        elif ctrl_z_check > 0:
            return ctrl_z_check - 1
        else:
            return 0

    def mode_3_remove_all(palm, finger, left, remove_check):
        # 60 means 60 frames to trigger 'REMOVE ALL'
        REMOVE_THRESHOLD = 60
        palm_th = np.array([-0.41607399, -0.20192736, 0.88662719])
        finger_th = np.array([-0.08736683, -0.96164491, -0.26001175])
        # print(ctrl_z_check, left)
        parameter = get_angle(palm, palm_th) + get_angle(finger, finger_th)
        if left == 3 and parameter < 2 and remove_check < REMOVE_THRESHOLD:
            return remove_check + 1
        elif remove_check == REMOVE_THRESHOLD:
            # N 프레임 쌓이면 전체 지움
            print('Remove all (E)')
            win32api.keybd_event(0x45, 0, 0, 0)  # E 누르기.
            time.sleep(0.03)
            win32api.keybd_event(0x45, 0, win32con.KEYEVENTF_KEYUP, 0)
            return 0
        else:
            return max(0, remove_check - 1)


    def mode_3_board(palm, finger, left, remove_check):
        # 60 means 60 frames to trigger 'REMOVE ALL'
        REMOVE_THRESHOLD = 30
        palm_th = np.array([-0.15196232, 0.23579129, -0.9598489])
        finger_th = np.array([-0.36294722, -0.91659405, -0.16770409])
        global BOARD_COLOR

        # print(ctrl_z_check, left)
        parameter = get_angle(palm, palm_th) + get_angle(finger, finger_th)
        if left == 7 and parameter < 1 and remove_check < REMOVE_THRESHOLD:
            return remove_check + 1
        elif remove_check == REMOVE_THRESHOLD:
            # N 프레임 쌓이면 전체 지움
            if BOARD_COLOR == 'b':
                print('BOARD ON (K)')
                win32api.keybd_event(0x4B, 0, 0, 0)  # K 누르기.
                time.sleep(0.03)
                win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)
                return 0
            elif BOARD_COLOR == 'w':
                print('BOARD ON (W)')
                win32api.keybd_event(0x57, 0, 0, 0)  # W 누르기.
                time.sleep(0.03)
                win32api.keybd_event(0x57, 0, win32con.KEYEVENTF_KEYUP, 0)
                return 0
        else:
            return 0

    class opcv(QThread):
        change_pixmap_signal = pyqtSignal(np.ndarray)
        mode_signal = pyqtSignal(int)

        R = np.array([0.22, -0.98, 0])
        G = np.array([0.73, -0.68, 0])
        B = np.array([0.95, -0.3, 0])
        O = np.array([0.9, 0.4, 0])
        COLOR_SET = {'R': R, 'G': G, 'B': B, 'O': O}

        BASE_LAYER = Image.open('./image/background.png')
        R_LAYER = './image/Red.png'
        G_LAYER = './image/Green.png'
        B_LAYER = './image/Blue.png'
        O_LAYER = './image/Orange.png'
        LAYER_PATH = {'R': R_LAYER, 'G': G_LAYER, 'B': B_LAYER, 'O': O_LAYER}
        LAYER_SET = {'R': Image.open(R_LAYER), 'G': Image.open(G_LAYER),
                     'B': Image.open(B_LAYER), 'O': Image.open(O_LAYER)}

        def __init__(self):
            super().__init__()

        def mode_3_pen_color(self, palm, finger, image):
            global pen_color

            palm_standard = [-0.29779509, -0.56894808, 0.76656126]
            if get_angle(palm, palm_standard) < 0.8:
                color_value = self.COLOR_SET.copy()

                pill_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                for key, value in self.COLOR_SET.items():
                    color_value[key] = get_angle(finger, value)
                if min(color_value.values()) < 0.6:
                    color_name_l = [k for k, v in color_value.items() if min(color_value.values()) == v]
                    if len(color_name_l) > 0:
                        color_name = color_name_l[0]
                        pill_image.paste(self.LAYER_SET[color_name], (0, 280), mask=self.LAYER_SET[color_name])
                    if color_name != pen_color:
                        if color_name == 'R':
                            win32api.keybd_event(0x52, 0, 0, 0)
                            win32api.keybd_event(0x52, 0, win32con.KEYEVENTF_KEYUP, 0)
                        if color_name == 'G':
                            win32api.keybd_event(0x47, 0, 0, 0)
                            win32api.keybd_event(0x47, 0, win32con.KEYEVENTF_KEYUP, 0)
                        if color_name == 'B':
                            win32api.keybd_event(0x42, 0, 0, 0)
                            win32api.keybd_event(0x42, 0, win32con.KEYEVENTF_KEYUP, 0)
                        if color_name == 'O':
                            win32api.keybd_event(0x4F, 0, 0, 0)
                            win32api.keybd_event(0x4F, 0, win32con.KEYEVENTF_KEYUP, 0)
                    pen_color = color_name

                else:
                    pill_image.paste(self.BASE_LAYER, (0, 280), self.BASE_LAYER)

                image = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

            return image

            # Color Set : R G B O
            # print(get_angle(finger, finger_vector_1), get_angle(finger, finger_vector_2))

        @pyqtSlot(int, int)
        def mode_setting(self, mode, mode_before):  # 1
            global MOUSE_USE, CLICK_USE, DRAG_USE, WHEEL_USE, mode_global, laser_state
            if mode != mode_before:
                self.mode_signal.emit(int(mode - 1))  # 2 / #2-4

                if mode == 1 and mode_global != mode:
                    MOUSE_USE = False
                    CLICK_USE = False
                    DRAG_USE = False
                    WHEEL_USE = False
                    laser_state = mode_2_off(mode_global, laser_state)
                    print('MODE 1, 대기 모드')
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 2020, 100, 0, 0)

                    # win32api.keybd_event(0x74, 0, 0, 0)  # F5 DOWN
                    # win32api.keybd_event(0x74, 0, win32con.KEYEVENTF_KEYUP, 0)  # F5 UP

                    mode_3_interrupt(mode_global)

                    mode_global = mode

                if mode == 2 and mode_global != mode:
                    MOUSE_USE = True
                    CLICK_USE = True
                    DRAG_USE = False
                    WHEEL_USE = False

                    if mode_global == 3:
                        # win32api.keybd_event(0xa2, 0, 0, 0)  # LEFT CTRL 누르기.
                        # win32api.keybd_event(0x31, 0, 0, 0)  # 1 누르기.
                        # time.sleep(0.1)
                        # win32api.keybd_event(0xa2, 0, win32con.KEYEVENTF_KEYUP, 0)
                        # win32api.keybd_event(0x31, 0, win32con.KEYEVENTF_KEYUP, 0)
                        win32api.keybd_event(0x1B, 0, 0, 0)  # ESC DOWN
                        win32api.keybd_event(0x1B, 0, win32con.KEYEVENTF_KEYUP, 0)  # ESC UP
                    print('MODE 2, 기본 발표 모드')
                    mode_3_interrupt(mode_global)
                    mode_global = mode

                if mode == 3 and mode_global != mode:
                    MOUSE_USE = True
                    CLICK_USE = False
                    DRAG_USE = True
                    WHEEL_USE = False
                    laser_state = mode_2_off(mode_global, laser_state)
                    # win32api.SetCursorPos((200, 200))
                    win32api.keybd_event(0xa2, 0, 0, 0)  # LEFT CTRL 누르기.
                    win32api.keybd_event(0x32, 0, 0, 0)  # 2 누르기.
                    time.sleep(0.1)
                    win32api.keybd_event(0xa2, 0, win32con.KEYEVENTF_KEYUP, 0)
                    win32api.keybd_event(0x32, 0, win32con.KEYEVENTF_KEYUP, 0)
                    print('MODE 3, 필기 발표 모드')  # 3, # 2-5
                    mode_global = mode

                if mode == 4 and mode_global != mode:
                    MOUSE_USE = True
                    CLICK_USE = True
                    DRAG_USE = True
                    WHEEL_USE = True
                    laser_state = mode_2_off(mode_global, laser_state)
                    print('MODE 4, 웹서핑 발표 모드')
                    mode_3_interrupt(mode_global)
                    mode_global = mode

        @pyqtSlot(bool)
        def send_img(self, bool_state):  # p를 보는 emit 함수
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap = self.capture
            # For webcam input:
            hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.8)
            # pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, upper_body_only=True)

            global width, height, static_gesture_num_l

            # TODO landmark 를 대응 인스턴스로 저장
            class Mark_pixel:
                def __init__(self, x, y, z=0, LR=0):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.LR = LR

                def __str__(self):
                    return str(self.x) + '   ' + str(self.y) + '   ' + str(self.z)

                def to_list(self):
                    """
                    :return: converted mark_pixel to list
                    """
                    return [self.x, self.y, self.z]

                def to_pixel(self):
                    global x_size
                    global y_size
                    return Mark_2d(self.x * x_size, self.y * y_size)

                def __sub__(self, other):
                    return self.x - other.x, self.y - other.y, self.z - other.z

            class Mark_2d:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def __str__(self):
                    return tuple(self.x, self.y)

                @staticmethod
                def mod_cursor_position(pos: tuple):
                    """
                    :param pos: position data (tuple)
                    :return: modified cursor position x, y (tuple)
                    """
                    x, y = pos[0], pos[1]
                    FULLSIZE = 1920, 1080
                    MOD_SIZE = 1600, 640
                    mod_x = x * FULLSIZE[0] / MOD_SIZE[0] - (FULLSIZE[0] - MOD_SIZE[0]) / 2
                    mod_x = max(0, mod_x);
                    mod_x = min(FULLSIZE[0], mod_x)
                    mod_y = y * FULLSIZE[1] / MOD_SIZE[1] - (FULLSIZE[1] - MOD_SIZE[1]) / 2
                    mod_y = max(0, mod_y);
                    mod_y = min(FULLSIZE[1], mod_y)
                    # print(mod_x, mod_y)
                    return int(mod_x) + 1920, int(mod_y)

                def mousemove(self):
                    if now_click == True:
                        cv2.circle(image, (int(self.x / 3), int(self.y / 2.25)), 5, (255, 0, 255), -1)
                    else:
                        cv2.circle(image, (int(self.x / 3), int(self.y / 2.25)), 5, (255, 255, 0), -1)
                    # self.x, self.y = convert_offset(self.x, self.y)
                    cursor_position = (int(self.x), int(self.y))
                    m_cursor_position = self.mod_cursor_position(cursor_position)
                    # print(m_cursor_position)
                    win32api.SetCursorPos(m_cursor_position)

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

            def hand_drag(landmark):
                """
                :param landmark: landmark or mark_p
                :return: nothing, but it change mouse position and click statement
                """
                global now_click
                global straight_line, rectangular, circle
                # print(now_click, straight_line, rectangular, circle)
                drag_threshold = 1
                if straight_line or rectangular or circle:
                    drag_threshold = drag_threshold * 1.3

                # print(get_distance(landmark[4], landmark[8], mode='3d') < drag_threshold * get_distance(landmark[4],
                #                                                                                      landmark[3],
                #                                                                                      mode='3d'),
                #       now_click)

                if get_distance(landmark[4], landmark[8], mode='3d') < drag_threshold * get_distance(landmark[4],
                                                                                                     landmark[3],
                                                                                                     mode='3d') \
                        and now_click == False:
                    print('drag on')
                    pos = win32api.GetCursorPos()
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, pos[0], pos[1], 0, 0)
                    # ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
                    now_click = True

                elif get_distance(landmark[4], landmark[8], mode='3d') > drag_threshold * get_distance(landmark[4],
                                                                                                       landmark[3],
                                                                                                       mode='3d') \
                        and now_click == True:
                    print('drag off')
                    pos = win32api.GetCursorPos()
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, pos[0], pos[1], 0, 0)
                    # ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                    now_click = False

            def hand_click(landmark, pixel):
                global now_click2

                if get_distance(landmark[4], landmark[10]) < get_distance(landmark[7],
                                                                          landmark[8]) and now_click2 == False:
                    print('click')
                    pos = win32api.GetCursorPos()
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, pos[0], pos[1], 0, 0)
                    print('click off')
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, pos[0], pos[1], 0, 0)
                    now_click2 = True
                    return -1
                if get_distance(landmark[4], landmark[10]) > get_distance(landmark[7],
                                                                          landmark[8]) and now_click2 == True:
                    now_click2 = False

                return 0

            """
            def blurFunction(src):
                with mp_face_detection.FaceDetection(
                        min_detection_confidence=0.5) as face_detection:  
                    "with 문, mp_face_detection.FaceDetection 클래스를 face_detection 으로 사용"
                    image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # image 파일의 BGR 색상 베이스를 RGB 베이스로 바꾸기
                    results = face_detection.process(image)  # 튜플 형태
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image_rows, image_cols, _ = image.shape
                    c_mask: np.ndarray = np.zeros((image_rows, image_cols), np.uint8)
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
            """
            before_c = Mark_pixel(0, 0, 0)
            pixel_c = Mark_pixel(0, 0, 0)
            hm_idx = False
            finger_open_ = [False for _ in range(5)]
            gesture_time = time.time()
            gesture = Gesture()
            gesture_mode = Gesture_mode()

            ctrl_z_check_number = 0
            remove_all_number = 0
            board_num = 0
            p_check_number = 0

            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
            cap.set(cv2.CAP_PROP_FPS, 60)
            before_time = time.time()

            click_tr = 0
            mode_before = 0
            p_key_ready = False
            mode = 0
            global gesture_check
            global mode_global
            global straight_line
            global rectangular
            global circle
            global REVERSE_MODE
            global laser_num, laser_state

            # TODO 변화량 모니터링
            from matplotlib import pyplot as plt
            from matplotlib import animation

            fig = plt.figure()
            ax = plt.axes(xlim=(0, 50), ylim=(-1, 1))
            line, = ax.plot([], [], 'b', lw=2)
            line2, = ax.plot([], [], 'g', lw=2)
            line3, = ax.plot([], [], 'r', lw=2)

            '''x가 파란색 y가 빨간색 z가 초록색'''
            max_points = 50
            line, = ax.plot(np.arange(max_points),
                            np.ones(max_points, dtype=np.float) * np.nan, lw=2)
            line2, = ax.plot(np.arange(max_points),
                             np.ones(max_points, dtype=np.float) * np.nan, lw=2)
            line3, = ax.plot(np.arange(max_points),
                             np.ones(max_points, dtype=np.float) * np.nan, lw=2)

            if VISUALIZE_GRAPH:
                def init():
                    return line,

                def animate():
                    y = gesture.palm_data[0][0]

                    old_y = line.get_ydata()
                    new_y = np.r_[old_y[1:], y]
                    line.set_ydata(new_y)

                    y2 = gesture.palm_data[0][1]

                    old_y2 = line2.get_ydata()
                    new_y2 = np.r_[old_y2[1:], y2]
                    line2.set_ydata(new_y2)

                    y3 = gesture.palm_data[0][2]

                    old_y3 = line3.get_ydata()
                    new_y3 = np.r_[old_y3[1:], y3]
                    line3.set_ydata(new_y3)
                    return line, line2, line3,

                anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=False)

                plt.show()

            while bool_state and cap.isOpened():
                # print('cam')
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # image = blurFunction(image)

                # x_size, y_size, channel = image.shape
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)
                # results_body = pose.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 몸!!
                '''
                if results_body.pose_landmarks:
                    mark_b = []
                    body_landmark = results_body.pose_landmarks.landmark
                    mp_drawing.draw_landmarks(
                        image, results_body.pose_landmarks, mp_pose.UPPER_BODY_POSE_CONNECTIONS)

                    for i in range(25):
                        mark_b.append(Mark_pixel(body_landmark[i].x, body_landmark[i].y,
                                                 body_landmark[i].z))
                    #print(mark_b_list)
                    #print(mark_b)
                    BM = Handmark(mark_b)  # BODYMARK
                '''
                # 손!!
                if results.multi_hand_landmarks:
                    mark_p_list = []

                    for hand_landmarks in results.multi_hand_landmarks:
                        '''hand_landmarks 는 감지된 손의 갯수만큼의 원소 수를 가진 list 자료구조'''
                        mark_p = []
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        for i in range(21):
                            mark_p.append(Mark_pixel(hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y,
                                                     hand_landmarks.landmark[i].z))
                        mark_p_list.append(mark_p)

                    for i in range(len(mark_p_list)):  # for 문 한 번 도는게 한 손에 대한 것임
                        LR_idx = results.multi_handedness[i].classification[0].label
                        # LR_idx = LR_idx[:-1]
                        # print(LR_idx)
                        image = cv2.putText(image, LR_idx[:], (
                            int(mark_p_list[i][17].x * image.shape[1]), int(mark_p_list[i][17].y * image.shape[0])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                        if REVERSE_MODE == True:
                            if len(LR_idx) == 4:
                                LR_idx = 'Right'
                            elif len(LR_idx) == 5:
                                LR_idx = 'Left'

                        mark_p_list[i].append(LR_idx)

                        mark_p = mark_p_list[i]
                        # print(len(mark_p_list), i)
                        # Handmark 정보 입력
                        # print(mark_p[-1], " / ", mark_p[0])

                        if len(mark_p) == 22 and hm_idx == False:
                            HM = Handmark(mark_p)
                            hm_idx = True
                        # print(HM.p_list[-1])
                        # palm_vector 저장
                        palm_vector = HM.get_palm_vector()
                        finger_vector = HM.get_finger_vector()

                        # mark_p 입력
                        if hm_idx:
                            HM.p_list = mark_p
                            # mark_p[-1] = mark_p[-1][:-1]
                            if USE_TENSORFLOW:
                                # print(len(HM.p_list[-1]))
                                if len(mark_p[-1]) == 4:
                                    f_p_list = HM.return_18_info()
                                    array_for_static_l[:] = f_p_list
                                    # print(array_for_static)
                                    static_gesture_num_l = value_for_static_l.value
                                if len(mark_p[-1]) == 5:
                                    f_p_list = HM.return_18_info()
                                    array_for_static_r[:] = f_p_list
                                    # print(array_for_static)
                                    static_gesture_num_r = value_for_static_r.value

                                # try:
                                #     static_gesture_drawing(static_gesture_num, mark_p[-1])
                                # except:
                                #     print('static_drawing error')
                                # print(static_gesture_num)
                            else:
                                finger_open_for_ml = np.ndarray.tolist(HM.return_finger_state())
                                # 정지 제스쳐 확인
                                # static_gesture_detect(finger_open_for_ml, mark_p[-1])
                            finger_open_ = HM.return_finger_state()

                        mark_p0 = mark_p[0].to_pixel()
                        mark_p5 = mark_p[5].to_pixel()

                        # pixel_c = mark_c.to_pixel()
                        if len(mark_p[-1]) == 5:
                            palm_vector = HM.get_palm_vector()
                            finger_vector = HM.get_finger_vector()

                            if mode_global == 2:
                                # MODE 2 LEFT ARROW
                                p_check_number = mode_2_pre(palm_vector, finger_vector,
                                                                 static_gesture_num_r, p_check_number)
                                # MODE 2 LASER POINTER
                                laser_hand = np.all(finger_open_ == np.array([1, 1, 1, 0, 0]))
                                laser_state, laser_num = mode_2_laser(laser_state, laser_num, laser_hand)

                            # MODE 3 CTRL + Z
                            if mode_global == 3:
                                ctrl_z_check_number = mode_3_ctrl_z(palm_vector, finger_vector,
                                                                         static_gesture_num_r, ctrl_z_check_number)
                                remove_all_number = mode_3_remove_all(palm_vector, finger_vector,
                                                                         static_gesture_num_r, remove_all_number)
                                board_num = mode_3_board(palm_vector, finger_vector,
                                                                      static_gesture_num_r, board_num)

                            pixel_c = mark_p5
                            # gesture updating
                            if len(mark_p) == 22:
                                # print(HM.p_list[-1])
                                gesture.update(HM, static_gesture_num_r)
                                # print(static_gesture_num)
                                try:
                                    # print(time.time() - gesture_time)
                                    # LRUD = gesture.gesture_LRUD()
                                    # print(LRUD)
                                    # print(gesture.gesture_data)
                                    # print(6. in gesture.gesture_data)
                                    if time.time() - gesture_time > 0.5 and USE_DYNAMIC == True:
                                        # 다이나믹 제스처
                                        detect_signal = gesture.detect_gesture()
                                    if detect_signal == -1:  # 디텍트했을때!
                                        gesture_time = time.time()
                                        detect_signal = 0
                                except:
                                    pass

                        if len(mark_p[-1]) == 5:
                            gesture_mode.update_right(static_gesture_num_r, palm_vector, finger_vector)

                        # 마우스 움직임, 드래그
                        if (get_distance(pixel_c, before_c) < get_distance(mark_p0, mark_p5)) and \
                                sum(finger_open_[3:]) == 0 and \
                                finger_open_[1] == 1 and \
                                len(mark_p[-1]) == 5 and \
                                MOUSE_USE == True:
                            pixel_c.mousemove()

                            # print(click_tr[2])

                            if finger_open_[2] != 1 and click_tr > -1 and DRAG_USE == True:
                                hand_drag(mark_p)

                            if finger_open_[2] != 1 and CLICK_USE == True:
                                if not now_click:
                                    click_tr = hand_click(mark_p, pixel_c)
                            # else:
                            # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, int(pixel_c.x), int(pixel_c.y), 0, 0)

                            # 마우스 휠
                            if finger_open_[2] == 1 and WHEEL_USE == True and get_angle(mark_p[5] - mark_p[8],
                                                                                        mark_p[5] - mark_p[12]) < 0.3:
                                pixel_c.wheel(before_c)

                        # MODE 3 색 변경
                        if len(mark_p[-1]) == 4:
                            gesture_mode.update_left(static_gesture_num_l, palm_vector, finger_vector)

                            # MODE CHANGE
                            palm_vector = HM.get_palm_vector()
                            finger_vector = HM.get_finger_vector()
                            mode = gesture_mode.select_mode(pixel_c)
                            self.mode_setting(mode, mode_before)
                            mode_before = mode

                            # 입력 모양 모니터링
                            # print(static_gesture_num_l, straight_line, rectangular, circle)

                            # 직선 그리기
                            if mode_global == 3 and len(mark_p[-1]) == 4 and static_gesture_num_l == 13:
                                straight_line = True
                                win32api.keybd_event(0xA0, 0, 0, 0)  # LShift 누르기.
                            else:
                                win32api.keybd_event(0xA0, 0, win32con.KEYEVENTF_KEYUP, 0)
                                straight_line = False

                            # 네모 그리기
                            if mode_global == 3 and len(mark_p[-1]) == 4 and static_gesture_num_l == 11:
                                rectangular = True
                                win32api.keybd_event(0xA2, 0, 0, 0)  # LCtrl 누르기.
                            else:
                                win32api.keybd_event(0xA2, 0, win32con.KEYEVENTF_KEYUP, 0)
                                rectangular = False

                            # 원 그리기
                            if mode_global == 3 and len(mark_p[-1]) == 4 and static_gesture_num_l == 1:
                                circle = True
                                win32api.keybd_event(0x09, 0, 0, 0)  # TAB 누르기.
                            else:
                                win32api.keybd_event(0x09, 0, win32con.KEYEVENTF_KEYUP, 0)
                                circle = False

                            # 펜 색 변경
                            if not now_click:
                                if mode_global == 3 and len(mark_p[-1]) == 4 and static_gesture_num_l == 6:
                                    image = self.mode_3_pen_color(palm_vector, finger_vector, image)

                        before_c = pixel_c

                FPS = round(1 / (time.time() - before_time), 2)

                before_time = time.time()
                image = cv2.putText(image, str(FPS), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
            global THEME
            self.MainWindow = MyWindow()
            MainWindow = self.MainWindow
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(870, 550)
            self.From_button = False

            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")

            self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
            self.groupBox.setGeometry(QRect(660, 20, 81, 301))
            if THEME == 'B':
                self.groupBox.setStyleSheet("color: rgb(255, 255, 255);")
            elif THEME == 'W':
                self.groupBox.setStyleSheet("color: rgb(0, 0, 0);")
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
            if THEME == 'b':
                self.groupBox_2.setStyleSheet("color: rgb(255, 255, 255);")
            elif THEME == 'w':
                self.groupBox_2.setStyleSheet("color: rgb(0, 0, 0);")
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
            self.label_2.setPixmap(QtGui.QPixmap("./image/default.jpg"))  # <-------------- 비디오 프레임이 들어가야함
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
            # M
            self.groupBox_2.setTitle(_translate("MainWindow", "활성 기능"))
            self.checkBox.setText(_translate("MainWindow", "마우스 움직임"))
            self.checkBox_2.setText(_translate("MainWindow", "마우스 클릭"))
            self.checkBox_3.setText(_translate("MainWindow", "드래그"))
            self.checkBox_4.setText(_translate("MainWindow", "스크롤"))

        @pyqtSlot(int)
        def push_button(self, integer):  # 2-1
            if integer != -1:
                B_list = [self.pushButton, self.pushButton_2,
                          self.pushButton_3, self.pushButton_4]
                if not B_list[integer].isChecked():
                    self.From_button = True
                    B_list[integer].toggle()  # #2-2
            else:
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
            if Button_list[integer].isChecked():  # 2-3
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
                            self.click_mode.emit(integer + 1, 1)
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

    class Setting_window(QtWidgets.QDialog):
        # def __init__(self):
        #     self.setupUI()

        def setupUi(self, Dialog):
            Dialog.setObjectName("Setting")
            Dialog.resize(400, 100)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("Image/setting.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            Dialog.setWindowIcon(icon)
            self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
            self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
            self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
            self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
            self.buttonBox.setObjectName("buttonBox")
            self.comboBox = QtWidgets.QComboBox(Dialog)
            self.comboBox.setGeometry(QtCore.QRect(230, 40, 121, 21))
            self.comboBox.setObjectName("comboBox")
            self.comboBox.addItem("")
            self.comboBox.addItem("")
            self.label = QtWidgets.QLabel(Dialog)
            self.label.setGeometry(QtCore.QRect(50, 40, 181, 16))
            font = QtGui.QFont()
            font.setFamily("서울남산 장체B")
            font.setPointSize(10)
            self.label.setFont(font)
            self.label.setObjectName("label")

            self.retranslateUi(Dialog)
            self.buttonBox.accepted.connect(Dialog.accept)
            self.buttonBox.rejected.connect(Dialog.reject)
            QtCore.QMetaObject.connectSlotsByName(Dialog)

        def retranslateUi(self, Dialog):
            _translate = QtCore.QCoreApplication.translate
            Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
            self.comboBox.setItemText(0, _translate("Dialog", "영어(English)"))
            self.comboBox.setItemText(1, _translate("Dialog", "한국어(Korean)"))
            self.label.setText(_translate("Dialog", "언어 선택(Language Selection)"))

    class MyWindow(QtWidgets.QMainWindow):

        power_off_signal = pyqtSignal(bool)

        def __init__(self):
            super().__init__()
            # self.setStyleSheet('''QMainWindow{background-color : rgb(0, 255, 0);}''')
            if THEME == 'b':
                self.setStyleSheet('''QMessageBox{background-color: rgb(225, 225, 225);}''')
                self.setStyleSheet('''QMainWindow{background-color : rgb(0, 0, 0);}''')
            elif THEME == 'w':
                self.setStyleSheet('''QMessageBox{background-color: rgb(0, 0, 0);}''')
                self.setStyleSheet('''QMainWindow{background-color : rgb(255, 255, 255);}''')
            self.msg = QMessageBox()

        def closeEvent(self, event):
            result = self.msg.question(self,
                                       "Confirm Exit...",
                                       "Are you sure you want to exit ?",
                                       self.msg.Yes | self.msg.No)
            if result == self.msg.Yes:
                self.power_off_signal.emit(False)
                event.accept()

            else:
                event.ignore()

    # app = QtWidgets.QApplication(sys.argv)
    # Form = QtWidgets.QWidget()
    # grabber = Grabber()
    # grabber.setupUi(grabber)
    # grabber.show()

    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setupUi()
    ui.MainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    print("This is util set program, it works well... maybe... XD")

    print('Running main_1_0.py...')
    from os import system

    system('python main_1_0.py')
