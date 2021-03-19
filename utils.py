import cv2
import mediapipe as mp
import pyautogui
import math
import win32api, win32con, time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from tensorflow import keras

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
#print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# For webcam input:
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

x_size, y_size = pyautogui.size().width, pyautogui.size().height


gesture_int = 0

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
        #print(self.palm_vector)
        return self.palm_vector

    def get_finger_vector(self):
        l0 = self._p_list[12] - self._p_list[9]
        self.finger_vector = np.array(l0)
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
        finger_angle_threshold = np.array([2.7, 1.7, 2.2, 2.2, 2])
        self.finger_state_angle = np.array(self.finger_angle_list > finger_angle_threshold, dtype=int)

        #TODO 각 손가락 거리정보 근거로 손가락 굽힘 판단
        self.finger_distance_list = np.array([get_distance(self.thumb[3], self.pinky[0]) / get_distance(self.index[0], self.pinky[0]),
                                     get_distance(self.index[3], self.index[0]) / get_distance(self.index[0], self.index[1]),
                                     get_distance(self.middle[3], self.middle[0]) / get_distance(self.middle[0], self.middle[1]),
                                     get_distance(self.ring[3], self.ring[0]) / get_distance(self.ring[0], self.ring[1]),
                                     get_distance(self.pinky[3], self.pinky[0]) / get_distance(self.pinky[0], self.pinky[1])])
        #print(self.finger_distance_list)
        finger_distance_threshold = np.array([1.5, 1.5, 1.5, 1.5, 2])
        self.finger_state_distance = np.array(self.finger_distance_list > finger_distance_threshold, dtype=int)

        # TODO 손가락과 손바닥 이용해 손가락 굽힘 판단
        self.hand_angle_list = np.array([self.get_angle(self.thumb[1] - self._p_list[0], self.thumb[3] - self.thumb[1]),
                                self.get_angle(self.index[0] - self._p_list[0], self.index[3] - self.index[0]),
                                self.get_angle(self.middle[0] - self._p_list[0], self.middle[3] - self.middle[0]),
                                self.get_angle(self.ring[0] - self._p_list[0], self.ring[3] - self.ring[0]),
                                self.get_angle(self.pinky[0] - self._p_list[0], self.pinky[3] - self.pinky[0])])
        #print(self.hand_angle_list)
        hand_angle_threshold = np.array([0.6, 1.5, 1.5, 1.5, 1.5])
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

if __name__ == '__main__':
    print("This is util set program, it works well... maybe... XD")

    print('Running main_63input.py...')
    from os import system
    system('python main_63input.py')