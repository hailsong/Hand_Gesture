import cv2
import mediapipe as mp
import math
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os

import pandas as pd

from multiprocessing import Process, Value, Array
import win32gui, win32console



series = "ㄴㄱㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅢㅐㅚㅟㅒㅖㅔ"

def process_static_gesture(array_for_static, value_for_static):
    """
    :param array_for_static: shared array between main process and static gesture detection process
    :param value_for_static: shared value between main process and static gesture detection process
    :return: NO RETURN BUT IT MODIFY SHARED ARR AND VAL
    """
    import keras
    MODEL_STATIC = keras.models.load_model(
        'keras_util/model_save/my_model_21_KSL.h5'
    )

    while True:
        input_ = np.copy(array_for_static[:])
        # print(input_)
        input_ = input_[np.newaxis]
        try:
            prediction = MODEL_STATIC.predict(input_)
            # print(f"predict : {prediction}\n", end='')
            if np.max(prediction[0]) > 0.5:
                value_for_static.value = np.argmax(prediction[0])
                # print(value_for_static.value)
            else:
                value_for_static.value = 0
        except:
            pass





width = 1280  # 너비
height = 720  # 높이
x_size = width
y_size = height

#TODO landmark를 대응 인스턴스로 저장
class mark_pixel():
    def __init__(self, x, y, z = 0, LR = 0):
        self.x = x
        self.y = y
        self.z = z
        self.LR = LR
    def __str__(self):
        return str(self.x) +'   '+ str(self.y) +'   ' + str(self.z)
    def to_list(self):
        return [self.x, self.y, self.z]
    def to_pixel(self):
        global x_size
        global y_size
        return Mark_2d(self.x * x_size, self.y * y_size)
    def __sub__(self, other):
        return self.x - other.x, self.y - other.y, self.z - other.z

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


class Mark_2d():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return tuple(self.x, self.y)


def vector_magnitude(one_D_array):
    return math.sqrt(np.sum(one_D_array * one_D_array))

def norm(p1):
    return math.sqrt((p1[0])**2 + (p1[1])**2 + (p1[2])**2)

def get_distance(p1, p2):
    try:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    except:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def get_center(p1, p2):
    return mark_pixel((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)


def static_gesture_detect(finger_open_, LR_index):
    global image


def initialize(shared_array, static_num):

    #print("Copyright 2021. INBODY inc. all rights reserved")
    #print("Contact : shi@inbody.com, HAIL SONG")

    # width = 1024 # 너비
    # height= 600 # 높이


    bpp = 3  # 표시 채널(grayscale:1, bgr:3, transparent: 4)

    img = np.full((height, width, bpp), 255, np.uint8)  # 빈 화면 표시

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection

    # For webcam input:
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    gesture_int = 0

    before_c = mark_pixel(0, 0, 0)
    pixel_c = mark_pixel(0, 0, 0)
    hm_idx = False
    finger_open_ = [False for _ in range(5)]
    gesture_time = time.time()

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            #print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            experiment_df = experiment_df.drop(experiment_df.index[0])

            # experiment_df.to_csv('video_output/' + name + '.csv', encoding='utf-8-sig')

            # exit()

            df_sum = pd.read_csv('../video_output/' + new_name + 'output_21.csv')
            # print(df_sum)
            # print(experiment_df)
            df_sum = pd.concat([df_sum, experiment_df])
            df_sum.to_csv('../video_output/' + new_name + 'output_21.csv', index=False)
            print('Saved dataframe to : ', '../video_output/' + new_name + 'output_21.csv')
            exit()

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        #x_size, y_size, channel = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            mark_p_list = []
            for hand_landmarks in results.multi_hand_landmarks: #hand_landmarks는 감지된 손의 갯수만큼의 원소 수를 가진 list 자료구조
                mark_p = []
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for i in range(21):
                    mark_p.append(mark_pixel(hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z))
                mark_p_list.append(mark_p)

            # TODO 지금 API에서 사용하는 자료형때문에 살짝 꼬였는데 mark_p(list)의 마지막 원소를 lR_idx(left or right)로 표현해놨음.
            for i in range(len(mark_p_list)): #for문 한 번 도는게 한 손에 대한 것임
                LR_idx = results.multi_handedness[i].classification[0].label
                image = cv2.putText(image, LR_idx[:], (int(mark_p_list[i][17].x * image.shape[1]), int(mark_p_list[i][17].y * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                mark_p_list[i].append(LR_idx)

                mark_c = get_center(mark_p[4], mark_p[8])

                mark_p = mark_p_list[i]
                #Handmark 정보 입력
                if len(mark_p) == 22 and hm_idx == False:
                    HM = Handmark(mark_p)
                    hm_idx = True

                #palm_vector 저장
                HM.get_palm_vector()
                HM.get_finger_vector()

                #mark_p 입력
                if hm_idx == True:
                    HM.p_list = mark_p
                    finger_open_ = np.ndarray.tolist(HM.return_finger_state())

                #정지 제스쳐 확인



                mark_p0 = mark_p[0].to_pixel()
                mark_p5 = mark_p[5].to_pixel()

            finger_info = HM.return_21_info()
            # print(finger_info)

            shared_array[:] = finger_info
            print(static_num.value)


        # image = cv2.resize(image, dsize = (0, 0), fx = 0.2, fy = 0.2)
        # image = cv2.resize(image, dsize = (0, 0), fx = 0.2, fy = =.2)


        b, g, r, a = 100, 200, 100, 70
        fontpath = "C:/Windows/Fonts/batang.ttc"
        font = ImageFont.truetype(fontpath, 100)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        if static_num.value != 0:
            text = str(series[static_num.value - 1])
            draw.text((60, 70), text, font=font, fill=(b, g, r, a))
            image = np.array(img_pil)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:

            break

    hands.close()
    cap.release()


if __name__ == "__main__":

    win32gui.ShowWindow(win32console.GetConsoleWindow(), 0)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("Copyright 2021. INBODY inc. all rights reserved")
    print("Contact : shi@inbody.com, HAIL SONG")
    print("Motion Presentation ver 1.5")

    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # width = 1024 # 너비
    # height= 600 # 높이
    shared_array = Array('d', [0. for _ in range(21)])
    static_num = Value('i', 0)

    process1 = Process(target=initialize, args=(shared_array, static_num))
    process2 = Process(target=process_static_gesture, args=(shared_array, static_num))

    # process4 = Process(target=process_load_window, args=(load_status,))

    # os.system('''start python load.pyw''')

    process1.start()
    process2.start()

    # process4.start()
    while process1.is_alive():
        pass
    process2.terminate()

    # process4.terminate()
    process1.join()
    process2.join()

    # process4.join()