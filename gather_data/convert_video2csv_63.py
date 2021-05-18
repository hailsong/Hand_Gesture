import argparse

parser = argparse.ArgumentParser(description='PUT FILENAME OF VIDEO')

parser.add_argument('--target', required=True, help='FILENAME OF VIDEO')
parser.add_argument('--webcam_mode', required=False, default=False, help='환경')

args = parser.parse_args()

print(args.target)
# print(args.webcam_mode)


import cv2
import mediapipe as mp
import pyautogui
import math
import win32api, win32con, time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import sys
from numpy.core._multiarray_umath import ndarray

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils2 import vector_magnitude, norm, get_distance, Handmark, Gesture

# Experiment 1
import pandas as pd
import keyboard
import sys

from mediapipe.framework.formats import location_data_pb2

# print('Static Hand Gesture experiment 2')
# print('실험에 참여해주셔서 감사합니다! 이름을 입력해주세요.')
name = str(args.target)[:-4]

split_list = name.split('/')
new_num = split_list[-1]
split_list = split_list[:-1]
new_name = ''
for _str in split_list:
    new_name = new_name + _str
    new_name = new_name + '/'

# name = 'ㅇㅇ'
# print('실험을 시작합니다. 본 실험은 14개의 손동작을 오른손, 왼손으로 각 3번씩 취해 그 결과를 확인하고 데이터를 수집합니다.')

init_list = [[0. for _ in range(4 + 21 * 3)], ]
init_list[0][:4] = ['dummy', 'dummy', 'dummy', True]
column_name = ['FILENAME', 'real', 'LR', 'match', ]
for i in range(21 * 3):
    column_name.append(str(i))
# print(column_name)
experiment_df = pd.DataFrame.from_records(init_list, columns=column_name)
# print(experiment_df)
key_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'q', 'w', 'e', 'r']


def keyboard_check(finger_state, name, LR_idx):
    global experiment_df

    _list = [[int(new_num), int(new_num), LR_idx[0], int(new_num)]]

    for mark_p in finger_state[:-1]:
        _xyz = mark_p.to_list()
        for xyz in _xyz:
            _list[0].append(xyz)
    # print(_list)
    # for data in finger_state:
    #     for i in range(5):
    #         _list[0].append(data[i])
    local_df = pd.DataFrame.from_records(_list, columns=column_name)
    # print(local_df)
    experiment_df = experiment_df.append(local_df, ignore_index=True)


'''
mark_pixel : 각각의 랜드마크
finger_open : 손 하나가 갖고있는 랜드마크들
Gesture : 손의 제스처를 판단하기 위한 랜드마크들의 Queue
'''

mouse_use = False


# TODO landmark를 대응 인스턴스로 저장
class mark_pixel():
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
        return mark_2d(self.x * x_size, self.y * y_size)

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y, self.z - other.z


class mark_2d():
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


def get_center(p1, p2):
    return mark_pixel((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)


def hand_click(landmark, pixel):
    x = pixel.x
    y = pixel.y
    # print(x, y)
    global nowclick
    if get_distance(landmark[4], landmark[8]) < get_distance(landmark[4], landmark[3]) and nowclick == False:
        print('click')
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, int(x), int(y), 0, 0)

        nowclick = True

    elif get_distance(landmark[4], landmark[8]) > 1.5 * get_distance(landmark[4], landmark[3]) and nowclick == True:
        print('click off')

        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, int(x), int(y), 0, 0)
        nowclick = False


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
hand_shape_name = ['바위', '따봉', '총', '3-1', '4-1', '보', '4-2', '3-2', '2', '1-2', '가위', '스파이더맨', '1-1', '3-3',
                   'Fuck you']


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


def BlurFunction(src):
    with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5) as face_detection:  # with 문, mp_face_detection.FaceDetection 클래스를 face_detection으로서 사용
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


if __name__ == "__main__":

    # print("Copyright 2021. INBODY inc. all rights reserved")
    # print("Contact : shi@inbody.com, HAIL SONG")

    # width = 1024 # 너비
    # height= 600 # 높이
    width = 960  # 너비
    height = 540  # 높이
    bpp = 3  # 표시 채널(grayscale:1, bgr:3, transparent: 4)

    img = np.full((height, width, bpp), 255, np.uint8)  # 빈 화면 표시

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_face_detection = mp.solutions.face_detection

    # For webcam input:
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, upper_body_only=True)
    print('../video_input/' + str(args.target))
    print(str(args.target))
    cap = cv2.VideoCapture('../video_input/' + str(args.target))

    x_size, y_size = pyautogui.size().width, pyautogui.size().height
    nowclick = False
    # print('x, y size : ', x_size, y_size)

    gesture_int = 0

    before_c = mark_pixel(0, 0, 0)
    pixel_c = mark_pixel(0, 0, 0)
    hm_idx = False
    finger_open_ = [False for _ in range(5)]
    gesture_time = time.time()
    gesture = Gesture()

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            # print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            experiment_df = experiment_df.drop(experiment_df.index[0])

            # experiment_df.to_csv('video_output/' + name + '.csv', encoding='utf-8-sig')

            # exit()
            '''
            df_sum = pd.read_csv('../video_output/' + new_name + 'output_63.csv')
            #print(df_sum)
            #print(experiment_df)
            df_sum = pd.concat([df_sum, experiment_df])
            '''
            experiment_df.to_csv('../video_output/' + new_name + new_num + '_63.csv')
            print('Saved dataframe to : ', '../video_output/' + new_name + new_num + '_63.csv')
            exit()

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = BlurFunction(image)
        # x_size, y_size, channel = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        # results_body = pose.process(image)

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
                    mark_p.append(mark_pixel(hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y,
                                             hand_landmarks.landmark[i].z))
                mark_p_list.append(mark_p)

            # TODO 지금 API에서 사용하는 자료형때문에 살짝 꼬였는데 mark_p(list)의 마지막 원소를 lR_idx(left or right)로 표현해놨음.
            for i in range(len(mark_p_list)):  # for 한 번 도는게 한 손에 대한 것임
                LR_idx = results.multi_handedness[i].classification[0].label
                image = cv2.putText(image, LR_idx[:], (
                    int(mark_p_list[i][17].x * image.shape[1]), int(mark_p_list[i][17].y * image.shape[0])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                mark_p_list[i].append(LR_idx)

                mark_c = get_center(mark_p[4], mark_p[8])

                mark_p = mark_p_list[i]
                # Handmark 정보 입력
                if len(mark_p) == 22 and not hm_idx:
                    HM = Handmark(mark_p)
                    hm_idx = True

                # palm_vector 저장
                HM.get_palm_vector()
                HM.get_finger_vector()

                # mark_p 입력
                if hm_idx:
                    HM.p_list = mark_p
                    finger_open_ = np.ndarray.tolist(HM.return_finger_state())

                # 정지 제스쳐 확인
                static_gesture_name = static_gesture_detect(finger_open_, mark_p[-1])

                keyboard_check(HM.p_list, name, LR_idx)

                mark_p0 = mark_p[0].to_pixel()
                mark_p5 = mark_p[5].to_pixel()

                # pixel_c = mark_c.to_pixel()
                if len(LR_idx) == 5:
                    pixel_c = mark_p5
                    # gesture updating
                    gesture.update(HM)
                    if len(mark_p) == 22:
                        # gesture.gesture_detect()
                        pass

                # 마우스 움직임, 클릭
                if (get_distance(pixel_c, before_c) < get_distance(mark_p0, mark_p5)) and \
                        sum(finger_open_[3:]) == 0 and \
                        finger_open_[1] == 1 and \
                        len(LR_idx) == 5 and \
                        mouse_use == True:
                    pixel_c.mousemove()

                    if finger_open_[2] != 1:
                        hand_click(hand_landmarks.landmark, pixel_c)
                    else:
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, int(pixel_c.x), int(pixel_c.y), 0, 0)
                    # 마우스 휠
                    if finger_open_[2] == 1:
                        pixel_c.wheel(before_c)

                before_c = pixel_c

            if gesture_int > 0 and time.time() - gesture_time > 1:  # gesture int로 gesture 겹쳐지는 현상 방지
                # print('gesture_idx triggered')
                gesture_int = 0
                gesture_time = time.time()
        image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            experiment_df = experiment_df.drop(experiment_df.index[0])
            experiment_df.to_csv('../video_output/' + name + '.csv', encoding='utf-8-sig')
            print('Saved dataframe to : ', 'experiment1_' + name + '.csv')
            break

    hands.close()
    cap.release()
