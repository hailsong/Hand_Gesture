import numpy as np
import cv2
import time
import datetime

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w = 640
h = 480

fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

POSENAME = '2'
FOLDERNAME = '/POSE_' + POSENAME + '/'
VIDEO_SIZE = 45 # 프레임
index = 1
record_sign = False
timer = time.time()
frame_num = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("VideoFrame", frame)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)

    if key == 27:
        break
    elif key == 24 or record_sign == True: #TODO CTRL + X!!!!!!!!!
        print("녹화 시작")
        record = True
        record_sign = False

        name = 'C:/code_____/Hand_Gesture_2/Hand_Gesture/video_input/LSTM_DATASET2' + FOLDERNAME + str(index) + ".avi"
        video = cv2.VideoWriter(name,
                                fourcc, 30.0, (w, h))
        index += 1

    elif frame_num > VIDEO_SIZE and record == True:
        print("녹화 중지")
        print("Saved to {}".format(name))
        record = False
        record_sign = True
        frame_num = 0
        video.release()
        print('1초 뒤 재녹화, {}번쨰 영상'.format(index))

        time.sleep(1)

    if record == True:
        print("녹화 중..")
        video.write(frame)
        frame_num += 1

cap.release()
cv2.destroyAllWindows()