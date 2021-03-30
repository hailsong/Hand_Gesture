import numpy as np
import cv2
import time
import datetime

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w = 640
h = 480
cap.set(cv2.CAP_PROP_FPS, 20)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

FOLDERNAME = '/POSE_1/'
VIDEO_SIZE = 1.5
index = 1
record_sign = False
timer = time.time()

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

        name = 'C:/code_____/Hand_Gesture/video_input/LSTM_DATASET' + FOLDERNAME + str(index) + ".avi"
        video = cv2.VideoWriter(name,
                                fourcc, 20.0, (w, h))
        index += 1
        timer = time.time()
    elif time.time() - timer > VIDEO_SIZE and record == True:
        print("녹화 중지")
        print("Saved to {}".format(name))
        record = False
        record_sign = True
        video.release()
        print('1초 뒤 재녹화, {}번쨰 영상'.format(index))
        time.sleep(1)

    if record == True:
        print("녹화 중..")
        video.write(frame)

cap.release()
cv2.destroyAllWindows()