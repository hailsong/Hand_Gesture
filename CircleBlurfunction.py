import cv2
import mediapipe as mp
import math
import numpy as np
from numpy.core._multiarray_umath import ndarray

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
from mediapipe.framework.formats import location_data_pb2

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
                try :
                    x1 = int((rect_start_point[0]+rect_end_point[0])/2)
                    y1 = int((rect_start_point[1]+rect_end_point[1])/2)
                    a = int(rect_end_point[0]-rect_start_point[0])
                    b = int(rect_end_point[1]-rect_start_point[1])
                    radius = int(math.sqrt(a*a+b*b)/2*0.7)
                    # 원 스펙 설정
                    cv2.circle(c_mask, (x1,y1), radius, (255,255,255), -1)
                except :
                    pass
            img_all_blurred = cv2.blur(image, (17,17))
            c_mask = cv2.cvtColor(c_mask, cv2.COLOR_GRAY2BGR)
            print(c_mask.shape)
            image = np.where(c_mask > 0, img_all_blurred, image)
    return image

face = cv2.imread("C:/Users/user/Desktop/CV/Image/together.jpg")
face = BlurFunction(face)
cv2.imshow('face',face)
cv2.waitKey(0)
cv2.destroyAllWindows()