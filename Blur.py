import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
from mediapipe.framework.formats import location_data_pb2

cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)



with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands, mp_face_detection.FaceDetection(min_detection_confidence = 0.5) as face_detection:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results1 = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_rows, image_cols, _ = image.shape
        if results1.detections:
            for detection in results1.detections:
                if not detection.location_data:
                    break
                if image.shape[2] != 3:
                    raise ValueError('Input image must contain three channel rgb data.')
                location = detection.location_data
                if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
                    raise ValueError(
                        'LocationData must be relative for this drawing funtion to work.')
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
                    x1 = int(rect_start_point[0])
                    x2 = int(rect_end_point[0])
                    y1 = int(rect_start_point[1])
                    y2 = int(rect_end_point[1])
                    dst = image.copy()
                    dst = dst[y1:y2,x1:x2]
                    image[y1:y2,x1:x2] = cv2.blur(dst, (9,9))
                except :
                    pass
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results2 = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results2.multi_hand_landmarks:
                for hand_landmarks in results2.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()