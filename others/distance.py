import cv2
import mediapipe as mp
import numpy as np
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils



# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
      point_1 = np.array([detection.location_data.relative_keypoints[0].x,
                          detection.location_data.relative_keypoints[0].y])
      point_2 = np.array([detection.location_data.relative_keypoints[1].x,
                          detection.location_data.relative_keypoints[1].y])
      dist = np.linalg.norm(point_1 - point_2)
      dist = round(1/dist, 2)
      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(image, str(dist), (10, 60), font, 2,(0,0,155), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()