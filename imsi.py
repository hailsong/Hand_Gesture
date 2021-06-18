import mediapipe as mp
import cv2
mp_iris = mp.solutions.iris
iris = mp_iris.Iris()
image = cv2.imread('dummy.jpg')
results = iris.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
print(results.face_landmarks_with_iris)
iris.close()