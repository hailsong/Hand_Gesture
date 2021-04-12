import cv2
import numpy as np
import math

back = np.zeros((500, 500, 3))


x, y = 110, 250
t = 0
while True:
    x += 100 * math.cos(t)
    y += 100 * math.sin(t)
    pos = (int(x), int(y))
    t += 0.1
    print(pos)
    back = np.zeros((500, 500, 3))
    back = cv2.putText(back, 'Pyeongchang Water', pos, cv2.FONT_HERSHEY_SIMPLEX, 1,((t) % 1 ,(t-0.3)%1 ,(t-0.6)%1), 2)
    print((t) % 1 ,(t-0.3)%1 ,(t-0.6)%1 )

    back = cv2.resize(back, (250, 250))
    cv2.imshow('aa', back)
    if cv2.waitKey(1) > 0:
        break

    x, y = 110, 250

cv2.destroyAllWindows()
