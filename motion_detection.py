import numpy as np
import cv2

cap = cv2.VideoCapture(0)
last_frame = cap.read()[1]
y, x = last_frame.shape[:2]
x = x//5
y = y//5
last_frame = cv2.resize(last_frame,(x, y), interpolation = cv2.INTER_CUBIC)
last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)


while(True):
    frame = cap.read()[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray,(x, y), interpolation = cv2.INTER_CUBIC)

    for i in range(0, y):
        for j in range(0, x):
            last_frame[i, j] = min(255, 2*int(int(last_frame[i, j])*.35 + int(gray[i, j])*.15))
            gray[i, j] = min(255, 2*int(abs(int(gray[i, j])-int(last_frame[i, j]))))

    cv2.imshow('diff',cv2.resize(gray,(x*5, y*5), interpolation = cv2.INTER_CUBIC))
    cv2.imshow('last',cv2.resize(last_frame,(x*5, y*5), interpolation = cv2.INTER_CUBIC))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
