import numpy as np
import cv2

cap = cv2.VideoCapture(0)
last_frame = cap.read()[1]
y, x = last_frame.shape[:2]
x = x//5
y = y//5
last_frame = cv2.resize(last_frame,(x, y), interpolation = cv2.INTER_CUBIC)
last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
UPPER_THRESH = 50
LOWER_THRESH = 50
global lowClick, highClick
lowClick = np.array([0,0,0])
highClick = np.array([10,10,10])
firstTime = True

COLOR_THRESH = 10
kernel = np.ones((3,3),np.uint8)

def getColor(event, y, x, flags, param):
    global lowClick, highClick
    if event == cv2.EVENT_LBUTTONDOWN:
        lowClick = np.array([hsv[x][y][0]-15, hsv[x][y][1]-15, hsv[x][y][2]-15])
        highClick = np.array([hsv[x][y][0]+15, hsv[x][y][1]+15, hsv[x][y][2]+15])
        print("Mouse clicked")
        print("lower:", lowClick)
        print("highter:", highClick)

while(True):
    frame = cap.read()[1]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsvObject = cv2.inRange(hsv, lowClick, highClick)
    gray = cv2.resize(gray,(x, y), interpolation = cv2.INTER_CUBIC)
    gray2 = cv2.resize(gray,(x, y), interpolation = cv2.INTER_CUBIC)
    
    for i in range(0, y):
        for j in range(0, x):
            blurred_pixel = int(int(last_frame[i, j])*.8 + int(gray[i, j])*.2)
            motion_pixel = min(255, 2*int(abs(int(gray[i, j])-int(last_frame[i, j]))))
            
            if motion_pixel > UPPER_THRESH:
                motion_pixel = 255
            elif motion_pixel < LOWER_THRESH:
                motion_pixel = 0

            last_frame[i, j] = blurred_pixel
            gray[i, j] = motion_pixel
            gray2[i,j] = motion_pixel

    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening,kernel,iterations = 3)
    eroded = cv2.erode(hsvObject, kernel, iterations = 1)
    dilationhsv = cv2.dilate(hsvObject,kernel,iterations = 3)
    cont, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.resize(frame,(x, y), interpolation = cv2.INTER_CUBIC)
    for contour in contours:
        cv2.drawContours(cont, contour, 0, (255,0,0), 3)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)
    cv2.imshow('diff',cv2.resize(gray2,(x*5, y*5), interpolation = cv2.INTER_CUBIC))
    cv2.imshow('hsv',cv2.resize(hsv,(x*5, y*5), interpolation = cv2.INTER_CUBIC))
    cv2.imshow('last',cv2.resize(last_frame,(x*5, y*5), interpolation = cv2.INTER_CUBIC))
    cv2.imshow('Contours',cv2.resize((255-cont),(x*5, y*5), interpolation = cv2.INTER_CUBIC))
    cv2.imshow('Original', cv2.resize(frame,(x*5, y*5), interpolation = cv2.INTER_CUBIC))
    if firstTime:
        cv2.setMouseCallback('hsv', getColor)
        firstTime = False

    cv2.imshow('HSV Object', cv2.resize(hsvObject,(x*5, y*5), interpolation = cv2.INTER_CUBIC))
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

