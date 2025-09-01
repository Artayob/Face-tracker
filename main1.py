import cv2
import numpy as np


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Error: Unable to access camera.")
    cap.release()
    exit()

frame = cv2.resize(frame, (640, 480))


roi = cv2.selectROI("Select Object", frame, False)
x, y, w, h = map(int, roi)
track_window = (x, y, w, h)


roi_frame = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    pts = cv2.boxPoints(ret)
    pts = pts.astype(int)  
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow("Object Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
