import cv2
import colourCalibrate as cc

img = cv2.imread("../img/sony1.jpg")

cv2.imshow("before calibration",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

calibratedImg = cc.calibrate(img)
cv2.imwrite("../img/calibratedImg.png",calibratedImg)
cv2.imshow("after calibration",calibratedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

