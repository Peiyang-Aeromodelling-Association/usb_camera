# coding=utf-8
import cv2
import time

# cap = cv2.VideoCapture(0)

# capture USB camera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))  # MJPG

display_size = (192 * 5, 108 * 5)

while True:
    ret, frame = cap.read()

    # if key is s, save to image_for_calibration
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("images_for_calibration/" + str(time.time()) + ".jpg", frame)
        print(f"saved image to images_for_calibration/{time.time()}.jpg")

    frame = cv2.resize(frame, display_size)
    cv2.imshow("record", frame)

cap.release()
cv2.destroyAllWindows()