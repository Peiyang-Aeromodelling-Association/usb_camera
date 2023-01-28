# coding=utf-8
import cv2
import time

# cap = cv2.VideoCapture(0)

# capture USB camera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_out = cv2.VideoWriter('./videos/output.mp4', fourcc, 20.0, (1920, 1080))  # MJPG, 20 fps, 1920x1080

display_size = (192 * 5, 108 * 5)
recording = False

while True:
    ret, frame = cap.read()

    # if key is r, start recording, if key is s, stop recording
    if cv2.waitKey(1) & 0xFF == ord('r'):
        recording = True
        print("recording...")
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        break

    if recording:
        video_out.write(frame)

    frame = cv2.resize(frame, display_size)
    cv2.imshow("record", frame)

print("finished recording")
video_out.release()
cap.release()
cv2.destroyAllWindows()
