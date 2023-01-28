# coding=utf-8
import cv2
import numpy as np
from pixel2angle import pixrel_to_angle

# 打开摄像头
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))  # MJPG

intrinsic_matrix = np.array([[2744.39055, 0.00000, 995.57729],
                             [0.00000, 2736.84056, 537.39599],
                             [0.00000, 0.00000, 1.00000]])

display_size = (192 * 5, 108 * 5)

while True:
    # 获取一帧图像
    ret, frame = cap.read()
    if not ret:
        break
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # print height and width of image
    h, w = gray.shape

    # 找到最亮的像素坐标
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    # 画绿点
    cv2.circle(frame, max_loc, 5, (0, 255, 0), -1)

    # 画一个横跨图片的十字线
    cv2.line(frame, (0, max_loc[1]), (frame.shape[1], max_loc[1]), (0, 255, 0), 1)
    cv2.line(frame, (max_loc[0], 0), (max_loc[0], frame.shape[0]), (0, 255, 0), 1)

    # 计算最亮像素的角度
    x_rel, y_rel = max_loc[0] / w, max_loc[1] / h
    x_angle_, y_angle_, theta_ = pixrel_to_angle(x_rel, y_rel, intrinsic_matrix)

    print(f"max_loc: {max_loc}, x_rel: {x_rel}, y_rel: {y_rel},"
          f" x_angle: {x_angle_}, y_angle: {y_angle_}, theta: {theta_}")

    # 显示角度
    cv2.putText(frame, f"x_angle={x_angle_}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"y_angle={y_angle_}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"theta={theta_}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 缩放到显示尺寸
    frame = cv2.resize(frame, display_size)

    # 显示图像
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头
cap.release()
cv2.destroyAllWindows()
