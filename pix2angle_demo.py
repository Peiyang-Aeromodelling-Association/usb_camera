# coding=utf-8
import cv2
import numpy as np
import math


def pixel_to_angle(x, y, intrinsic_matrix, in_radians=False):
    """
    It converts a pixel coordinate to an angle coordinate

    :param x: x-coordinate of the pixel
    :param y: The y-coordinate of the pixel
    :param intrinsic_matrix: The intrinsic matrix of the camera
    :param in_radians: If True, the output will be in radians. Otherwise, it will be in degrees, defaults to False
    (optional)
    :return: The x and y angles of the pixel in the image.
    """
    # 将像素坐标转换为归一化坐标
    normalized_coord = np.linalg.inv(intrinsic_matrix).dot([x, y, 1])
    # 将归一化坐标转换为欧拉角
    x_angle = math.atan2(normalized_coord[0], normalized_coord[2])
    y_angle = math.atan2(normalized_coord[1], normalized_coord[2])
    # get the angle between pixel and optical axis
    angle = math.atan2(math.sqrt(normalized_coord[0] ** 2 + normalized_coord[1] ** 2), normalized_coord[2])
    if in_radians:
        return x_angle, y_angle, angle
    else:
        return math.degrees(x_angle), math.degrees(y_angle), math.degrees(angle)


def pixrel_to_angle(x_rel, y_rel, intrinsic_matrix, in_radians=False):
    # convert relative coordinate to absolute pixel coordinate
    x = x_rel * intrinsic_matrix[0][2] * 2
    y = y_rel * intrinsic_matrix[1][2] * 2
    return pixel_to_angle(x, y, intrinsic_matrix, in_radians)


# 打开摄像头
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))  # MJPG

# TODO: change this to your camera's intrinsic matrix
intrinsic_matrix = np.array([[2693.08823, 0.00000, 1054.17649],
                             [0.00000, 2702.62836, 503.34463],
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

    print(f"max_loc: {max_loc},\t x_rel: {x_rel:.5f},\t y_rel: {y_rel:.5f},"
          f"\t x_angle: {x_angle_:.5f},\t y_angle: {y_angle_:.5f},\t theta: {theta_:.5f}")

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
