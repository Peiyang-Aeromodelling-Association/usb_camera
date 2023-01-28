# coding=utf-8
import cv2
import time
import numpy as np

# cap = cv2.VideoCapture(0)

# capture USB camera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))

display_size = (192 * 5, 108 * 5)

equalizeHist = False
denoise = True
close = True
open = True

# HSV config
value_min = 90
value_max = 255

saturation_red_min = 90
saturation_min = 100
saturation_max = 255

close_iterations = 5
open_iterations = 1

# print the hsv of the pixel of mouse
def print_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        display_hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
        print(f"HSV: {display_hsv[y, x]}")


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', print_hsv)

while True:
    ret, frame = cap.read()

    # 直方图均衡化
    if equalizeHist:
        r, g, b = cv2.split(frame)
        rH, gH, bH = cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)
        frame = cv2.merge((rH, gH, bH))

    # 去噪
    if denoise:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    # 使用 cv2.cvtColor() 函数将图像转换为 HSV 颜色空间，这样可以更容易地识别颜色
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 使用 numpy 数组运算和 cv2.inRange() 函数创建红色颜色范围的掩模
    lower_red_1 = np.array([0, saturation_red_min, value_min])
    upper_red_1 = np.array([6, saturation_max, value_max])
    lower_red_2 = np.array([170, saturation_red_min, value_min])
    upper_red_2 = np.array([180, saturation_max, value_max])
    mask_red = cv2.inRange(hsv, lower_red_1, upper_red_1) + cv2.inRange(hsv, lower_red_2, upper_red_2)

    lower_blue = np.array([90, saturation_min, value_min])
    upper_blue = np.array([125, saturation_max, value_max])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = mask_red + mask_blue

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    if open:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_list = []  # return value
    result = frame.copy()
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 100:
            continue
        epsl = 0.01 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsl, True)  # 多边形拟合
        x, y, w, h = cv2.boundingRect(approx)  # 外接矩形
        # if (float(w / h) < 2) and (float(w / h) > 0.5):  # 筛选出长宽比在0.5-2之间的矩形
        rotated_roi = cv2.minAreaRect(approx)
        roi_list.append(((x, y, w, h), rotated_roi, area))

    # arrange roi_list according to the roi area
    roi_sorted = sorted(roi_list, key=lambda k: k[2], reverse=True)  # descend

    # plot top roi_max
    roi_max = 10
    for i, (roi, rotated_roi, area) in enumerate(roi_sorted):
        if i >= roi_max:
            break
        roi_x, roi_y, roi_w, roi_h = roi
        result = cv2.rectangle(result, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 1)  # ROI in green
        rotated_box = cv2.boxPoints(rotated_roi)
        rotated_box = np.int0(rotated_box)
        result = cv2.drawContours(result, [rotated_box], 0, (255, 0, 0), 2)

    masked = cv2.bitwise_and(frame, frame, mask=mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        equalizeHist = not equalizeHist
        print('equalizeHist =', equalizeHist)
    elif key == ord('d'):
        denoise = not denoise
        print('denoise =', denoise)
    elif key == ord('c'):
        close = not close
        print('close =', close)

    frame = cv2.resize(masked, display_size)
    result = cv2.resize(result, display_size)
    display = np.hstack((frame, result))
    cv2.imshow("frame", display)

cap.release()
cv2.destroyAllWindows()
