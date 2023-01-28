# coding=utf-8
import cv2
import numpy as np
import glob
from tqdm import tqdm
import os

display_size = (192 * 5, 108 * 5)

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("images_for_calibration/*.jpg")
# print(f"images = {images}")
count = 0
for i,fname in enumerate(tqdm(images)):
    # print(f"{i}/{len(images)}")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size = gray.shape[::-1]
    # ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    # print(ret)

    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        # cv2.drawChessboardCorners(img, (8, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        img = cv2.resize(img, display_size)
        cv2.imshow('img', img)
        cv2.waitKey(20)
    else:
        # delete the image
        os.remove(fname)

print(f"len(obj_points) = {len(obj_points)}")
print(f"len(img_points) = {len(img_points)}")
cv2.destroyAllWindows()

print("calibrating...")

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print(f"mtx = [[{mtx[0][0]:.5f}, {mtx[0][1]:.5f}, {mtx[0][2]:.5f}], \n"
      f"[{mtx[1][0]:.5f}, {mtx[1][1]:.5f}, {mtx[1][2]:.5f}], \n"
      f"[{mtx[2][0]:.5f}, {mtx[2][1]:.5f}, {mtx[2][2]:.5f}]]")  # 内参数矩阵
# print("dist:\n", dist)
print(f"dist = [{dist[0][0]:.5f}, {dist[0][1]:.5f}, {dist[0][2]:.5f}, {dist[0][3]:.5f}, {dist[0][4]:.5f}]") # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
# print("tvecs:\n", tvecs ) # 平移向量  # 外参数

print(f"center point: u={mtx[0][2]:.2f}, v={mtx[1][2]:.2f}")
print(f"fx={mtx[0][0]:.2f}, fy={mtx[1][1]:.2f}")

print("-----------------------------------------------------")