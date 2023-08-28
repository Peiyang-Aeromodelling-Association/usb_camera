# coding=utf-8
import cv2
import numpy as np
import glob
from tqdm import tqdm
import os

# Parameters
chessboard_size = (9, 6)  # 棋盘格尺寸，交叉点个数
display_size = (192 * 5, 108 * 5)

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("images_for_calibration/*.jpg")

for i, fname in enumerate(tqdm(images)):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_size[0], chessboard_size[1]), None)

    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (chessboard_size[0], chessboard_size[1]), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
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

print("reprojection error: ", ret)  # 得到重投影误差, RMS

print(f"mtx = \t[[{mtx[0][0]:.5f}, \t{mtx[0][1]:.5f}, \t{mtx[0][2]:.5f}], \n"
      f"\t[{mtx[1][0]:.5f}, \t{mtx[1][1]:.5f}, \t{mtx[1][2]:.5f}], \n"
      f"\t[{mtx[2][0]:.5f}, \t{mtx[2][1]:.5f}, \t{mtx[2][2]:.5f}]]")  # 内参数矩阵

print(f"dist = [{dist[0][0]:.5f}, {dist[0][1]:.5f}, {dist[0][2]:.5f}, {dist[0][3]:.5f}, {dist[0][4]:.5f}]") # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

print(f"center point: u={mtx[0][2]:.2f}, v={mtx[1][2]:.2f}")

print("-----------------------------------------------------")