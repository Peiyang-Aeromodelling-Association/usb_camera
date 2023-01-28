# coding=utf-8
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

#
# if __name__ == '__main__':
#     intrinsic_matrix = np.array([[2645.68633, 0.00000, 906.53962],
#                                  [0.00000, 2650.27742, 449.54990],
#                                  [0.00000, 0.00000, 1.00000]])
#     print(pixel_to_angle(200.53962, 449.54990, intrinsic_matrix))
