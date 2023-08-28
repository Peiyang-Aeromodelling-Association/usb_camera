# USB摄像头内参标定

内参标定时，棋盘格的宽度没有要求。

## 标定流程

1. 新建`images_for_calibrations`文件夹。
2. 修改`record_images.py`中相机index，运行`python record_images.py`，按`s`键拍摄棋盘格图片。
3. 运行`python calibrate.py`。该脚本读取`images_for_calibrations`文件夹下的图片，进行标定，输出内参矩阵和畸变系数。

## 亮斑追踪demo

运行`python pix2angle_demo.py`，会显示亮斑（如手机摄像头）的角度。