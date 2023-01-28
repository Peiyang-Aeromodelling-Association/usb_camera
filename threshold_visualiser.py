# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:14:44 2022
@author: JAMES Fei
"""
import cv2
import numpy as np
import PySimpleGUI as sg


def color_filter(inputimg, LOW=None, UPE=None, mod="lab"):
    """
    通过颜色上下限，过滤图像
    """
    if inputimg.shape[2] == 3:
        if mod == 'rgb' or mod == 0:
            rhl = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)
        elif mod == 'hsv' or mod == 1:
            rhl = cv2.cvtColor(inputimg, cv2.COLOR_BGR2HSV)
        elif mod == 'lab' or mod == 2:
            rhl = cv2.cvtColor(inputimg, cv2.COLOR_BGR2LAB)
        # 变换图像 = cv2.GaussianBlur(变换图像, (7, 7), 0)
        if type(LOW) == type([]):
            LOW = np.array(LOW)
        if type(UPE) == type([]):
            UPE = np.array(UPE)
        mask = cv2.inRange(rhl, LOW, UPE)
        output = cv2.bitwise_and(inputimg, inputimg, mask=mask)
        # print(output.shape)
        return output
    else:
        print("不是RGB图像")


def find_hsv_range(color="红1"):
    hsv_colormap = {
        "黑": {
            "lower": np.array([0, 0, 0]),
            "upper": np.array([180, 180, 46])
        },
        "灰": {
            "lower": np.array([0, 0, 46]),
            "upper": np.array([180, 43, 220])
        },
        "白": {
            "lower": np.array([0, 0, 221]),
            "upper": np.array([180, 30, 225])
        },
        "红1": {
            "lower": np.array([156, 43, 46]),
            "upper": np.array([180, 255, 225])
        },
        "红2": {
            "lower": np.array([0, 43, 46]),
            "upper": np.array([10, 255, 225])
        },
        "橙": {
            "lower": np.array([11, 43, 46]),
            "upper": np.array([25, 255, 225])
        },
        "黄": {
            "lower": np.array([26, 43, 46]),
            "upper": np.array([34, 255, 225])
        },
        "黄1": {
            "lower": np.array([20, 180, 221]),
            "upper": np.array([31, 255, 225])
        },
        "绿": {
            "lower": np.array([35, 43, 46]),
            "upper": np.array([77, 255, 225])
        },
        "青": {
            "lower": np.array([78, 43, 46]),
            "upper": np.array([99, 255, 225])
        },
        "蓝": {
            "lower": np.array([100, 43, 46]),
            "upper": np.array([124, 255, 225])
        },
        "紫": {
            "lower": np.array([125, 43, 46]),
            "upper": np.array([155, 255, 225])
        },
    }
    if type(color) == type(''):
        if color in hsv_colormap.keys():
            lower = hsv_colormap[color]["lower"]
            upper = hsv_colormap[color]["upper"]
            return True, lower, upper
    else:
        print("wrong color type")
        return False


def color_filterTH(inputimg, mblock, isoutput=True, module=None, **args):
    # output=None

    modnum = {
        'lab': 2,
        'rgb': 0,
        "hsv": 1,
        1: "hsv",
        0: 'rgb',
        2: 'lab'

    }

    argTH = {
        "LOW": [50, 0, 0],
        "UPE": [180, 100, 50],
        "mod": 1
    }

    if type(mblock["args"]) == type({}):
        for k, v in mblock["args"].items():
            if k in argTH.keys():
                argTH[k] = v

    layout = [
        [sg.Text('色彩空间', size=(10, 1)), sg.Radio('RGB', "333", key="-RGB-"),
         sg.Radio('LAB', "333", key="-LAB-"),
         sg.Radio('HSV', "333", default=True, key="-HSV-")],
        [sg.Radio('手动查找', "111", default=True, key="-manhand-"),
         sg.Radio('专家查找', "111", key="-exphand-"),
         sg.Combo(("黑", "灰", "白", "红1", "红2", "橙", "黄", "绿", "青", "蓝", "紫"), default_value="红1",
                  size=(10, 1), key='-hsvcolorrange-')],
        [sg.Text('L(R/H)低', size=(10, 1)),
         sg.Slider((0, 255), argTH["LOW"][0], 1, orientation='h', size=(20, 15), key='-lrhlow-'),
         sg.Text('L(R/H)高', size=(10, 1)),
         sg.Slider((0, 255), argTH["UPE"][0], 1, orientation='h', size=(20, 15), key='-lrhup-'),
         ],
        [sg.Text('A(G/S)低', size=(10, 1)),
         sg.Slider((0, 255), argTH["LOW"][1], 1, orientation='h', size=(20, 15), key='-agslow-'),
         sg.Text('A(G/S)高', size=(10, 1)),
         sg.Slider((0, 255), argTH["UPE"][1], 1, orientation='h', size=(20, 15), key='-agsup-'),
         ],
        [sg.Text('B(B/V)低', size=(10, 1)),
         sg.Slider((0, 255), argTH["LOW"][2], 1, orientation='h', size=(20, 15), key='-bbvlow-'),
         sg.Text('B(B/V)高', size=(10, 1)),
         sg.Slider((0, 255), argTH["UPE"][2], 1, orientation='h', size=(20, 15), key='-bbvup-')
         ],
        [sg.Button('Exit')],
        [sg.Image(filename='', key='-IMAGE-')],

    ]
    win = sg.Window('颜色选择器TH', layout, disable_close=True)

    def flash(win):
        win_active = True
        event = True
        mod = modnum[argTH["mod"]]
        event, values = win.read(timeout=20)
        if mod == "lab":
            win['-LAB-'].update(value=True)
        if mod == "rgb":
            win['-RGB-'].update(value=True)
        if mod == "hsv":
            win['-HSV-'].update(value=True)

        while win_active:
            event, values = win.read(timeout=20)
            if values["-RGB-"]:
                mod = 'rgb'
                win['-lrhlow-'].update(range=(0, 255))
                if values['-lrhlow-'] < 0:
                    win['-lrhlow-'].update(value=0)
                win['-lrhup-'].update(range=(0, 255))
                if values['-lrhup-'] < 0:
                    win['-lrhup-'].update(value=5)

                win['-agslow-'].update(range=(0, 255))
                if values['-agslow-'] < 0:
                    win['-agslow-'].update(value=0)
                win['-agsup-'].update(range=(0, 255))
                if values['-agsup-'] < 0:
                    win['-agsup-'].update(value=5)

                win['-bbvlow-'].update(range=(0, 255))
                if values['-bbvlow-'] < 0:
                    win['-bbvlow-'].update(value=0)
                win['-bbvup-'].update(range=(0, 255))
                if values['-bbvup-'] < 0:
                    win['-bbvup-'].update(value=5)

            if values["-LAB-"]:
                mod = 'lab'
                win['-lrhlow-'].update(range=(0, 255))
                if values['-lrhlow-'] < 0:
                    win['-lrhlow-'].update(value=0)
                win['-lrhup-'].update(range=(0, 255))
                if values['-lrhup-'] < 0:
                    win['-lrhup-'].update(value=5)

                win['-agslow-'].update(range=(0, 255))
                if values['-agslow-'] < 0:
                    win['-agslow-'].update(value=0)
                win['-agsup-'].update(range=(0, 255))
                if values['-agsup-'] < 0:
                    win['-agsup-'].update(value=5)

                win['-bbvlow-'].update(range=(0, 255))
                if values['-bbvlow-'] < 0:
                    win['-bbvlow-'].update(value=0)
                win['-bbvup-'].update(range=(0, 255))
                if values['-bbvup-'] < 0:
                    win['-bbvup-'].update(value=5)

            if values["-HSV-"]:
                mod = 'hsv'
                win['-lrhlow-'].update(range=(0, 180))
                if values['-lrhlow-'] > 180:
                    win['-lrhlow-'].update(value=180)
                win['-lrhup-'].update(range=(0, 180))
                if values['-lrhup-'] > 180:
                    win['-lrhup-'].update(value=180)

                win['-agslow-'].update(range=(0, 255))
                if values['-agslow-'] < 0:
                    win['-agslow-'].update(value=0)
                win['-agsup-'].update(range=(0, 255))
                if values['-agsup-'] < 0:
                    win['-agsup-'].update(value=5)

                win['-bbvlow-'].update(range=(0, 255))
                if values['-bbvlow-'] < 0:
                    win['-bbvlow-'].update(value=0)
                win['-bbvup-'].update(range=(0, 255))
                if values['-bbvup-'] < 0:
                    win['-bbvup-'].update(value=5)

            LOW = np.array([values['-lrhlow-'], values['-agslow-'], values['-bbvlow-']])
            UPE = np.array([values['-lrhup-'], values['-agsup-'], values['-bbvup-']])

            if values["-HSV-"] and values["-exphand-"]:
                mod = 'hsv'
                res, LOW, UPE = find_hsv_range(color=values['-hsvcolorrange-'])
                win['-lrhlow-'].update(value=LOW[0])
                win['-agslow-'].update(value=LOW[1])
                win['-bbvlow-'].update(value=LOW[2])
                win['-lrhup-'].update(value=UPE[0])
                win['-agsup-'].update(value=UPE[1])
                win['-bbvup-'].update(value=UPE[2])
                if res:
                    pass
                else:
                    LOW = np.array([int(values['-lrhlow-']), int(values['-agslow-']), int(values['-bbvlow-'])])
                    UPE = np.array([int(values['-lrhup-']), int(values['-agsup-']), int(values['-bbvup-'])])

            output = color_filter(inputimg, LOW=LOW, UPE=UPE, mod=mod)

            if type(mblock["args"]) == type({}):
                mblock["args"]["LOW"] = LOW.tolist()
                mblock["args"]["UPE"] = UPE.tolist()
                mblock["args"]["mod"] = modnum[mod]

            if type(mblock["args"]) == type([]):
                if len(mblock["args"]):
                    mblock["args"].clear()
                    mblock["args"].append({"LOW": LOW.tolist()})
                    mblock["args"].append({"UPE": UPE.tolist()})
                    mblock["args"].append({"mod": mod})

                else:
                    mblock["args"].append({"LOW": LOW.tolist()})
                    mblock["args"].append({"UPE": UPE.tolist()})
                    mblock["args"].append({"mod": mod})

            output, ratio = resizeoutput(output)
            imgbytes = cv2.imencode('.png', output)[1].tobytes()
            win['-IMAGE-'].update(data=imgbytes)
            # win['-OUTPUT-'].update("video window:"+str(task0.is_alive()))
            if event is None or event == 'Exit':
                win.close()
                win_active = False
                break

    flash(win)


def resizeoutput(output, maxw=800, maxh=800):
    H = output.shape[0]
    W = output.shape[1]
    ratio = None
    if W >= H:
        if W > maxw:
            gsizew = maxw
            gsizeh = int(H / W * maxw)
            ratio = maxw / W
        else:
            gsizew = W
            gsizeh = H
            ratio = 1
    else:
        if H > maxh:
            gsizeh = maxh
            gsizew = int(W / H * maxh)
            ratio = maxh / H
        else:
            gsizew = W
            gsizeh = H
            ratio = 1
    pic = cv2.resize(output, (gsizew, gsizeh), interpolation=cv2.INTER_LINEAR)
    return pic, ratio


if __name__ == '__main__':
    mblock = {
    }

    img = cv2.imread('images_for_calibration/1674048759.2812796.jpg')
    display_size = (192 * 5, 108 * 5)
    img = cv2.resize(img, display_size, interpolation=cv2.INTER_LINEAR)
    mock = {
        "args": {}

    }
    color_filterTH(img.copy(), mock, isoutput=False)
    LOW = mock["args"]["LOW"]
    UPE = mock["args"]["UPE"]
    mod = mock["args"]["mod"]
    print("找到的参数为", mock)