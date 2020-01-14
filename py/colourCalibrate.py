#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu  Dec 12 2019
@author: Mingkon
@E-mail: zhangmingge@shdingyun.com

"""


import numpy as np
import cv2
from cv2 import aruco
from PIL import Image

ALPHA = 0.8 # Magnitude of the CCM

def getColourBlocks(block00, img, step_x, step_y, delta_x, delta_y):
    ColourBlocks = []
    for i in range(0,6):
        for j in range(0,4):
            block_x = block00[0] + j * step_x + delta_x
            block_y = block00[1] + i * step_y + delta_y
            side = int(step_x / 4)
            ColourBlock = img[
                          int(block_y) - side:int(block_y) + side,
                          int(block_x) - side:int(block_x) + side, :]
            ColourBlocks.append(
                (ColourBlock[:, :, 2].mean(),  # r channel
                 ColourBlock[:, :, 1].mean(),  # g channel
                 ColourBlock[:, :, 0].mean())  # b channel
            )
            # cv2.imshow("ColourBlock_{:0>2d}".format(i), ColourBlock)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    return ColourBlocks


def detectColourBlock(img):
    """
    :param img:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)

    # aruco_dict = aruco.generateCustomDictionary(36, 5)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    corner = {}
    for i in range(len(ids)):
        c = corners[i][0]
        corner[str(ids[i][0])] = c

    corner04 = corner['4']
    corner03 = corner['3']
    corner02 = corner['2']
    corner01 = corner['1']

    top_start = corner01[:, 0].mean(), corner01[:, 1].mean()  # the position centre of the top left binary code
    top_end = corner02[:, 0].mean(), corner02[:, 1].mean()  # the position centre of the top right binary code
    bottom_start = corner03[:, 0].mean(), corner03[:, 1].mean()  # the position centre of the bottom left binary code
    bottom_end = corner04[:, 0].mean(), corner04[:, 1].mean()  # the position centre of the bottom right binary code




    length_x = top_end[0] - top_start[0]
    length_y = bottom_start[1] - top_start[1]
    shift_x = bottom_start[0] - top_start[0]
    shift_y = top_end[1] - top_start[1]


    step_x = int(length_x * 6 / 31)
    step_y = int(length_y * 6 / 43)
    delta_x = int(shift_x / 7)
    delta_y = int(shift_y / 5)

    top_block00 = (int(top_start[0] + 1.083 * step_x),int(top_start[1] + 1.083 * step_y))

    colour_blocks = getColourBlocks(top_block00, img, step_x, step_y, delta_x, delta_y)

    colour_blocks = np.array(colour_blocks)
    print("colour_blocks shape: ", colour_blocks.shape)  ### 24 * 3

    return colour_blocks

## for the PIL Image
def sRGB2XYZ(img):
    # D50
    # rgb2xyz = (0.4360747  0.3850649  0.1430804, 0,
    #            0.2225045  0.7168786  0.0606169, 0,
    #            0.0139322  0.0971045  0.7141733, 0)
    # D 65
    rgb2xyz = (
        0.412391, 0.357584, 0.180481, 0,
        0.212639, 0.715169, 0.072192, 0,
        0.019331, 0.119195, 0.950532, 0
    )
    return img.convert("RGB", rgb2xyz)

## for the PIL Image
def XYZ2sRGB(img):
    # D50
    # xyz2rgb = (3.1338561 -1.6168667 -0.4906146, 0,
    #            -0.9787684  1.9161415  0.0334540, 0,
    #            0.0719453 -0.2289914  1.4052427, 0)
    # D65
    xyz2rgb = (3.240970, -1.537383, -0.498611, 0,
               -0.969244, 1.875968, 0.041555, 0,
               0.055630, -0.203977, 1.056972, 0)
    return img.convert("RGB", xyz2rgb)


## for the numpy array
def conv_sRGB2XYZ(rgb):
    # D 50
    # M = np.array([[0.4360747  0.3850649  0.1430804]
    #               [0.2225045  0.7168786  0.0606169]
    #               [0.0139322  0.0971045  0.7141733]])
    # D 65
    M = np.array([[0.412391, 0.357584, 0.180481],
                  [0.212639, 0.715169, 0.072192],
                  [0.019331, 0.119195, 0.950532]])
    return np.dot(M, rgb.T).T


def correctColor(img, ccm):
    return img.convert("RGB", tuple(ccm.transpose().flatten()))


def gamma_table(gamma_r, gamma_g, gamma_b, gain_r=1.0, gain_g=1.0, gain_b=1.0):
    r_tbl = [min(255, int((x / 255.) ** (gamma_r) * gain_r * 255.)) for x in range(256)]
    g_tbl = [min(255, int((x / 255.) ** (gamma_g) * gain_g * 255.)) for x in range(256)]
    b_tbl = [min(255, int((x / 255.) ** (gamma_b) * gain_b * 255.)) for x in range(256)]
    return r_tbl + g_tbl + b_tbl


def applyGamma(img, gamma=2.2):
    inv_gamma = 1. / gamma
    return img.point(gamma_table(inv_gamma, inv_gamma, inv_gamma))


def deGamma(img, gamma=2.2):
    return img.point(gamma_table(gamma, gamma, gamma))


def calibrate(img):
    ## get colour blocks
    stdRGBBlocks = np.array(
        [(255, 0, 0),
         (220, 20, 60),
         (250, 128, 114),
         (255, 99, 71),
         (255, 165, 0),
         (255, 215, 0),
         (255, 255, 0),
         (173, 255, 47),
         (0, 255, 0),
         (0, 128, 0),
         (0, 255, 255),
         (30, 144, 255),
         (0, 0, 255),
         (0, 0, 139),
         (138, 43, 226),
         (128, 0, 128),
         (255, 0, 255),
         (255, 20, 147),
         (255, 192, 203),
         (255, 255, 224),
         (255, 255, 255),
         (192, 192, 192),
         (128, 128, 128),
         (0, 0, 0)]
    ) / 255.0
    imgRGBBlocks = detectColourBlock(img) / 255.0

    # Degamma
    gamma = 1.0  # 2.0
    std_linear = np.power(stdRGBBlocks, gamma)
    img_linear = np.power(imgRGBBlocks, gamma)

    # XYZ
    std_xyz = conv_sRGB2XYZ(std_linear)
    img_xyz = conv_sRGB2XYZ(img_linear)

    # Solve
    # img_xyz * ccm == std_xyz
    # (24, 3 + 1) * (4, 3) = (24 * 3)
    img_xyz_hm = np.append(img_xyz, np.ones((24, 1)), axis=1)
    ccm = np.linalg.pinv(img_xyz_hm).dot(std_xyz) * ALPHA

    input_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # from opencv formate to PIL formate

    #  PIL read image: input_img = Image.open(args.input, 'r').convert("RGB")
    input_img = deGamma(input_img, gamma=gamma)
    input_img = sRGB2XYZ(input_img)
    input_img = correctColor(input_img, ccm)
    input_img = XYZ2sRGB(input_img)
    input_img = applyGamma(input_img, gamma=gamma)

    output_img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

    return output_img
