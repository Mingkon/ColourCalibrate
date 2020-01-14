#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu  Dec 12 2019
@author: Mingkon
@E-mail: zhangmingge@shdingyun.com

"""




import numpy as np
import cv2
import PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd



"""生成二维码"""
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)

# fig = plt.figure()
# nx = 4
# ny = 3
# for i in range(1, nx*ny+1):
#     ax = fig.add_subplot(ny,nx, i)
#     img = aruco.drawMarker(aruco_dict,i, 700)
#     plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
#     ax.axis("off")
#
# plt.savefig("markers.jpeg")
# plt.show()
# plt.close()
for i in range(1,5):
    img = aruco.drawMarker(aruco_dict,i, 700)
    cv2.imwrite("aruco_DICT_6X6_250_{:0>3d}.png".format(i),img)
    
    

"""生成彩色条"""
top_bar = np.zeros((140,1460,3),dtype='uint8')
cv2.imshow("top base:",top_bar)
cv2.waitKey(0)
cv2.destroyAllWindows()



###############################
#######  colour code  #########
### red	#FF0000	(255,0,0)
### crimson	#DC143C	(220,20,60)
### salmon	#FA8072	(250,128,114)
### tomato	#FF6347	(255,99,71)
### orange	#FFA500	(255,165,0)
### gold	#FFD700	(255,215,0)
### yellow	#FFFF00	(255,255,0)
### green yellow	#ADFF2F	(173,255,47)
### lime	#00FF00	(0,255,0)
### green	#008000	(0,128,0)
### aqua	#00FFFF	(0,255,255)
### dodger blue	#1E90FF	(30,144,255)
###############################
top_colours = [
    (255,0,0),
    (220,20,60),
    (250,128,114),
    (255,99,71),
    (255,165,0),
    (255,215,0),
    (255,255,0),
    (173,255,47),
    (0,255,0),
    (0,128,0),
    (0,255,255),
    (30,144,255)
]

for i in range(12):
    colour = top_colours[i]
    top_bar[20:120,20+i*120:120+i*120,0] = colour[2]
    top_bar[20:120,20+i*120:120+i*120,1] = colour[1]
    top_bar[20:120,20+i*120:120+i*120,2] = colour[0]
    cv2.imshow("top base {:0>2d}".format(i),top_bar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#cv2.imwrite("top_bar.jpg",top_bar)
#cv2.imshow("top base",top_bar)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#### bottom bar
    
bottom_bar = np.zeros((140,1460,3),dtype='uint8')
cv2.imshow("bottom base ",bottom_bar)
cv2.waitKey(0)
cv2.destroyAllWindows()

###############################
#######  colour code  #########
### blue	#0000FF	(0,0,255)
### dark blue	#00008B	(0,0,139)
### blue violet	#8A2BE2	(138,43,226)
### purple	#800080	(128,0,128)
### magenta / fuchsia	#FF00FF	(255,0,255)
### deep pink	#FF1493	(255,20,147)
### pink	#FFC0CB	(255,192,203)
### light yellow	#FFFFE0	(255,255,224)
### white	#FFFFFF	(255,255,255)
### silver	#C0C0C0	(192,192,192)
### gray / grey	#808080	(128,128,128)
### black	#000000	(0,0,0)
###############################
bottom_colours = [
    (0,0,255),
    (0,0,139),
    (138,43,226),
    (128,0,128),
    (255,0,255),
    (255,20,147),
    (255,192,203),
    (255,255,224),
    (255,255,255),
    (192,192,192),
    (128,128,128),
    (0,0,0)
]

for i in range(12):
    colour = bottom_colours[i]
    bottom_bar[20:120,20+i*120:120+i*120,0] = colour[2]
    bottom_bar[20:120,20+i*120:120+i*120,1] = colour[1]
    bottom_bar[20:120,20+i*120:120+i*120,2] = colour[0]
    cv2.imshow("bottom base {:0>2d}".format(i),bottom_bar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#cv2.imwrite("bottom_bar.jpg",bottom_bar)
#print("bottom_bar shape: ", bottom_bar.shape)

binary_code01 = cv2.imread("aruco_DICT_6X6_250_001.png")
binary_code02 = cv2.imread("aruco_DICT_6X6_250_002.png")
binary_code03 = cv2.imread("aruco_DICT_6X6_250_003.png")
binary_code04 = cv2.imread("aruco_DICT_6X6_250_004.png")
#print("binary_code01.shape: ", binary_code01.shape)

binary_code01_rez = cv2.resize(binary_code01,(140,140))
binary_code02_rez = cv2.resize(binary_code02,(140,140))
binary_code03_rez = cv2.resize(binary_code03,(140,140))
binary_code04_rez = cv2.resize(binary_code04,(140,140))
#cv2.imshow("binary_code01_rez",binary_code01_rez)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

top_binary_code_colour_bar = np.zeros((140,1780,3),dtype='uint8')
top_binary_code_colour_bar[:,0:140,:] = binary_code01_rez
top_binary_code_colour_bar[:,140:160,:] = 255
top_binary_code_colour_bar[:,160:1620,:] = top_bar
top_binary_code_colour_bar[:,1620:1640,:] = 255
top_binary_code_colour_bar[:,1640:,:] = binary_code02_rez
cv2.imshow("top_binary_code_colour_bar",top_binary_code_colour_bar)
cv2.waitKey(0)
cv2.destroyAllWindows()

final_top_bar = np.zeros((160,1800,3),dtype='uint8')
final_top_bar[:,:,:] = 255
final_top_bar[10:-10,10:-10,:] = top_binary_code_colour_bar
cv2.imwrite("final_top_bar.jpg",final_top_bar)
cv2.imshow("final_top_bar",final_top_bar)
cv2.waitKey(0)
cv2.destroyAllWindows()


bottom_binary_code_colour_bar = np.zeros((140,1780,3),dtype='uint8')
bottom_binary_code_colour_bar[:,0:140,:] = binary_code03_rez
bottom_binary_code_colour_bar[:,140:160,:] = 255
bottom_binary_code_colour_bar[:,160:1620,:] = bottom_bar
bottom_binary_code_colour_bar[:,1620:1640,:] = 255
bottom_binary_code_colour_bar[:,1640:,:] = binary_code04_rez
cv2.imshow("bottom_binary_code_colour_bar",bottom_binary_code_colour_bar)
cv2.waitKey(0)
cv2.destroyAllWindows()

final_bottom_bar = np.zeros((160,1800,3),dtype='uint8')
final_bottom_bar[:,:,:] = 255
final_bottom_bar[10:-10,10:-10,:] = bottom_binary_code_colour_bar
cv2.imwrite("final_bottom_bar.jpg",final_bottom_bar)
cv2.imshow("final_bottom_bar",final_bottom_bar)
cv2.waitKey(0)
cv2.destroyAllWindows()

final_bar = np.zeros((800,1800,3),dtype='uint8')
final_bar[:,:,:] = 255
final_bar[100:260,:,:] = final_top_bar
final_bar[540:700,:,:] = final_bottom_bar
cv2.imwrite("final_bar.jpg",final_bar)
cv2.imshow("final_bar",final_bar)
cv2.waitKey(0)
cv2.destroyAllWindows()
