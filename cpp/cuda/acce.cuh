//
// Created by mingkon on 1/10/20.
//

#ifndef COLOR_CALIBRATE_CPP_ACCE_H
#define COLOR_CALIBRATE_CPP_ACCE_H

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


extern "C"
void useCUDA();

extern "C"
void getRGBs(std::vector<float> * res,cv::Mat * Image);

extern "C"
void getAver();

extern "C"
void setWB(cv::Mat & img, cv::Mat & processedImg,
           float blueAver, float greenAver, float redAver, int _image_height, int _image_width);


#endif //COLOR_CALIBRATE_CPP_ACCE_H
