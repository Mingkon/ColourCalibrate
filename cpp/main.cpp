//
////  Created by Mingkon Chang on 19/12/2019.
////  Copyright © 2019 Mingkon Chang. All rights reserved.
//
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ColorBar.h"
#include "ColorCalibrate.h"

int main()
{
    //// generate color bar
    ColorBar color_bar;
    std::string save_path = "/home/mingkon/Desktop/colourCalibrate/img/";
    color_bar.generateColorBar(save_path);
    std::string img_path = "/home/mingkon/Desktop/colourCalibrate/img/sony1.jpg";
    cv::Mat img = cv::imread(img_path.c_str(),cv::IMREAD_COLOR);
    std::cout <<"img size:"<< img.size()<<std::endl;
//    cv::imshow("Image",img);
//    cv::waitKey();
//    cv::destroyAllWindows();

    //// calibrate image with color bar generated above
    cv::Mat calibratedImg = cv::Mat(img.size(),img.type());

    ColorCalibrate cc(img);
    cc.executeCalibrate(calibratedImg,save_path);
    cc.showStdImgColorBlocks();

    // show the calibrated image
    cv::imshow("calibratedImage",calibratedImg);
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}
