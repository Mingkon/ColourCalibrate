//
////  Created by Mingkon on 12/19/19.
////  Copyright © 2019 Mingkon Chang. All rights reserved.
//

#ifndef COLOR_CALIBRATE_CPP_COLORCALIBRATE_H
#define COLOR_CALIBRATE_CPP_COLORCALIBRATE_H


#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Core>
#include <Eigen/LU>



class ColorCalibrate
{
public:
    ColorCalibrate(const cv::Mat & Image);

    void detectImgRGBColorBlocks(cv::Mat & imgRGBBlocks);
    void getColourBlocks(const cv::Point2i & start_point, const int & step_x, const int & step_y,const int & delta_x, const int & delta_y, cv::Mat & color_blocks);

    void applyGamma(cv::Mat & img ,cv::Mat & resImg,float gamma_r = 2.2, float  gamma_g = 2.2, float gamma_b = 2.2,
                    float gain_r = 1.0, float gain_g = 1.0, float gain_b = 1.0);
    void deGamma(cv::Mat & img ,cv::Mat & resImg, float gamma_r = 2.2, float gamma_g = 2.2, float gamma_b = 2.2,
                 float gain_r = 1.0, float gain_g = 1.0, float gain_b = 1.0);

    void showStdImgColorBlocks();

    void computeCCM(cv::Mat& uncorrected, cv::Mat& reference,cv::Mat& CCM,
                    float gamma_r = 2.2, float gamma_g = 2.2, float gamma_b = 2.2,
                    float gain_r = 1.0, float gain_g = 1.0, float gain_b = 1.0);

    void correctColor(cv::Mat & img, cv::Mat & corrected, const cv::Mat & ccm);
    void applyWhiteBalancePR(cv::Mat img, cv::Mat & processedImg, float white_rate);


    void calibrate(cv::Mat& correctedImg,const cv::Mat & ccm,
    float gamma_r = 2.2, float gamma_g = 2.2, float gamma_b = 2.2,
    float gain_r = 1.0, float gain_g = 1.0, float gain_b = 1.0);

    void executeCalibrate(cv::Mat& calibratedImg,std::string& path);

private:
    cv::Mat _Image;
    cv::Mat _stdRGBColorBlocks;
    cv::Mat _imgRGBColorBlocks;
    int _image_width;
    int _image_height;
    const float _ALPHA = 0.5;
};
#endif //COLOR_CALIBRATE_CPP_COLORCALIBRATE_H
