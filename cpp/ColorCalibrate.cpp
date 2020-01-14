//
////  Created by Mingkon on 12/19/19.
////  Copyright © 2019 Mingkon Chang. All rights reserved.
//

#include <iostream>
#include <algorithm>
#include <numeric>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ColorCalibrate.h"

bool compare(float A, float B)
{
    return A > B;
}

extern "C"
void useCUDA();

extern "C"
void getRGBs();

extern "C"
void getAver();

extern "C"
void setWB(cv::Mat & img, cv::Mat & processedImg,
           float blueAver, float greenAver, float redAver, int _image_height, int _image_width);

ColorCalibrate::ColorCalibrate(const cv::Mat & Image)
{
    _Image = Image;
    _image_height = Image.size().height;
    _image_width  = Image.size().width;


    uchar arrStandardColor[6][4][3]
            = {  { {255, 0, 0},
                 {220, 20, 60},
                 {250, 128, 114},
                 {255, 99, 71} },


                 { {255, 165, 0},
                 {255, 215, 0},
                 {255, 255, 0},
                 {173, 255, 47} },


                 { {0, 255, 0},
                 {0, 128, 0},
                 {0, 255, 255},
                 {30, 144, 255} },


                 { {0, 0, 255},
                 {0, 0, 139},
                 {138, 43, 226},
                 {128, 0, 128} },


                 { {255, 0, 255},
                 {255, 20, 147},
                 {255, 192, 203},
                 {255, 255, 224} },


                 { {255, 255, 255},
                 {192, 192, 192},
                 {128, 128, 128},
                 {0, 0, 0} }
             };

    cv::Mat StandardColor = cv::Mat(cv::Size(4,6), CV_8UC3);

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            StandardColor.at<cv::Vec3b>(i,j).val[0] =  arrStandardColor[i][j][0];
            StandardColor.at<cv::Vec3b>(i,j).val[1] =  arrStandardColor[i][j][1];
            StandardColor.at<cv::Vec3b>(i,j).val[2] =  arrStandardColor[i][j][2];
        }
    }
    _stdRGBColorBlocks = StandardColor;
}

void ColorCalibrate::getColourBlocks(const cv::Point2i & start_point,
        const int & step_x,const int & step_y, const int & delta_x, const int & delta_y,cv::Mat & color_blocks)
{

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 4; j++) {


            int block_x = start_point.x + j * step_x + delta_x;
            int block_y = start_point.y + i * step_y + delta_y;
            int side = step_x / 4;
            cv::Mat roi = _Image(cv::Range(block_y - side, block_y + side),
                                 cv::Range(block_x - side, block_x + side));

            // show the detect roi
//            cv::imshow("roi",roi);
//            cv::waitKey();
//            cv::destroyAllWindows();

            cv::Scalar roi_mean = cv::mean(roi);

            color_blocks.at<cv::Vec3b>(i, j).val[0] = uchar(roi_mean.val[2]);// R channel
            color_blocks.at<cv::Vec3b>(i, j).val[1] = uchar(roi_mean.val[1]);// G channel
            color_blocks.at<cv::Vec3b>(i, j).val[2] = uchar(roi_mean.val[0]);// B channel
        }
    }
//    std::cout <<"color_blocks:"<<std::endl;
//    std::cout << color_blocks << std::endl;
}

void ColorCalibrate::detectImgRGBColorBlocks(cv::Mat & imgRGBBlocks)
{
    ////detect the marker
//    cv::Mat inputImage;
//    cv::cvtColor(_Image,inputImage,cv::COLOR_BGR2GRAY);
//    cv::imshow("inputImage",inputImage);
//    cv::waitKey();
//    cv::destroyAllWindows();
    std::vector< int > markerIds;
    std::vector< std::vector<cv::Point2f> > markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
    //cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
    cv::Mat gray_img;
    cv::cvtColor(_Image,gray_img,cv::COLOR_BGR2GRAY);
    cv::aruco::detectMarkers(gray_img, dictionary, markerCorners, markerIds);

    if( markerCorners.size() < 3){
        std::cout<<"Error: cann't find two more binary markers!!!"<<std::endl;
        std::cout<<"Please make the colour checker bigger or closer !"<<std::endl;
    }


    //// show the detect results
//    cv::Mat outputImage = _Image;
//    cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
//    cv::imwrite("detect results.png",outputImage);
//    cv::imshow("detect results",outputImage);
//    cv::waitKey();
//    cv::destroyAllWindows();



    ////handel the detect results
    bool TAG01 = false;
    bool TAG02 = false;
    bool TAG03 = false;
    bool TAG04 = false;

    std::vector<cv::Point2f> corner01,corner02,corner03,corner04;
    for (int i = 0; i < markerIds.size(); i ++ )
    {
        if(1 == markerIds[i])
        {
            corner01 = markerCorners[i];
            TAG01 = true;
        }
        else if (2 == markerIds[i])
        {
            TAG02 = true;
            corner02 = markerCorners[i];
        }
        else if (3 == markerIds[i])
        {
            TAG03 = true;
            corner03 = markerCorners[i];
        }
        else if (4 == markerIds[i])
        {
            TAG04 = true;
            corner04 = markerCorners[i];
        }

    }

    const cv::Point2f zero(0,0);
    cv::Point2f top_start = accumulate(corner01.begin(),corner01.end(),zero)/4; // the position centre of the top left binary code
    cv::Point2f top_end   = accumulate(corner02.begin(),corner02.end(),zero)/4; // the position centre of the top right binary code
    cv::Point2f bottom_start = accumulate(corner03.begin(),corner03.end(),zero)/4;  // the position centre of the bottom left binary code
    cv::Point2f bottom_end   = accumulate(corner04.begin(),corner04.end(),zero)/4;  // the position centre of the bottom right binary code

    //// see colour bar creation document for the details

    float length_x, length_y, shift_x, shift_y;

//    if (TAG01 && TAG02) {
//        length_x = top_end.x - top_start.x;
//    }
//    else if (TAG03 && TAG04) {
//        length_x = bottom_end.x - bottom_start.x;
//    }
//
//    if (TAG01 && TAG03) {
//        length_y = bottom_start.y - top_start.y;
//    }
//    else if (TAG02 && TAG04) {
//        length_y = bottom_end.y - top_end.y;
//    }
//
//    if (TAG01 && TAG04) {
//        length_x = bottom_end.x - top_start.x;
//        length_y = bottom_end.y - top_start.y;
//    }

    length_x = top_end.x - top_start.x;
    length_y = bottom_start.y - top_start.y;
    shift_x  = bottom_start.x - top_start.x;
    shift_y  = top_end.y - top_start.y;

    int step_x = int(length_x * 6 / 31);
    int step_y = int(length_y * 6 / 43);
    int delta_x = int(shift_x / 7);
    int delta_y = int(shift_y / 5);



    cv::Point2i top_block00; // the  position centre of top first colour block
    top_block00.x = int(top_start.x + 1.083 * step_x);
    top_block00.y = int(top_start.y + 1.083 * step_y);


    cv::Mat  arrImgColor = cv::Mat(cv::Size(4,6),CV_8UC3);
    getColourBlocks(top_block00, step_x, step_y, delta_x, delta_y, arrImgColor);

    imgRGBBlocks = arrImgColor;
    _imgRGBColorBlocks = arrImgColor;


}

void ColorCalibrate::showStdImgColorBlocks()
{
    cv::Mat canvas = cv::Mat::zeros(cv::Size(220,340),_Image.type());
    cv::Mat stdBGRColorBlocks,imgBGRColorBlocks;
    cv::cvtColor(_stdRGBColorBlocks,stdBGRColorBlocks,cv::COLOR_RGB2BGR);
    cv::cvtColor(_imgRGBColorBlocks,imgBGRColorBlocks,cv::COLOR_RGB2BGR);

//    std::cout << "_stdRGBColorBlocks:"<< std::endl;
//    std::cout << _stdRGBColorBlocks << std::endl;
//    std::cout << "_imgRGBColorBlocks:"<< std::endl;
//    std::cout << _imgRGBColorBlocks << std::endl;

    for (int i = 0; i < stdBGRColorBlocks.rows; i++)
    {
        for (int j = 0; j < stdBGRColorBlocks.cols; j++)
        {
            int xx00 = 20;
            int yy00 = 20;
            int step = 50;

//            uchar std_r = _stdRGBColorBlocks.at<cv::Vec3b>(i, j).val[0];
//            uchar std_b = _stdRGBColorBlocks.at<cv::Vec3b>(i, j).val[0];
//            uchar std_g = _stdRGBColorBlocks.at<cv::Vec3b>(i, j).val[0];
              cv::Mat std_roi = cv::Mat(cv::Size(40,20),_Image.type());
              cv::Mat img_roi = cv::Mat(cv::Size(40,20),_Image.type());

              for (int m = 0; m < 20; m++)
              {
                  for (int n = 0; n < 40; n++)
                  {
                      std_roi.at<cv::Vec3b>(m,n) = stdBGRColorBlocks.at<cv::Vec3b>(i, j);
                      img_roi.at<cv::Vec3b>(m,n) = imgBGRColorBlocks.at<cv::Vec3b>(i, j);
                  }
              }

//              std::cout << "stdBGRColorBlocks.at<cv::Vec3b>("<<i+1<<","<<j+1<<")" << std::endl;
//              std::cout << stdBGRColorBlocks.at<cv::Vec3b>(i, j) << std::endl;
//              std::cout << "imgBGRColorBlocks.at<cv::Vec3b>("<<i+1<<","<<j+1<<")" << std::endl;
//              std::cout << imgBGRColorBlocks.at<cv::Vec3b>(i, j) << std::endl;

//              cv::imshow("img_roi",img_roi);
//              cv::waitKey();
//              cv::destroyAllWindows();

              cv::Rect std_roi_rect = cv::Rect(xx00 + j * step,yy00 + i * step,40,20);
              cv::Rect img_roi_rect = cv::Rect(xx00 + j * step,yy00 + i * step + 20,40,20);

              std_roi.copyTo(canvas(std_roi_rect));
              img_roi.copyTo(canvas(img_roi_rect));

        }
    }

    cv::imshow("stdimgColorBlocks",canvas);
    cv::waitKey();
    cv::destroyAllWindows();
}

void ColorCalibrate::applyGamma(cv::Mat & img  ,cv::Mat & gammaImg,
                                float  gamma_r, float gamma_g , float gamma_b,
                                float gain_r, float gain_g, float gain_b)
{

    for (int i = 0; i<img.rows; i++)
    {
        for (int j = 0; j<img.cols; j++)
        {
            float blue = static_cast<float>(img.at<cv::Vec3b>(i, j).val[0]);
            float green = static_cast<float>(img.at<cv::Vec3b>(i, j).val[1]);
            float red = static_cast<float>(img.at<cv::Vec3b>(i, j).val[2]);

            // 恢复到和人眼特性一致的颜色空间中，即非线性RGB空间
            int blue_gamma = static_cast<int>(powf((blue / 255.0), 1.0 / gamma_b)*gain_b*255.0);
            int green_gamma = static_cast<int>(powf((green / 255.0), 1.0 / gamma_g)*gain_g*255.0);
            int red_gamma = static_cast<int>(powf((red / 255.0), 1.0 / gamma_r)*gain_r*255.0);

            blue_gamma = (blue_gamma>255) ? 255 : blue_gamma;
            green_gamma = (green_gamma>255) ? 255 : green_gamma;
            red_gamma = (red_gamma>255) ? 255 : red_gamma;

            gammaImg.at<cv::Vec3b>(i, j).val[0] = uchar(blue_gamma);
            gammaImg.at<cv::Vec3b>(i, j).val[1] = uchar(green_gamma);
            gammaImg.at<cv::Vec3b>(i, j).val[2] = uchar(red_gamma);
        }
    }
}

void ColorCalibrate::deGamma(cv::Mat & img ,cv::Mat & degammaImg,
        float gamma_r, float gamma_g, float gamma_b,
                             float gain_r, float gain_g, float gain_b)
{
    // 由于相机读取到的数据和人眼特性一致，并不处在线性RGB空间中，即已经在线性基础上乘上了一个因子：1/2.2,
    // 因此degamma需要将其恢复到线性RGB空间中，以便进行颜色校正和白平衡调节
    float inv_gamma_r = 1 / gamma_r;
    float inv_gamma_g = 1 / gamma_g;
    float inv_gamma_b = 1 / gamma_b;

    applyGamma(img,degammaImg, inv_gamma_r, inv_gamma_g, inv_gamma_b, gain_r, gain_g, gain_b);


}


void ColorCalibrate::computeCCM(cv::Mat& uncorrected, cv::Mat& reference,cv::Mat& CCM,
        float gamma_r, float gamma_g, float gamma_b,
        float gain_r, float gain_g, float gain_b)
{
        /* DeGamma */
        cv::Mat linearUncorrected = cv::Mat(uncorrected.size(),uncorrected.type());
        deGamma(uncorrected,linearUncorrected, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);

        cv::Mat linearReference = cv::Mat(reference.size(),reference.type());
        deGamma(reference,linearReference, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);

        /* LinearBGR -> XYZ */
        cv::Mat uncorrectedXYZ = cv::Mat(linearUncorrected.size(), CV_8UC3);
        cv::Mat referenceXYZ = cv::Mat(linearReference.size(), CV_8UC3);
        cv::cvtColor(linearUncorrected, uncorrectedXYZ, cv::COLOR_BGR2XYZ);
        cv::cvtColor(linearReference, referenceXYZ, cv::COLOR_BGR2XYZ);

        /* Solve */
        std::vector<uchar> vUncorrectedXYZ, vReferenceXYZ;

        vUncorrectedXYZ.assign(uncorrectedXYZ.datastart, uncorrectedXYZ.dataend);
        vReferenceXYZ.assign(referenceXYZ.datastart, referenceXYZ.dataend);

        cv::Mat UncorrectedXYZ = cv::Mat(cv::Size(3, uncorrectedXYZ.rows*uncorrectedXYZ.cols),
                                         CV_8UC1, vUncorrectedXYZ.data());
        cv::Mat ReferenceXYZ = cv::Mat(cv::Size(3, referenceXYZ.rows*referenceXYZ.cols),
                                       CV_8UC1, vReferenceXYZ.data());

        UncorrectedXYZ.convertTo(UncorrectedXYZ, CV_32FC1);
        ReferenceXYZ.convertTo(ReferenceXYZ, CV_32FC1);

        //cv::Mat homoUncorrectedXYZ = cv::Mat::ones(cv::Size(UncorrectedXYZ.cols + 1, UncorrectedXYZ.rows), CV_32FC1);
        cv::Mat homoUncorrectedXYZ = cv::Mat::ones(cv::Size(UncorrectedXYZ.cols, UncorrectedXYZ.rows), CV_32FC1);
        UncorrectedXYZ.col(0).copyTo(homoUncorrectedXYZ.col(0));
        UncorrectedXYZ.col(1).copyTo(homoUncorrectedXYZ.col(1));
        UncorrectedXYZ.col(2).copyTo(homoUncorrectedXYZ.col(2));

        cv::Mat invHomoUncorrectedXYZ;
        cv::invert(homoUncorrectedXYZ, invHomoUncorrectedXYZ, cv::DECOMP_SVD);

        CCM = invHomoUncorrectedXYZ * ReferenceXYZ * _ALPHA;

}

void ColorCalibrate::correctColor(cv::Mat & img, cv::Mat & corrected, const cv::Mat & ccm)
{
    std::vector<uchar> img_vec;
    img_vec.assign(img.datastart, img.dataend);
    cv::Mat uncorrected = cv::Mat(cv::Size(3, img.rows*img.cols), CV_8UC1, img_vec.data());
    uncorrected.convertTo(uncorrected, CV_32FC1);

    //cv::Mat homoUncorrected = cv::Mat::ones(cv::Size(uncorrected.cols + 1, uncorrected.rows), CV_32FC1);
    cv::Mat homoUncorrected = cv::Mat::ones(cv::Size(uncorrected.cols, uncorrected.rows), CV_32FC1);
    uncorrected.col(0).copyTo(homoUncorrected.col(0));
    uncorrected.col(1).copyTo(homoUncorrected.col(1));
    uncorrected.col(2).copyTo(homoUncorrected.col(2));

    corrected = homoUncorrected * ccm;
}

void  ColorCalibrate::applyWhiteBalancePR(cv::Mat img, cv::Mat & processedImg, float white_rate)
{
    processedImg = cv::Mat(img.size(), img.type());

    std::vector<float> RGBs;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float rgb = static_cast<float>(img.at<cv::Vec3b>(i, j).val[0]
                                           + img.at<cv::Vec3b>(i, j).val[1]
                                           + img.at<cv::Vec3b>(i, j).val[2]);
            RGBs.push_back(rgb);
        }
    }

    int maxNum = round(RGBs.size() * white_rate);

    std::sort(RGBs.begin(), RGBs.end());// compare);

    float threshold = RGBs[maxNum - 1];


        int amount = 0;
        float redSum = 0, greenSum = 0, blueSum = 0;

        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                float rgb = static_cast<float>(img.at<cv::Vec3b>(i, j).val[0]
                                               + img.at<cv::Vec3b>(i, j).val[1]
                                               + img.at<cv::Vec3b>(i, j).val[2]);
                if (rgb >= threshold) {
                    amount++;
                    blueSum += static_cast<float>(img.at<cv::Vec3b>(i, j).val[0]);
                    greenSum += static_cast<float>(img.at<cv::Vec3b>(i, j).val[1]);
                    redSum += static_cast<float>(img.at<cv::Vec3b>(i, j).val[2]);
                }
            }
        }

        float blueAver = blueSum / amount;
        float greenAver = greenSum / amount;
        float redAver = redSum / amount;

    //// check if there is any GPU available

    int nDevices;
    struct cudaDeviceProp * 	prop;
    cudaError_t res = cudaGetDeviceProperties	(prop, nDevices);
    std::cout<<"GPU numbers:"<<nDevices<<std::endl;

    if (0 != nDevices)
    {
        std::cout<<"Using GPU"<<std::endl;

        setWB(
                (cv::Mat&) img, (cv::Mat&) processedImg,
                (float) blueAver, (float) greenAver, (float) redAver,
                (int)  _image_height, (int)  _image_width);
    }
    else {
        std::cout << "Using CPU" << std::endl;

        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                float blue = static_cast<float>(img.at<cv::Vec3b>(i, j).val[0] / blueAver * 255);
                float green = static_cast<float>(img.at<cv::Vec3b>(i, j).val[1] / greenAver * 255);
                float red = static_cast<float>(img.at<cv::Vec3b>(i, j).val[2] / redAver * 255);

                if (blue > 255) {
                    blue = 255;
                } else if (blue < 0) {
                    blue = 0;
                }
                if (green > 255) {
                    green = 255;
                } else if (green < 0) {
                    green = 0;
                }
                if (red > 255) {
                    red = 255;
                } else if (red < 0) {
                    red = 0;
                }

                processedImg.at<cv::Vec3b>(i, j).val[0] = static_cast<uchar>(blue);
                processedImg.at<cv::Vec3b>(i, j).val[1] = static_cast<uchar>(green);
                processedImg.at<cv::Vec3b>(i, j).val[2] = static_cast<uchar>(red);
            }
        }
    }
}


void ColorCalibrate::calibrate(cv::Mat& correctedImg,const cv::Mat & ccm,
                       float gamma_r, float gamma_g, float gamma_b,
                       float gain_r, float gain_g, float gain_b)
{
    cv::Mat linearBGR = cv::Mat(_Image.size(),_Image.type());
    cv::Mat img = _Image;

    time_t start = time(NULL);
    deGamma(img, linearBGR,gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);
    time_t end = time(NULL);
    std::cout <<"degamma time spent: "<<end - start<< std::endl;
    cv::Mat XYZ = cv::Mat(_Image.size(),_Image.type());
    cv::cvtColor(linearBGR, XYZ, cv::COLOR_BGR2XYZ);
    cv::Mat corrected;
    correctColor(XYZ,corrected, ccm);
    corrected.convertTo(corrected, CV_8UC1);

    std::vector<uchar> corrected_vec;
    corrected_vec.assign(corrected.datastart, corrected.dataend);
    cv::Mat correctedXYZ = cv::Mat(img.size(), img.type(), corrected_vec.data());

    cv::Mat correctedLinearBGR;
    cv::cvtColor(correctedXYZ, correctedLinearBGR, cv::COLOR_XYZ2BGR);

    // 以后需要加white rate参数
    cv::Mat wbCorrectedLinearBGR;
    time_t start_wb = time(NULL);
    applyWhiteBalancePR(correctedLinearBGR, wbCorrectedLinearBGR, 0.2);
    time_t end_wb = time(NULL);
    std::cout <<"applyWhiteBalancePR time spent: "<<end_wb - start_wb<< std::endl;
    // 确定degamma和gamma的意义

    cv::Mat correctedBGR = cv::Mat(wbCorrectedLinearBGR.size(),wbCorrectedLinearBGR.type());
    time_t start_ag = time(NULL);
    applyGamma(wbCorrectedLinearBGR,correctedBGR, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);
    time_t end_ag = time(NULL);
    std::cout <<"applyGamma time spent: "<<end_ag - start_ag<< std::endl;
    correctedImg = correctedBGR;
}

void ColorCalibrate::executeCalibrate(cv::Mat& calibratedImg,std::string& output_path)
{
    cv::Mat imgRGBColorBlocks;
    detectImgRGBColorBlocks(imgRGBColorBlocks);

    cv::Mat imgBGRColorBlocks,stdBGRColorBlocks;
    cv::cvtColor(imgRGBColorBlocks,imgBGRColorBlocks,cv::COLOR_RGB2BGR);
    cv::cvtColor(_stdRGBColorBlocks,stdBGRColorBlocks,cv::COLOR_RGB2BGR);

    float gamma = 2.2;
    cv::Mat ccm;
    //computeCCM(imgRGBColorBlocks, _stdRGBColorBlocks, ccm,gamma, gamma, gamma); // RGB order
    computeCCM(imgBGRColorBlocks, stdBGRColorBlocks, ccm,gamma, gamma, gamma); // BGR order
    std::cout << "CCM: " << std::endl;
    std::cout << ccm << std::endl;

    calibrate(calibratedImg, ccm);

    cv::imwrite(output_path + "calibratedImg_sony1.jpg", calibratedImg);

}
