//
////  Created by Mingkon on 12/19/19.
////  Copyright © 2019 Mingkon Chang. All rights reserved.
//
#include <iostream>
#include <opencv2/freetype.hpp>
#include <string>
#include "ColorBar.h"


ColorBar::ColorBar()
{
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
    cv::cvtColor(_stdRGBColorBlocks,_stdBGRColorBlocks,cv::COLOR_RGB2BGR);
}

void ColorBar::generateColorBar(const std::string & save_colorbar_path)
{

    const int SCALE = 4;
    //// generate binary code markers
    cv::Mat markerImage001,markerImage002,markerImage003,markerImage004;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
    cv::aruco::drawMarker(dictionary, 1, 240 * SCALE, markerImage001);
    cv::aruco::drawMarker(dictionary, 2, 240 * SCALE, markerImage002);
    cv::aruco::drawMarker(dictionary, 3, 240 * SCALE, markerImage003);
    cv::aruco::drawMarker(dictionary, 4, 240 * SCALE, markerImage004);


    //// write binary code markers
    cv::imwrite(save_colorbar_path + "aruco_DICT_6X6_50_01.png",markerImage001);
    cv::imwrite(save_colorbar_path + "aruco_DICT_6X6_50_02.png",markerImage002);
    cv::imwrite(save_colorbar_path + "aruco_DICT_6X6_50_03.png",markerImage003);
    cv::imwrite(save_colorbar_path + "aruco_DICT_6X6_50_04.png",markerImage004);

    //// read binary code markers
    cv::Mat markerImage01,markerImage02,markerImage03,markerImage04;
    markerImage01 = cv::imread(save_colorbar_path + "aruco_DICT_6X6_50_01.png");
    markerImage02 = cv::imread(save_colorbar_path + "aruco_DICT_6X6_50_02.png");
    markerImage03 = cv::imread(save_colorbar_path + "aruco_DICT_6X6_50_03.png");
    markerImage04 = cv::imread(save_colorbar_path + "aruco_DICT_6X6_50_04.png");

//    cv::imshow("markerImage01",markerImage01);
//    cv::waitKey();
//    cv::destroyAllWindows();

    cv::Mat canvas    = cv::Mat(cv::Size(1600 * SCALE,2080 * SCALE),
            CV_8UC3,cv::Scalar(255,255,255));


    //// background part
    cv::Rect black_rect = cv::Rect(20 * SCALE,20 * SCALE,1560 * SCALE,2040 * SCALE);
    cv::Mat zeros = cv::Mat(cv::Size(1560 * SCALE,2040 * SCALE),CV_8UC3,cv::Scalar(200,200,200));
    zeros.copyTo(canvas(black_rect));


    //// binary code marker part
    cv::Rect binary01_rect = cv::Rect(60 * SCALE,60 * SCALE,240 * SCALE,240 * SCALE);
    cv::Rect binary02_rect = cv::Rect(1300 * SCALE,60 * SCALE,240 * SCALE,240 * SCALE);
    cv::Rect binary03_rect = cv::Rect(60 * SCALE,1780 * SCALE,240 * SCALE,240 * SCALE);
    cv::Rect binary04_rect = cv::Rect(1300 * SCALE,1780 * SCALE,240 * SCALE,240 * SCALE);

    markerImage01.copyTo(canvas(binary01_rect));
    markerImage02.copyTo(canvas(binary02_rect));
    markerImage03.copyTo(canvas(binary03_rect));
    markerImage04.copyTo(canvas(binary04_rect));



//    cv::imshow("top_binary_code_colour_bar",top_binary_code_colour_bar);
//    cv::waitKey();
//    cv::destroyAllWindows();

    //// creat from standard colour code
    int idx = 0;
    for (int i = 0; i < _stdBGRColorBlocks.rows; i++)
    {
        for (int j = 0; j < _stdBGRColorBlocks.cols; j++)
        {
            int xx00 = 340 * SCALE;
            int yy00 = 340 * SCALE;
            int step = 240 * SCALE;

            idx = i * _stdBGRColorBlocks.cols + j;

//            uchar std_r = _stdRGBColorBlocks.at<cv::Vec3b>(i, j).val[0];
//            uchar std_b = _stdRGBColorBlocks.at<cv::Vec3b>(i, j).val[0];
//            uchar std_g = _stdRGBColorBlocks.at<cv::Vec3b>(i, j).val[0];
            cv::Mat std_roi = cv::Mat(cv::Size(200 * SCALE,200 * SCALE),CV_8UC3);
            //cv::Mat img_roi = cv::Mat(cv::Size(40,20),CV_8UC3);

            for (int m = 0; m < 200 * SCALE; m++)
            {
                for (int n = 0; n < 200 * SCALE; n++)
                {
                    std_roi.at<cv::Vec3b>(m,n) = _stdBGRColorBlocks.at<cv::Vec3b>(i, j);
                    //img_roi.at<cv::Vec3b>(m,n) = imgBGRColorBlocks.at<cv::Vec3b>(i, j);
                }
            }

//              std::cout << "stdBGRColorBlocks.at<cv::Vec3b>("<<i+1<<","<<j+1<<")" << std::endl;
//              std::cout << stdBGRColorBlocks.at<cv::Vec3b>(i, j) << std::endl;
//              std::cout << "imgBGRColorBlocks.at<cv::Vec3b>("<<i+1<<","<<j+1<<")" << std::endl;
//              std::cout << imgBGRColorBlocks.at<cv::Vec3b>(i, j) << std::endl;

//              cv::imshow("std_roi",std_roi);
//              cv::waitKey();
//              cv::destroyAllWindows();

            cv::Rect std_roi_rect = cv::Rect(xx00 + j * step, yy00 + i * step,200 * SCALE,200 * SCALE);
            std_roi.copyTo(canvas(std_roi_rect));


        }
    }


    //// handel the sides

    int maB = 174;
    int maG = 164;
    int maR = 144;
    int miB = 56;
    int miG = 50;
    int miR = 38;

    cv::Mat side_bar = cv::Mat(cv::Size(240 * SCALE,1400 * SCALE),CV_8UC3);

    float stepB = float(maB - miB) / float(1400 * SCALE);
    float stepG = float(maG - miG) / float(1400 * SCALE);
    float stepR = float(maR - miR) / float(1400 * SCALE);

    for (int h = 0; h < side_bar.rows; h++)
    {
        uchar B = uchar(maB - stepB * h);
        uchar G = uchar(maG - stepG * h);
        uchar R = uchar(maR - stepR * h);

        for (int w =0; w < side_bar.cols; w++)
        {
            side_bar.at<cv::Vec3b>(h, w).val[0] = B;
            side_bar.at<cv::Vec3b>(h, w).val[1] = G;
            side_bar.at<cv::Vec3b>(h, w).val[2] = R;
        }
    }

    cv::Rect side_bar_rect_L = cv::Rect(60 * SCALE,340 * SCALE,240 * SCALE,1400 * SCALE);
    cv::Rect side_bar_rect_R = cv::Rect(1300 * SCALE,340 * SCALE,240 * SCALE,1400 * SCALE);

    side_bar.copyTo(canvas(side_bar_rect_L));
    side_bar.copyTo(canvas(side_bar_rect_R));

    const std::string eng_name = "TriMetaPixel Colour Checker";
    const std::string chn_name = "三元素色卡";

    cv::Point eng_pos = cv::Point2i(430 * SCALE,200 * SCALE); // 文本框的左下角
    cv::Point chn_pos = cv::Point2i(430 * SCALE,1440 * SCALE);
    cv::Point lft_up_pos = cv::Point2i(150 * SCALE,450 * SCALE);
    cv::Point rht_up_pos = cv::Point2i(1390 * SCALE,450 * SCALE);
    cv::Point lft_down_pos = cv::Point2i(100 * SCALE,1600 * SCALE);
    cv::Point rht_down_pos = cv::Point2i(1340 * SCALE,1600 * SCALE);
    int fontFace = cv::FONT_HERSHEY_COMPLEX; // 字体 (如cv::FONT_HERSHEY_PLAIN)
    double fontScale = 1.5 * SCALE; // 尺寸因子，值越大文字越大
    double fontScale_sd = 1.8 * SCALE;
    cv::Scalar color = cv::Scalar(255,255,255); // 线条的颜色（RGB）
    int thickness = 8 * SCALE; // 线条宽度
    int thickness_sd = 10 * SCALE;
    int lineType = 8; // 线型（4邻域或8邻域，默认8邻域）

    cv::putText(canvas,eng_name,eng_pos,fontFace,fontScale,color,thickness,lineType);
    cv::putText(canvas,"UP",lft_up_pos,fontFace,fontScale_sd,color,thickness_sd,lineType);
    cv::putText(canvas,"UP",rht_up_pos,fontFace,fontScale_sd,color,thickness_sd,lineType);
    cv::putText(canvas,"DOWN",lft_down_pos,fontFace,fontScale_sd,color,thickness_sd,lineType);
    cv::putText(canvas,"DOWN",rht_down_pos,fontFace,fontScale_sd,color,thickness_sd,lineType);



//    cv::imshow("canvas",canvas);
//    cv::waitKey();
//    cv::destroyAllWindows();

    cv::imwrite(save_colorbar_path + "colorBarChecker.png",canvas);

}