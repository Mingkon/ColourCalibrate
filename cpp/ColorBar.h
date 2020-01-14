//
////  Created by Mingkon on 12/19/19.
////  Copyright © 2019 Mingkon Chang. All rights reserved.
//

#ifndef COLOR_CALIBRATE_CPP_COLORBAR_H
#define COLOR_CALIBRATE_CPP_COLORBAR_H

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>




class ColorBar {
public:
    ColorBar();
    void generateColorBar(const std::string & save_path);
private:
    cv::Mat _stdRGBColorBlocks;
    cv::Mat _stdBGRColorBlocks;

};


#endif //COLOR_CALIBRATE_CPP_COLORBAR_H
