/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef HOUGH_CIRCLES_ALT_HPP_
#define HOUGH_CIRCLES_ALT_HPP_

#include <opencv4/opencv2/opencv.hpp>

namespace houghtf
{
    struct EstimatedCircle
    {
        EstimatedCircle() { accum = 0; }
        EstimatedCircle(cv::Vec3f _c, int _accum) : c(_c), accum(_accum) {}
        cv::Vec3f c;
        int accum;
    };

    struct CircleData
    {
        CircleData()
        {
            rw = 0;
            weight = 0;
            mask = 0;
        }
        double rw;
        int weight;
        uint64 mask;
    };

    static int circle_popcnt(uint64 val)
    {
#ifdef CV_POPCNT_U64
        return CV_POPCNT_U64(val);
#else
        val -= (val >> 1) & 0x5555555555555555ULL;
        val = (val & 0x3333333333333333ULL) + ((val >> 2) & 0x3333333333333333ULL);
        val = (val + (val >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
        return (int)((val * 0x0101010101010101ULL) >> 56);
#endif
    }

    enum
    {
        HOUGH_CIRCLES_ALT_BLOCK_SIZE = 10,
        HOUGH_CIRCLES_ALT_MAX_CLUSTERS = 10
    };

    void HoughCirclesAlt(const cv::Mat &img, std::vector<EstimatedCircle> &circles, double dp, double rdMinDist,
                                double minRadius, double maxRadius, double cannyThreshold, double minCos2);
}
#endif