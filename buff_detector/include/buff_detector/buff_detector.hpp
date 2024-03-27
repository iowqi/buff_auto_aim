#ifndef BUFF_DETECTOR_HPP_
#define BUFF_DETECTOR_HPP_

#include <opencv2/core.hpp>

#include <vector>

#include "buff_detector/targets.hpp"

namespace buff_ws {

    typedef std::vector<std::vector<cv::Point>> Contours;

    class BuffDetector {
    public:
        struct LightParams {
            int sample_offset;
            double dencity_ratio;
            int color_thres;
        };
        cv::Mat merge_binary_img;
        cv::Mat processed_img;

        int binary_thres;
        int detect_color;
        LightParams l;

        cv::Mat bgr_img;
        cv::Mat binary_img;

        BuffDetector(const int &bin_thres, const int &color, LightParams lp);
        Target detect(cv::Mat &input);
        void drawResults(cv::Mat &img);
        cv::Point offsetPoint(cv::Point &firstp, cv::Point &nextp, const int &offset);

    private:
        Contours contours_;
        Contours target_contours_;
        void preprocessImage();
        void findTarget();
        int judgeContourColor(const cv::Mat &input, std::vector<cv::Point> &contour);
    };

} // namespace buff_ws

#endif