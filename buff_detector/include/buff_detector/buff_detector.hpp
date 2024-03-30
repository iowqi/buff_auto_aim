#ifndef BUFF_DETECTOR_HPP_
#define BUFF_DETECTOR_HPP_

#include <opencv2/core.hpp>

#include <vector>

#include "buff_detector/targets.hpp"

namespace buff_ws {

    typedef std::vector<std::vector<cv::Point>> Contours;
    typedef std::vector<cv::Point> Points;

    class BuffDetector {
    public:
        struct LightParams {
            int sample_offset;
            double dencity_ratio;
            int color_thres;
        };

        struct TargetParams {
            double square_torrance;
            int normalize_resolution;
            double confidence_threshold;
        };

        std::vector<cv::Mat> classifier_templates;

        cv::Mat merge_binary_img;
        cv::Mat processed_img;

        int binary_thres;
        int detect_color;
        LightParams l;
        TargetParams t;

        cv::Mat bgr_img;
        cv::Mat gray_img;
        cv::Mat binary_img;

        BuffDetector(const int &bin_thres, const int &color, LightParams &lp, TargetParams &tp, std::vector<cv::Mat> &ct);
        Target detect(cv::Mat &input);
        void drawResults(cv::Mat &img);

    private:
        Contours contours_;
        Contours target_contours_;
        std::vector<cv::RotatedRect> other_targets_;
        std::vector<Target> targets_;
        std::vector<Target>::iterator rlogo_;
        std::vector<Target>::iterator bullseye_;
        void preprocessImage();
        void findTarget();
        cv::Point offsetPoint(cv::Point &firstp, cv::Point &nextp, const int &offset);
        bool isLight(Points &contour);
        void getMaskedROI(Points &contour, cv::Mat &roi_img_masked);
        std::pair<TargetType, double> classifyTarget(cv::Mat &input);
    };

} // namespace buff_ws

#endif