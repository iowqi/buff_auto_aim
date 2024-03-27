#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "buff_detector/buff_detector.hpp"

namespace buff_ws {
    BuffDetector::BuffDetector(const int &bin_thres, const int &color, LightParams lp) : binary_thres(bin_thres), detect_color(color), l(lp) {
    }

    Target BuffDetector::detect(cv::Mat &input) {
        bgr_img = input;
        preprocessImage();
        findTarget();
        return Target();
    }

    void BuffDetector::drawResults(cv::Mat &img) {
        static auto COLOR_OPPO = cv::Scalar((detect_color == RED) * 255, (detect_color == RED) * 200, (detect_color == BLUE) * 255);
        static auto COLOR_GREEN = cv::Scalar(0, 255, 0);
        static auto COLOR_GRAY = cv::Scalar(100, 100, 100);
        static auto TARGET_OBJ_COLORS = std::vector<cv::Scalar>({COLOR_OPPO, (128, 128, 128), (0, 255, 0)});
        cv::drawContours(img, target_contours_, -1, COLOR_OPPO, 1);
        for (auto contour : target_contours_) {
            int size = contour.size();
            for (int j = 0, k = size - 1; j < size; k = j++) {
                auto offset_p = offsetPoint(contour[k], contour[j], l.sample_offset);
                if (offset_p.x < 0 || offset_p.x > img.cols || offset_p.y < 0 || offset_p.y > img.rows)
                    continue;
                auto &p_at = img.at<cv::Vec3b>(offset_p);
                if ((detect_color == RED && (p_at[2] - p_at[0]) >= l.color_thres) || (detect_color == BLUE && (p_at[0] - p_at[2]) >= l.color_thres))
                    cv::drawMarker(img, offset_p, COLOR_GREEN, cv::MARKER_CROSS, 1);
                else
                    cv::drawMarker(img, offset_p, COLOR_GRAY, cv::MARKER_CROSS, 1);
            }
        }
        // cv::imshow("result", img);
        // cv::waitKey(1);
    }

    void BuffDetector::preprocessImage() {
        cv::cvtColor(bgr_img, binary_img, cv::COLOR_BGR2GRAY);
        cv::threshold(binary_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);
    }

    cv::Point BuffDetector::offsetPoint(cv::Point &firstp, cv::Point &nextp, const int &delta) {
        auto vec = nextp - firstp;
        auto norm = cv::norm(vec);
        if (norm > 0)
            return cv::Point(-vec.y, vec.x) * delta / norm + firstp;
        else
            return firstp;
    }

    int BuffDetector::judgeContourColor(const cv::Mat &input, std::vector<cv::Point> &contour) {
        int color_count = 0;
        int size = contour.size();
        if (size < 2)
            return false;
        for (int j = 0, k = size - 1; j < size; k = j++) {
            auto offset_p = offsetPoint(contour[k], contour[j], l.sample_offset);
            if (offset_p.x < 0 || offset_p.x > input.cols || offset_p.y < 0 || offset_p.y > input.rows)
                continue;
            auto &p_at = input.at<cv::Vec3b>(offset_p);
            if ((detect_color == RED && (p_at[2] - p_at[0]) >= l.color_thres) || (detect_color == BLUE && (p_at[0] - p_at[2]) >= l.color_thres))
                color_count++;
        }
        return (double)color_count / size > l.dencity_ratio;
    }

    void BuffDetector::findTarget() {
        contours_ = Contours();
        target_contours_ = Contours();
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(this->binary_img, contours_, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        for (auto cnt : contours_) {
            if (judgeContourColor(this->bgr_img, cnt))
                target_contours_.push_back(cnt);
        }
    }
} // namespace buff_ws
