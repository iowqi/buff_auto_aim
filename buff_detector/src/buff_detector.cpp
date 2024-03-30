#include <exception>
#include <iomanip>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "buff_detector/buff_detector.hpp"

namespace buff_ws {
    BuffDetector::BuffDetector(const int &bin_thres, const int &color, LightParams &lp, TargetParams &tp, std::vector<cv::Mat> &ct) : binary_thres(bin_thres), detect_color(color), l(lp), t(tp), classifier_templates(ct) {
    }

    Target BuffDetector::detect(cv::Mat &input) {
        bgr_img = input;
        preprocessImage();
        findTarget();
        return Target();
    }

    void BuffDetector::drawResults(cv::Mat &img) {
        auto COLOR_OPPO = cv::Scalar((detect_color == RED) * 255, (detect_color == RED) * 200, (detect_color == BLUE) * 255);
        static auto COLOR_GREEN = cv::Scalar(0, 255, 0);
        static auto COLOR_GRAY = cv::Scalar(100, 100, 100);
        static auto COLOR_YELLOW = cv::Scalar(0, 255, 255);
        static auto TARGET_OBJ_COLORS = std::vector<cv::Scalar>({COLOR_GREEN, COLOR_YELLOW, COLOR_GRAY});
        static int shift = 3;
        static int factor = 1 << shift;
        cv::drawContours(img, target_contours_, -1, COLOR_OPPO, 2);
        for (auto t : other_targets_) {
            cv::Point2f vertices[4];
            t.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(img, vertices[i], vertices[(i + 1) % 4], COLOR_GRAY, 2);
            }
        }
        for (auto t : targets_) {
            cv::circle(img, t.center * factor + cv::Point2f(0.5, 0.5), 10 * factor, TARGET_OBJ_COLORS[static_cast<int>(t.type)], 2, cv::LINE_AA, shift);
            cv::putText(
                img, t.classfication_result, cv::Point(t.center.x - t.roi_img.cols / 2, t.center.y - t.roi_img.rows / 2), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                TARGET_OBJ_COLORS[static_cast<int>(t.type)], 2);
        }
        // for (auto contour : target_contours_) {
        //     int size = contour.size();
        //     for (int j = 0, k = size - 1; j < size; k = j++) {
        //         auto offset_p = offsetPoint(contour[k], contour[j], l.sample_offset);
        //         if (offset_p.x < 0 || offset_p.x > img.cols || offset_p.y < 0 || offset_p.y > img.rows)
        //             continue;
        //         auto &p_at = img.at<cv::Vec3b>(offset_p);
        //         if ((detect_color == RED && (p_at[2] - p_at[0]) >= l.color_thres) || (detect_color == BLUE && (p_at[0] - p_at[2]) >= l.color_thres))
        //             cv::drawMarker(img, offset_p, COLOR_GREEN, cv::MARKER_CROSS, 1);
        //         else
        //             cv::drawMarker(img, offset_p, COLOR_GRAY, cv::MARKER_CROSS, 1);
        //     }
        // }
        // cv::imshow("result", img);
        // cv::waitKey(1);
    }

    void BuffDetector::preprocessImage() {
        cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);
    }

    cv::Point BuffDetector::offsetPoint(cv::Point &firstp, cv::Point &nextp, const int &delta) {
        auto vec = nextp - firstp;
        auto norm = cv::norm(vec);
        if (norm > 0)
            return cv::Point(-vec.y, vec.x) * delta / norm + firstp;
        else
            return firstp;
    }

    bool BuffDetector::isLight(Points &contour) {
        int color_count = 0;
        int size = contour.size();
        for (int j = 0, k = size - 1; j < size; k = j++) {
            auto offset_p = offsetPoint(contour[k], contour[j], l.sample_offset);
            if (offset_p.x < 0 || offset_p.x > bgr_img.cols || offset_p.y < 0 || offset_p.y > bgr_img.rows)
                continue;
            auto &p_at = bgr_img.at<cv::Vec3b>(offset_p);
            if ((detect_color == RED && (p_at[2] - p_at[0]) >= l.color_thres) || (detect_color == BLUE && (p_at[0] - p_at[2]) >= l.color_thres))
                color_count++;
        }
        if ((double)color_count / size < l.dencity_ratio)
            return false;
        target_contours_.emplace_back(contour);
        return true;
    }

    void BuffDetector::getMaskedROI(Points &mask_contour, cv::Mat &roi_img_masked) {
        auto roi = cv::boundingRect(mask_contour);
        auto roi_img = gray_img(roi);
        cv::Mat mask = cv::Mat::zeros(roi_img.size(), CV_8UC1);
        cv::drawContours(mask, std::vector<std::vector<cv::Point>>({mask_contour}), 0, cv::Scalar(255), cv::FILLED, 8, std::vector<cv::Vec4i>(), 0, -roi.tl());
        cv::bitwise_and(roi_img, mask, roi_img_masked);
    }

    std::pair<TargetType, double> BuffDetector::classifyTarget(cv::Mat &input) {
        std::vector<double> scores;
        for (auto &t : classifier_templates) {
            cv::Mat res;
            double score;
            cv::resize(input, res, cv::Size(this->t.normalize_resolution, this->t.normalize_resolution));
            cv::matchTemplate(res, t, res, cv::TM_CCOEFF_NORMED);
            cv::minMaxLoc(res, nullptr, &score);
            scores.emplace_back(score);
        }
        auto it = std::max_element(scores.begin(), scores.end());
        int argmax = std::distance(scores.begin(), it);
        auto score = scores[argmax];
        if (score > t.confidence_threshold) {
            switch (argmax) {
            case 0:
                return {TargetType::RLOGO, score};
            case 1:
                return {TargetType::BULLSEYE, score};
            case 2:
                return {TargetType::ACTIVATED, score};
            }
        }
        return {TargetType::INVALID, score};
    }

    void BuffDetector::findTarget() {
        contours_ = Contours();
        target_contours_ = Contours();
        targets_ = std::vector<Target>();
        other_targets_ = std::vector<cv::RotatedRect>();
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(this->binary_img, contours_, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        for (auto contour : contours_) {
            if (contour.size() < 10) continue;
            if (isLight(contour)) {
                auto box = cv::minAreaRect(contour);
                if (abs(box.size.width / box.size.height - 1) < t.square_torrance) {
                    Points hull;
                    cv::convexHull(contour, hull);
                    cv::Mat roi_img;
                    getMaskedROI(hull, roi_img);
                    auto type = classifyTarget(roi_img);
                    if (type.first != TargetType::INVALID) {
                        auto target = Target(hull);
                        target.roi_img = roi_img;
                        target.type = type.first;
                        std::stringstream result_ss;
                        result_ss << TARGET_TYPE_STR[static_cast<int>(type.first)] << ": " << std::fixed << std::setprecision(1)
                                  << type.second * 100.0 << "%";
                        target.classfication_result = result_ss.str();
                        target.confidence = type.second;
                        targets_.emplace_back(target);
                    } else
                        other_targets_.emplace_back(box);
                } else
                    other_targets_.emplace_back(box);
            }
        }

    }
} // namespace buff_ws