#ifndef BUFF_TARGETS_HPP_
#define BUFF_TARGETS_HPP_

#include <opencv2/core.hpp>

namespace buff_ws {
    const int RED = 0;
    const int BLUE = 1;

    enum class TargetType {
        RLOGO = 0,
        BULLSEYE,
        ACTIVATED,
        INVALID
    };
    const std::string TARGET_TYPE_STR[4] = {"rlogo", "bullseye", "activated", "invalid"};

    struct Target {
        Target() = default;
        explicit Target(std::vector<cv::Point> contour) {
            auto mu = cv::moments(contour);
            center.x = mu.m10 / mu.m00;
            center.y = mu.m01 / mu.m00;
        }
        cv::Mat roi_img;
        TargetType type;
        cv::Point2f center;
        float confidence;
        std::string classfication_result;
    };

} // namespace buff_ws
#endif