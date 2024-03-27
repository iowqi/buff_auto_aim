#ifndef BUFF_TARGETS_HPP_
#define BUFF_TARGETS_HPP_

#include <opencv2/core.hpp>

namespace buff_ws {
    const int RED = 0;
    const int BLUE = 1;

    enum class TargetType {
        RLOGO,
        BULLSEYE,
        ACTIVATED
    };
    const std::string TARGET_TYPE_STR[3] = {"rlogo", "bullseye", "activated"};

    struct Target : public cv::RotatedRect {
        Target() = default;
        Target(cv::RotatedRect box) {
            center = box.center;
            size = box.size;
            angle = box.angle;
        }

        TargetType type;

        float confidence;
        std::string classfication_result;
    };

} // namespace buff_ws
#endif