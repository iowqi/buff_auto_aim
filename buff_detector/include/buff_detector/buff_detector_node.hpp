#ifndef _BUFF_DETECTOR_NODE_HPP_
#define _BUFF_DETECTOR_NODE_HPP_

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

// OpenCV
#include <opencv2/core.hpp>

#include "buff_detector/buff_detector.hpp"
#include "buff_detector/targets.hpp"

namespace buff_ws {
    class BuffDetectorNode : public rclcpp::Node {
    public:
        BuffDetectorNode(const rclcpp::NodeOptions &options);

    private:
        std::unique_ptr<BuffDetector> buff_detector_;

        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

        // Camera info part
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
        cv::Point2f cam_center_;
        std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;

        // Debug information
        bool debug_;
        std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
        std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
        image_transport::Publisher binary_img_pub_;
        image_transport::Publisher result_img_pub_;

        void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

        std::unique_ptr<BuffDetector> initBuffDetector();
        Target detectTargets(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg);
        void createDebugPublishers();
        void destroyDebugPublishers();
    };
} // namespace buff_ws

#endif