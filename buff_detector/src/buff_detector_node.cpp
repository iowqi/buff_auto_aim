#include <cv_bridge/cv_bridge.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "buff_detector/buff_detector_node.hpp"

namespace buff_ws {
    BuffDetectorNode::BuffDetectorNode(const rclcpp::NodeOptions &options) : Node("buff_detector", options) {
        RCLCPP_INFO(this->get_logger(), "BuffDetectorNode Init.");

        buff_detector_ = initBuffDetector();

        // Debug Publishers
        debug_ = this->declare_parameter("debug", false);
        if (debug_)
            createDebugPublishers();

        // Debug param change moniter
        debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
        debug_cb_handle_ = debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter &p) {
            debug_ = p.as_bool();
            RCLCPP_INFO(this->get_logger(), "Debug Param Changed: %d", debug_);
            debug_ ? createDebugPublishers() : destroyDebugPublishers();
        });

        cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
                cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
                cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
                cam_info_sub_.reset();
            });

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&BuffDetectorNode::imageCallback, this, std::placeholders::_1));
    }

    void BuffDetectorNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto targets = detectTargets(msg);
    }

    std::unique_ptr<BuffDetector> BuffDetectorNode::initBuffDetector() {
        rcl_interfaces::msg::ParameterDescriptor param_desc;
        param_desc.integer_range.resize(1);
        param_desc.integer_range[0].step = 1;
        param_desc.integer_range[0].from_value = 0;
        param_desc.integer_range[0].to_value = 255;
        int binary_thres = this->declare_parameter("binary_thres", 80, param_desc);

        param_desc.description = "0-RED, 1-BLUE";
        param_desc.integer_range[0].from_value = 0;
        param_desc.integer_range[0].to_value = 1;
        auto detect_color = this->declare_parameter("detect_color", RED, param_desc);

        BuffDetector::LightParams l_params = {
            .sample_offset = declare_parameter("light.sample_offset", 2),
            .dencity_ratio = declare_parameter("light.dencity_ratio", 0.5),
            .color_thres = declare_parameter("light.color_thres", 80)};

        auto detector = std::make_unique<BuffDetector>(binary_thres, detect_color, l_params);
        return detector;
    }

    Target BuffDetectorNode::detectTargets(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg) {
        auto img = cv_bridge::toCvShare(img_msg, "bgr8")->image;

        buff_detector_->binary_thres = get_parameter("binary_thres").as_int();
        buff_detector_->detect_color = get_parameter("detect_color").as_int();

        auto targets = buff_detector_->detect(img);

        auto final_time = this->now();
        auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

        if (debug_) {
            binary_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "mono8", buff_detector_->binary_img).toImageMsg());
            buff_detector_->drawResults(img);
            // Draw camera center
            cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
            // Draw latency
            std::stringstream latency_ss;
            latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
            auto latency_s = latency_ss.str();
            cv::putText(img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            result_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "bgr8", img).toImageMsg());
        }

        return Target();
    }

    void BuffDetectorNode::createDebugPublishers() {
        binary_img_pub_ = image_transport::create_publisher(this, "/buff_detector/binary_img");
        result_img_pub_ = image_transport::create_publisher(this, "/buff_detector/result_img");
    }

    void BuffDetectorNode::destroyDebugPublishers() {
        binary_img_pub_.shutdown();
        result_img_pub_.shutdown();
    }

} // namespace buff_ws

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(buff_ws::BuffDetectorNode);