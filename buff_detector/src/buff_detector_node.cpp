#include <cv_bridge/cv_bridge.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "buff_detector/buff_detector_node.hpp"

namespace buff_ws
{
BuffDetectorNode::BuffDetectorNode(const rclcpp::NodeOptions &options) : Node("buff_detector", options)
{
    RCLCPP_INFO(this->get_logger(), "BuffDetectorNode Init.");

    // Debug Publishers
    debug_ = this->declare_parameter("debug", false);
    if (debug_)
        createDebugPublishers();

    // Debug param change moniter
    debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    debug_cb_handle_ = debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter &p) {
        debug_ = p.as_bool();
        debug_ ? createDebugPublishers() : destroyDebugPublishers();
    });

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&BuffDetectorNode::imageCallback, this, std::placeholders::_1));
}

void BuffDetectorNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "Image Received.");
    const cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::imshow("Image", image);
    cv::waitKey(1);
    return;
}

void BuffDetectorNode::createDebugPublishers()
{
    return;
}

void BuffDetectorNode::destroyDebugPublishers()
{
    return;
}

} // namespace buff_ws

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(buff_ws::BuffDetectorNode);