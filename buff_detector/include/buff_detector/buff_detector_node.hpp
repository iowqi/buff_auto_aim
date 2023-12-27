#ifndef _BUFF_DETECTOR_NODE_HPP_
#define _BUFF_DETECTOR_NODE_HPP_

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

// OpenCV
#include <opencv2/core.hpp>

namespace buff_ws
{
class BuffDetectorNode : public rclcpp::Node
{
  public:
    BuffDetectorNode(const rclcpp::NodeOptions &options);

  private:
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

    // Debug information
    bool debug_;
    std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void createDebugPublishers();
    void destroyDebugPublishers();
};
} // namespace buff_ws

#endif