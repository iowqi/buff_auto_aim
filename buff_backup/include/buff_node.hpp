#ifndef BUFF_NODE_HPP_
#define BUFF_NODE_HPP_

#include <bits/stdc++.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>
#include <dynamic_reconfigure/server.h>
#include <opencv4/opencv2/core.hpp>
#include <uart_process_2/uart_receive.h>
#include <sagitari_debug/sagitari_img_debug.h>
#include <buff/BuffConfig.h>
#include <buff.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <std_msgs/Float32.h>
// #include <message_filters/subscriber.h>
// #include <EulerAngle.h>
// #include <Eigen/Core>
// #include <Eigen/Dense>

class BuffNode
{
public:
    BuffNode();

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber camraw_img_sub_;
    image_transport::Publisher merge_img_pub_;
    image_transport::Publisher roi_img_pub_;
    image_transport::Publisher result_img_pub_;
    ros::Publisher uart_pub_;
    ros::Publisher radius_pub_;
    ros::Publisher angle_pub_;
    ros::Publisher velocity_pub_;
    ros::Publisher filtered_v_pub_;
    ros::Publisher polyfit_v_pub_;
    ros::Publisher predict_v_pub_;
    ros::Publisher predict_a_pub_;
    uart_process_2::uart_send uart_sent_;       // 串口发送数据
    uart_process_2::uart_receive uart_receive_; // 串口接受数据

    bool debug_ = false;
    double polyfit_v = 0;
    double predict_v = 0;
    dynamic_reconfigure::Server<buff::BuffConfig> *dsrv_;
    void reconfigureCB(buff::BuffConfig &config, uint32_t level);
    void initDebugger();

    Buff *buff_;
    void initBuff();
    void onCameraRawImageReceived(const sensor_msgs::ImageConstPtr &msg);

    void drawTriangle(cv::Mat &image, std::vector<cv::Point2f> &triangle, const cv::Scalar &color, const int &thickness = 1, const int &lineType = 8);
    void drawRotatedRect(cv::Mat &image, cv::RotatedRect &rect, const cv::Scalar &color, const int &thickness = 1, const int &lineType = 8);
};

#endif