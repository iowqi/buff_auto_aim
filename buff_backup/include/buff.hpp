#ifndef BUFF_HPP_
#define BUFF_HPP_

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <uart_process_2/uart_send.h>
#include <uart_process_2/uart_receive.h>
#include <sensor_msgs/Imu.h>
#include <cmath>
#include <Eigen/Dense>

#include "hough_circles_alt.hpp"

#include "EulerAngle.h"
// #include "coord_solver.hpp"

const float CAMERA_FOCUS = 60;
const float im_real_weights = 3.75;
const double limit_yaw_angle_val = 40;
const double limit_pitch_angle_val = 20;
const double distance_ = 2;
const double focus = 60.0;

class Buff
{

public:
    enum class IdentityColor
    {
        IDENTITY_RED,
        IDENTITY_BLUE
    };
    struct ImgprocParams
    {
        int light_th;
        int color_th;
        int close_size;
    };
    struct LogoParams
    {
        int min_size;
        int max_size;
        double min_fill_ratio;
        double square_tolerance;
        double min_area_ratio;
        double shift_x;
        double shift_y;
    };
    struct BuffParams
    {
        int min_length;
        double aspect_tolerance;
        double fill_tolerance;
        double shape_tolerance;
        double symmetry_tolerance;
        double arrow_like_tolerance;
        double circle_accuracy;
    };
    struct AbstractTatget : public cv::RotatedRect
    {
        explicit AbstractTatget(cv::RotatedRect &box, std::vector<cv::Point> &points);
        void formatAngle(cv::RotatedRect &box);
        std::vector<cv::Point2f> triangle;
        double area;
        double fill_ratio;
        double aspect_ratio;
        double area_ratio;
    };
    struct KFParams
    {
        double last_P; // 上次估算协方差	--e(ESTk-1)上次协方差
        double now_P;  // 当前估算协方差	--预测e(ESTk)当前估算协方差
        double out;    // 卡尔曼滤波器输出
        double Kg;     // 卡尔曼增益 --Kk
        double Q;      // 过程噪声协方差 --影响收敛速度
        double R;      // 观测噪声协方差 --e(MEAk)测量误差 --影响收敛速度
    };
    struct PolyFitParams
    {
        int degree;
        int buffer_len;
        double fit_offset;
    };
    IdentityColor id_color;

    ImgprocParams i;
    LogoParams r;
    BuffParams b;
    KFParams kfp;
    PolyFitParams pfp;

    cv::Size fsize;
    cv::Mat gray;
    cv::Mat bin_gray;
    cv::Mat bin_bgr;
    cv::Mat bin_and;
    cv::Mat target_roi;
    cv::Mat kernel;
    std::vector<cv::Mat> bgr_channels;            // 存储通道
    std::vector<std::vector<cv::Point>> contours; // 存储轮廓
    std::vector<AbstractTatget> logos;
    std::vector<AbstractTatget> buffs;
    std::vector<houghtf::EstimatedCircle> circles;

    AbstractTatget *logo_now; // 当前目标
    AbstractTatget *buff_now;
    AbstractTatget *logo_last; // 前一目标，用于同时找到两个目标时的取舍
    AbstractTatget *buff_last;
    cv::Point2f target_center = cv::Point(0, 0);
    int direction = 1;
    double radius = 0;
    double time_start = 0;
    double time_now = 0;
    double time_last = 0;
    double angle_now = 0;
    double angle_last = 0;
    double a_velocity = 0;
    double dt;
    double base_line;
    double latency = 0;
    double scale;
    double a_predict = 0;
    bool start_predict = false;

    std::deque<double> buffer_v;
    std::deque<double> buffer_t;
    Eigen::VectorXd coefficients;
    // double timestamp;
    cv::Point dest_point = cv::Point(0, 0);
    // int frame = 0;
    // bool hasTarget = 0;
    // bool direction = true; // true为顺时针
    // double last_angle = 0xFFFF;
    // double last_angle_time = 0xFFFF;
    // double speed_arr[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // int speed_cnt = 0;
    // float time_delay = 0.2;
    // double angle_offset = 0.14 * CV_PI; //(At Speed Of 25m/s)

    // CoordSolver coord_solver;
    bool disabled = false;
    bool windmill_spd_mode = true;

    Buff(const IdentityColor &init_colorid, const LogoParams &init_r, const ImgprocParams &init_i, const BuffParams &init_b, const KFParams &init_kfp, const PolyFitParams &init_pfp);

    // EulerAngle getAngle(cv::Point center, cv::Point dest_point);
    // void targetTo(const EulerAngle& currentAngle, double distance, bool hasTarget, int attackFlag, int predictLatency);

    Buff &operator<<(cv::Mat &input);
    void updateData();
    double getPredictV(double &input, Eigen::VectorXd &coeffs, double offset = 0, double scale = 1);

private:
    void imgproc(cv::Mat &frame);
    void shiftCenter(AbstractTatget &target, float fx, float fy);
    bool isLogo(AbstractTatget &target);
    bool isBuff(AbstractTatget &target);
    bool isIsoscelesTopAcuteTriangle(std::vector<cv::Point2f> &triangle);
    cv::Point2f findBullseye(AbstractTatget &target);
    bool findTarget();
    void predictTarget();
    void update(const uart_process_2::uart_receive &receive);
    double kalmanFilter(KFParams *kfp, double input);
    void polyFit(std::deque<double> &input_t, std::deque<double> &input_v, Eigen::VectorXd &coeffs, int &degree, int &buffer_len);
    double getPredictAngle(double &input, double &latency, int n, Eigen::VectorXd &coeffs, double offset = 0);
    double predict_angle(double angle, bool direction, bool windmill_spd_mode);
    double getRadian(cv::Point2f center, double radius, cv::Point2f point);
    double calc_speed(double time_now, double angle);
    cv::Point2f getMidPoint(cv::Point2f p1, cv::Point2f p2);
    cv::Point2f getTriangleCentroid(std::vector<cv::Point2f> &triangle);
    float euclideanDist(cv::Point2f &a, cv::Point2f &b);
    float euclideanDistSquare(cv::Point2f &a, cv::Point2f &b);
    int findNearestPoint(std::vector<cv::Point2f> &points, cv::Point2f &point);

    EulerAngle getAngle(cv::Point center, cv::Point dest_point);
    void targetTo(const EulerAngle &currentAngle, double distance, bool hasTarget, int attackFlag, int predictLatency = 0);
};
#endif