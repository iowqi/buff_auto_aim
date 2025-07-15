#include "buff.hpp"
#include "EulerAngle.h"

template <typename T1, typename T2>
void log(T1 name, T2 text)
{
    std::cerr << name << ":" << text << std::endl;
}

Buff::Buff(const IdentityColor &init_colorid, const LogoParams &init_r, const ImgprocParams &init_i, const BuffParams &init_b, const KFParams &init_kfp, const PolyFitParams &init_pfp)
    : id_color(init_colorid), r(init_r), i(init_i), b(init_b), kfp(init_kfp), pfp(init_pfp)
{
}

Buff &Buff::operator<<(cv::Mat &input)
{
    time_now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    imgproc(input);
    if (findTarget())
    {
        radius = euclideanDist(logo_now->center, target_center);
        angle_now = getRadian(logo_now->center, radius, target_center);
        double da = angle_now - angle_last;
        if (!start_predict)
        {
            time_start = time_last = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            start_predict = true;
        }
        if (std::fabs(da) < 0.1)
        {
            direction = da > 0 ? 1 : -1;
            double dt = (time_now - time_last) * 0.001 / 5;
            a_velocity = da / dt;
        }
    }
    if (start_predict)
    {
        std::cerr << "Direction:" << direction << std::endl;
        kalmanFilter(&kfp, a_velocity);
        predictTarget();
        a_predict = angle_now + getPredictAngle(dt, latency, std::ceil(latency / dt), coefficients, pfp.fit_offset);
        dest_point.x = (logo_now->center.x + (radius * cos(a_predict)));
        dest_point.y = (logo_now->center.y - (radius * sin(a_predict)));
    }
    return *this;
}

void Buff::updateData()
{
    logo_last = logo_now;
    buff_last = buff_now;
    time_last = time_now;
    angle_last = angle_now;
}

void Buff::imgproc(cv::Mat &frame)
{
    // 灰度通道裁剪
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bin_gray, i.light_th, 255, cv::THRESH_BINARY);
    // bgr通道裁剪
    cv::split(frame, bgr_channels);
    if (id_color == IdentityColor::IDENTITY_RED)
        cv::subtract(bgr_channels[2], bgr_channels[0], bin_bgr);
    else
        cv::subtract(bgr_channels[0], bgr_channels[2], bin_bgr);
    cv::threshold(bin_bgr, bin_bgr, i.color_th, 255, cv::THRESH_BINARY);
    // 求交集、形态学处理
    cv::bitwise_and(bin_gray, bin_bgr, bin_and);
    cv::morphologyEx(bin_and, bin_and, cv::MORPH_CLOSE, kernel);
}

Buff::AbstractTatget::AbstractTatget(cv::RotatedRect &box, std::vector<cv::Point> &points) : cv::RotatedRect(box)
{
    formatAngle(*this);
    this->area = this->size.area();
    this->area_ratio = cv::minEnclosingTriangle(points, this->triangle) / this->area;
    this->fill_ratio = cv::contourArea(points) / this->area;
    this->aspect_ratio = this->size.aspectRatio();
}

void Buff::shiftCenter(AbstractTatget &target, float fx, float fy)
{
    target.center.x += target.size.width * fx;
    target.center.y += target.size.height * fy;
}

bool Buff::isLogo(Buff::AbstractTatget &target)
{
    bool is_solid = target.fill_ratio > r.min_fill_ratio;
    bool is_square = fabs(1.0 - target.aspect_ratio) < r.square_tolerance && target.area_ratio > r.min_area_ratio;
    bool is_small = target.size.width < r.max_size;

    bool is_logo = is_solid && is_square && is_small;
    return is_logo;
}

bool Buff::isBuff(Buff::AbstractTatget &target)
{
    bool is_large = target.size.height > b.min_length;
    bool is_long_rectangle = fabs(0.5 - target.aspect_ratio) < b.aspect_tolerance;
    bool is_half_fill = fabs(0.5 - target.fill_ratio) < b.fill_tolerance;
    bool is_shape_like = fabs(1.0 - target.area_ratio) < b.shape_tolerance;
    bool is_triangle_match = isIsoscelesTopAcuteTriangle(target.triangle);
    bool is_buff = is_long_rectangle && is_large && is_half_fill && is_shape_like && is_triangle_match;
    return is_buff;
}

cv::Point2f Buff::findBullseye(AbstractTatget &target)
{
    int size = target.size.width * 0.8; // 0.8 full
    cv::Point2f relative_arr(size / 2, size / 2);
    cv::Point2f ps[4];
    target.points(ps);
    auto centroid = getTriangleCentroid(target.triangle);
    auto close_point = euclideanDistSquare(ps[0], centroid) < euclideanDistSquare(ps[1], centroid) ? ps[0] : ps[1];
    auto roi_center = getMidPoint(close_point, getMidPoint(ps[2], ps[3])) * (1 - .15) + centroid * .15; // shifted
    auto roi_orig = roi_center - relative_arr;
    target_roi = gray(cv::Rect(roi_orig, cv::Size(size, size)));
    cv::GaussianBlur(target_roi, target_roi, cv::Size(5, 5), 0);
    double min, max;
    cv::minMaxLoc(target_roi, &min, &max);
    houghtf::HoughCirclesAlt(target_roi, circles, 1, 5, 0, 0, int(max), b.circle_accuracy);
    int len = circles.size();
    if (len > 0)
        if (len <= 2)
            return cv::Point2f(circles[0].c[0], circles[0].c[1]) + roi_orig;
        else
        {
            std::vector<cv::Point2f> points;
            for (int i = 0; i < len; i++)
                points.emplace_back(cv::Point2f(circles[i].c[0], circles[i].c[1]));
            return points[findNearestPoint(points, relative_arr)] + roi_orig;
        }
    else
        return roi_center;
}

bool Buff::findTarget()
{
    contours = std::vector<std::vector<cv::Point>>(); // 存储轮廓
    logos = std::vector<AbstractTatget>();
    buffs = std::vector<AbstractTatget>();
    cv::findContours(bin_and, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        if (contours[i].size() < 10)
            continue;
        auto rect = cv::minAreaRect(contours[i]);
        if (rect.size.width < r.min_size || rect.size.height < r.min_size)
            continue;
        AbstractTatget target(rect, contours[i]);
        if (isLogo(target))
            logos.emplace_back(target);
        else if (isBuff(target))
            buffs.emplace_back(target);
    }
    int l_size = logos.size(), b_size = buffs.size();
    if (l_size)
    {
        logo_now = &logos[0];
        shiftCenter(*logo_now, r.shift_x, r.shift_y);
        // if (l_size > 1)
        ; // TODO: Multiple target selection
    }
    if (b_size)
    {
        buff_now = &buffs[0];
        target_center = findBullseye(*buff_now);
        // if (b_size > 1)
        ; // TODO: Multiple target selection
    }
    return l_size && b_size;
}

double Buff::kalmanFilter(KFParams *kfp, double input)
{
    // 预测协方差方程：k时刻系统估算协方差 = k-1时刻的系统协方差 + 过程噪声协方差
    kfp->now_P = kfp->last_P + kfp->Q;
    // 卡尔曼增益方程：卡尔曼增益 = k1-1时刻系统估算协方差 / （k时刻系统估算协方差 + 观测噪声协方差）
    kfp->Kg = kfp->now_P / (kfp->now_P + kfp->R);
    // 更新最优值方程：k时刻状态变量的最优值 = 状态变量的预测值 + 卡尔曼增益 * （测量值 - 状态变量的预测值）
    kfp->out = kfp->out + kfp->Kg * (input - kfp->out); // 因为这一次的预测值就是上一次的输出值
    // 更新协方差方程: 本次的系统协方差付给 kfp->last_P 威下一次运算准备。
    kfp->last_P = (1.0 - kfp->Kg) * kfp->now_P;
    std::cerr << "KFOut:" << kfp->out << std::endl;
    return kfp->out;
}
void Buff::polyFit(std::deque<double> &input_t, std::deque<double> &input_v, Eigen::VectorXd &coeffs, int &degree, int &buffer_len)
{
    Eigen::MatrixXd A(buffer_len, degree + 1);
    Eigen::VectorXd b(buffer_len);
    for (int i = 0; i < buffer_len; i++)
    {
        b(i) = input_v[i];
        for (int j = 0; j <= degree; j++)
            A(i, j) = pow(input_t[i], j);
    }
    coeffs = A.colPivHouseholderQr().solve(b);
}

double Buff::getPredictV(double &input, Eigen::VectorXd &coeffs, double offset, double scale)
{
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); i++)
        result += coeffs(i) * pow(input + offset, i);
    // result = (result - direction * base_line) * scale + direction * base_line;
    return (result * scale + direction * base_line * (1 - scale));
}

double Buff::getPredictAngle(double &input, double &latency, int n, Eigen::VectorXd &coeffs, double offset)
{
    double dest = input + latency;
    double step = latency / n;
    // Simpson’s rule with adaptive scaling factor
    double sum = getPredictV(input, coeffs, offset) + getPredictV(dest, coeffs, offset, scale);
    for (int i = 1; i < n; i += 2)
    {
        double dif = i * step;
        double x = input + dif;
        double s = std::min(0.9 * pow((dif - 1), 2) + 0.15, 1.0);
        sum += 4 * getPredictV(x, coeffs, offset, s);
    }
    for (int i = 2; i < n; i += 2)
    {
        double dif = i * step;
        double x = input + i * step;
        double s = std::min(0.9 * pow((dif - 1), 2) + 0.15, 1.0);
        sum += 2 * getPredictV(x, coeffs, offset, s);
    }
    return sum * step / 3.0;
}

void Buff::predictTarget()
{
    dt = (time_now - time_start) * 0.001 / 5;
    buffer_v.push_back(kfp.out);
    buffer_t.push_back(dt);
    int fit_size = buffer_t.size();
    polyFit(buffer_t, buffer_v, coefficients, pfp.degree, fit_size);
    if (fit_size >= pfp.buffer_len)
    {
        buffer_v.pop_front();
        buffer_t.pop_front();
    }

    // Update the IMU message
    /* Eigen::Matrix3d ImuRotationMatrix = quat.matrix().cast<double>();            // IMU旋转矩阵    cast<double>()转换成double类型
     Eigen::Vector3d ImuEulerAngle = ImuRotationMatrix.eulerAngles(2, 1, 0);   // IMU欧拉角

    PnPInfo armor_center = coordsolver.pnp(armor_points_pixel, ImuRotationMatrix);

    Eigen::Vector3d hit_point_world = {0, 0, 0};
    Eigen::Vector3d hit_point_camera = {0, 0, 0};
    Eigen::Vector2d hit_point_pixel = {0, 0};

     // 把拟合大符预测出来的角度算出来的点坐标（Eigen），当作击打点，进行动态补偿
    hit_point_world[0] = dest_point.x;
    hit_point_world[1] = dest_point.y;
     hit_point_world = (armor_center.rmat * hit_point_world) + armor_center.world_p;
    hit_point_camera = coordsolver.WorldToCamera(hit_point_world, ImuRotationMatrix);
    hit_point_pixel = coordsolver.CameraToPixel(hit_point_camera);

    cv::Point2f hit_point_2d;
    hit_point_2d.x = hit_point_pixel[0];
    hit_point_2d.y = hit_point_pixel[1];
     cv::circle(input, hit_point_2d, 5, Scalar(255, 255, 0), 2);  // 在图上画出动态补偿的点

    // 得到yaw pitch 发送给电控
    auto target_armor_angle = coordsolver.getCompensation(hit_point_world, ImuRotationMatrix);*/

    //     EulerAngle targe_armor_angle = this->getAngle(cv::Point(bin_gray.cols / 2, bin_gray.rows / 2), dest_point);

    //     log("yaw", targe_armor_angle.yaw);
    //     log("pitch", targe_armor_angle.pitch);
    //     log("frame", frame);

    //     log("windmill_spd_mode", this->windmill_spd_mode);
    //     log("targetColor", (this->id_color == IdentityColor::IDENTITY_RED));
    //     if (radius < 200 && logo_now->center.x > 0 && logo_now->center.y > 0 && logo_now->center.x < this->bin_gray.cols && logo_now->center.y < this->bin_gray.rows && dest_point.x > 0 && dest_point.y > 0 && dest_point.x < this->bin_gray.cols && dest_point.y < this->bin_gray.rows)
    //     {
    //         hasTarget = 1;
    //         targetTo(targe_armor_angle, distance_, 1, 1);
    //     }
    //     else
    //     {
    //         std::cout << "radius too large" << std::endl;
    //         targetTo({0, 0}, distance_, 0, 0);
    //     }

    //     if ((frame % 8) == 0)
    //     {
    //         this->last_angle = this->getRadian(logo_now->center, radius, target_center);
    //         this->last_angle_time = time_now;
    //     }
    // }
    // if (!hasTarget)
    //     targetTo({0, 0}, distance_, 0, 0);
}

void Buff::update(const uart_process_2::uart_receive &receive)
{
    // printf("%d\n",receive.mod);
    if (receive.mod >= 10 && receive.mod <= 13)
    {
        if (this->disabled)
        { // 从非打符状态切换到打符状态
            this->disabled = false;
        }
        switch (receive.mod)
        {
        case 10:
            this->windmill_spd_mode = false;
            // this->direction = CLOCKWISE;
            break;
        case 11:
            this->windmill_spd_mode = false;
            // this->direction = COUNTERCLOCKWISE;
            break;
        case 12:
            this->windmill_spd_mode = true;
            // this->direction = CLOCKWISE;
            break;
        case 13:
            this->windmill_spd_mode = true;
            // this->direction = COUNTERCLOCKWISE;
            break;
        default:
            break;
        }
    }
    else
    {
        this->disabled = true; // This stops feeding new images.
    }
    // 和自瞄相反
    if (receive.red_blue == 1)
    {
        // std::cout << "寻找目标颜色：红色" << std::endl;
        this->id_color = IdentityColor::IDENTITY_RED;
    }
    else
    {
        // std::cout << "寻找目标颜色：蓝色" << std::endl;
        this->id_color = IdentityColor::IDENTITY_BLUE;
    }
}

double Buff::predict_angle(double angle, bool direction, bool windmill_spd_mode)
{
    // double res;
    // double angle_addon;
    // if (windmill_spd_mode)
    // {
    //     double timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() * 0.001;
    //     // angle_addon = -((angle) + (0.416667 * cos(1.884 * (timestamp))) - (1.305 * timestamp));
    //     // res = (((-0.416667) * cos(1.884 * (timestamp + this->time_delay))) + (1.305 * (timestamp + this->time_delay))+ (direction ? angle_addon*-1 : angle_addon));
    //     // res = 2*angle - res;
    //     angle_addon = 1.826 / 1.942 * (sin(0.971 * this->time_delay + 1.942 * timestamp) * sin(0.971 * this->time_delay)) + 1.177 * time_delay;
    //     res = angle + (direction ? (angle_addon) * -1 : (angle_addon));
    // }
    // else
    // {
    //     angle_addon = 0.01 * CV_PI;
    //     res = angle + (direction ? (this->angle_offset + angle_addon) * -1 : (this->angle_offset + angle_addon));
    // }
    // // if(windmill_spd_mode) res *= -1;
    // return res;
    // /// return angle;
}

double Buff::getRadian(cv::Point2f center, double radius, cv::Point2f point)
{
    double angle = asin((center.y - point.y) / radius);
    if (point.x - center.x < 0)
        angle = CV_PI - angle;
    if (angle < 0)
        angle = 2 * CV_PI + angle;
    return angle;
}

double Buff::calc_speed(double time_now, double angle)
{
    // double speed_tmp = fabs((angle / (time_now - this->last_angle_time)) * 9.549297);
    // if (speed_tmp < 30)
    // {
    //     this->speed_arr[this->speed_cnt % 20] = fabs((angle / (time_now - this->last_angle_time)) * 9.549297);
    //     this->speed_cnt++;
    // }
    // if (this->speed_cnt > 19)
    // {
    //     double speed;
    //     for (int i = 0; i < 20; i++)
    //         speed += this->speed_arr[i];
    //     speed /= 20;
    //     return speed;
    // }
    // return 0;
}

cv::Point2f Buff::getMidPoint(cv::Point2f p1, cv::Point2f p2)
{
    return cv::Point2f((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
}

cv::Point2f Buff::getTriangleCentroid(std::vector<cv::Point2f> &triangle)
{
    return cv::Point2f((triangle[0] + triangle[1] + triangle[2]) / 3);
}

void Buff::AbstractTatget::formatAngle(cv::RotatedRect &box)
{
    if (box.size.width > box.size.height)
    {
        std::swap(box.size.height, box.size.width);
        box.angle += 90;
    }
}

bool Buff::isIsoscelesTopAcuteTriangle(std::vector<cv::Point2f> &triangle)
{
    int index;
    float min = std::numeric_limits<float>::max();
    float length[3];
    for (int i = 0; i < 3; i++)
    {
        length[i] = euclideanDistSquare(triangle[i], triangle[(i + 1) % 3]);
        if (min > length[i])
        {
            min = length[i];
            index = i;
        }
    }
    std::swap(length[index], length[0]);
    bool is_equal = abs(1 - length[1] / length[2]) < b.symmetry_tolerance;
    bool is_longer = abs(0.6 - min / std::max(length[1], length[2])) < b.arrow_like_tolerance;
    return is_equal && is_longer;
}

float Buff::euclideanDistSquare(cv::Point2f &a, cv::Point2f &b)
{
    cv::Point2f diff = a - b;
    return (diff.x * diff.x + diff.y * diff.y);
}

float Buff::euclideanDist(cv::Point2f &a, cv::Point2f &b)
{
    cv::Point2f diff = a - b;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

int Buff::findNearestPoint(std::vector<cv::Point2f> &points, cv::Point2f &point)
{
    int index;
    float min = std::numeric_limits<float>::max();
    for (int i = 0; i < points.size(); i++)
    {
        float distance = euclideanDistSquare(points[i], point);
        if (min > distance)
        {
            min = distance;
            index = i;
        }
    }
    return index;
}