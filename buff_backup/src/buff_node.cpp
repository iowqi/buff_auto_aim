#include "buff.hpp"
#include "buff_node.hpp"

BuffNode::BuffNode() : it_(nh_)
{
	camraw_img_sub_ = it_.subscribe("DahuaCamera/LowDims", 1, &BuffNode::onCameraRawImageReceived, this);
	uart_pub_ = nh_.advertise<uart_process_2::uart_send>("uart_send", 1);

	initBuff();

	ros::NodeHandle nh("~");
	nh.getParam("debug_", debug_);
	if (debug_)
		initDebugger();
}

void BuffNode::initBuff()
{
	Buff::IdentityColor id_color = Buff::IdentityColor::IDENTITY_RED;
	Buff::ImgprocParams i_params = {
		.light_th = 130,
		.color_th = 100,
		.close_size = 5};
	Buff::LogoParams r_params = {
		.min_size = 5,
		.max_size = 50,
		.min_fill_ratio = .70,
		.square_tolerance = .15,
		.min_area_ratio = 1.45,
		.shift_x = 0.08,
		.shift_y = 0.33};
	Buff::BuffParams b_params = {
		.min_length = 100,
		.aspect_tolerance = .10,
		.fill_tolerance = .20,
		.shape_tolerance = .20,
		.symmetry_tolerance = .30,
		.arrow_like_tolerance = .17,
		.circle_accuracy = .85};
	Buff::KFParams kf_params = {
		.last_P = 0.02,
		.now_P = 0,
		.out = 0,
		.Kg = 0,
		.Q = 0.001,
		.R = 0.3};
	Buff::PolyFitParams pf_params = {
		.degree = 3,
		.buffer_len = 100,
		.fit_offset = 0.15};
	buff_ = new Buff(id_color, r_params, i_params, b_params, kf_params, pf_params);
	buff_->base_line = 1.1775;
	buff_->scale = 0.5;
	buff_->kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(buff_->i.close_size * 2 + 1, buff_->i.close_size * 2 + 1));
}

void BuffNode::reconfigureCB(buff::BuffConfig &config, uint32_t level)
{
	buff_->id_color = Buff::IdentityColor(config.identity_color);
	buff_->i.light_th = config.light_threshold;
	buff_->i.color_th = config.color_threshold;
	buff_->i.close_size = config.close_kernel_size;
	buff_->r.min_size = config.logo_min_size;
	buff_->r.max_size = config.logo_max_size;
	buff_->r.min_fill_ratio = config.logo_min_fill_ratio;
	buff_->r.square_tolerance = config.logo_square_tolerance;
	buff_->r.min_area_ratio = config.logo_min_area_ratio;
	buff_->r.shift_x = config.logo_shift_x;
	buff_->r.shift_y = config.logo_shift_y;
	buff_->b.min_length = config.buff_min_length;
	buff_->b.aspect_tolerance = config.buff_aspect_tolerance;
	buff_->b.fill_tolerance = config.buff_fill_tolerance;
	buff_->b.shape_tolerance = config.buff_shape_tolerance;
	buff_->b.symmetry_tolerance = config.buff_symmetry_tolerance;
	buff_->b.arrow_like_tolerance = config.buff_arrow_like_tolerance;
	buff_->b.circle_accuracy = config.buff_circle_accuracy;
	buff_->kfp.Q = config.kf_Q;
	buff_->kfp.R = config.kf_R;
	buff_->pfp.degree = config.fit_degree;
	buff_->pfp.fit_offset = config.fit_offset;
	buff_->latency = config.latency;
	buff_->scale = std::min(0.9 * pow((config.latency - 1), 2) + 0.15, 1.0);

	buff_->kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(buff_->i.close_size * 2 + 1, buff_->i.close_size * 2 + 1));
}

void BuffNode::initDebugger()
{
	std::cerr << "\033[41;37m debug mode \033[0m" << std::endl;
	this->dsrv_ = new dynamic_reconfigure::Server<buff::BuffConfig>(ros::NodeHandle("buff_node/debug_params"));
	dynamic_reconfigure::Server<buff::BuffConfig>::CallbackType cb = boost::bind(
		&BuffNode::reconfigureCB, this, _1, _2);
	this->dsrv_->setCallback(cb);
	this->merge_img_pub_ = it_.advertise("buff_node/debug_image/merge", 1);
	this->roi_img_pub_ = it_.advertise("buff_node/debug_image/roi", 1);
	this->result_img_pub_ = it_.advertise("buff_node/debug_image/result", 1);
	this->radius_pub_ = nh_.advertise<std_msgs::Float32>("buff_node/debug_msgs/radius", 1);
	this->angle_pub_ = nh_.advertise<std_msgs::Float32>("buff_node/debug_msgs/angle", 1);
	this->velocity_pub_ = nh_.advertise<std_msgs::Float32>("buff_node/debug_msgs/velocity", 1);
	this->filtered_v_pub_ = nh_.advertise<std_msgs::Float32>("buff_node/debug_msgs/filtered", 1);
	this->polyfit_v_pub_ = nh_.advertise<std_msgs::Float32>("buff_node/debug_msgs/polyfit", 1);
	this->predict_v_pub_ = nh_.advertise<std_msgs::Float32>("buff_node/debug_msgs/predict_v", 1);
	this->predict_a_pub_ = nh_.advertise<std_msgs::Float32>("buff_node/debug_msgs/predict_a", 1);
}

void BuffNode::onCameraRawImageReceived(const sensor_msgs::ImageConstPtr &msg)
{
	auto start = std::chrono::system_clock::now();
	cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
	*buff_ << frame;
	buff_->updateData();

	// buff_->logo_last = buff_->logo_now;
	// buff_->buff_last = buff_->buff_now;
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cerr << "\r - Timing: Total time elapsed: \033[41;37m" << std::to_string(duration) << "\033[0m ms." << std::endl;
	std::cerr << "---------------------------------------------------" << std::endl;
	if (debug_)
	{
		cv::Mat merge, target;
		cv::Mat f_contours = frame.clone();
		std::vector<cv::Mat> channels;
		cv::split(frame, channels);
		if (buff_->id_color == Buff::IdentityColor::IDENTITY_RED) // 融合通道图像
		{
			channels[0] = buff_->bin_gray;
			channels[1] = buff_->bin_and;
			channels[2] = buff_->bin_bgr;
		}
		else
		{
			channels[0] = buff_->bin_bgr;
			channels[1] = buff_->bin_and;
			channels[2] = buff_->bin_gray;
		}
		cv::merge(channels, merge);

		cv::drawContours(f_contours, buff_->contours, -1, cv::Scalar(128, 128, 128), 1, cv::LINE_AA);
		for (int i = 0; i < buff_->logos.size(); i++)
		{
			drawRotatedRect(f_contours, buff_->logos[i], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			drawTriangle(f_contours, buff_->logos[i].triangle, cv::Scalar(128, 128, 128), 1, cv::LINE_AA);
		}
		for (int i = 0; i < buff_->buffs.size(); i++)
		{
			drawRotatedRect(f_contours, buff_->buffs[i], cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
			drawTriangle(f_contours, buff_->buffs[i].triangle, cv::Scalar(128, 128, 128), 1, cv::LINE_AA);
		}

		if (buff_->buffs.size())
		{
			cv::cvtColor(buff_->target_roi, target, cv::COLOR_GRAY2BGR);
			cv::drawMarker(target, cv::Point(target.cols / 2, target.rows / 2), cv::Scalar(255, 0, 0), cv::MARKER_TILTED_CROSS);
			for (size_t i = 0; i < buff_->circles.size(); i++)
			{
				cv::Vec3f cc = buff_->circles[i].c;
				cv::circle(target, cv::Point(cc[0], cc[1]), cc[2], cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // 圆周
				cv::drawMarker(target, cv::Point(cc[0], cc[1]), cv::Scalar(0, 255, 0), cv::MARKER_CROSS);  // 圆心
			}
			cv::circle(f_contours, buff_->target_center, 20, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
			cv::drawMarker(f_contours, buff_->target_center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 1);
			roi_img_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", target).toImageMsg());

			if (buff_->logos.size())
			{
				cv::circle(f_contours, buff_->dest_point, 20, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
				cv::drawMarker(f_contours, buff_->dest_point, cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 20, 1);
				cv::circle(f_contours, buff_->logo_now->center, buff_->radius, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
			}
		}

		merge_img_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", merge).toImageMsg());
		result_img_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", f_contours).toImageMsg());

		if (buff_->start_predict)
		{
			double dt = (buff_->time_now - buff_->time_start) * 0.001 / 5;
			if (buff_->angle_now - buff_->angle_last > 0)
			{
				polyfit_v = std::min(std::max(buff_->getPredictV(dt, buff_->coefficients, buff_->pfp.fit_offset), -10.0), 12.5);
				predict_v = std::min(std::max((buff_->getPredictV(dt, buff_->coefficients, (buff_->pfp.fit_offset + buff_->latency), buff_->scale)), -10.0), 12.5);
			}
			else
			{
				polyfit_v = std::min(std::max(buff_->getPredictV(dt, buff_->coefficients, buff_->pfp.fit_offset), -12.5), 10.0);
				predict_v = std::min(std::max((buff_->getPredictV(dt, buff_->coefficients, (buff_->pfp.fit_offset + buff_->latency), buff_->scale)), -12.5), 10.0);
			}
		}

		std_msgs::Float32 radius_msg, angle_msg, velocity_msg, filtered_msg, polyfit_msg, predict_v_msg, predict_a_msg;
		radius_msg.data = buff_->radius;
		angle_msg.data = buff_->angle_now;
		velocity_msg.data = buff_->a_velocity;
		filtered_msg.data = buff_->kfp.out;
		polyfit_msg.data = polyfit_v;
		predict_v_msg.data = predict_v;
		predict_a_msg.data = buff_->a_predict;
		radius_pub_.publish(radius_msg);
		angle_pub_.publish(angle_msg);
		velocity_pub_.publish(velocity_msg);
		filtered_v_pub_.publish(filtered_msg);
		polyfit_v_pub_.publish(polyfit_msg);
		predict_v_pub_.publish(predict_v_msg);
		predict_a_pub_.publish(predict_a_msg);
	}
}

void BuffNode::drawTriangle(cv::Mat &image, std::vector<cv::Point2f> &triangle, const cv::Scalar &color, const int &thickness, const int &lineType)
{
	std::vector<std::vector<cv::Point>> tmpContours; // 构建轮廓线
	std::vector<cv::Point> contours;
	for (int i = 0; i != 3; ++i)
		contours.emplace_back(cv::Point2i(triangle[i]));
	tmpContours.insert(tmpContours.end(), contours);
	std::stringstream ss; // 转换字符串格式
	std::string str;
	ss << int(cv::contourArea(contours));
	ss >> str;
	cv::putText(image, str, contours[0], cv::FONT_HERSHEY_PLAIN, thickness, color, thickness, lineType);
	cv::drawContours(image, tmpContours, 0, color, thickness, lineType);
}

void BuffNode::drawRotatedRect(cv::Mat &image, cv::RotatedRect &rect, const cv::Scalar &color, const int &thickness, const int &lineType)
{
	cv::Point2f ps[4]; // 提取旋转矩形的四个角点
	rect.points(ps);
	std::vector<std::vector<cv::Point>> tmpContours; // 构建轮廓线
	std::vector<cv::Point> contours;
	for (int i = 0; i != 4; ++i)
		contours.emplace_back(cv::Point2i(ps[i]));
	tmpContours.insert(tmpContours.end(), contours);
	std::stringstream ss; // 转换字符串格式
	std::string str;
	ss << int(rect.angle) << "|" << int((rect.size.width / rect.size.height) * 100) << "|" << int(rect.size.area());
	ss >> str;
	cv::putText(image, str, contours[0], cv::FONT_HERSHEY_PLAIN, thickness, color, thickness, lineType); // 绘制部分
	cv::line(image, contours[0], contours[1], color, thickness * 2, lineType);
	cv::drawContours(image, tmpContours, 0, color, thickness, lineType);
}

void failSafe(int)
{
	std::cerr << "[FailSafe] I'm dying." << std::endl;
	/*const EulerAngle failSafeAngle = {0, 0};
	sagitari.targetTo(failSafeAngle, failSafeAngle, 0, false, 0);*/
}

int main(int argc, char *argv[])
{
	signal(SIGINT, failSafe);
	signal(SIGABRT, failSafe);

	ros::init(argc, argv, "buff_node");
	BuffNode buff_node;
	ros::Rate rate(200);

	while (ros::ok())
	{
		ros::spinOnce();
		rate.sleep();
	}

	return 0;
}
