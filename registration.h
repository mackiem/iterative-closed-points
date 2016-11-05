#pragma once
#include "opencv2/core/core.hpp"

class Registration
{
public:
	cv::Mat load_depth_img(std::string filename);
	cv::Mat create_point_cloud_mat(cv::Mat& depth_img, cv::Mat& K);
	std::vector<cv::Vec3d> create_point_cloud(cv::Mat& depth_img, cv::Mat& K);

	cv::Mat construct_M(const std::vector<cv::Vec3d>& pts, cv::Vec3d& c);

	void icp(std::vector<cv::Vec3d>& pc_P, std::vector<cv::Vec3d>& pc_Q, 
		 std::vector<cv::Vec3d>& transformed_pc_Q );

	void write_point_cloud(std::vector<cv::Vec3d>& pc_P, std::string prefix);

	Registration();
	~Registration();
};

