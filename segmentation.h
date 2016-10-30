#pragma once

#include "opencv2/core/core.hpp"

class Segmentation {
	std::vector<std::vector<int>> histogram_;
	std::vector<int> total_no_of_pixels_per_channel_;
	void create_historgram(const cv::Mat& img);
	std::vector<int> run_otsu();
	void create_textures(const cv::Mat& img, std::vector<cv::Mat>& texture_imgs);
public:
	void threshold_on_rgb(const cv::Mat& img, std::vector<cv::Mat>& masks);
	void threshold_on_texture(const cv::Mat& img, std::vector<cv::Mat>& masks);
	Segmentation();
	~Segmentation();
};

class Contour {
public:
	void trace_contours(const cv::Mat& img, cv::Mat& mask) const;
	
};

