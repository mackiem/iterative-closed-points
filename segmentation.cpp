#include "Segmentation.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define CHANNELS 3
#define GREY_LEVELS 255

void Segmentation::create_historgram(const cv::Mat& img) {
	cv::Mat channels[CHANNELS];
	cv::split(img, channels);

	// init and cleanup
	histogram_.clear();
	histogram_.resize(CHANNELS);
	total_no_of_pixels_per_channel_.clear();
	total_no_of_pixels_per_channel_.resize(CHANNELS);

	for (int ch = 0; ch < CHANNELS; ++ch) {
		histogram_[ch].resize(256);
		for (int j = 0; j < 256; ++j) {
			histogram_[ch][j] = 0;
		}
		total_no_of_pixels_per_channel_[ch] = 0;
	}

	// populate historgram
	int totacount = 0;
	for (int ch = 0; ch < CHANNELS; ++ch) {
		for (int row = 0; row < img.rows; ++row) {
			for (int col = 0; col < img.cols; ++col) {
				int channel_grey_level = channels[ch].at<unsigned char>(row, col);
				// let's ignore 0 grey level, as thresholded images will contain majority of 0s
				if (channel_grey_level > 0) {
					histogram_[ch][channel_grey_level]++;
					total_no_of_pixels_per_channel_[ch]++;
				}
			}
		}
	}


}

std::vector<int> Segmentation::run_otsu() {

	// total no of pixels
	std::vector<int> max_thresholds(CHANNELS);

	// calculate mu(T)
	for (int ch = 0; ch < CHANNELS; ++ch) {
		double mu_T = 0.0;

		double prob[GREY_LEVELS];

		for (int g = 0; g < GREY_LEVELS; ++g) {
			prob[g] = histogram_[ch][g] / static_cast<double>(total_no_of_pixels_per_channel_[ch]);
			mu_T += g * prob[g];
		}

		double max_sigma_sqr_k = 0.0;
		int max_k = 0;
		for (int g = 0; g < GREY_LEVELS; ++g) {
			double omega_k = 0.0;
			double mu_k = 0.0;
			for (int k = 0; k < g; ++k) {
				omega_k += prob[k];
				mu_k += k * prob[k];
			}
			double sigma_sqr_k = 0.0;
			sigma_sqr_k = std::pow(mu_T * omega_k - mu_k, 2) / (omega_k * (1 - omega_k));
			if (max_sigma_sqr_k  < sigma_sqr_k) {
				max_sigma_sqr_k = sigma_sqr_k;
				max_k = g;
			}
		}
		max_thresholds[ch] = max_k;
	}

	return max_thresholds;
}

void Segmentation::create_textures(const cv::Mat& img, std::vector<cv::Mat>& texture_imgs) {
	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, CV_BGR2GRAY);

	std::vector<int> kernels = { 3, 5, 7 };
	texture_imgs.resize(CHANNELS);

	for (int k = 0; k < kernels.size(); ++k) {

		int kernel_size = kernels[k];

		//for (int ch = 0; ch < CHANNELS; ++ch) {
		cv::Mat padded_img;
		cv::copyMakeBorder(gray_img, padded_img, kernel_size - 2, kernel_size - 2, kernel_size - 2, kernel_size - 2, cv::BORDER_REPLICATE);

		cv::Mat texture_img = gray_img.clone();
		for (int row = 0; row < img.rows; ++row) {
			for (int col = 0; col < img.cols; ++col) {
				int padded_row = row + kernel_size - 2;
				int padded_col = col + kernel_size - 2;
				cv::Rect window(padded_col - (kernel_size - 2), padded_row - (kernel_size - 2), kernel_size, kernel_size);
				cv::Mat roi = padded_img(window);
				cv::Scalar mean;
				cv::Scalar stddev;
				cv::meanStdDev(roi, mean, stddev);
				// variance as texture value
				texture_img.at<unsigned char>(row, col) = std::round(std::pow(stddev[0], 2));
			}
		}
		texture_imgs[k] = texture_img;
	}
}


void Segmentation::threshold_on_rgb(const cv::Mat& img, std::vector<cv::Mat>& masks) {
	create_historgram(img);
	auto max_thresholds = run_otsu();

	cv::Mat channels[CHANNELS];
	cv::split(img, channels);

	masks.resize(CHANNELS);
	for (int ch = 0; ch < CHANNELS; ++ch) {
		cv::threshold(channels[ch], masks[ch], max_thresholds[ch], 255, CV_THRESH_BINARY);
	}
}

void Segmentation::threshold_on_texture(const cv::Mat& img, std::vector<cv::Mat>& masks) {
	std::vector<cv::Mat> texture_imgs;
	create_textures(img, texture_imgs);

	cv::Mat output_texture_img;
	cv::merge(texture_imgs, output_texture_img);

	threshold_on_rgb(output_texture_img, masks);
}

Segmentation::Segmentation()
{

}


Segmentation::~Segmentation()
{
}

void Contour::trace_contours(const cv::Mat& img, cv::Mat& mask) const {
	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, CV_BGR2GRAY);

	cv::Mat padded_img;
	cv::copyMakeBorder(gray_img, padded_img, 1, 1, 1, 1, cv::BORDER_REPLICATE);


	int kernel_size = 3;

	mask = cv::Mat::zeros(gray_img.rows, gray_img.cols, CV_8U);

	for (int row = 0; row < img.rows; ++row) {
		for (int col = 0; col < img.cols; ++col) {
			int padded_row = row + kernel_size - 2;
			int padded_col = col + kernel_size - 2;
			cv::Rect window(padded_col - (kernel_size - 2), padded_row - (kernel_size - 2), kernel_size, kernel_size);
			cv::Mat roi = padded_img(window);

			// check if pixel is not zero
			unsigned char grayscale_val = roi.at<unsigned char>(1, 1);
			if (grayscale_val > 0) {
				// check all values d8 connectivity
				int d8_sum = 0;
				for (int kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
					for (int kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {
						if (!(kernel_row == 1 && kernel_col == 1)) {
							unsigned char d8_gray_scale = roi.at<unsigned char>(kernel_row, kernel_col);
							if (d8_gray_scale > 0) {
								d8_sum++;
							}
						}
					}
				}
				// if d8 values are greater than 1 and not all pixels are connected it's a contour
				if (d8_sum > 0 && d8_sum < 8) {
					mask.at<unsigned char>(row, col) = 255;
				}
			}
		}
	}
}

