#include "Registration.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/contrib/contrib.hpp>
#include <fstream>


cv::Mat Registration::load_depth_img(std::string filename) {
	const int rows = 424;
	const int cols = 512;
	cv::Mat depth_map(rows, cols, CV_32F);
	std::ifstream file(filename);
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			float val;
			file >> val;
			depth_map.at<float>(row, col) = val;
		}
	}

	double min;
	double max;
	cv::minMaxIdx(depth_map, &min, &max);
	cv::Mat adjMap;
	// expand your range to 0..255. Similar to histEq();
	depth_map.convertTo(adjMap, CV_8UC1, 255 / (max - min), -min);

	// this is great. It converts your grayscale image into a tone-mapped one, 
	// much more pleasing for the eye
	// function is found in contrib module, so include contrib.hpp 
	// and link accordingly
	cv::Mat falseColorsMap;
	cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);

	cv::imshow("Out", falseColorsMap);

	//cv::Mat show_img;
	//cv::convertScaleAbs(depth_img, show_img);
	//cv::imshow("depth_img", show_img);

	return depth_map;
}

cv::Mat Registration::create_point_cloud_mat(cv::Mat& depth_img, cv::Mat& K) {

	cv::Mat K_inv = K.inv();

	cv::Mat point_cloud(depth_img.rows, depth_img.cols, CV_64FC3);

	for (int row = 0; row < depth_img.rows; ++row) {
		for (int col = 0; col < depth_img.cols; ++col) {
			cv::Vec3d pxl_coord(col, row, 1.0);
			float depth_val = depth_img.at<float>(row, col);
			cv::Mat pt_3D = depth_val * K_inv * cv::Mat(pxl_coord);
			point_cloud.at<cv::Vec3d>(row, col) = cv::Vec3d(pt_3D);
		}
	}

	double min;
	double max;
	cv::minMaxIdx(point_cloud, &min, &max);
	cv::Mat adjMap;
	// expand your range to 0..255. Similar to histEq();
	point_cloud.convertTo(adjMap, CV_8UC3, 255 / (max - min), -min);

	cv::imshow("Out1", adjMap);
	return point_cloud;
}

std::vector<cv::Vec3d> Registration::create_point_cloud(cv::Mat& depth_img, cv::Mat& K) {

	cv::Mat K_inv = K.inv();

	std::vector<cv::Vec3d> pc;

	for (int row = 0; row < depth_img.rows; ++row) {
		for (int col = 0; col < depth_img.cols; ++col) {
			cv::Vec3d pxl_coord(col, row, 1.0);
			float depth_val = depth_img.at<float>(row, col);

			if (depth_val > 1e-8) {
				cv::Mat pt_3D = depth_val * K_inv * cv::Mat(pxl_coord);
				pc.push_back(cv::Vec3d(pt_3D));
			}
		}
	}

	return pc;
}

cv::Mat Registration::construct_M(const std::vector<cv::Vec3d>& pts, cv::Vec3d& c) {
	cv::Mat M(3, pts.size(), CV_64F);
	cv::Vec3d centroid;
	for (auto& pt : pts) {
		centroid += pt;
	}

	centroid /= (double)pts.size();
	c = centroid;

	for (int i = 0; i < pts.size(); ++i) {
		auto diff = pts[i] - centroid;
		for (int k = 0; k < 3; ++k) {
			M.at<double>(k, i) = diff[k];
		}
	}
	return M;
}

void Registration::icp(std::vector<cv::Vec3d>& pc_P, std::vector<cv::Vec3d>& pc_Q,
		 std::vector<cv::Vec3d>& transformed_pc_Q ) {

	struct PCDiff {
		int p_i;
		int q_i;

		double dist;
	};

	double threshold = 0.1;

	std::vector<cv::Vec3d> P_prime;
	std::vector<cv::Vec3d> Q_prime;

	for (int p_i = 0; p_i < pc_P.size(); ++p_i) {

		cv::Vec3d pt_P = pc_P[p_i];

		if (cv::norm(pt_P) > 1e-8) {
			std::vector<PCDiff> pc_diffs;
			for (int q_i = 0; q_i < pc_Q.size(); ++q_i) {
				cv::Vec3d pt_Q = pc_Q[q_i];
				if (cv::norm(pt_Q) > 1e-8) {
					double distance = cv::norm(pt_P - pt_Q);
					if (distance < threshold) {
						PCDiff point_cloud_diff;
						point_cloud_diff.p_i = p_i;
						point_cloud_diff.q_i = q_i;
						point_cloud_diff.dist = distance;
						pc_diffs.push_back(point_cloud_diff);
					}
				}
			}

			if (pc_diffs.size() > 0) {
				// sort by euclidean distance
				std::sort(pc_diffs.begin(), pc_diffs.end(), [](const PCDiff& left, const PCDiff& right)
				{
					return left.dist < right.dist;
				});

				// pick 1st as the correspondence pair, which is guaranteed to exist
				auto closest_point_pair = pc_diffs[0];
				P_prime.push_back(pt_P);
				Q_prime.push_back(pc_Q[closest_point_pair.q_i]);
			}
		}
	}

	// construct M_q, M_p
	cv::Vec3d P_c, Q_c;
	cv::Mat M_p = construct_M(P_prime, P_c);
	cv::Mat M_q = construct_M(Q_prime, Q_c);

	cv::Mat C = M_q * M_p.t();

	cv::Mat u, sigma, vt;
	cv::SVD::compute(C, sigma, u, vt);

	cv::Mat R = vt.t() * u.t();
	cv::Mat rotated_Q_c = R * cv::Mat(Q_c);
	cv::Vec3d t = P_c - cv::Vec3d(rotated_Q_c);

	cv::Mat T(4, 4, CV_64F);

	cv::Mat tmp = T(cv::Rect(0, 0, 3, 3));
	R.copyTo(tmp);

	for (int k = 0; k < 3; ++k) {
		T.at<double>(k, 3) = t[k];
	}

	transformed_pc_Q.resize(pc_Q.size());
	for (int i = 0; i < pc_Q.size(); ++i) {
		cv::Vec3d pt_Q = pc_Q[i];
		cv::Vec4d homogenous_pt(pt_Q[0], pt_Q[1], pt_Q[2], 1.0);

		cv::Mat homogenous_t_Q_mat = T * cv::Mat(homogenous_pt);
		cv::Vec4d homogenous_t_Q = cv::Vec4d(homogenous_t_Q_mat);

		cv::Vec3d transformed_pt_Q(homogenous_t_Q[0], homogenous_t_Q[1], homogenous_t_Q[2]);
		transformed_pt_Q /= homogenous_t_Q[3];
		transformed_pc_Q[i] = transformed_pt_Q;
	}
	
}

void Registration::write_point_cloud(std::vector<cv::Vec3d>& pc_P, std::string prefix) {

	std::ofstream filename(prefix + ".txt");
	cv::Mat point_cloud_mat(3, pc_P.size(), CV_64F);
	for (int col = 0; col < pc_P.size(); ++col) {
		for (int i = 0; i < 3; ++i) {
			point_cloud_mat.at<double>(i, col) = pc_P[col][i];
		}
	}

	filename << cv::format(point_cloud_mat, "matlab");

}

void Registration::icp(cv::Mat& pc_P, cv::Mat& pc_Q) {

	struct PCDiff {
		int col_p;
		int row_p;

		int col_q;
		int row_q;

		double dist;
	};

	double threshold = 0.1;

	std::vector<cv::Vec3d> P_prime;
	std::vector<cv::Vec3d> Q_prime;



	for (int row_p = 0; row_p < pc_P.rows; ++row_p) {
		for (int col_p = 0; col_p < pc_P.cols; ++col_p) {
			cv::Vec3d pt_P = pc_P.at<cv::Vec3d>(row_p, col_p);
			if (cv::norm(pt_P) > 1e-8) {
				std::vector<PCDiff> pc_diffs;
				for (int row_q = 0; row_q < pc_Q.rows; ++row_q) {
					for (int col_q = 0; col_q < pc_Q.cols; ++col_q) {
						cv::Vec3d pt_Q = pc_Q.at<cv::Vec3d>(row_q, col_q);
						if (cv::norm(pt_Q) > 1e-8) {
							double distance = cv::norm(pt_P - pt_Q);
							if (distance < threshold) {
								PCDiff point_cloud_diff;
								point_cloud_diff.col_p = col_p;
								point_cloud_diff.row_p = row_p;

								point_cloud_diff.col_q = col_q;
								point_cloud_diff.row_q = row_q;

								point_cloud_diff.dist = distance;
								pc_diffs.push_back(point_cloud_diff);
							}
						}
					}
				}

				if (pc_diffs.size() > 0) {
					// sort by euclidean distance
					std::sort(pc_diffs.begin(), pc_diffs.end(), [](const PCDiff& left, const PCDiff& right)
					{
						return left.dist < right.dist;
					});
					
					// pick 1st as the correspondence pair, which is guaranteed to exist
					auto closest_point_pair = pc_diffs[0];
					P_prime.push_back(pt_P);
					Q_prime.push_back(pc_Q.at<cv::Vec3d>(closest_point_pair.row_q, closest_point_pair.row_q));
				}
			}
		}
	}


	// construct M_q, M_p
	cv::Vec3d P_c, Q_c;
	cv::Mat M_p = construct_M(P_prime, P_c);
	cv::Mat M_q = construct_M(Q_prime, Q_c);

	cv::Mat C = M_q * M_p.t();

	cv::Mat u, sigma, vt;
	cv::SVD::compute(C, sigma, u, vt);

	cv::Mat R = vt.t() * u.t();
	cv::Mat rotated_Q_c = R * cv::Mat(Q_c);
	cv::Vec3d t = P_c - cv::Vec3d(rotated_Q_c);

	cv::Mat T(4, 4, CV_64F);

	cv::Mat tmp = T(cv::Rect(0, 0, 3, 3));
	R.copyTo(tmp);

	for (int k = 0; k < 3; ++k) {
		T.at<double>(k, 3) = t[k];
	}
	

}

Registration::Registration()
{
}


Registration::~Registration()
{
}

int main(int argc, char** argv) {

	std::vector<std::string> filenames = { "depthImage1ForHW.txt", "depthImage2ForHW.txt" };
	std::vector<std::vector<cv::Vec3d>> point_clouds;

	Registration registration;

	for (auto& filename : filenames) {
		cv::Mat depth_img = registration.load_depth_img(filename);

		// K
		cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
		K.at<double>(0, 0) = 365;
		K.at<double>(0, 2) = 256;
		K.at<double>(1, 1) = 365;
		K.at<double>(1, 2) = 212;
		K.at<double>(2, 2) = 1;

		std::vector<cv::Vec3d> point_cloud = registration.create_point_cloud(depth_img, K);
		point_clouds.push_back(point_cloud);
	}

	std::vector<cv::Vec3d> P = point_clouds[0];
	std::vector<cv::Vec3d> Q = point_clouds[1];

	registration.write_point_cloud(P, "P");
	registration.write_point_cloud(Q, "Q");

	const int M = 20;
	for (int i = 0; i < M; ++i) {
		std::vector<cv::Vec3d> transformed_Q;
		registration.icp(P, Q, transformed_Q);
		Q = transformed_Q;
	}

	registration.write_point_cloud(Q, "transformed_Q");
	



	//std::vector<cv::Mat> channels;
	//cv::split(point_cloud, channels);

	//std::string axes[] = { "x", "y", "z" };
	//for (int i = 0; i < 3; ++i) {
	//	std::ofstream xstream(axes[i] + ".txt");
	//	xstream << cv::format(channels[i], "matlab");
	//}

	cv::waitKey();
	
}