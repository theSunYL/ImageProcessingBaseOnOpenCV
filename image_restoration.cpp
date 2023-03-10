#include "image_restoration.h"
#include <iostream>
#include <random>
#include <algorithm>
#define _USE_MATH_DEFINES // 使用math.h中的M_PI宏定义需要
#include <math.h>
using namespace std;

Mat ImageRestoration::get_hist_img(Mat src)
{
	Mat hist;
	float range[] = { 0, 256 };
	const int channels[] = { 0 };
	const int bins[] = { 256 };
	const float* ranges[] = { range };
	cv::calcHist(&src, 1, channels, Mat(), hist, 1, bins, ranges, true, false);
	double max_val = 0;
	minMaxLoc(hist, 0, &max_val);
	int scale = 3;
	int hist_w = 256;
	int hist_h = 512;
	Mat histImg = Mat::zeros(Size(hist_w*scale, hist_h), CV_8UC1);
	for(int i=0; i<hist.rows; ++i){
		int hight = hist.at<float>(i)/ max_val * hist_h;
		//cout <<  i << " : " << hist.at<float>(i) << endl;
		rectangle(histImg, Point(i * scale, hist_h - hight), Point((i + 1) * scale, hist_h), Scalar(255, 255, 255), -1);
	}
	return histImg;
}

Mat ImageRestoration::add_gauss_noise(const Mat& src, double mean, double stddev)
{
	Mat noise(src.size(), CV_32FC1);
	cv::randn(noise, mean, stddev);
	Mat ret = src.clone();
	ret.convertTo(ret, CV_32FC1);
	cv::add(ret, noise, ret);
	cv::normalize(ret, ret, 0, 255, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

Mat ImageRestoration::add_rayleigh_noise(const Mat& src, double a)
{
	Mat noise(src.size(), CV_32FC1);
	get_rayleigh_noise(noise, a);
	Mat ret = src.clone();
	ret.convertTo(ret, CV_32FC1);
	cv::add(ret, noise, ret);
	cv::normalize(ret, ret, 0, 255, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

Mat ImageRestoration::add_uniform_noise(const Mat& src, int a, int b)
{
	Mat noise(src.size(), CV_32FC1);
	get_uniform_noise(noise, a, b);
	Mat ret = src.clone();
	ret.convertTo(ret, CV_32FC1);
	cv::add(ret, noise, ret);
	cv::normalize(ret, ret, 0, 255, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

Mat ImageRestoration::add_salt_noise(const Mat& src, double p)
{
	std::random_device rd;//硬件生成随机数种子
	double u, x;
	std::mt19937 dre(rd());
	int h = src.rows;
	int w = src.cols;
	uniform_real_distribution<double> dh(0, h);
	uniform_real_distribution<double> dw(0, w);
	int noise_count = h * w * p;
	Mat ret = src.clone();
	for (int i = 0; i < noise_count; ++i) {
		int x = dh(dre);
		int y = dw(dre);
		ret.at<uchar>(x, y) = 255;
	}
	return ret;
}

Mat ImageRestoration::add_pepper_noise(const Mat& src, double p)
{
	random_device rd;//硬件生成随机数种子
	double u, x;
	mt19937 dre(rd());
	int h = src.rows;
	int w = src.cols;
	uniform_real_distribution<double> dh(0, h);
	uniform_real_distribution<double> dw(0, w);
	int noise_count = h * w * p;
	Mat ret = src.clone();
	for (int i = 0; i < noise_count; ++i) {
		int x = dh(dre);
		int y = dw(dre);
		ret.at<uchar>(x, y) = 0;
	}
	return ret;
}

Mat ImageRestoration::add_salt_pepper_noise(const Mat& src, double salt_p, double pepper_p)
{
	Mat ret = add_salt_noise(src, salt_p);
	ret = add_pepper_noise(ret, pepper_p);
	return ret;
}

Mat ImageRestoration::add_ireland_noise(const Mat& src, double a, double b)
{
	//TODO
	return src;
}

Mat ImageRestoration::add_exponential_noise(const Mat& src, double a)
{
	//TODO
	return src;
}

Mat ImageRestoration::add_move_blur(const Mat& src, double vx, double vy, double exposure_time)
{
	//TODO：效果不好
	//计算傅里叶变换，不能补边
	vector<Mat> plane{Mat_<float>(src), Mat::zeros(src.size(), CV_32FC1)};
	Mat complex_img;
	merge(plane.data(), 2, complex_img);
	dft(complex_img, complex_img);
	split(complex_img, plane);
	//产生运动模糊退化模板
	vector<Mat> move_plane{ Mat::zeros(src.size(), CV_32FC1), Mat::zeros(src.size(), CV_32FC1) };
	for(int i=0; i<src.rows; ++i){
		for(int j=0; j<src.cols; ++j){
			double sigma = (i * vy + j * vx) * M_PI;
			double rel = (exposure_time * sin(sigma)) / sigma * cos(sigma);
			double img = -(exposure_time * sin(sigma)) / sigma * sin(sigma);
			move_plane[0].at<float>(i, j) = rel;
			move_plane[1].at<float>(i, j) = img;
		}
	}
	vector<Mat> move_img_fft{ Mat::zeros(src.size(), CV_32FC1), Mat::zeros(src.size(), CV_32FC1) };
	//计算并处理NaN像素
	for(int i=0; i<move_img_fft[0].rows; ++i){
		for(int j=0; j<move_img_fft[1].cols; ++j){
			double a1 = plane[0].at<float>(i, j) * move_plane[0].at<float>(i, j) - plane[1].at<float>(i, j) * move_plane[1].at<float>(i, j);
			double a2 = plane[0].at<float>(i, j) * move_plane[1].at<float>(i, j) + plane[1].at<float>(i, j) * move_plane[0].at<float>(i, j);
			if(std::isnan(a1)){
				a1 = 0;
			}
			if(std::isnan(a2)){
				a2 = 0;
			}
			move_img_fft[0].at<float>(i, j) = a1;
			move_img_fft[1].at<float>(i, j) = a2;
		}
	}
	Mat move_ifft;
	merge(move_img_fft.data(), 2, move_ifft);
	idft(move_ifft, move_ifft);
	split(move_ifft, move_img_fft);
	magnitude(move_img_fft[0], move_img_fft[1], move_img_fft[0]);//得到幅度谱
	normalize(move_img_fft[0], move_img_fft[0], 0, 1, NORM_MINMAX);
	return move_img_fft[0];
}

void ImageRestoration::get_rayleigh_noise(Mat& noise, double a)
{
	long long seed = noise.rows * noise.cols / 2;
	double u, x;
	default_random_engine dre(seed);
	uniform_real_distribution<double> d(0, 1);
	for (int i = 0; i < noise.rows; ++i) {
		for (int j = 0; j < noise.cols; ++j) {
			noise.at<float>(i, j) = a * sqrt(-2 * log(d(dre)));
		}
	}
	return;
}

void ImageRestoration::get_uniform_noise(Mat& noise, int a, int b)
{
	random_device rd;
	mt19937 dre(rd());
	double u, x;
	uniform_real_distribution<double> d(a, b + 1);
	for (int i = 0; i < noise.rows; ++i) {
		for (int j = 0; j < noise.cols; ++j) {
			noise.at<float>(i, j) = d(dre);
		}
	}
	return;
}

Mat ImageRestoration::mean_blur(const Mat& src, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	pad.convertTo(pad, CV_32FC1);
	double size = double(s.width) * s.height;
	Mat ret = Mat::zeros(src.size(), CV_32FC1);
	for(int i=top; i<src.rows+top; ++i){
		for(int j=left; j<src.cols+left; ++j){
			double pixel_data = 0;
			for(int k=-top; k<=top; ++k){
				for(int m=-left; m<=left; ++m){
					pixel_data += pad.at<float>(i + k, j + m);
				}
			}
			pixel_data /= size;
			ret.at<float>(i-top, j-left) = pixel_data;
		}
	}
	normalize(ret, ret, 0, 255, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

Mat ImageRestoration::geometric_mean_blur(const Mat& src, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	double sz = 1.0 / (s.width * s.height);
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	pad.convertTo(pad, CV_32FC1);
	normalize(pad, pad, 1, 2, NORM_MINMAX);	//避免
	Mat ret(src.size(), CV_32FC1);
	for(int i=top; i<src.rows+top; ++i){
		for(int j=left; j<src.cols+left; ++j){
			double pixel_data = 1;
			//滤波核内部相乘操作
			for (int k = -top; k < top + 1; ++k) {
				for(int m=-left; m<left+1; ++m){
					pixel_data = pixel_data * pad.at<float>(i + k, j + m);
				}
			}
			pixel_data = pow(pixel_data, sz);
			ret.at<float>(i-top, j-left) = pixel_data;
		}
	}
	normalize(ret, ret, 0, 255, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

Mat ImageRestoration::harmonic_mean_blur(const Mat& src, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	pad.convertTo(pad, CV_32FC1);
	//add(pad, Scalar::all(1), pad);//避免像素值0溢出，加上偏置
	double size = double(s.width) * s.height;
	Mat ret(src.size(), CV_32FC1);
	for (int i = top; i < src.rows + top; ++i) {
		for (int j = left; j < src.cols + left; ++j) {
			double pixel_data = 0;
			for (int k = -top; k <= top; ++k) {
				for (int m = -left; m <= left; ++m) {
					pixel_data += (1.0 / pad.at<float>(i + k, j + m));
				}
			}
			pixel_data = size/pixel_data;
			ret.at<float>(i-top, j-left) = pixel_data;
		}
	}
	normalize(ret, ret, 0, 255, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

Mat ImageRestoration::antiharmonic_mean_blur(const Mat& src, int Q, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	pad.convertTo(pad, CV_32FC1);
	//add(pad, Scalar::all(1), pad);//避免像素值0溢出，加上偏置
	double size = double(s.width) * s.height;
	Mat ret(src.size(), CV_32FC1);
	for (int i = top; i < src.rows + top; ++i) {
		for (int j = left; j < src.cols + left; ++j) {
			double pixel_data_1 = 0;
			double pixel_data_2 = 0;
			for (int k = -top; k <= top; ++k) {
				for (int m = -left; m <= left; ++m) {
					pixel_data_1 += pow(pad.at<float>(i + k, j + m), Q + 1);
					pixel_data_2 += pow(pad.at<float>(i + k, j + m), Q);
				}
			}
			ret.at<float>(i-top, j-left) = pixel_data_1 / pixel_data_2;
		}
	}
	normalize(ret, ret, 0, 255, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

Mat ImageRestoration::median_mean_blur(const Mat& src, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	Mat temp = pad.clone();
	vector<int> vec(s.height * s.width);
	int index = 0;
	Mat ret = Mat::zeros(src.size(), CV_8UC1);
	for (int i = top; i < src.rows + top; ++i) {
		for (int j = left; j < src.cols + left; ++j) {
			index = 0;
			for (int k = -top; k <= top; ++k) {
				for (int m = -left; m <= left; ++m) {
					vec[index] = pad.at<uchar>(i + k, j + m);
					++index;
				}
			}
			sort(vec.begin(), vec.end());
			ret.at<uchar>(i - top, j - left) = vec[vec.size() / 2];
		}
	}
	return ret;
}

Mat ImageRestoration::max_mean_blur(const Mat& src, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	Mat temp = pad.clone();
	vector<int> vec(s.height * s.width);
	int index = 0;
	Mat ret = Mat::zeros(src.size(), CV_8UC1);
	for (int i = top; i < src.rows + top; ++i) {
		for (int j = left; j < src.cols + left; ++j) {
			index = 0;
			for (int k = -top; k <= top; ++k) {
				for (int m = -left; m <= left; ++m) {
					vec[index] = pad.at<uchar>(i + k, j + m);
					++index;
				}
			}
			sort(vec.begin(), vec.end());
			ret.at<uchar>(i-top, j-left) = vec.back();
		}
	}
	return ret;
}

Mat ImageRestoration::min_mean_blur(const Mat& src, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	vector<int> vec(s.height * s.width);
	int index = 0;
	Mat ret = Mat::zeros(src.size(), CV_8UC1);
	for (int i = top; i < src.rows + top; ++i) {
		for (int j = left; j < src.cols + left; ++j) {
			index = 0;
			for (int k = -top; k <= top; ++k) {
				for (int m = -left; m <= left; ++m) {
					vec[index] = pad.at<uchar>(i + k, j + m);
					++index;
				}
			}
			sort(vec.begin(), vec.end());
			ret.at<uchar>(i - top, j - left) = vec[0];
		}
	}
	return ret;
}

Mat ImageRestoration::mid_point_blur(const Mat& src, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	vector<int> vec(s.height * s.width);
	int index = 0;
	Mat ret = Mat::zeros(src.size(), CV_8UC1);
	for(int i=top; i<src.rows+top; ++i){
		for(int j=left; j<src.cols+left; ++j){
			index = 0;
			for(int k=-top; k<= top; ++k){
				for(int m=-left; m<=left; ++m){
					vec[index] = pad.at<uchar>(i + top, j + left);
					++index;
				}
			}
			sort(vec.begin(), vec.end());
			ret.at<uchar>(i - top, j - left) = (int)(vec[0] + vec.back()) / 2;
		}
	}
	return ret;
}

Mat ImageRestoration::modified_alpha_mean_blur(const Mat& src, int d, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	Mat ret = Mat::zeros(src.size(), CV_8UC1);
	vector<int> vec(s.height * s.width);
	int index = 0;
	for(int i=top; i<src.rows+top; ++i){
		for(int j=left; j<src.cols+left; ++j){
			for(int k=-top; k<=top; ++k){
				for(int m=-left; m<=left; ++m){
					vec[index] = pad.at<uchar>(i+k, j+m);
					++index;
				}
			}
			index = 0;
			sort(vec.begin(), vec.end());
			double value = 0;
			int count = 0;
			for(int n=d/2; n<vec.size()-d/2; ++n){
				value += vec[n];
				++count;
			}
			value = floor(value / count + 0.5);
			ret.at<uchar>(i - top, j - left) = value;
		}
	}
	return ret;
}

Mat ImageRestoration::adaptive_local_noise_reduce_blur(const Mat& src, double sigma_n, Size s)
{
	int top = s.height / 2;
	int left = s.width / 2;
	Mat pad = copy_boarder(src, top, top, left, left, 0);
	Mat ret = Mat::zeros(src.size(), CV_8UC1);
	vector<int> vec(s.width * s.height);
	int index = 0;
	for(int i=top; i<src.rows+top; ++i){
		for(int j=left; j<src.cols+left; ++j){
			index = 0;
			for(int k=-top; k<=top; ++k){
				for(int m=-left; m<=left; ++m){
					vec[index] = pad.at<uchar>(i + k, j + m);
					++index;
				}
			}
			double mean = 0, sig = 0;
			cal_mean_var(vec, mean, sig);
			double val = pad.at<uchar>(i, j) - sigma_n / sig * (pad.at<uchar>(i, j) - mean);
			if(val <0){
				val = 0;
			}
			ret.at<uchar>(i - top, j - left) = floor(val + 0.5);
		}
	}
	return ret;
}

Mat ImageRestoration::adaptive_median_blur(const Mat & src, int s_max)
{
	int w = s_max / 2;
	int h = s_max / 2;
	Mat pad = copy_boarder(src, h, h, w, w, 0);
	Mat ret = Mat::zeros(src.size(), CV_8UC1);
	for(int i=h; i<src.rows+h; ++i){
		for(int j=w; j<src.cols+w; ++j){
			int ksize = 3;
			while(ksize <= s_max){
				vector<int> vec;
				int k = -ksize / 2, m = -ksize / 2;
				for(int k=-ksize/2; k<=ksize/2; ++k){
					for(int m=-ksize/2; m<=ksize/2; ++m){
						vec.push_back(pad.at<uchar>(i + k, j + m));
					}
				}
				sort(vec.begin(), vec.end());
				int mid = vec[vec.size() / 2];
				if (mid > vec[0] && mid < vec.back()) {
					// level B
					int cur_pixel = pad.at<uchar>(i, j);
					if (cur_pixel > vec[0] && cur_pixel < vec.back()) {
						ret.at<uchar>(i - h, j - w) = cur_pixel;
					} else {
						ret.at<uchar>(i - h, j - w) = mid;
					}
					break;
				}
				else {
					//level A
					if(ksize < s_max){
						++ksize;
					} else {
						ret.at<uchar>(i - h, j - w) = mid;
						break;
					}
				}
			}
		}
	}
	return ret;
}

Mat ImageRestoration::copy_boarder(const Mat& src, int top, int bottom, int left, int right, int border_type, const int copy_val)
{	
	Mat ret(Size(src.cols + left + right, src.rows + top + bottom), src.type(), Scalar::all(copy_val));
	src.copyTo(ret(Rect(top, right, src.cols, src.rows)));
	if(border_type == 0){
		//以边界进行复制填充
		for(int i=0; i<top; i++){
			//上
			src.row(0).copyTo(ret(Range(i, i + 1), Range(left, ret.cols - right)));
		}
		for(int i = src.rows + top; i< ret.rows; ++i){
			//下
			src.row(src.rows - 1).copyTo(ret(Range(i, i + 1), Range(left, ret.cols - right)));
		}
		for(int i=0; i<left; ++i){
			//左
			ret.col(left).copyTo(ret(Range(0, ret.rows), Range(i, i + 1)));
		}
		for(int i=src.cols +  left; i< ret.cols; ++i){
			//右
			ret.col(src.cols).copyTo(ret(Range(0, ret.rows), Range(i, i + 1)));
		}
	} else if(border_type == 1){
		//以固定copy_value进行填充
	}
	return ret;
}
