#include "image_restoration.h"
#include <iostream>
using namespace std;
Mat ImageRestoration::get_hist_img(Mat src)
{
	Mat hist;
	float range[] = { 0, 255 };
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
		//cout << hist.at<float>(i) << endl;
		rectangle(histImg, Point(i * scale, hist_h - hight), Point((i + 1) * scale, hist_h), Scalar(255, 255, 255), -1);
	}
	return histImg;
}

Mat ImageRestoration::add_gauss_noise(const Mat& src, double mean, double stddev)
{
	//Ìí¼Ó¸ßË¹ÔëÉù
	Mat noise(src.size(), CV_32FC1);
	cv::randn(noise, mean, stddev);
	Mat ret = src.clone();
	ret.convertTo(ret, CV_32FC1);
	cv::add(ret, noise, ret);
	cv::normalize(ret, ret, 0, 255, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}