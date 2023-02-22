#ifndef IMAGE_RESTORATION_H
#define IMAGE_RESTORATION_H

#include <opencv2/opencv.hpp>
using namespace cv;

class ImageRestoration
{
public:
	/// @brief 绘制直方图
	/// @param src 灰度图像,类型为CV_8UC1
	/// @return 灰度直方图
	static Mat get_hist_img(Mat src);
	/// @brief 添加高斯噪声
	/// p(z) = 1/(sqrt(2*pi)*stddev)*exp(-(z-mean)/(2*stddev^2))
	/// @param src 源图像
	/// @param mean 高斯函数均值
	/// @param stddev 高斯函数标准差
	/// @return 添加噪声后归一化图像
	static Mat add_gauss_noise(const Mat& src, double mean, double stddev);
private:

};

#endif // !IMAGE_RESTORATION_H