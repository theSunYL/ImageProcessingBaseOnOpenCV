#ifndef IMAGE_RESTORATION_H
#define IMAGE_RESTORATION_H

#include <opencv2/opencv.hpp>
using namespace cv;

class ImageRestoration
{
public:
	/// @brief ����ֱ��ͼ
	/// @param src �Ҷ�ͼ��,����ΪCV_8UC1
	/// @return �Ҷ�ֱ��ͼ
	static Mat get_hist_img(Mat src);
	/// @brief ��Ӹ�˹����
	/// p(z) = 1/(sqrt(2*pi)*stddev)*exp(-(z-mean)/(2*stddev^2))
	/// @param src Դͼ��
	/// @param mean ��˹������ֵ
	/// @param stddev ��˹������׼��
	/// @return ����������һ��ͼ��
	static Mat add_gauss_noise(const Mat& src, double mean, double stddev);
private:

};

#endif // !IMAGE_RESTORATION_H