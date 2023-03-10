#ifndef IMAGE_RESTORATION_H
#define IMAGE_RESTORATION_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
using namespace cv;
//TODO:ά���˲���Լ����С���˷��˲������ξ�ֵ�˲�
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
	/// @return ���������ͼ��[0,255]
	static Mat add_gauss_noise(const Mat& src, double mean, double stddev);
	/// @brief �����������
	/// PDF:f(x) = x/a^2 * exp(-x^2/(2*a^2)) 
	/// @param src Դͼ��
	/// @param a 
	/// @return ���������ͼ��[0,255]
	static Mat add_rayleigh_noise(const Mat& src, double a);
	/// @brief ��Ӱ���������
	/// PDF:f(x) = (a^b*x^(b-1))/(b-1)!*exp(-a*x)
	/// @param src Դͼ��
	/// @param a 
	/// @param b 
	/// @return 
	static Mat add_ireland_noise(const Mat& src, double a, double b);
	/// @brief ���ָ������
	/// PDF: f(x) = a*exp(-a*x)
	/// @param src Դͼ��
	/// @param a 
	/// @return ���������ͼ��[0,255]
	static Mat add_exponential_noise(const Mat& src, double a);
	/// @brief ��Ӿ�������
	/// PDF: f(x) = 1/(b-a)
	/// @param src Դͼ��
	/// @param a 
	/// @param b 
	/// @return ���������ͼ��[0,255]
	static Mat add_uniform_noise(const Mat& src, int a, int b);
	/// @brief �����������
	/// @param src Դͼ��
	/// @param p �������ָ���
	/// @return ���������ͼ��
	static Mat add_salt_noise(const Mat& src, double p);
	/// @brief ��Ӻ���������
	/// @param src Դͼ��
	/// @param p �������ָ���
	/// @return ���������ͼ��
	static Mat add_pepper_noise(const Mat& src, double p);
	/// @brief ��ӽ�������
	/// ��ɫ��������ɫ������
	/// @param src Դͼ��
	/// @return ���������ͼ��
	static Mat add_salt_pepper_noise(const Mat& src, double salt_p, double pepper_p);
	/// @brief ����˶�ģ��
	/// @param src Դͼ��
	/// @param vx x���ƶ��ٶ�
	/// @param vy y���ƶ��ٶ�
	/// @param exposure_time �ع�ʱ��
	/// @return ����˶�ģ����ͼ��
	static Mat add_move_blur(const Mat& src, double vx, double vy, double exposure_time);
	/// @brief ������ֵ�˲���
	/// ÿ���˲��������˲���ͼ���������������صľ�ֵ
	/// @param src Դͼ��
	/// @param s �˲��˳ߴ磬����Ϊ�����ߴ�
	/// @return �˲���ͼ��
	static Mat mean_blur(const Mat& src, Size s);
	/// @brief ���ξ�ֵ�˲���
	/// ÿ���˲��������˲���ͼ����������������֮����1/mn����
	/// @param src Դͼ��
	/// @param s �˲��˳ߴ�,�����������ߴ�
	/// @return �˲���ͼ��
	static Mat geometric_mean_blur(const Mat& src, Size s);
	/// @brief г��ƽ���˲���
	/// @param src Դͼ��
	/// @param s �˲��˳ߴ磬������
	/// @return �˲���ͼ��
	static Mat harmonic_mean_blur(const Mat& src, Size s);
	/// @brief ��г��ƽ���˲���
	/// @param src Դͼ��
	/// @param s �˲��˳ߴ磬������
	/// @return �˲���ͼ��
	static Mat antiharmonic_mean_blur(const Mat& src, int Q, Size s);
	/// @brief ��ֵ�˲���
	/// @param src Դͼ��
	/// @param s �˲�����
	/// @return �˲���ͼ��
	static Mat median_mean_blur(const Mat& src, Size s);
	/// @brief ���ֵ�˲���
	/// @param src Դͼ��
	/// @param s �˲����˳ߴ�
	/// @return �˲���ͼ��
	static Mat max_mean_blur(const Mat& src, Size s);
	/// @brief ��Сֵ�˲���
	/// @param src Դͼ��
	/// @param s �˲����˳ߴ�
	/// @return �˲���ͼ��
	static Mat min_mean_blur(const Mat& src, Size s);
	/// @brief �е��˲���
	/// @param src Դͼ��
	/// @param s �˲����˳ߴ�
	/// @return �˲���ͼ��
	static Mat mid_point_blur(const Mat& src, Size s);
	/// @brief ������������ֵ�˲���
	/// @param src Դͼ��
	/// @param d �Ҷ�������Χ
	/// @param s �˲����˳ߴ�
	/// @return �˲���ͼ��
	static Mat modified_alpha_mean_blur(const Mat& src, int d, Size s);
	/// @brief �ֲ������˲���
	/// @param src Դͼ��
	/// @param sigma_n ��������
	/// @param s �˲��˳ߴ�
	/// @return �˲���ͼ�� 
	static Mat adaptive_local_noise_reduce_blur(const Mat& src, double sigma_n, Size s);
	/// @brief ����Ӧ��ֵ�˲���
	/// @param src Դͼ��
	/// @param s_max �˲������ߴ�
	/// @return �˲���ͼ��
	static Mat adaptive_median_blur(const Mat& src, int s_max);
private:
	/// @brief ����PDF�����ֲ�������
	/// PDF:f(x) = x/a^2 * exp(-x^2/(2*a^2)) 
	/// ��ֵ:a*sqrt(pi/2);����:(2-pi/2)*a^2
	/// first:ʹ����任����������=2��ָ���ֲ����������y����PDFΪf(y) = 1/2*exp(-y/2)
	/// second:ͨ���任x=a*sqrt(y)���������ֲ����������x
	/// (1) ������ȷֲ������u
	/// (2) ����y = -2ln(u)
	/// (3) ����x = a*sqrt(y)
	/// @param nosie ����ͼ��
	/// @param a ����
	/// @return 
	static void get_rayleigh_noise(Mat& noise, double a);
	/// @brief ��ȡ��������ͼ��
	/// f(x) = 1/(b-a)
	/// @param noise ����ͼ��
	/// @param a 
	/// @param b 
	static void get_uniform_noise(Mat& noise, int a, int b);
	/// @brief ����ͼ��߽�
	/// @param src Դͼ��
	/// @param top ͼ���Ϸ���������
	/// @param bottom ͼ���·���������
	/// @param left ͼ�������������
	/// @param right ͼ���ұ���������
	/// @param border_type ����߽����ͣ�0�����Ʊ߽磻1���Թ̶�ֵ
	/// @param copy_val �Թ̶�������߽�ʱʹ�õ�ֵ
	/// @return ����߽��ͼ��
	static Mat copy_boarder(const Mat& src, int top, int bottom, int left, int right, int border_type, const int copy_val = 0);

	/// @brief �����ֵ����
	/// @tparam T ��������
	/// @param vec ����
	/// @param mean ��ֵ
	/// @param std ����
	template<class T>
	static void cal_mean_var(const std::vector<T>& vec, double& mean, double& var);
};
template<class T>
void ImageRestoration::cal_mean_var(const std::vector<T>&vec, double& mean, double& var)
{
	if(vec.size() == 0){
		mean = 0;
		var = 0;
		return;
	}
	double sum = 0.0;
	for(int i=0; i<vec.size(); ++i){
		sum += vec[i];
	}
	mean = sum / vec.size();
	sum = 0;
	for(int i=0; i<vec.size(); ++i){
		sum += pow(vec[i] - mean,2);
	}
	var = sum / vec.size();
}

#endif // !IMAGE_RESTORATION_H