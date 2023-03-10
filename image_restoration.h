#ifndef IMAGE_RESTORATION_H
#define IMAGE_RESTORATION_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
using namespace cv;
//TODO:维纳滤波、约束最小二乘方滤波、几何均值滤波
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
	/// @return 添加噪声后图像[0,255]
	static Mat add_gauss_noise(const Mat& src, double mean, double stddev);
	/// @brief 添加瑞利噪声
	/// PDF:f(x) = x/a^2 * exp(-x^2/(2*a^2)) 
	/// @param src 源图像
	/// @param a 
	/// @return 添加噪声后图像[0,255]
	static Mat add_rayleigh_noise(const Mat& src, double a);
	/// @brief 添加爱尔兰噪声
	/// PDF:f(x) = (a^b*x^(b-1))/(b-1)!*exp(-a*x)
	/// @param src 源图像
	/// @param a 
	/// @param b 
	/// @return 
	static Mat add_ireland_noise(const Mat& src, double a, double b);
	/// @brief 添加指数噪声
	/// PDF: f(x) = a*exp(-a*x)
	/// @param src 源图像
	/// @param a 
	/// @return 添加噪声后图像[0,255]
	static Mat add_exponential_noise(const Mat& src, double a);
	/// @brief 添加均匀噪声
	/// PDF: f(x) = 1/(b-a)
	/// @param src 源图像
	/// @param a 
	/// @param b 
	/// @return 添加噪声后图像[0,255]
	static Mat add_uniform_noise(const Mat& src, int a, int b);
	/// @brief 添加盐粒噪声
	/// @param src 源图像
	/// @param p 噪声出现概率
	/// @return 添加噪声后图像
	static Mat add_salt_noise(const Mat& src, double p);
	/// @brief 添加胡椒粒噪声
	/// @param src 源图像
	/// @param p 噪声出现概率
	/// @return 添加噪声后图像
	static Mat add_pepper_noise(const Mat& src, double p);
	/// @brief 添加椒盐噪声
	/// 白色盐粒；黑色胡椒粒
	/// @param src 源图像
	/// @return 添加噪声后图像
	static Mat add_salt_pepper_noise(const Mat& src, double salt_p, double pepper_p);
	/// @brief 添加运动模糊
	/// @param src 源图像
	/// @param vx x轴移动速度
	/// @param vy y轴移动速度
	/// @param exposure_time 曝光时间
	/// @return 添加运动模糊后图像
	static Mat add_move_blur(const Mat& src, double vx, double vy, double exposure_time);
	/// @brief 算术均值滤波器
	/// 每个滤波像素是滤波核图像区域中所有像素的均值
	/// @param src 源图像
	/// @param s 滤波核尺寸，必须为奇数尺寸
	/// @return 滤波后图像
	static Mat mean_blur(const Mat& src, Size s);
	/// @brief 几何均值滤波器
	/// 每个滤波像素是滤波核图像区域中所有像素之积的1/mn次幂
	/// @param src 源图像
	/// @param s 滤波核尺寸,必须是奇数尺寸
	/// @return 滤波后图像
	static Mat geometric_mean_blur(const Mat& src, Size s);
	/// @brief 谐波平均滤波器
	/// @param src 源图像
	/// @param s 滤波核尺寸，必须是
	/// @return 滤波后图像
	static Mat harmonic_mean_blur(const Mat& src, Size s);
	/// @brief 反谐波平均滤波器
	/// @param src 源图像
	/// @param s 滤波核尺寸，必须是
	/// @return 滤波后图像
	static Mat antiharmonic_mean_blur(const Mat& src, int Q, Size s);
	/// @brief 中值滤波器
	/// @param src 源图像
	/// @param s 滤波器核
	/// @return 滤波后图像
	static Mat median_mean_blur(const Mat& src, Size s);
	/// @brief 最大值滤波器
	/// @param src 源图像
	/// @param s 滤波器核尺寸
	/// @return 滤波后图像
	static Mat max_mean_blur(const Mat& src, Size s);
	/// @brief 最小值滤波器
	/// @param src 源图像
	/// @param s 滤波器核尺寸
	/// @return 滤波后图像
	static Mat min_mean_blur(const Mat& src, Size s);
	/// @brief 中点滤波器
	/// @param src 源图像
	/// @param s 滤波器核尺寸
	/// @return 滤波后图像
	static Mat mid_point_blur(const Mat& src, Size s);
	/// @brief 修正阿尔法均值滤波器
	/// @param src 源图像
	/// @param d 灰度修正范围
	/// @param s 滤波器核尺寸
	/// @return 滤波后图像
	static Mat modified_alpha_mean_blur(const Mat& src, int d, Size s);
	/// @brief 局部降噪滤波器
	/// @param src 源图像
	/// @param sigma_n 噪声方差
	/// @param s 滤波核尺寸
	/// @return 滤波后图像 
	static Mat adaptive_local_noise_reduce_blur(const Mat& src, double sigma_n, Size s);
	/// @brief 自适应中值滤波器
	/// @param src 源图像
	/// @param s_max 滤波器最大尺寸
	/// @return 滤波后图像
	static Mat adaptive_median_blur(const Mat& src, int s_max);
private:
	/// @brief 生成PDF瑞利分布的数据
	/// PDF:f(x) = x/a^2 * exp(-x^2/(2*a^2)) 
	/// 均值:a*sqrt(pi/2);方差:(2-pi/2)*a^2
	/// first:使用逆变换产生参数β=2的指数分布的随机变量y，其PDF为f(y) = 1/2*exp(-y/2)
	/// second:通过变换x=a*sqrt(y)产生瑞利分布的随机变量x
	/// (1) 计算均匀分布随机数u
	/// (2) 计算y = -2ln(u)
	/// (3) 计算x = a*sqrt(y)
	/// @param nosie 噪声图像
	/// @param a 参数
	/// @return 
	static void get_rayleigh_noise(Mat& noise, double a);
	/// @brief 获取均匀噪声图像
	/// f(x) = 1/(b-a)
	/// @param noise 噪声图像
	/// @param a 
	/// @param b 
	static void get_uniform_noise(Mat& noise, int a, int b);
	/// @brief 扩增图像边界
	/// @param src 源图像
	/// @param top 图像上方扩充行数
	/// @param bottom 图像下方扩充行数
	/// @param left 图像左边扩充列数
	/// @param right 图像右边扩充列数
	/// @param border_type 扩充边界类型；0：复制边界；1：以固定值
	/// @param copy_val 以固定至扩充边界时使用的值
	/// @return 扩充边界后图像
	static Mat copy_boarder(const Mat& src, int top, int bottom, int left, int right, int border_type, const int copy_val = 0);

	/// @brief 计算均值方差
	/// @tparam T 数据类型
	/// @param vec 数据
	/// @param mean 均值
	/// @param std 方差
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