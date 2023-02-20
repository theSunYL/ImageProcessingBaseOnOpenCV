#ifndef FREQUENCY_FILTER_H
#define FREQUENCY_FILTER_H

#include <opencv2/opencv.hpp>
using namespace cv;

/// @brief 频域图像处理
class FrequencyFilter 
{
public:
	/// @brief 理想低通滤波
	/// @param src 源图像
	/// @param d0 截止频率
	/// @return 滤波后图像
	static Mat ideal_lowpass(const Mat& src, float d0);
	/// @brief 理想高通滤波
	/// @param src 源图像
	/// @param d0 截止频率
	/// @return 滤波后图像
	static Mat ideal_highpass(const Mat& src, float d0);
	/// @brief 巴特沃斯低通滤波
	/// @param src 源图像
	/// @param d0 截止频率
	/// @param n 阶数，曲线陡峭程度，n越大曲线越陡峭，振铃效应明显
	/// @return 滤波后图像
	static Mat butterworth_lowpass(const Mat& src, float d0, int n);
	/// @brief 巴特沃斯高通滤波
	/// @param src 源图像
	/// @param d0 截止频率
	/// @param n 阶数，曲线陡峭程度，n越大曲线越陡峭，振铃效应明显
	/// @return 滤波后图像
	static Mat butterworth_highpass(const Mat& src, float d0, int n);
	/// @brief 高斯低通滤波
	/// @param src 源图像
	/// @param sigma 截止频率
	/// @return 滤波后图像
	static Mat gaussian_lowpass(const Mat& src, float sigma);
	/// @brief 高斯高通滤波
	/// @param src 源图像
	/// @param sigma 截至频率
	/// @return 滤波后图像
	static Mat gaussian_highpass(const Mat& src, float sigma);
	/// @brief 频域laplace边缘提取
	/// @param src 源图像 
	/// @return 边缘图像
	static Mat laplace_edge(const Mat& src);
	/// @brief laplace图像锐化
	/// @param src 源图像
	/// @return 锐化后图像
	static Mat laplace_sharpen(const Mat& src);
	/// @brief 同态滤波：抑制低频，放大高频，减少图像光照变化并锐化细节
	/// @param src 源图像
	/// @param c 高斯变化陡峭程度
	/// @param sigma 低频与高频截至频率
	/// @param gamma_l 低频权重
	/// @param gamma_h 高频权重
	/// @return 处理后图像
	static Mat homomoriphic_filter(const Mat& src, float c, float sigma, float gamma_l, float gamma_h);
	/// @brief 高斯带阻滤波器
	/// @param src 源图像
	/// @param R 带阻滤波中心频率
	/// @param W 带阻滤波宽度
	/// @return 处理后图像
	static Mat gauss_BE_filter(const Mat& src, float R, float W);
	/// @brief 巴特沃斯带阻滤波器
	/// @param src 源图像
	/// @param R 带阻滤波中心频率
	/// @param W 带阻滤波宽度
	/// @param N 阶数，曲线陡峭程度
	/// @return 处理后图像
	static Mat butterworth_BE_filter(const Mat& src, float R, float W, int N);
	/// @brief 理想带阻滤波器
	/// @param src 源图像
	/// @param R 带阻滤波中心频率
	/// @param W 带阻滤波宽度
	/// @return 处理后图像
	static Mat idel_BE_filter(const Mat& src, float R, int W);
	/// @brief 巴特沃斯陷波滤波器
	/// @param src 源图像
	/// @param R 陷波滤波器截止频率
	/// @param uk 陷波频率位于频谱行数
	/// @param rk 陷波频率位于频谱列数
	/// @param N 陷波滤波器阶数
	/// @return 处理后图像
	static Mat butterworth_NF_filter(const Mat& src, int R, int uk, int rk, int N);
	/// @brief 多点巴特沃斯陷波滤波器
	/// @param src 源图像
	/// @param R 陷波滤波器截止频率
	/// @param uk 多点陷波频率位于频谱行数
	/// @param rk 多点陷波频率位于频谱列数
	/// @param N 陷波滤波器阶数
	/// @return 处理后图像
	static Mat butterworth_NF_filter(const Mat& src, int R, std::vector<int> uk, std::vector<int> rk, int N);
	/// @brief 获取图像频谱
	/// @param src 源图像
	/// @return 频谱
	static Mat get_frequency_spectrum(const Mat& src);
	/// @brief 获取图像功率谱
	/// @param src 源图像
	/// @return 功率谱
	static Mat get_power_spectrum(const Mat& src);
	/// @brief 频域滤波函数
	/// @param src 源图像
	/// @param kernel 频域滤波器核
	/// @return 滤波后的归一化图像
	static Mat frequency_filter(const Mat& src, const Mat& kernel);
	
private:
	/// @brief 在图像上和左扩充图像至DFT最优尺寸
	/// @param src 源图像
	/// @return 处理后图像
	static Mat image_make_border(const Mat& src);
	/// @brief 缩减为DFT扩充图像至原来尺寸
	/// @param src 
	/// @param fft_img 
	/// @return 处理后图像
	static Mat image_reduce_border(const Mat& src, const Mat& fft_img);
	/// @brief 获取理想低通滤波器核
	/// @param src 源图像
	/// @param d0 截止频率
	/// @return 理想低通滤波器核
	static Mat get_ideal_lowpass_kernel(const Mat& src, float d0);
	/// @brief 获取理想高通滤波器核
	/// @param src 源图像
	/// @param d0 截止频率
	/// @return 理想高通滤波器核
	static Mat get_ideal_highpass_kernel(const Mat& src, float d0);
	/// @brief 获取高斯低通滤波器核
	/// H(u,v) = exp(-D(u,v)^2/(2*sigma^2))
	/// @param src 源图像
	/// @param sigma 截至频率
	/// @return 高斯低通滤波器核
	static Mat get_gauss_lowpass_kernel(const Mat& src, float sigma);
	/// @brief 获取高斯高通滤波器核
	/// H(u,v) = 1-exp(-D(u,v)^2/(2*sigma^2))
	/// @param src 源图像
	/// @param sigma 截止频率
	/// @return 高斯高通滤波器核
	static Mat get_gauss_highpass_kernel(const Mat& src, float sigma);
	/// @brief 获取巴特沃斯低通滤波器核
	/// H(u,v) = 1/(1+(D(u,v)/sigma)^(2N))
	/// @param src 源图像
	/// @param sigma 截止频率
	/// @param N 阶数
	/// @return 巴特沃斯低通滤波器核
	static Mat get_butterworth_lowpass_kernel(const Mat& src, float sigma, int N);
	/// @brief 获取巴特沃斯高通滤波器核
	/// H(u,v) = 1/(1+(sigma/D(u,v))^(2N))
	/// @param src 源图像
	/// @param sigma 截止频率
	/// @param N 阶数
	/// @return 巴特沃斯高通滤波器核
	static Mat get_butterworth_highpass_kernel(const Mat& src, float sigma, int N);
	/// @brief 获取laplace滤波核
	/// H(u,v) = -4*pi*D(u,v)^2
	/// @param src 源图像
	/// @return laplace滤波核
	static Mat get_laplace_kernel(const Mat& src);
	/// @brief 获取同态滤波核
	/// H(u,v) = (gamma_h-gamma_l)[1-exp(-c*D(u,v)^2/D0^2)] + gamma_l
	/// @param src 源图像
	/// @param c 高斯变化陡峭程度
	/// @param sigma 低频与高频截至频率
	/// @param gamma_l 低频权重
	/// @param gamma_h 高频权重
	/// @return 同态滤波核
	static Mat get_homoriphic_kernel(const Mat& src, float c, float sigma, float gamma_l, float gamma_h);
	/// @brief 获取高斯带阻滤波器核
	/// H(u,v) = 1-exp(-((D(u,v)^2 - R^2)/(D(u,v)*W))^2)
	/// @param src 源图像
	/// @param R 带阻中心频率
	/// @param W 带阻宽度
	/// @return 高斯带通滤波器核
	static Mat get_gauss_BE_kernel(const Mat& src, float R, float W);
	/// @brief 获取巴特沃斯带阻滤波器核
	/// @param src 源图像
	/// @param R 带阻滤波中心频率
	/// @param W 带阻滤波宽度
	/// @param N 阶数
	/// @return 巴特沃斯带阻滤波器核
	static Mat get_butterworth_BE_kernel(const Mat& src, float R, float W, int N);
	/// @brief 获取理想带阻滤波器核
	/// @param src 源图像
	/// @param R 带阻滤波中心频率
	/// @param W 带阻宽度
	/// @return 理想带阻滤波器核
	static Mat get_idel_BE_kernel(const Mat& src, float R, float W);
	/// @brief 获取巴特沃斯陷波滤波器核
	/// @param src 源图像
	/// @param R 陷波滤波器截止频率
	/// @param uk 陷波频率位于频谱行数
	/// @param rk 陷波频率位于频谱列数
	/// @param N 陷波滤波器阶数
	/// @return 巴特沃斯陷波滤波器核
	static Mat get_butterworth_NF_filter(const Mat& src, int R, int uk, int rk, int N);
};

#endif