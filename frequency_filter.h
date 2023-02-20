#ifndef FREQUENCY_FILTER_H
#define FREQUENCY_FILTER_H

#include <opencv2/opencv.hpp>
using namespace cv;

/// @brief Ƶ��ͼ����
class FrequencyFilter 
{
public:
	/// @brief �����ͨ�˲�
	/// @param src Դͼ��
	/// @param d0 ��ֹƵ��
	/// @return �˲���ͼ��
	static Mat ideal_lowpass(const Mat& src, float d0);
	/// @brief �����ͨ�˲�
	/// @param src Դͼ��
	/// @param d0 ��ֹƵ��
	/// @return �˲���ͼ��
	static Mat ideal_highpass(const Mat& src, float d0);
	/// @brief ������˹��ͨ�˲�
	/// @param src Դͼ��
	/// @param d0 ��ֹƵ��
	/// @param n ���������߶��ͳ̶ȣ�nԽ������Խ���ͣ�����ЧӦ����
	/// @return �˲���ͼ��
	static Mat butterworth_lowpass(const Mat& src, float d0, int n);
	/// @brief ������˹��ͨ�˲�
	/// @param src Դͼ��
	/// @param d0 ��ֹƵ��
	/// @param n ���������߶��ͳ̶ȣ�nԽ������Խ���ͣ�����ЧӦ����
	/// @return �˲���ͼ��
	static Mat butterworth_highpass(const Mat& src, float d0, int n);
	/// @brief ��˹��ͨ�˲�
	/// @param src Դͼ��
	/// @param sigma ��ֹƵ��
	/// @return �˲���ͼ��
	static Mat gaussian_lowpass(const Mat& src, float sigma);
	/// @brief ��˹��ͨ�˲�
	/// @param src Դͼ��
	/// @param sigma ����Ƶ��
	/// @return �˲���ͼ��
	static Mat gaussian_highpass(const Mat& src, float sigma);
	/// @brief Ƶ��laplace��Ե��ȡ
	/// @param src Դͼ�� 
	/// @return ��Եͼ��
	static Mat laplace_edge(const Mat& src);
	/// @brief laplaceͼ����
	/// @param src Դͼ��
	/// @return �񻯺�ͼ��
	static Mat laplace_sharpen(const Mat& src);
	/// @brief ̬ͬ�˲������Ƶ�Ƶ���Ŵ��Ƶ������ͼ����ձ仯����ϸ��
	/// @param src Դͼ��
	/// @param c ��˹�仯���ͳ̶�
	/// @param sigma ��Ƶ���Ƶ����Ƶ��
	/// @param gamma_l ��ƵȨ��
	/// @param gamma_h ��ƵȨ��
	/// @return �����ͼ��
	static Mat homomoriphic_filter(const Mat& src, float c, float sigma, float gamma_l, float gamma_h);
	/// @brief ��˹�����˲���
	/// @param src Դͼ��
	/// @param R �����˲�����Ƶ��
	/// @param W �����˲����
	/// @return �����ͼ��
	static Mat gauss_BE_filter(const Mat& src, float R, float W);
	/// @brief ������˹�����˲���
	/// @param src Դͼ��
	/// @param R �����˲�����Ƶ��
	/// @param W �����˲����
	/// @param N ���������߶��ͳ̶�
	/// @return �����ͼ��
	static Mat butterworth_BE_filter(const Mat& src, float R, float W, int N);
	/// @brief ��������˲���
	/// @param src Դͼ��
	/// @param R �����˲�����Ƶ��
	/// @param W �����˲����
	/// @return �����ͼ��
	static Mat idel_BE_filter(const Mat& src, float R, int W);
	/// @brief ������˹�ݲ��˲���
	/// @param src Դͼ��
	/// @param R �ݲ��˲�����ֹƵ��
	/// @param uk �ݲ�Ƶ��λ��Ƶ������
	/// @param rk �ݲ�Ƶ��λ��Ƶ������
	/// @param N �ݲ��˲�������
	/// @return �����ͼ��
	static Mat butterworth_NF_filter(const Mat& src, int R, int uk, int rk, int N);
	/// @brief ��������˹�ݲ��˲���
	/// @param src Դͼ��
	/// @param R �ݲ��˲�����ֹƵ��
	/// @param uk ����ݲ�Ƶ��λ��Ƶ������
	/// @param rk ����ݲ�Ƶ��λ��Ƶ������
	/// @param N �ݲ��˲�������
	/// @return �����ͼ��
	static Mat butterworth_NF_filter(const Mat& src, int R, std::vector<int> uk, std::vector<int> rk, int N);
	/// @brief ��ȡͼ��Ƶ��
	/// @param src Դͼ��
	/// @return Ƶ��
	static Mat get_frequency_spectrum(const Mat& src);
	/// @brief ��ȡͼ������
	/// @param src Դͼ��
	/// @return ������
	static Mat get_power_spectrum(const Mat& src);
	/// @brief Ƶ���˲�����
	/// @param src Դͼ��
	/// @param kernel Ƶ���˲�����
	/// @return �˲���Ĺ�һ��ͼ��
	static Mat frequency_filter(const Mat& src, const Mat& kernel);
	
private:
	/// @brief ��ͼ���Ϻ�������ͼ����DFT���ųߴ�
	/// @param src Դͼ��
	/// @return �����ͼ��
	static Mat image_make_border(const Mat& src);
	/// @brief ����ΪDFT����ͼ����ԭ���ߴ�
	/// @param src 
	/// @param fft_img 
	/// @return �����ͼ��
	static Mat image_reduce_border(const Mat& src, const Mat& fft_img);
	/// @brief ��ȡ�����ͨ�˲�����
	/// @param src Դͼ��
	/// @param d0 ��ֹƵ��
	/// @return �����ͨ�˲�����
	static Mat get_ideal_lowpass_kernel(const Mat& src, float d0);
	/// @brief ��ȡ�����ͨ�˲�����
	/// @param src Դͼ��
	/// @param d0 ��ֹƵ��
	/// @return �����ͨ�˲�����
	static Mat get_ideal_highpass_kernel(const Mat& src, float d0);
	/// @brief ��ȡ��˹��ͨ�˲�����
	/// H(u,v) = exp(-D(u,v)^2/(2*sigma^2))
	/// @param src Դͼ��
	/// @param sigma ����Ƶ��
	/// @return ��˹��ͨ�˲�����
	static Mat get_gauss_lowpass_kernel(const Mat& src, float sigma);
	/// @brief ��ȡ��˹��ͨ�˲�����
	/// H(u,v) = 1-exp(-D(u,v)^2/(2*sigma^2))
	/// @param src Դͼ��
	/// @param sigma ��ֹƵ��
	/// @return ��˹��ͨ�˲�����
	static Mat get_gauss_highpass_kernel(const Mat& src, float sigma);
	/// @brief ��ȡ������˹��ͨ�˲�����
	/// H(u,v) = 1/(1+(D(u,v)/sigma)^(2N))
	/// @param src Դͼ��
	/// @param sigma ��ֹƵ��
	/// @param N ����
	/// @return ������˹��ͨ�˲�����
	static Mat get_butterworth_lowpass_kernel(const Mat& src, float sigma, int N);
	/// @brief ��ȡ������˹��ͨ�˲�����
	/// H(u,v) = 1/(1+(sigma/D(u,v))^(2N))
	/// @param src Դͼ��
	/// @param sigma ��ֹƵ��
	/// @param N ����
	/// @return ������˹��ͨ�˲�����
	static Mat get_butterworth_highpass_kernel(const Mat& src, float sigma, int N);
	/// @brief ��ȡlaplace�˲���
	/// H(u,v) = -4*pi*D(u,v)^2
	/// @param src Դͼ��
	/// @return laplace�˲���
	static Mat get_laplace_kernel(const Mat& src);
	/// @brief ��ȡ̬ͬ�˲���
	/// H(u,v) = (gamma_h-gamma_l)[1-exp(-c*D(u,v)^2/D0^2)] + gamma_l
	/// @param src Դͼ��
	/// @param c ��˹�仯���ͳ̶�
	/// @param sigma ��Ƶ���Ƶ����Ƶ��
	/// @param gamma_l ��ƵȨ��
	/// @param gamma_h ��ƵȨ��
	/// @return ̬ͬ�˲���
	static Mat get_homoriphic_kernel(const Mat& src, float c, float sigma, float gamma_l, float gamma_h);
	/// @brief ��ȡ��˹�����˲�����
	/// H(u,v) = 1-exp(-((D(u,v)^2 - R^2)/(D(u,v)*W))^2)
	/// @param src Դͼ��
	/// @param R ��������Ƶ��
	/// @param W ������
	/// @return ��˹��ͨ�˲�����
	static Mat get_gauss_BE_kernel(const Mat& src, float R, float W);
	/// @brief ��ȡ������˹�����˲�����
	/// @param src Դͼ��
	/// @param R �����˲�����Ƶ��
	/// @param W �����˲����
	/// @param N ����
	/// @return ������˹�����˲�����
	static Mat get_butterworth_BE_kernel(const Mat& src, float R, float W, int N);
	/// @brief ��ȡ��������˲�����
	/// @param src Դͼ��
	/// @param R �����˲�����Ƶ��
	/// @param W ������
	/// @return ��������˲�����
	static Mat get_idel_BE_kernel(const Mat& src, float R, float W);
	/// @brief ��ȡ������˹�ݲ��˲�����
	/// @param src Դͼ��
	/// @param R �ݲ��˲�����ֹƵ��
	/// @param uk �ݲ�Ƶ��λ��Ƶ������
	/// @param rk �ݲ�Ƶ��λ��Ƶ������
	/// @param N �ݲ��˲�������
	/// @return ������˹�ݲ��˲�����
	static Mat get_butterworth_NF_filter(const Mat& src, int R, int uk, int rk, int N);
};

#endif