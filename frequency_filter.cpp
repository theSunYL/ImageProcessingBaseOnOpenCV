#include "frequency_filter.h"
#define _USE_MATH_DEFINES
#include "math.h"
#include <vector>

Mat FrequencyFilter::frequency_filter(const Mat& src, const Mat& kernel)
{
	std::vector<Mat> plane{ Mat_<float>(src), Mat::zeros(src.size(), CV_32FC1) };
	Mat complexI;
	merge(plane.data(), 2, complexI);
	dft(complexI, complexI);
	/*中心化*/
	split(complexI, plane);
	Mat re = plane[0];
	int cx = re.cols / 2;
	int cy = re.rows / 2;
	Mat q0(re, Rect(0, 0, cx, cy)); //左上方
	Mat q1(re, Rect(cx, 0, cx, cy)); //右上方
	Mat q2(re, Rect(0, cy, cx, cy)); //左下方
	Mat q3(re, Rect(cx, cy, cx, cy)); //右下方
	//左上方和右下方交换
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	//右上方和左下方交换
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	Mat im = plane[1];
	cx = im.cols / 2;
	cy = im.rows / 2;
	Mat q4(im, Rect(0, 0, cx, cy)); //左上方
	Mat q5(im, Rect(cx, 0, cx, cy)); //右上方
	Mat q6(im, Rect(0, cy, cx, cy)); //左下方
	Mat q7(im, Rect(cx, cy, cx, cy)); //右下方
	//左上方和右下方交换
	q4.copyTo(tmp);
	q7.copyTo(q4);
	tmp.copyTo(q7);
	//右上方和左下方交换
	q5.copyTo(tmp);
	q6.copyTo(q5);
	tmp.copyTo(q6);
	/*进行实际滤波操作*/
	Mat filter_re, filter_im, filter;
	multiply(plane[0], kernel, filter_re);
	multiply(plane[1], kernel, filter_im);
	std::vector<Mat> plane1{ filter_re, filter_im };
	merge(plane1.data(), 2, filter);
	idft(filter, filter);
	split(filter, plane);
	magnitude(plane[0], plane[1], plane[0]);//得到幅度谱
	normalize(plane[0], plane[0], 0, 1, NORM_MINMAX);
	return plane[0];
}

Mat FrequencyFilter::get_frequency_spectrum(const Mat& src)
{
	Mat pad = image_make_border(src);
	std::vector<Mat> plane{ Mat_<float>(pad), Mat::zeros(pad.size(), CV_32FC1) };
	Mat complexI;
	merge(plane.data(), 2, complexI);
	dft(complexI, complexI);
	split(complexI, plane);
	//中心化
	Mat re = plane[0];
	int cx = re.cols / 2;
	int cy = re.rows / 2;
	Mat q0(re, Rect(0, 0, cx, cy));
	Mat q1(re, Rect(cx, 0, cx, cy));
	Mat q2(re, Rect(0, cy, cx, cy));
	Mat q3(re, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	Mat im = plane[1];
	q0 = Mat(im, Rect(0, 0, cx, cy));
	q1 = Mat(im, Rect(cx, 0, cx, cy));
	q2 = Mat(im, Rect(0, cy, cx, cy));
	q3 = Mat(im, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//频谱定义： log(1+abs(sqrt(re^2 + im^2)))
	plane[0] = plane[0].mul(plane[0]);
	plane[1] = plane[1].mul(plane[1]);
	cv::sqrt(plane[0] + plane[1], plane[0]);
	cv::log(1 + cv::abs(plane[0]), plane[0]);
	//归一化便于显示
	normalize(plane[0], plane[0], 0, 1, NORM_MINMAX);
	return plane[0];
}

Mat FrequencyFilter::get_power_spectrum(const Mat& src)
{
	std::vector<Mat> plane{ Mat_<float>(src), Mat::zeros(src.size(), CV_32FC1) };
	Mat complexI;
	merge(plane.data(), 2, complexI);
	dft(complexI, complexI);
	split(complexI, plane);
	//中心化
	Mat re = plane[0];
	int cx = re.cols / 2;
	int cy = re.rows / 2;
	Mat q0(re, Rect(0, 0, cx, cy));
	Mat q1(re, Rect(cx, 0, cx, cy));
	Mat q2(re, Rect(0, cy, cx, cy));
	Mat q3(re, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	Mat im = plane[1];
	q0 = Mat(im, Rect(0, 0, cx, cy));
	q1 = Mat(im, Rect(cx, 0, cx, cy));
	q2 = Mat(im, Rect(0, cy, cx, cy));
	q3 = Mat(im, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//功率谱定义：re^2 + im^2
	plane[0] = plane[0].mul(plane[0]);
	plane[1] = plane[1].mul(plane[1]);
	plane[0] = plane[0] + plane[1];
	//归一化便于显示
	normalize(plane[0], plane[0], 0, 1, NORM_MINMAX);
	return plane[0];
}

Mat FrequencyFilter::homomoriphic_filter(const Mat& src, float c, float sigma, float gamma_l, float gamma_h)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_homoriphic_kernel(pad, c, sigma, gamma_l, gamma_h);
	pad = pad + 1;
	cv::log(pad, pad);
	Mat fimg = frequency_filter(pad, kernel);
	cv::exp(fimg, fimg);
	normalize(fimg, fimg, 0, 1, NORM_MINMAX);
	Mat ret = image_reduce_border(src, fimg);
	return ret;
}

Mat FrequencyFilter::gauss_BE_filter(const Mat& src, float R, float W)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_gauss_BE_kernel(pad, R, W);
	imshow("kernrl", kernel);
	Mat fimg = frequency_filter(pad, kernel);
	Mat ret = image_reduce_border(src, fimg);
	return ret;
}

Mat FrequencyFilter::butterworth_BE_filter(const Mat& src, float R, float W, int N)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_butterworth_BE_kernel(pad, R, W, N);
	imshow("kernel", kernel);
	Mat fimg = frequency_filter(pad, kernel);
	Mat ret = image_reduce_border(src, fimg);
	return ret;
}

Mat FrequencyFilter::idel_BE_filter(const Mat& src, float R, int W)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_idel_BE_kernel(pad, R, W);
	imshow("kernel", kernel);
	Mat fimg = frequency_filter(pad, kernel);
	Mat ret = image_reduce_border(src, fimg);
	return ret;
}

Mat FrequencyFilter::butterworth_NF_filter(const Mat& src, int R, int uk, int rk, int N)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_butterworth_NF_filter(pad, R, uk, rk, N);
	imshow("kernel", kernel);
	Mat fimg = frequency_filter(pad, kernel);
	Mat ret = image_reduce_border(src, fimg);
	return ret;
}

Mat FrequencyFilter::butterworth_NF_filter(const Mat& src, int R, std::vector<int> uk, std::vector<int> rk, int N)
{
	Mat pad = image_make_border(src);
	std::vector<Mat> kernels;
	for(int i=0; i<uk.size(); ++i){
		kernels.push_back(get_butterworth_NF_filter(pad, R, uk[i], rk[i], N));
	}
	Mat kernel = kernels[0].clone();
	if(kernels.size() != 1){
		for(int i=1; i<kernels.size(); ++i){
			cv::multiply(kernel, kernels[i], kernel);
		}
	}
	imshow("kernel", kernel);
	Mat fimg = frequency_filter(pad, kernel);
	Mat ret = image_reduce_border(src, fimg);
	return ret;
}

Mat FrequencyFilter::ideal_lowpass(const Mat& src, float d0)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_ideal_lowpass_kernel(pad, d0);
	imshow("kernel", kernel);
	Mat ifft = frequency_filter(pad, kernel);
	return image_reduce_border(src, ifft);
}

Mat FrequencyFilter::ideal_highpass(const Mat& src, float d0)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_ideal_highpass_kernel(pad, d0);
	imshow("kernel", kernel);
	Mat ifft = frequency_filter(pad, kernel);
	return image_reduce_border(src, ifft);
}

Mat FrequencyFilter::butterworth_lowpass(const Mat& src, float d0, int n)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_butterworth_lowpass_kernel(pad, d0, n);
	imshow("kernel", kernel);
	Mat fft_img = frequency_filter(pad, kernel);
	return image_reduce_border(src, fft_img);
}	
/// @brief 巴特沃斯高通滤波
/// @param src 源图像
/// @param d0 截止频率
/// @param n 阶数，曲线陡峭程度，n越大曲线越陡峭，振铃效应明显
/// @return 滤波后图像
Mat FrequencyFilter::butterworth_highpass(const Mat& src, float d0, int n)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_butterworth_highpass_kernel(pad, d0, n);
	imshow("kernel", kernel);
	Mat fft_img = frequency_filter(pad, kernel);
	return image_reduce_border(src, fft_img);
}

Mat FrequencyFilter::gaussian_lowpass(const Mat& src, float sigma)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_gauss_lowpass_kernel(pad, sigma);
	imshow("kernel", kernel);
	Mat fft_img = frequency_filter(pad, kernel);
	return image_reduce_border(src, fft_img);
}

Mat FrequencyFilter::gaussian_highpass(const Mat& src, float sigma)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_gauss_highpass_kernel(pad, sigma);
	imshow("kernel", kernel);
	Mat fft_img = frequency_filter(pad, kernel);
	return image_reduce_border(src, fft_img);
}

Mat FrequencyFilter::laplace_edge(const Mat& src)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_laplace_kernel(pad);
	imshow("kernel", kernel);
	Mat fft_img = frequency_filter(pad, kernel);
	return image_reduce_border(src, fft_img);
}

Mat FrequencyFilter::laplace_sharpen(const Mat& src)
{
	Mat pad = image_make_border(src);
	Mat kernel = get_laplace_kernel(pad);
	imshow("kernel", kernel);
	Mat iidft = frequency_filter(pad, kernel);
	imshow("iidft", iidft);
	Mat result(iidft.size(), CV_32F);
	//注意原图像和边缘图像必须放缩至同一范围进行操作
	normalize(iidft, iidft, 0, 1, NORM_MINMAX); 
	normalize(pad, pad, 0, 1, NORM_MINMAX);
	result = iidft + pad;
	normalize(result, result, 0, 1, NORM_MINMAX);
	return image_reduce_border(src, result);
}

Mat FrequencyFilter::image_make_border(const Mat& src)
{
	int m = getOptimalDFTSize(src.rows);
	int n = getOptimalDFTSize(src.cols);
	Mat pad;
	copyMakeBorder(src, pad, m-src.rows, 0, n-src.cols, 0, BORDER_CONSTANT, Scalar::all(0));
	pad.convertTo(pad, CV_32F);
	return pad;
}

Mat FrequencyFilter::image_reduce_border(const Mat& src, const Mat& fft_img)
{
	if(src.size() == fft_img.size()){
		return fft_img;
	}
	Mat ret = fft_img(Rect(fft_img.cols - src.cols, fft_img.rows - src.rows, src.cols, src.rows));
	return ret;
}

Mat FrequencyFilter::get_ideal_lowpass_kernel(const Mat& src, float d0)
{
	Mat kernel(src.size(), CV_32F);
	int cx = src.rows / 2;
	int cy = src.cols / 2;
	for(int i=0; i<src.rows; ++i){
		for(int j=0; j<src.cols; ++j){
			double dis = sqrt(pow(i - cx, 2) + pow(j - cy, 2));
			if(dis <= d0){
				kernel.at<float>(i, j) = 1;
			} else{
				kernel.at<float>(i, j) = 0;
			}
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_ideal_highpass_kernel(const Mat& src, float d0)
{
	Mat kernel(src.size(), CV_32F);
	int cx = src.rows / 2;
	int cy = src.cols / 2;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			double dis = sqrt(pow(i - cx, 2) + pow(j - cy, 2));
			if (dis < d0) {
				kernel.at<float>(i, j) = 0;
			}
			else {
				kernel.at<float>(i, j) = 1;
			}
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_gauss_lowpass_kernel(const Mat& src, float sigma)
{
	Mat kernel(src.size(), CV_32F);
	int cx = src.rows / 2;
	int cy = src.cols / 2;
	float d0 = 2 * pow(sigma,2);
	for(int i=0; i<src.rows; ++i){
		for(int j=0; j<src.cols; ++j){
			double d = pow(float(i - cx), 2) + pow(float(j - cy), 2);
			kernel.at<float>(i, j) = expf(-d / d0);
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_gauss_highpass_kernel(const Mat& src, float sigma)
{
	Mat kernel(src.size(), CV_32F);
	int cx = src.rows / 2;
	int cy = src.cols / 2;
	float d0 = 2 * pow(sigma, 2);
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			double d = pow(float(i - cx), 2) + pow(float(j - cy), 2);
			kernel.at<float>(i, j) = 1 - expf(-d / d0);
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_butterworth_lowpass_kernel(const Mat& src, float sigma, int N)
{
	Mat kernel(src.size(), CV_32F);
	int cx = src.rows / 2;
	int cy = src.cols / 2;
	float d0 = sigma;
	for(int i=0; i<src.rows; ++i){
		for(int j=0; j<src.cols; ++j){
			float d = sqrt(pow(i - cx, 2) + pow(j - cy, 2));
			kernel.at<float>(i, j) = (float)1 / (1 + pow(d / d0, 2 * N));
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_butterworth_highpass_kernel(const Mat& src, float sigma, int N)
{
	Mat kernel(src.size(), CV_32F);
	int cx = src.rows / 2;
	int cy = src.cols / 2;
	float d0 = sigma;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			float d = sqrt(pow(i - cx, 2) + pow(j - cy, 2));
			kernel.at<float>(i, j) = (float)1 / (1 + pow(d0 / d, 2 * N));
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_laplace_kernel(const Mat& src)
{
	Mat kernel(src.size(), CV_32F);
	int cx = src.rows / 2;
	int cy = src.cols / 2;
	for(int i=0; i<src.rows; ++i){
		for(int j=0; j<src.cols; ++j){
			kernel.at<float>(i,j) = -4 * pow(M_PI, 2) * (pow(i-cx, 2) + pow(j-cy,2));
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_homoriphic_kernel(const Mat& src, float c, float sigma, float gamma_l, float gamma_h)
{
	Mat kernel(src.size(), CV_32FC1);
	int cx = kernel.rows / 2;
	int cy = kernel.cols / 2;
	float d0 = pow(sigma, 2);
	for(int i=0; i<kernel.rows; ++i){
		for(int j=0; j<kernel.cols; ++j){
			float dis = pow(i - cx, 2) + pow(j - cy, 2);
			kernel.at<float>(i, j) = (gamma_h - gamma_l) * (1 - exp(-c * dis / d0)) + gamma_l;
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_gauss_BE_kernel(const Mat& src, float R, float W)
{
	Mat kernel(src.size(), CV_32FC1);
	int cx = kernel.rows / 2;
	int cy = kernel.cols / 2;
	float d0 = pow(R, 2);
	for(int i=0; i<kernel.rows; ++i){
		for(int j=0; j<kernel.cols; ++j){
			float dis = sqrt(pow(i - cx, 2) + pow(j - cy, 2));
			kernel.at<float>(i, j) = 1 - exp(-pow((pow(dis,2) - d0) / (double(dis) * W),2));
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_butterworth_BE_kernel(const Mat& src, float R, float W, int N)
{
	Mat kernel(src.size(), CV_32FC1);
	int cx = kernel.rows / 2;
	int cy = kernel.cols / 2;
	float d0 = pow(R, 2);
	for(int i=0; i<kernel.rows; ++i){
		for(int j=0; j<kernel.cols; ++j){
			float dis = sqrt(pow(i - cx, 2) + pow(j - cy, 2));
			kernel.at<float>(i, j) = 1.0 / (1 + pow((double(dis) * W) / (pow(dis, 2) - d0), 2*N));
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_idel_BE_kernel(const Mat& src, float R, float W)
{
	Mat kernel(src.size(), CV_32FC1);
	int cx = kernel.rows / 2;
	int cy = kernel.cols / 2;
	for(int i=0; i < kernel.rows; ++i){
		for(int j=0; j<kernel.cols; ++j){
			float dis = sqrt(pow(i - cx, 2) + pow(j - cy, 2));
			if(dis >= R-W/2 && dis <= R+W/2){
				kernel.at<float>(i, j) = 0;
			} else {
				kernel.at<float>(i, j) = 1;
			}
		}
	}
	return kernel;
}

Mat FrequencyFilter::get_butterworth_NF_filter(const Mat& src, int R, int uk, int rk, int N)
{
	Mat kernel(src.size(), CV_32FC1);
	int cx = kernel.rows / 2;
	int cy = kernel.cols / 2;
	long int N2 = 2*long int(N);
	for (int i = 0; i < kernel.rows; ++i){
		for(int j=0; j < kernel.cols; ++j){
			float pos_dis = sqrt(pow(i - cx + (cx - uk), 2) + pow(j - cy + (cy - rk), 2));
			float neg_dis = sqrt(pow(i - cx - (cx - uk), 2) + pow(j - cy - (cy - rk), 2));
			kernel.at<float>(i, j) = (1.0 / (1 + pow(R / pos_dis, N2))) * (1.0 / (1 + pow(R / neg_dis, N2)));
		}
	}
	return kernel;
}
