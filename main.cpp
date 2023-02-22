#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "frequency_filter.h"
#include "image_restoration.h"
using namespace std;
using namespace cv;
int main()
{
	Mat src = imread("./img/Fig0503.tif",0);
	if(src.data == 0){
		cout << "No img read" << endl;
		return 0;
	}
	imshow("src", src);
	Mat noise = ImageRestoration::add_gauss_noise(src, 126, 20);
	imshow("gauss_noise", noise);

	std::cout << noise.type() << std::endl;
	auto hist_img = ImageRestoration::get_hist_img(noise);
	imshow("hist", hist_img);
    waitKey();
	return 0;
}
