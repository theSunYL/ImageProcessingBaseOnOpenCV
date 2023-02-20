#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "frequency_filter.h"
using namespace std;
using namespace cv;
int main()
{
	Mat src = imread("./img/Fig0464.tif",0);
	if(src.data == 0){
		cout << "No img read" << endl;
		return 0;
	}
	imshow("src", src);
	Mat frequency_spectrum = FrequencyFilter::get_frequency_spectrum(src);
	imshow("frequency_spectrum", frequency_spectrum);
	Mat power_spectrum = FrequencyFilter::get_power_spectrum(src);
	imshow("power_spectrum", power_spectrum);
    waitKey();
	return 0;
}
