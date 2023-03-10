#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "frequency_filter.h"
#include "image_restoration.h"
using namespace std;
using namespace cv;
int main()
{
	Mat src = imread("./img/Fig0526(a).tif",0);
	if(src.data == 0){
		cout << "No img read" << endl;
		return -1;
	}
	imshow("src", src);
	Mat move_blur = ImageRestoration::add_move_blur(src, 0.001, 0.001, 1);
	imshow("move_blur", move_blur);
    waitKey();
	return 0;
}
