#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv/cxmisc.h>


cv::Mat calcIntegralImage_naive(cv::Mat &image){
    cv::Mat IntegralImage = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
    for(int h = 0; h < image.rows; h++){
        for(int w = 0; w < image.cols; w++){
            for (int j = 0; j<=h; j++){
                for (int i = 0; i<=w; i++){
                    IntegralImage.at<double>(h, w) += image.at<double>(j, i);
                } }
        } }
    return IntegralImage;
}


cv::Mat calcIntegralImage_useprev(cv::Mat &image){
    cv::Mat IntegralImage = cv::Mat::zeros(image.rows, image.cols,CV_64FC1);

    int integral_stride=IntegralImage.cols;
    int src_stride=image.cols; //1 channel
    double *integral = IntegralImage.ptr<double>(0);
    double *src = image.ptr<double>(0);
    for(int h = 0; h < image.rows; h++){
        int nh = h - 1;
        double row_sum=0;
        for(int w = 0; w < image.cols; w++){
            int nw = w - 1;
            if (nh >= 0){
                integral[w] += integral[w - integral_stride]; }
            row_sum+= src[w];
            integral[w] += row_sum;
        }
        integral += integral_stride;
        src += src_stride;
    }
    return IntegralImage;
}


cv::Mat calcIntegralImage_padding(cv::Mat &image){
    cv::Mat IntegralImage = cv::Mat::zeros(image.rows+1, image.cols+1,CV_64FC1);
    double *integral = IntegralImage.ptr<double>(1)+1; //Start from (1,1)
    double *src = image.ptr<double>(0);
    int integral_stride=IntegralImage.cols;
    int src_stride=image.cols;
    for(int h = 0; h < image.rows; h++){
        int row_sum = 0;
        for(int w = 0; w < image.cols; w++){
            row_sum += src[w];
            integral[w] = row_sum + integral[w - integral_stride];
        }
        src += src_stride;
        integral += integral_stride;
    }
    IntegralImage = IntegralImage(cv::Rect(1,1,image.cols,image.rows));
    return IntegralImage;
}


int main() {
    cv::TickMeter meter;
    cv::Mat sample = cv::Mat::eye(200, 200, CV_64FC1);

    meter.start();
    calcIntegralImage_naive(sample);
    meter.stop();
    std::cout << "calcIntegralImage_naive (200*200): "<< meter << std::endl;

    meter.reset();
    meter.start();
    calcIntegralImage_useprev(sample);
    meter.stop();
    std::cout << "calcIntegralImage_useprev (200*200): "<< meter << std::endl;

    meter.reset();
    meter.start();
    calcIntegralImage_padding(sample);
    meter.stop();
    std::cout << "calcIntegralImage_padding (200*200): "<< meter << std::endl;


    sample = cv::Mat::eye(5000, 5000, CV_64FC1);
    meter.reset();
    meter.start();
    calcIntegralImage_useprev(sample);
    meter.stop();
    std::cout << "calcIntegralImage_useprev (5000*5000): "<< meter << std::endl;

    meter.reset();
    meter.start();
    calcIntegralImage_padding(sample);
    meter.stop();
    std::cout << "calcIntegralImage_padding (5000*5000): "<< meter << std::endl;


    return 0;
}