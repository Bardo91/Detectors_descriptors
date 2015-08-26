///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//		Test feature detector and descriptors.
//		Author: Pablo R.S.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	

#include <fstream>
#include <iostream>
#include <string>

#include <core/time/time.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace BOViL;
using namespace std;

// Wrapper class for FAST method, in opencv is not implemented as a class.
class FAST_wrapper {
public:
	void detect(const Mat &_image, vector<KeyPoint> &_kps){
		FAST(_image, _kps, 9);
	}

	static Ptr<FAST_wrapper> create() {
		return Ptr<FAST_wrapper>();
	}
};


const vector<string>	folderPaths	= {	"C:/Users/Pablo RS/Downloads/Matching dataset/graffiti/graffiti/",
										"C:/Users/Pablo RS/Downloads/Matching dataset/cars/cars/",
										"C:/Users/Pablo RS/Downloads/Matching dataset/boat_d/boat/",
										"C:/Users/Pablo RS/Downloads/Matching dataset/bikes/bikes/" };
const vector<string>	imgNames	= {	"img_1.ppm", 
										"img_2.ppm", 
										"img_3.ppm", 
										"img_4.ppm"};
const vector<Size>		imgSizes	= {Size(300, 200), Size(640, 480), Size(800,600)};

//---------------------------------------------------------------------------------------------------------------------
template<class Detector_> 
double computeDetectorTime(string _imgPath, Size _imgSize, unsigned _repetitions);
template<typename Operator_>
void detectorTimes();

//---------------------------------------------------------------------------------------------------------------------
int main(int _argc, char ** _argv) {
	detectorTimes<computeDetectorTime>();

	// Descriptors speed.
	
	// Detector Repeatability.

	// 
}

//---------------------------------------------------------------------------------------------------------------------
template<class Detector_> 
double computeDetectorTime(string _imgPath, Size _imgSize, unsigned _repetitions) {
	double avgTime = 0.0;

	Mat image = imread(_imgPath, CV_LOAD_IMAGE_GRAYSCALE);
	resize(image, image, _imgSize);
	Ptr<Detector_> detector = Detector_::create();

	STime *timer = STime::get();
	double tstart = 0;

	for (unsigned i = 0; i < _repetitions; i++) {
		vector<KeyPoint> keypoints;
		tstart = timer->getTime();
		detector->detect(image, keypoints);
		avgTime += timer->getTime() - tstart;
	}

	return avgTime/_repetitions;
}

//---------------------------------------------------------------------------------------------------------------------
template<class Detector_> 
double computeDescriptorTime(string _imgPath, Size _imgSize, unsigned _repetitions) {
	double avgTime = 0.0;

	Mat image = imread(_imgPath, CV_LOAD_IMAGE_GRAYSCALE);
	resize(image, image, _imgSize);

	Ptr<Detector_> descriptor = Detector_::create();
	Ptr<SIFT> detectorSIFT = SIFT::create();

	STime *timer = STime::get();
	double tstart = 0;

	for (unsigned i = 0; i < _repetitions; i++) {
		vector<KeyPoint> keypoints;
		Mat descriptors;
		detectorSIFT->detect(image, keypoints);
		tstart = timer->getTime();
		descriptor->compute(image, keypoints, descriptors);
		avgTime += timer->getTime() - tstart;
	}

	return avgTime/_repetitions;
}

//---------------------------------------------------------------------------------------------------------------------
template<template <typename Descriptor_> typename Operator_>
void detectorTimes() {
	// Open stream file
	ofstream detectorTimes("detector_times.txt");
	unsigned repetitions = 10;
	vector<vector<double>> times;
	times.resize(imgSizes.size());

	std::cout << "Computing detector times" << std::endl;
	for (int i = 0; i < imgSizes.size(); i++) {
		times[i].resize(7);
		std::cout << "--> Size: " << imgSizes[i].width << "x" << imgSizes[i].height << std::endl; 
		for (string folder : folderPaths) {
			std::cout << "----> Folder: " << folder << std::endl;
			for (string imgName : imgNames) {
				std::cout << "------> Image: " << imgName << std::endl;
				times[i][0] += Operator_<xfeatures2d::SIFT>	(folder + imgName, imgSizes[i], repetitions);
				times[i][1] += Operator_<xfeatures2d::SURF>	(folder + imgName, imgSizes[i], repetitions);
				times[i][2] += Operator_<FAST_wrapper>		(folder + imgName, imgSizes[i], repetitions);
				times[i][3] += Operator_<ORB>				(folder + imgName, imgSizes[i], repetitions);
				times[i][4] += Operator_<BRISK>				(folder + imgName, imgSizes[i], repetitions);
				times[i][5] += Operator_<KAZE>				(folder + imgName, imgSizes[i], repetitions);
				times[i][6] += Operator_<AKAZE>				(folder + imgName, imgSizes[i], repetitions);
				//times[0] += computeDetectorTime<xfeatures2d::LATCH>(folder+imgName, repetitions); 777 Not implemented yet in opencv 3.0
			}
		}

		unsigned totalImages = folderPaths.size()*imgNames.size();
		for (double &time : times[i]) {
			time /= totalImages;
		}
	}

	for (unsigned j = 0; j < times[0].size(); j++) {
		for (unsigned i = 0; i < times.size(); i++) {
			detectorTimes << times[i][j] << "\t";
		}
		detectorTimes << std::endl;
	}

	detectorTimes.flush();
	detectorTimes.close();
}

//---------------------------------------------------------------------------------------------------------------------