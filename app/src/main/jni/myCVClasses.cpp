#pragma warning(disable: 4996)
#include "opencv2/opencv.hpp"
#include <algorithm>
#include "myCVClasses.hpp"
#include <android/log.h>

#define LOG_TAG "Arc2"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)


namespace myCV {
/****************************************************************************************************************************/
/*//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/
/*//////////////////////////////////////////////////CAMERACALIBRATION///////////////////////////////////////////////////////*/
/*//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/
/****************************************************************************************************************************/

	void CameraCalibration::Initialisation() {

		if (!_objectPointsSet) {

			std::vector<cv::Point3f> tmp;
			//Compute the the three dimensional world-coordinates
			for (float h = 0.0f; h < _board_h; h += 1.0f) {
				for (float w = 0.0f; w < _board_w; w += 1.0f) {
					tmp.push_back(cv::Point3f(h, w, 0.0f));
				}
			}
			//Put all the world-coordinates for each frame in a vector
			for (int n = 0; n < _nImages; ++n) {
				_objectPoints.push_back(tmp);
			}

		}

		//GrabFrames();
	}

	void CameraCalibration::GrabFrames(cv::Mat _capture, cv::Mat _Rgbcapture) {

		std::vector<cv::Point2f> corners;
		cv::Size board_sz(_board_w, _board_h);
		cv::Size winSize(5, 5);
		cv::Size zeroZone(-1, -1);
		int cornerCount = 0, index = 0;
		bool found = false;

		//cv::Mat img, imgText, grayImg;

		//img.create((int)_capture.get(CV_CAP_PROP_FRAME_HEIGHT), (int)_capture.get(CV_CAP_PROP_FRAME_WIDTH), CV_8UC3);
		//imgText.create((int)_capture.get(CV_CAP_PROP_FRAME_HEIGHT), (int)_capture.get(CV_CAP_PROP_FRAME_WIDTH), CV_8UC3);
		//grayImg.create((int)_capture.get(CV_CAP_PROP_FRAME_HEIGHT), (int)_capture.get(CV_CAP_PROP_FRAME_WIDTH), CV_8UC1);

		//_capture >> img;
		//img.copyTo(imgText);

		//Init stringstream for the image counter text
		//std::ostringstream ostr;

		//cv::namedWindow("Grab_Frames");

		//ostr.str("");
		//ostr << frames << "/" << _nImages;

		//Find Chessboardcorners on the original Image "img"
		found = cv::findChessboardCorners(_capture, board_sz, corners,
											  CV_CALIB_CB_ADAPTIVE_THRESH |
											  CV_CALIB_CB_FILTER_QUADS);
		//LOGI("found:%d", found);
		if(found){
			//Draw the visual output on the scribble Image "imgText"
			cv::drawChessboardCorners(_Rgbcapture, board_sz, cv::Mat(corners), found);
			//cv::putText(imgText, ostr.str().c_str(), cv::Point(50,50),cv::FONT_HERSHEY_COMPLEX|cv::FONT_ITALIC, 1.0, cv::Scalar(0,0,0));
			//cv::imshow("Grab_Frames", imgText);

			//Picture has been chosen by the user

			//cv::cvtColor(img, grayImg, CV_BGR2GRAY);

			cv::cornerSubPix(_capture, corners, winSize, zeroZone, cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

			_imagePoints.push_back(corners);

			_frames++;
		}

		//LOGI("test 1");
        if(_frames == _nImages) {
			//Take the next Frame and copy it to the visual output frame "imgText"
			//_capture >> img;
			//img.copyTo(imgText);

			//Compute matrices
			cv::calibrateCamera(_objectPoints, _imagePoints, _capture.size(), _intrinsicsMatrix,
								_distortionCoeffs, _rvecs, _tvecs);
			_initialisation = true;
			//cv::destroyWindow("Grab_Frames");
			LOGI("_initialisation:%d", _initialisation);
		}
		//LOGI("test 2");

	}

	void CameraCalibration::setObjectPoints(std::vector<std::vector<cv::Point3f>> l_objectPoints) {

		_objectPoints = l_objectPoints;
		_objectPointsSet = true;
	}

	bool CameraCalibration::getInitialisation() {
		return _initialisation;
	}

	cv::Mat CameraCalibration::getIntrinsicsMatrix() const {
		return _intrinsicsMatrix;
	}

	cv::Mat CameraCalibration::getDistortionCoeffs() const {
		return _distortionCoeffs;
	}

	std::vector<cv::Mat> CameraCalibration::getRotationVectors() {
		return _rvecs;
	}

	std::vector<cv::Mat> CameraCalibration::getTranslationVectors() {
		return _tvecs;
	}
}