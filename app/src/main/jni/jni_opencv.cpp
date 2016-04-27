#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <android/log.h>
#include "myCVClasses.hpp"

#define LOG_TAG "Arc"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace std;
using namespace cv;
using namespace std;

#define CAP_WIDTH  640
#define CAP_HEIGHT 480



Size boardSize( 9, 6 );
Size imageSize(CAP_WIDTH, CAP_HEIGHT);


std::vector<cv::Point2f> corners;								//Vector which will contain the corner coordinates for each camera frame
std::vector<cv::Point3f> _3DPoints;								//Vector that contains the 3D coordinates for each chessboard corner
myCV::CameraCalibration *camCalib;

int nImages = 2;
bool done = false;

extern "C"
{

    JNIEXPORT void JNICALL
    Java_org_opencv_samples_cameracalibration_CameraCalibrationActivity_Init(JNIEnv * , jobject)
    {
        //Initialising the 3D-Points for the chessboard
        float rot = 0.0f;
        float a = 0.2f;								//The widht/height of each square of the chessboard object
        Point3f _3DPoint;
        float y = (((boardSize.height-1.0f)/2.0f)*a)+(a/2.0f);
        float x = 0.0f;
        for(int h = 0; h < boardSize.height; h++, y+=a){
            x = (((boardSize.width-2.0f)/2.0f)*(-a))-(a/2.0f);
            for(int w = 0; w < boardSize.width; w++, x+=a){
                _3DPoint.x = x;
                _3DPoint.y = y;
                _3DPoint.z = 0.0f;
                _3DPoints.push_back(_3DPoint);
            }
        }
        camCalib = new myCV::CameraCalibration(boardSize.width, boardSize.height, nImages);
        //camCalib(boardSize.width, boardSize.height, nImages);		//create an object which handels the camera calibration
        camCalib->Initialisation();
    }

    JNIEXPORT void JNICALL
    Java_org_opencv_samples_cameracalibration_CameraCalibrationActivity_Segment(JNIEnv * , jobject , jlong addrGray, jlong addrRgba )
    {
        Mat& mGr  = *(Mat*)addrGray;
        Mat& mRgb = *(Mat*)addrRgba;

        //The size of the chessboard which should be searched during the camera calibration plus
        //the number of images which should be taken

        bool found = false;								//Will be used to determine wether a chessboard has been
        //found in the camera frame or not
        cv::Mat R(3,1,CV_64F), T(3,1,CV_64F);							//Declaration of rotation and translation Vector
        if(!camCalib->getInitialisation()) {
            camCalib->GrabFrames(mGr, mRgb);
        }
        else {

            corners.clear();

            cv::Mat M = camCalib->getIntrinsicsMatrix();
            cv::Mat D = camCalib->getDistortionCoeffs();


            //Search for a chessboard in the camera frame and save his corners
            found = cv::findChessboardCorners(mGr, boardSize, corners);
            //LOGI("found:%d", found);
            if(found)
            { //A chessboard has been found
                cv::cornerSubPix(mGr, corners, cv::Size(5, 5), cv::Size(-1,-1), cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
                cv::drawChessboardCorners(mRgb, boardSize, cv::Mat(corners), found);	//Draw the chessboard corners on the camera frame

                cv::solvePnP(cv::Mat(_3DPoints), cv::Mat(corners), M, D, R, T);		//Calculate the Rotation and Translation vector
                double theta = 0;

                //Calculate the angle of the rotation
                theta = sqrt((R.at<double>(0,0)*R.at<double>(0,0))+
                     (R.at<double>(1,0)*R.at<double>(1,0))+
                     (R.at<double>(2,0)*R.at<double>(2,0)));
                corners.clear();

                LOGI("theta:%d, R1: %d, R2: %d, R3: %d", (theta*180.0f)/3.14159f, R.at<double>(0,0), R.at<double>(1,0), R.at<double>(2,0));
                LOGI("T1: %d, T2: %d, T3: %d",  -T.at<double>(0,0) , (-T.at<double>(1,0)-1.0), (-T.at<double>(2,0)+5.0));

            }
        }
    }

}

