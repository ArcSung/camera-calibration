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



Size boardSize( 7, 5 );
Size imageSize(CAP_WIDTH, CAP_HEIGHT);
const int CUBE_SIZE = 5;
const int CHESS_SIZE = 25;


std::vector<cv::Point2f> corners;								//Vector which will contain the corner coordinates for each camera frame
std::vector<cv::Point3f> _3DPoints;								//Vector that contains the 3D coordinates for each chessboard corner
vector<Point3f> objectCorners, contoursFindedObjectPoints;
vector< vector<Point2f> > contoursFinded;
myCV::CameraCalibration *camCalib;
Mat rvecs, tvecs;
cv::Mat cameraMatrix;
cv::Mat distCoeffs;
Point sPoint, tPoint;

int nImages = 5;
bool done = false;

extern "C"
{

    class Cube{
        public :
            vector<Point3f> srcPoints3D;
            vector<Point2f> dstPoints2D;
        Cube(){
            LOGI("cube init starts");
            float CUBExCHESS = CUBE_SIZE * CHESS_SIZE;
            for(int i=0;i<8;i++){
                switch(i){
                    case 0:
                        srcPoints3D.push_back(Point3f(0,0,0));
                    break;
                    case 1:
                        srcPoints3D.push_back(Point3f(CUBExCHESS,0,0));
                    break;
                    case 2:
                        srcPoints3D.push_back(Point3f(CUBExCHESS,CUBExCHESS,0));
                    break;
                    case 3:
                        srcPoints3D.push_back(Point3f(0,CUBExCHESS,0));
                    break;
                    case 4:
                        srcPoints3D.push_back(Point3f(0,0,CUBExCHESS));
                    break;
                    case 5:
                        srcPoints3D.push_back(Point3f(CUBExCHESS,0,CUBExCHESS));
                    break;
                    case 6:
                        srcPoints3D.push_back(Point3f(CUBExCHESS,CUBExCHESS,CUBExCHESS));
                    break;
                    case 7:
                        srcPoints3D.push_back(Point3f(0,CUBExCHESS,CUBExCHESS));
                    break;
                    default:
                    break;
            }
        }
        LOGI("cube init ends");
    }
};

inline double dist(Point a, Point b){
        return sqrt( (a.x-b.x) * (a.x-b.x) + (a.y-b.y) * (a.y-b.y) );
    }

    String int2str(int i) {
         char* s;
         sprintf(s,"%d",i);

        return (string)s;
    }

    void drawMarkerContours(Mat image, Mat mgray){
        LOGI("drawMarkerContours starts");
        cv::Mat bin_img;
        cv::threshold(mgray, bin_img, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(bin_img,
            contours,
            hierarchy,
            cv::RETR_TREE,
            cv::CHAIN_APPROX_SIMPLE);

        vector< vector< Point> >::iterator itc = contours.begin();
        for(;itc!=contours.end();){  //remove some contours size <= 100
            if(itc->size() <= 100)
                itc = contours.erase(itc);
            else
                ++itc;
        }


        LOGI("approx poly");
        vector< vector< Point> > contours_poly( contours.size() );
        for( int i = 0; i < contours.size(); i++ )
        {
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true ); // let contiurs more smooth
        }

        itc = contours_poly.begin();
        for(;itc!=contours_poly.end();){ // if contour has not four points, then we remove it
            if(itc->size() != 4)
                itc = contours_poly.erase(itc);
            else
                ++itc;
        }


        LOGI("resize contours_poly");
        int csize = contours_poly.size();
        vector< Point > centers;
        vector< bool > isValid;
        for(int i=0;i<csize;i++){
            double x = 0, y = 0;
            for(int j=0;j<4;j++){
                x += contours_poly[i][j].x;
                y += contours_poly[i][j].y;
            }
            centers.push_back( Point( x/4, y/4 ) );
        }

        for(int i=0;i<csize;i++){
            bool ok = false;
            for(int j=0;j<csize;j++){
                if(i == j) continue;
                if(dist(centers[i], centers[j]) < 20){
                    ok = true;
                    break;
                }
            }
            if(ok) isValid.push_back(true);
            else isValid.push_back(false);
        }

        itc = contours_poly.begin();
        for(int i=0;itc!=contours_poly.end();++i){
            if(!isValid[i])
                itc = contours_poly.erase(itc);
            else
                ++itc;
        }

        contoursFinded.clear();

        for(int i=0;i<contours_poly.size();i++){
            vector< Point2f > temp;
            for(int j=0;j<contours_poly[i].size();j++){
                temp.push_back( Point2f(contours_poly[i][j].x, contours_poly[i][j].y) );
                LOGI("contours_poly[%d][%d] = (%d, %d)", i, j, contours_poly[i][j].x, contours_poly[i][j].y);
                circle(image, cvPoint(contours_poly[i][j].x, contours_poly[i][j].y), 3, CV_RGB(0, 0, 255), 3, CV_AA);
            }
            contoursFinded.push_back(temp);
        }

        LOGI("draw contour");
        /*cv::drawContours(image,
            contours_poly,
            -1,
            Scalar(128,128,0,255),
            2);*/

        bin_img.release();
        LOGI("drawMarkerContours ends");
    }


    JNIEXPORT void JNICALL
    Java_org_opencv_samples_cameracalibration_CameraCalibrationActivity_Init(JNIEnv * , jobject)
    {
        //Initialising the 3D-Points for the chessboard
        float rot = 0.0f;
        float a = 0.2f;								//The widht/height of each square of the chessboard object
        Point3f _3DPoint;
        float y = (((boardSize.height-1.0f)/2.0f)*a)+(a/2.0f);
        float x = 0.0f;
        /*for(int h = 0; h < boardSize.height; h++, y+=a){
            x = (((boardSize.width-2.0f)/2.0f)*(-a))-(a/2.0f);
            for(int w = 0; w < boardSize.width; w++, x+=a){
                _3DPoint.x = x;
                _3DPoint.y = y;
                _3DPoint.z = 0.0f;
                _3DPoints.push_back(_3DPoint);
            }
        }*/
        camCalib = new myCV::CameraCalibration(boardSize.width, boardSize.height, nImages);
        //camCalib(boardSize.width, boardSize.height, nImages);		//create an object which handels the camera calibration
        camCalib->Initialisation();

        int x2 = 100;

        contoursFindedObjectPoints.push_back(
                Point3f(0, 0, 0.0f));
        contoursFindedObjectPoints.push_back(
                Point3f(0, x2, 0.0f));
        contoursFindedObjectPoints.push_back(
                Point3f(x2, x2, 0.0f));
        contoursFindedObjectPoints.push_back(
                Point3f(x2, 0, 0.0f));

        FileStorage fs2("/sdcard/CameraCalib.xml", FileStorage::READ);

// first method: use (type) operator on FileNode.
        int frameCount = (int)fs2["frameCount"];

        std::string date;
// second method: use FileNode::operator >>
        fs2["calibrationDate"] >> date;

        fs2["cameraMatrix"] >> cameraMatrix;
        fs2["distCoeffs"] >> distCoeffs;
        fs2.release();

        if(frameCount == nImages)
        {
            done = true;
        }
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
        if(!done) {
            camCalib->GrabFrames(mGr, mRgb);

            FileStorage fs("/sdcard/CameraCalib.xml", FileStorage::WRITE);

            fs << "frameCount" << nImages;
            time_t rawtime; time(&rawtime);
            fs << "calibrationDate" << asctime(localtime(&rawtime));
            fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
            fs << "features" << "[";
            for( int i = 0; i < 3; i++ )
            {
                int x = rand() % 640;
                int y = rand() % 480;
                uchar lbp = rand() % 256;

                fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
                for( int j = 0; j < 8; j++ )
                    fs << ((lbp >> j) & 1);
                fs << "]" << "}";
            }
            fs << "]";
            fs.release();

            done = camCalib->getInitialisation();

        }
        else {

            Cube mCube = Cube();
            drawMarkerContours(mRgb, mGr);

            if(contoursFinded.size() > 0) {

                Mat m1(contoursFinded[0]);
                Mat m2(contoursFindedObjectPoints);


                LOGI("src size: %d", contoursFinded[0].size());
                LOGI("obj size: %d", contoursFindedObjectPoints.size());

                //contoursFinded[0].

                LOGI("solvePnP");
                cv::solvePnP(m2,
                             m1,
                             cameraMatrix,
                             distCoeffs,
                             rvecs,
                             tvecs);


                LOGI("projectPoints");
                projectPoints(mCube.srcPoints3D, rvecs, tvecs, cameraMatrix, distCoeffs, mCube.dstPoints2D);

                /*LOGI("points::::::::::::");
                for(int i=0;i<8;i++){
                    LOGI("%f,%f,%f --> %f,%f",
                         mCube.srcPoints3D[i].x, mCube.srcPoints3D[i].y, mCube.srcPoints3D[i].z,
                         mCube.dstPoints2D[i].x, mCube.dstPoints2D[i].y);
                }*/

                sPoint = mCube.dstPoints2D[0];
                tPoint = mCube.dstPoints2D[1];
                cv::line(mRgb, sPoint, tPoint, Scalar(0,0,255,255),5,8,0);

                sPoint = mCube.dstPoints2D[0];
                tPoint = mCube.dstPoints2D[4];
                cv::line(mRgb, sPoint, tPoint, Scalar(255,0,0,255),5,8,0);

                sPoint = mCube.dstPoints2D[0];
                tPoint = mCube.dstPoints2D[3];
                cv::line(mRgb, sPoint, tPoint, Scalar(0,255,0,255),5,8,0);

                /*sPoint = Point((mCube.dstPoints2D[0].x+ mCube.dstPoints2D[2].x)/2, (mCube.dstPoints2D[0].y+ mCube.dstPoints2D[2].y)/2);
                tPoint = Point((mCube.dstPoints2D[0].x+ mCube.dstPoints2D[1].x)/2, (mCube.dstPoints2D[0].y+ mCube.dstPoints2D[1].y)/2);
                cv::line(mRgb, sPoint, tPoint, Scalar(255,0,0,255),5,8,0);

                tPoint = Point((mCube.dstPoints2D[0].x+ mCube.dstPoints2D[3].x)/2, (mCube.dstPoints2D[0].y+ mCube.dstPoints2D[3].y)/2);
                cv::line(mRgb, sPoint, tPoint, Scalar(0,255,0,255),5,8,0);

                tPoint = Point((mCube.dstPoints2D[4].x+ mCube.dstPoints2D[6].x)/2, (mCube.dstPoints2D[4].y+ mCube.dstPoints2D[6].y)/2);
                cv::line(mRgb, sPoint, tPoint, Scalar(0,0,255,255),5,8,0);*/

                LOGI("calcue roll, pitch yaw");

                Mat rmat;
                char str[200];
                Rodrigues(rvecs,rmat);
                double roll, pitch, yaw;

                roll = atan2(rmat.at<double>(1,0),rmat.at<double>(0,0));
                pitch = -asin(rmat.at<double>(2,0));
                yaw = atan2(rmat.at<double>(2,1),rmat.at<double>(2,2));

                sprintf(str,"roll: %d ; pitch: %d ; yaw: %d", (int)(roll*180/3.1415), (int)(pitch*180/3.1415), (int)(yaw*180/3.1415));

                //LOGI("roll: %d ; pitch: %d ; yaw: %d", (int)(roll*180/3.1415), (int)(pitch*180/3.1415), (int)(yaw*180/3.1415));
                putText(mRgb, str,
                        Point(10,mRgb.rows - 40), FONT_HERSHEY_PLAIN, 2, CV_RGB(0,255,0));

                m1.release();
                m2.release();
                rmat.release();

            }
            LOGI("drawProcessing ends");


        }
    }

                /*corners.clear();

                cv::Mat M = camCalib->getIntrinsicsMatrix();
                cv::Mat D = camCalib->getDistortionCoeffs();


                //Search for a chessboard in the camera frame and save his corners
                found = cv::findChessboardCorners(mGr, boardSize, corners);
                LOGI("found:%d", found);
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
                   // LOGI("found a");

                    //Vec3d eulerAngles;
                    //getEulerAngles(R,eulerAngles);
                  //  LOGI("found b");

                    LOGI("theta:%d, R1: %d, R2: %d, R3: %d", (theta*180.0f)/3.14159f, R.at<double>(0,0), R.at<double>(1,0), R.at<double>(2,0));
                    //LOGI("T1: %d, T2: %d, T3: %d",  -T.at<double>(0,0) , (-T.at<double>(1,0)-1.0), (-T.at<double>(2,0)+5.0));
                   // LOGI("yaw: %d, pitch: %d, roll: %d",  eulerAngles[1] , eulerAngles[0], eulerAngles[2]);

                }*/

}

