#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
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

vector<Point3f> contoursFindedObjectPoints;
vector< vector<Point2f> > contoursFinded;
myCV::CameraCalibration *camCalib;
Mat rvecs, tvecs;
cv::Mat cameraMatrix;
cv::Mat distCoeffs;
Point sPoint, tPoint;
int tlx, tly, brx, bry;
Point2f LL1, LL2, LL3, LL4;

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
            int minsum = 10000, maxsum = 0, mindiff = 10000, maxdiff = 10000;
            Point2f L1, L2, L3, L4;
            int tl1, br1;

            for(int j=0;j<contours_poly[i].size();j++){
                //temp.push_back( Point2f(contours_poly[i][j].x, contours_poly[i][j].y) );
                //circle(image, cvPoint(contours_poly[i][j].x, contours_poly[i][j].y), 3, CV_RGB(0, 0, 255), 3, CV_AA);
                if(contours_poly[i][j].x+ contours_poly[i][j].y < minsum)
                {
                    minsum = contours_poly[i][j].x+ contours_poly[i][j].y;
                    L1 = Point2f(contours_poly[i][j].x, contours_poly[i][j].y);
                    tl1 = j;
                }

                if(contours_poly[i][j].x+ contours_poly[i][j].y > maxsum)
                {
                    maxsum = contours_poly[i][j].x+ contours_poly[i][j].y;
                    L3 = Point2f(contours_poly[i][j].x, contours_poly[i][j].y);
                    br1 = j;
                }
            }

            for(int j=0;j<contours_poly[i].size();j++){

                if(j == br1 || j == tl1)
                    continue;
                if(contours_poly[i][j].x < mindiff)
                {
                    mindiff = contours_poly[i][j].x;
                    L4 = Point2f(contours_poly[i][j].x, contours_poly[i][j].y);
                }

                if(contours_poly[i][j].y < maxdiff)
                {
                    maxdiff = contours_poly[i][j].y;
                    L2 = Point2f(contours_poly[i][j].x, contours_poly[i][j].y);
                }
            }

            tlx = (L1.x < L4.x) ? L1.x : L4.x;
            tly = (L1.y < L2.y) ? L1.y : L2.y;
            brx = (L3.x > L2.x) ? L3.x : L2.x;
            bry = (L3.y > L4.y) ? L3.y : L4.y;

            LL1 = L1;
            LL2 = L2;
            LL3 = L3;
            LL4 = L4;

            temp.push_back(L1);
            temp.push_back(L2);
            temp.push_back(L3);
            temp.push_back(L4);
            circle(image, L1, 3, CV_RGB(0, 0, 255), 3, CV_AA);
            circle(image, L2, 3, CV_RGB(0, 0, 255), 3, CV_AA);
            circle(image, L3, 3, CV_RGB(0, 0, 255), 3, CV_AA);
            circle(image, L4, 3, CV_RGB(0, 0, 255), 3, CV_AA);
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
        camCalib = new myCV::CameraCalibration(boardSize.width, boardSize.height, nImages);
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

                int ROIW = brx - tlx - 1;
                int ROIH = bry - tly - 1;

                Mat SrcROI = mRgb(Rect(tlx, tly, ROIW, ROIH));

                // 設定變換[之前]與[之後]的坐標 (左上,左下,右下,右上)
                const int nOffset=200;
                cv::Point2f pts1[] = {LL1,LL4,LL3,LL2};
                cv::Point2f pts2[] = {cv::Point2f(0,0),cv::Point2f(0,ROIH),cv::Point2f(ROIW,ROIH),cv::Point2f(ROIW,0)};

                cv::Mat perspective_matrix = cv::getPerspectiveTransform(pts1, pts2);
                cv::Mat dst_img;
                // 變換
                cv::warpPerspective(mRgb, dst_img, perspective_matrix, Size(ROIW, ROIH), cv::INTER_LINEAR);


                LOGI("src size: %d", contoursFinded[0].size());
                LOGI("obj size: %d", contoursFindedObjectPoints.size());


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

                if(ROIW < mRgb.rows/2 && ROIH < mRgb.cols/2) {
                    LOGI("ROI %d, %d", ROIW, ROIH);
                    dst_img.copyTo(mRgb(Rect(0, 0, ROIW, ROIH)));
                }

                m1.release();
                m2.release();
                rmat.release();
                SrcROI.release();

            }
            LOGI("drawProcessing ends");


        }
    }

}

