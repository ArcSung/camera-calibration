package org.opencv.samples.cameracalibration;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs; // imread, imwrite, etc

import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Array;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

public class CameraCalibrationActivity extends Activity implements CvCameraViewListener2 {
    private static final String    TAG = "OCVSample::Activity";

    static{if (!OpenCVLoader.initDebug()) {
        // Handle initialization error
        Log.i("opencv", "opecv init fail");
    }
    else {
        Log.i("opencv", "opecv init successful");
        //System.loadLibrary("mixed_sample"); // load other native libraries
    }}


    private Mat mRgba;
    private Mat mGray;


    private List<Camera.Size>      mResolutionList;
    private CameraView             mOpenCvCameraView;


    ImageButton ImageButton_TakePicture;
    ImageButton ImageButton_Cancel;
    ImageButton ImageButton_Check;
    ImageButton ImageButton_Gralloc;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("mixed_sample");

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public CameraCalibrationActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_main);

        //Camera UI
        mOpenCvCameraView = (CameraView) findViewById(R.id.mainactivity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        ImageButton_TakePicture = (ImageButton)findViewById(R.id.camera_takepicture);
        ImageButton_Cancel      = (ImageButton)findViewById(R.id.camera_cancel);
        ImageButton_Check       = (ImageButton)findViewById(R.id.camera_check);
        ImageButton_Gralloc     = (ImageButton)findViewById(R.id.camera_gralloc);

        ImageButton_TakePicture.setOnClickListener(ImageButtonlistener);
        ImageButton_Cancel.setOnClickListener(ImageButtonlistener);
        ImageButton_Check.setOnClickListener(ImageButtonlistener);
        ImageButton_Gralloc.setOnClickListener(ImageButtonlistener);

        ImageButton_Check.setVisibility(View.GONE);
        ImageButton_Cancel.setVisibility(View.GONE);

        Init();

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        //OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);

        Camera.Size resolution = null;
        int id = 0;
        mResolutionList = mOpenCvCameraView.getResolutionList();
        for(; id <= mResolutionList.size(); id++)
        {
            resolution = mResolutionList.get(id);
            if(resolution.width == 640 && resolution.height == 480)
                break;
        }

        if(id > mResolutionList.size())
            resolution = mResolutionList.get(0);

        mOpenCvCameraView.setResolution(resolution);
        resolution = mOpenCvCameraView.getResolution();
        String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
        Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
        Log.i(TAG, "resolution" + Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString());
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Segment(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr());

        return mRgba;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        return true;
    }


    private void Inttobitmap (int pixels[]) {
        Bitmap bitmap = Bitmap.createBitmap(140, 40, Bitmap.Config.ARGB_8888);
        //Log.i("ArcSung", "Inttobitmap start");
        int tempvalue;
        String test = new String();
        for(int z=0; z<=6; z++)
            for(int i=0; i < 40; i++)
            {
                test = "";
                for(int j=0; j < 20; j++)
                {
                    //if(j%2 == 0) {
                    tempvalue = 0;
                    tempvalue += (pixels[j + i*20+800*z] << 16);
                    tempvalue += (pixels[j + i*20+800*z] << 8);
                    tempvalue += (pixels[j + i*20+800*z]);
                    //if((pixels[j + i*140]) > 0)
                    //    test +='1';
                    //else
                    //    test +='0';

                    //}else
                    //{
                    //    tempvalue = 0x0000FFFF;
                    //}
                    bitmap.setPixel(j+z*20, i, tempvalue|0xFF000000);
                }
                Log.i("ArcSung", test);
            }
        Savebitmap(bitmap);
    }

    private void Inttomat (int pixels[]) {
        Mat testmat = new Mat(20, 40, CvType.CV_8UC1);
        int tempvalue;
        for(int i=0; i < 20; i++)
        {
            for(int j=0; j < 40; j++)
            {
                testmat.put(j, i, pixels[j + i*j]);
            }
        }
        SaveImage(testmat);
    }

    private void Savebitmap (Bitmap bitmap) {

        Log.i("ArcSung", "Savebitmap start");
        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        String filename = currentDateandTime+".jpg";
        File file = new File(path, filename);

        try {
            FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 90, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        /*if (bool == true)
            Toast.makeText(this, "Save Success", Toast.LENGTH_SHORT).show();
        else
            Toast.makeText(this, "Save fail", Toast.LENGTH_SHORT).show();*/
    }

    private void SaveImage (Mat mat) {
        Mat mIntermediateMat = new Mat();

        Imgproc.cvtColor(mat, mIntermediateMat, Imgproc.COLOR_RGB2BGR, 3);

        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        String filename = currentDateandTime+".jpg";
        File file = new File(path, filename);

        Boolean bool = null;
        filename = file.toString();
        bool = Imgcodecs.imwrite(filename, mIntermediateMat);

        if (bool == true)
            Toast.makeText(this, "Save Success", Toast.LENGTH_SHORT).show();
        else
            Toast.makeText(this, "Save fail", Toast.LENGTH_SHORT).show();
    }

    private View.OnClickListener ImageButtonlistener =new View.OnClickListener(){

        @Override
        public void onClick(View v) {
            // TODO Auto-generated method stub
            switch(v.getId()){
                case R.id.camera_takepicture:
                    SaveImage(mRgba);
                    //ImageButton_Cancel.setVisibility(View.VISIBLE);
                    //ImageButton_Check.setVisibility(View.VISIBLE);
                    //ImageButton_Gralloc.setVisibility(View.GONE);
                    //ImageButton_TakePicture.setVisibility(View.GONE);
                    //Camera_button_check = true;
                    break;

                /*case R.id.camera_cancel:
                    mOpenCvCameraView.onButtonCheck(Camera_start_preview);
                    ImageButton_Cancel.setVisibility(View.GONE);
                    ImageButton_Check.setVisibility(View.VISIBLE);
                    ImageButton_Gralloc.setVisibility(View.VISIBLE);
                    ImageButton_TakePicture.setVisibility(View.VISIBLE);
                    Camera_button_check = false;
                    break;
                case R.id.camera_check:
                    mOpenCvCameraView.onButtonCheck(Camera_save_picture);
                    mOpenCvCameraView.onButtonCheck(Camera_start_preview);
                    SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
                    String currentDateandTime = sdf.format(new Date());
                    String fileName = Environment.getExternalStorageDirectory().getPath() +
                            "/sample_picture_" + currentDateandTime + ".jpg";
                    mOpenCvCameraView.takePicture(fileName);
                    ImageButton_Cancel.setVisibility(View.GONE);
                    ImageButton_Check.setVisibility(View.GONE);
                    ImageButton_Gralloc.setVisibility(View.VISIBLE);
                    ImageButton_TakePicture.setVisibility(View.VISIBLE);
                    Camera_button_check = false;
                    break;*/


                case R.id.camera_gralloc:
                    //cameraView.onButtonCheck(Camera_stop_preview);
                    //Intent intent = new Intent();
                    //intent.setClass(CameraPreviewActivity.this,GridviewActivity.class);
                    //startActivityForResult(intent, 0);
                    break;
            }
        }
    };

    public static boolean createDirIfNotExists(String path) {
        boolean ret = true;

        File file = new File(Environment.getExternalStorageDirectory(), path);
        if (!file.exists()) {
            if (!file.mkdirs()) {
                Log.e("TravellerLog :: ", "Problem creating Image folder");
                ret = false;
            }
        }
        return ret;
    }

    @Override
    public void onBackPressed() {
        //Display alert message when back button has been pressed
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        backButtonHandler();
        return;
    }

    public void backButtonHandler() {
        AlertDialog.Builder alertDialog = new AlertDialog.Builder(
                CameraCalibrationActivity.this);
        // Setting Dialog Title
        alertDialog.setTitle("Leave application?");
        // Setting Dialog Message
        alertDialog.setMessage("Are you sure you want to leave the application?");
        // Setting Icon to Dialog
        // Setting Positive "Yes" Button
        alertDialog.setPositiveButton("YES",
                new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        finish();
                    }
                });
        // Setting Negative "NO" Button
        alertDialog.setNegativeButton("NO",
                new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        // Write your code here to invoke NO event
                        if (mOpenCvCameraView != null)
                            mOpenCvCameraView.enableView();
                        dialog.cancel();
                    }
                });
        // Showing Alert Message
        alertDialog.show();
    }

    public native void  Segment(long matAddrGr, long matAddrRgba);
    public native void  Init();
}
