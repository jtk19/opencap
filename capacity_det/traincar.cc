#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <unistd.h>


#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>

#include "traincar.h"


#define DATA_DIR		"/home/data/train/"
#define MODEL_DIR		"/usr/local/opt/opencv/data/haarcascades/"


using namespace std;
using namespace cv;



string original_win = "Original";
string face_detect_win = "Face Detect";

string face_cascade_model = string(MODEL_DIR) + "haarcascade_frontalface_alt.xml";
string profile_cascade_model = string(MODEL_DIR) + "haarcascade_profileface.xml";
string body_cascade_model = string(MODEL_DIR) + "haarcascade_upperbody.xml";

CascadeClassifier face_cascade;
CascadeClassifier profile_cascade;
CascadeClassifier body_cascade;

string image_name[] = { "train1.jpg", "train2.jpg"};
size_t numdir = 2;



bool faceDetect( cv::Mat *frame );


	
int main( int argc, char *argv[] )
{
	cout<< "Welcome to Train Crowds."<< endl;

	namedWindow( original_win, WINDOW_NORMAL | WINDOW_KEEPRATIO );
	namedWindow( face_detect_win, WINDOW_NORMAL | WINDOW_KEEPRATIO );
	cv::moveWindow( original_win, 1260, 0);
	cv::resizeWindow( original_win, 640, 910 );
	cv::moveWindow( face_detect_win, 600, 0);
	cv::resizeWindow( face_detect_win, 640, 910 );
	cout<< "Displays initialised."<< endl;


	// Load the cascade model for face detection.
	if( !face_cascade.load( face_cascade_model ) )
	{
		cerr<< "--(!)Error loading face model: "<< face_cascade_model<< endl;
		return -1;
	}
	else
	{
		cout<< "Face cascade model loaded: "<< face_cascade_model<< endl;
	}
	// Load the cascade model for eyes detection.
	if ( !profile_cascade.load( profile_cascade_model ) )
	{
		cerr<< "--(!)Error loading profile model: "<< profile_cascade_model<< endl;
		return -1;
	}
	else
	{
		cout<< "Profile cascade model loaded: "<< profile_cascade_model<< endl;
	}
	// Load the cascade model for full-body detection.
	if ( !body_cascade.load( body_cascade_model ) )
	{
		cerr<< "--(!)Error loading body model: "<< body_cascade_model<< endl;
		return -1;
	}
	else
	{
		cout<< "Body cascade model loaded: "<< body_cascade_model<< endl;
	}


	cout<< "Running detection for train images."<< endl;

	for ( size_t i = 0; i < numdir; ++i )
	{
		string file = string( DATA_DIR ) + image_name[i];
		Mat img = imread( file.c_str() );

		imshow( original_win, img);
		waitKey(1);

		faceDetect( &img );

	}

	return 0;


}



// Assumption: There is only one model in the frame.
bool faceDetect( cv::Mat *frame )
{
	vector<Rect> faces, profiles, bodies;
	Mat	frame_gray, frame_gray1;

	cvtColor( *frame, frame_gray, CV_RGB2GRAY );
	//equalizeHist( frame_gray1, frame_gray );
	//frame_gray1.release();
	Mat facem = frame->clone();


	// Detect face.
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size( 6, 6) );
	for( size_t i = 0; i < faces.size(); i++ )
	{
		//Point fcentre( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );

		rectangle( facem, faces[i], Scalar( 255, 0, 0 ), 2 );
	}
	cout<< faces.size()<< " faces detected."<< endl;

	/*
	// Detect profile faces.
	profile_cascade.detectMultiScale( frame_gray, profiles, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size( 24, 30) );
	for( size_t i = 0; i < profiles.size(); i++ )
	{
		//Point fcentre( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );

		rectangle( facem, profiles[i], Scalar( 0, 255, 0 ), 2 );
	}

	cout<< profiles.size()<< " profiles detected."<< endl;


	// detect body
	//body_cascade_model.detectMultiScale( frame_gray, bodies, 1.1, 2, 18|9, Size(10,20));
	body_cascade.detectMultiScale( frame_gray, bodies, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size( 8, 8) );
	for( size_t i = 0; i < bodies.size(); i++ )
	{
		//Point fcentre( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );

		rectangle( facem, bodies[i], Scalar( 0, 0, 255 ), 2 );
	}
	*/

	static int count = 1;
	stringstream ss;
	ss << DATA_DIR<< "train_res_"<< count++<< ".jpg";
	imwrite( ss.str(), facem );

	imshow( face_detect_win, facem );
	waitKey(0);


	return true;

}
