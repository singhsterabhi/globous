#include "opencv2/opencv.hpp"
#include "iostream"
#include<string>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

using namespace cv;
using namespace std;

void GammaCorrection(Mat& src, Mat& dst, float fGamma);

int main(int, char**)
{
    VideoCapture cap ( "live_clahe.avi" ); // open the default camera
    if( ! cap.isOpened () )  // check if we succeeded
        return -1;

    /* Mat edges; */
   // namedWindow ( "tree" , 1 );
    double frnb ( cap.get ( CV_CAP_PROP_FRAME_COUNT ) );
    double fps (cap.get(CV_CAP_PROP_FPS));
    std::cout << "frame count = " << frnb << endl;
    int i;
    int codec = CV_FOURCC('M', 'J', 'P', 'G');
    Mat src;
    cap>>src;
    if (src.empty()) {
        cerr << "ERROR! blank frame grabbed\n";
        return -1;
    }
    bool isColor = (src.type() == CV_8UC3);
    //--- INITIALIZE VIDEOWRITER
    VideoWriter writer;
    string filename = "./live.avi";             // name of the output video file
    writer.open(filename, codec, fps, src.size(), isColor);
    // check if we succeeded
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }
    for(i=1;i<=frnb;i++) {
      Mat frame;
        if (!cap.read(src)) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        Mat dst(src.rows, src.cols, src.type());
        GammaCorrection(src, dst,2);
        writer.write(dst);
        cout << i <<endl;
       // imshow("Live", src);
        if (waitKey(5) >= 0)
            break;
    }

    return 0;
}


void GammaCorrection(Mat& src, Mat& dst, float fGamma)
{
	CV_Assert(src.data);

	// accept only char type matrices
	CV_Assert(src.depth() != sizeof(uchar));

	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

			  MatIterator_<uchar> it, end;
			  for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
				  //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
				  *it = lut[(*it)];

			  break;
	}
	case 3:
	{

			  MatIterator_<Vec3b> it, end;
			  for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
			  {

				  (*it)[0] = lut[((*it)[0])];
				  (*it)[1] = lut[((*it)[1])];
				  (*it)[2] = lut[((*it)[2])];
			  }

			  break;

	}
	}
}
