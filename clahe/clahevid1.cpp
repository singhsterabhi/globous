#include "opencv2/opencv.hpp"
#include "iostream"
#include<string>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

using namespace cv;
using namespace std;



int main(int, char**)
{
    cout << "start\n";
    VideoCapture cap ( "vid.MP4" ); // open the default camera
    
    cout << cap.isOpened(); // print 1 if ok
	
	if( ! cap.isOpened() )  // check if we succeeded
	{
	   cout<<"-1\n";
	return -1;
	}

	double frnb ( cap.get ( CV_CAP_PROP_FRAME_COUNT ) );
    double fps (cap.get(CV_CAP_PROP_FPS));

    std::cout << "frame count = " << frnb << endl;

    int i;
    int codec = CV_FOURCC('H', '2', '6', '4');
    Mat src;


    cap>>src;
    if (src.empty()) {
        cerr << "ERROR! blank frame grabbed\n";
        return -1;
    }
    

    bool isColor = (src.type() == CV_8UC3);

    //--- INITIALIZE VIDEOWRITER
    VideoWriter writer;
    string filename = "live_clahe1.avi";             // name of the output video file
    cout << filename ;
    cout << "\n";
    writer.open(filename, codec, fps, src.size(), isColor);

    // check if we succeeded
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    for(i=1;i<=frnb;i++) {
        // Mat frame;
        // cout << cap.read(src);
        // cout << "\nhi\n";
        // if (!cap.read(src)) {
        //     cerr << "ERROR! blank frame grabbed\n";
        //     continue;
        // }
        cap>>src;
	    if (src.empty()) {
	        cerr << "ERROR! blank frame grabbed\n";
	        // return -1;
	        continue;
	    }

        // Mat dst(src.rows, src.cols, src.type());
        //GammaCorrection(src, dst,2);


        cv::Mat lab_image;
        cv::cvtColor(src, lab_image, CV_BGR2Lab);
        // Extract the L channel
        std::vector<cv::Mat> lab_planes(3);
        cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

        // apply the CLAHE algorithm to the L channel
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        cv::Mat dst;
        clahe->apply(lab_planes[0], dst);

        // Merge the the color planes back into an Lab image
        dst.copyTo(lab_planes[0]);
        cv::merge(lab_planes, lab_image);

        // convert back to RGB
        cv::Mat image_clahe;
        cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

        // display the results  (you might also want to see lab_planes[0] before and after)
        writer.write(image_clahe);
        cout << i <<endl;
        if (waitKey(5) == 0)
            cout << "\n breaking \n";
                //break;
    }

    return 0;
}
