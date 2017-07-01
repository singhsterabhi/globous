#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char ** argv){
    int t=0, emtfrm=0;
    VideoCapture cap("vid.MP4"); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    
    // Mat edges;
    namedWindow("edges",CV_WINDOW_NORMAL);
    while(true){
        t++;
        Mat frame;
        cap >> frame;
        // cout<<t<<" ";
        if (frame.empty()) {
            emtfrm++;
            cerr << "ERROR! blank frame grabbed\n";
            // break;
            if(emtfrm>10) break;
            continue;
        }
        emtfrm=0;
        // cvtColor(frame, edges, COLOR_BGR2GRAY);
        // GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        imshow("edges",frame);
        int k=waitKey(10);
        
        if ( k>= 0 && k!=255)
        {   
            cout<<k<<endl;
            break;
        }    
        }
    return 0;
}