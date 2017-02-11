#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SDL/SDL.h"
/*
 * A simple 'getting started' interface to the ARDrone, v0.2 
 * author: Tom Krajnik
 * The code is straightforward,
 * check out the CHeli class and main() to see 
 */
#include <stdlib.h>
#include "CHeli.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

// Here we will store points
vector<Point> points;
bool stop = false;
CRawImage *image;
CHeli *heli;
float pitch, roll, yaw, height;
int hover=0;
// Joystick related
SDL_Joystick* m_joystick;
bool useJoystick;
int joypadRoll, joypadPitch, joypadVerticalSpeed, joypadYaw;
bool navigatedWithJoystick, joypadTakeOff, joypadLand, joypadHover;
string ultimo = "init";

int Px, PMx;
int Py, PMy;
int vR, vMR;
int vG, vMG;
int vB, vMB;
int C1Px=0;
int C2Px=0;
int C3Px=0;
int vC1, vC2, vC3;

Mat imagenClick;
//Variable donde se almacenara la imagen congelada
Mat frozenImageBGR;
Mat frozenImageYIQ;
Mat frozenImageHSV;



IplImage* convertImageRGBtoYIQ(const IplImage *imageRGB)
{
    float fR, fG, fB;
    float fY, fI, fQ;
    const float FLOAT_TO_BYTE = 255.0f;
    const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;
    const float MIN_I = -0.5957f;
    const float MIN_Q = -0.5226f;
    const float Y_TO_BYTE = 255.0f;
    const float I_TO_BYTE = 255.0f / (MIN_I * -2.0f);
    const float Q_TO_BYTE = 255.0f / (MIN_Q * -2.0f);

    // Create a blank YIQ image
    IplImage *imageYIQ = cvCreateImage(cvGetSize(imageRGB), 8, 3);
    if (!imageYIQ || imageRGB->depth != 8 || imageRGB->nChannels != 3) {
        printf("ERROR in convertImageRGBtoYIQ()! Bad input image.\n");
        exit(1);
    }

    int h = imageRGB->height;           // Pixel height
    int w = imageRGB->width;            // Pixel width
    int rowSizeRGB = imageRGB->widthStep;       // Size of row in bytes, including extra padding.
    char *imRGB = imageRGB->imageData;      // Pointer to the start of the image pixels.
    int rowSizeYIQ = imageYIQ->widthStep;       // Size of row in bytes, including extra padding.
    char *imYIQ = imageYIQ->imageData;      // Pointer to the start of the image pixels.
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            // Get the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
            uchar *pRGB = (uchar*)(imRGB + y*rowSizeRGB + x*3);
            int bB = *(uchar*)(pRGB+0); // Blue component
            int bG = *(uchar*)(pRGB+1); // Green component
            int bR = *(uchar*)(pRGB+2); // Red component

            // Convert from 8-bit integers to floats
            fR = bR * BYTE_TO_FLOAT;
            fG = bG * BYTE_TO_FLOAT;
            fB = bB * BYTE_TO_FLOAT;
            // Convert from RGB to YIQ,
            // where R,G,B are 0-1, Y is 0-1, I is -0.5957 to +0.5957, Q is -0.5226 to +0.5226.
            fY =    0.299 * fR +    0.587 * fG +    0.114 * fB;
            fI = 0.595716 * fR - 0.274453 * fG - 0.321263 * fB;
            fQ = 0.211456 * fR - 0.522591 * fG + 0.311135 * fB;
            // Convert from floats to 8-bit integers
            int bY = (int)(0.5f + fY * Y_TO_BYTE);
            int bI = (int)(0.5f + (fI - MIN_I) * I_TO_BYTE);
            int bQ = (int)(0.5f + (fQ - MIN_Q) * Q_TO_BYTE);

            // Clip the values to make sure it fits within the 8bits.
            if (bY > 255)
                bY = 255;
            if (bY < 0)
                bY = 0;
            if (bI > 255)
                bI = 255;
            if (bI < 0)
                bI = 0;
            if (bQ > 255)
                bQ = 255;
            if (bQ < 0)
                bQ = 0;

            // Set the YIQ pixel components
            uchar *pYIQ = (uchar*)(imYIQ + y*rowSizeYIQ + x*3);
            *(pYIQ+0) = bY;     // Y component
            *(pYIQ+1) = bI;     // I component
            *(pYIQ+2) = bQ;     // Q component
        }
    }
    return imageYIQ;
}
/*
 * This method flips horizontally the sourceImage into destinationImage. Because it uses 
 * "Mat::at" method, its performance is low (redundant memory access searching for pixels).
 */
void flipImageBasic(const Mat &sourceImage, Mat &destinationImage)
{
    if (destinationImage.empty())
        destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

    for (int y = 0; y < sourceImage.rows; ++y)
        for (int x = 0; x < sourceImage.cols / 2; ++x)
            for (int i = 0; i < sourceImage.channels(); ++i)
            {
                destinationImage.at<Vec3b>(y, x)[i] = sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i];
                destinationImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i] = sourceImage.at<Vec3b>(y, x)[i];
            }
}

void drawPolygonWithPoints() {
    if (imagenClick.data) {
        int thickness=2;
        int lineType=8;
        int previous=0;
        Scalar color=Scalar( 0, 0, 255 );
     /* Draw all points */
        for (int current = 0; current < (int) points.size(); ++current) {
            circle(imagenClick, (Point)points[current], 5, color, CV_FILLED);
            if (current>0) {
                line( imagenClick, points[previous],points[current],color,thickness,lineType);
                previous++;
            }

        }
    }
}

Mat blackAndWhite(const Mat &sourceImage) {
    Mat destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
    for (int y = 0; y < sourceImage.rows; ++y)
        for (int x = 0; x < sourceImage.cols; ++x) {
            int value=sourceImage.at<Vec3b>(y, x)[0]*0.1+sourceImage.at<Vec3b>(y, x)[1]*0.3+sourceImage.at<Vec3b>(y, x)[2]*0.6;
            Vec3b intensity(value, value, value);
            destinationImage.at<Vec3b>(y, x) = intensity;
        }
    return destinationImage;
}

double yiqMat[3][3] = {
    {0.114, 0.587, 0.299},
    {-0.332, -0.274, 0.596},
    {0.312, -0.523, 0.211}
};
Mat bgr2yiq(const Mat &sourceImage) {
    Mat destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
    for (int y = 0; y < sourceImage.rows; ++y)
        for (int x = 0; x < sourceImage.cols; ++x) {
            // bgr to yiq conversion
            double yiq[3];
            for (int i=0;i<3;i++) {
                yiq[i]=0;
                for (int j=0;j<3;j++) {
                    yiq[i] += yiqMat[i][j] * sourceImage.at<Vec3b>(y, x)[j];
                }
            }
            // normalize values
            yiq[0] = yiq[0]; // Y
            yiq[1] = (yiq[1] + 154.53)*255/306.51; // I
            yiq[2] = (yiq[2] + 133.365)*255/266.73; //Q

            Vec3b intensity(yiq[2], yiq[1], yiq[0]);
            destinationImage.at<Vec3b>(y, x) = intensity;

        }
    return destinationImage;
}
// Convert CRawImage to Mat
void rawToMat( Mat &destImage, CRawImage* sourceImage)
{	
	uchar *pointerImage = destImage.ptr(0);
	
	for (int i = 0; i < 240*320; i++)
	{
		pointerImage[3*i] = sourceImage->data[3*i+2];
		pointerImage[3*i+1] = sourceImage->data[3*i+1];
		pointerImage[3*i+2] = sourceImage->data[3*i];
	}
}

//codigo del click en pantalla
void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param)
{
    uchar* destination;
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN: //CLICK
            Px=x;
            Py=y;
            destination = (uchar*) imagenClick.ptr<uchar>(Py);
            vB=destination[Px * 3];
            vG=destination[Px*3+1];
            vR=destination[Px*3+2];
	    vC1=vB;
	    vC2=vG;
	    vC3=vR;
             points.push_back(Point(x, y));
            break;
        case CV_EVENT_MOUSEMOVE: //Desplazamiento de flecha
            PMx=x;
            PMy=y;
            destination = (uchar*) imagenClick.ptr<uchar>(PMy);
            vMB=destination[PMx * 3];
            vMG=destination[PMx*3+1];
            vMR=destination[PMx*3+2];
            break;
        case CV_EVENT_LBUTTONUP:
            break;
        case CV_EVENT_RBUTTONDOWN:
        //flag=!flag;
            break;
        
    }
}
//codigo del click en pantalla
void C1CoordinatesCallback(int event, int x, int y, int flags, void* param)
{
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            vC1=x;
            break;
        case CV_EVENT_MOUSEMOVE:
            C1Px=x;
            break;
    }
}
//codigo del click en pantalla
void C2CoordinatesCallback(int event, int x, int y, int flags, void* param)
{
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            vC2=x;
            break;
        case CV_EVENT_MOUSEMOVE:
            C2Px=x;
            break;
    }
}
//codigo del click en pantalla
void C3CoordinatesCallback(int event, int x, int y, int flags, void* param)
{
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            vC3=x;
            break;
        case CV_EVENT_MOUSEMOVE:
            C3Px=x;
            break;
    }
}
int main(int argc,char* argv[])
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
	//establishing connection with the quadcopter
	// heli = new CHeli();
	
	//this class holds the image from the drone	
	// image = new CRawImage(320,240);
	
	// Initial values for control	
    pitch = roll = yaw = height = 0.0;
    joypadPitch = joypadRoll = joypadYaw = joypadVerticalSpeed = 0.0;

	// Destination OpenCV Mat	
	Mat currentImage;// = Mat(240, 320, CV_8UC3);
	// Show it	
	//imshow("ParrotCam", currentImage);

    // Initialize joystick
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK);
    useJoystick = SDL_NumJoysticks() > 0;
    if (useJoystick)
    {
        SDL_JoystickClose(m_joystick);
        m_joystick = SDL_JoystickOpen(0);
    }

    namedWindow("Click");
    setMouseCallback("Click", mouseCoordinatesExampleCallback);
    namedWindow("C1"); //Histograma Ch1
    setMouseCallback("C1", C1CoordinatesCallback);
    namedWindow("C2");//Histograma Ch2
    setMouseCallback("C2", C2CoordinatesCallback);
    namedWindow("C3");//Histograma Ch3
    setMouseCallback("C3", C3CoordinatesCallback);

    while (stop == false)
    {

        // Clear the console
        printf("\033[2J\033[1;1H");

        if (useJoystick)
        {
            SDL_Event event;
            SDL_PollEvent(&event);

            joypadRoll = SDL_JoystickGetAxis(m_joystick, 2);
            joypadPitch = SDL_JoystickGetAxis(m_joystick, 3);
            joypadVerticalSpeed = SDL_JoystickGetAxis(m_joystick, 1);
            joypadYaw = SDL_JoystickGetAxis(m_joystick, 0);
            joypadTakeOff = SDL_JoystickGetButton(m_joystick, 1);
            joypadLand = SDL_JoystickGetButton(m_joystick, 2);
            joypadHover = SDL_JoystickGetButton(m_joystick, 0);
        }

        // prints the drone telemetric data, helidata struct contains drone angles, speeds and battery status
        printf("===================== Parrot Basic Example =====================\n\n");
        fprintf(stdout, "Angles  : %.2lf %.2lf %.2lf \n", helidata.phi, helidata.psi, helidata.theta);
        fprintf(stdout, "Speeds  : %.2lf %.2lf %.2lf \n", helidata.vx, helidata.vy, helidata.vz);
        fprintf(stdout, "Battery : %.0lf \n", helidata.battery);
        fprintf(stdout, "Hover   : %d \n", hover);
        fprintf(stdout, "Joypad  : %d \n", useJoystick ? 1 : 0);
        fprintf(stdout, "  Roll    : %d \n", joypadRoll);
        fprintf(stdout, "  Pitch   : %d \n", joypadPitch);
        fprintf(stdout, "  Yaw     : %d \n", joypadYaw);
        fprintf(stdout, "  V.S.    : %d \n", joypadVerticalSpeed);
        fprintf(stdout, "  TakeOff : %d \n", joypadTakeOff);
        fprintf(stdout, "  Land    : %d \n", joypadLand);
        fprintf(stdout, "Navigating with Joystick: %d \n", navigatedWithJoystick ? 1 : 0);
        cout<<"Pos X: "<<Px<<" Pos Y: "<<Py<<" Valor RGB: ("<<vR<<","<<vG<<","<<vB<<")"<<endl;
        cout<<"Histogramas "<<Py<<" Valor C1 C2 C3: ("<<vC1<<","<<vC2<<","<<vC3<<")"<<endl;

        cap >> currentImage;
        resize(currentImage, currentImage, Size(320, 240), 0, 0, cv::INTER_CUBIC);
        imshow("ParrotCam", currentImage);
        currentImage.copyTo(imagenClick);
        // put Text
        ostringstream textStream;
        textStream<<"X: "<<PMx<<" Y: "<<PMy<<" RGB: ("<<vMR<<","<<vMG<<","<<vMB<<")";
	//Pone texto en la Mat imageClick y el stream textStream lo pone en la posision
        putText(imagenClick, textStream.str(), cvPoint(5,15), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(0,0,0), 1, CV_AA);
        drawPolygonWithPoints();
        imshow("Click", imagenClick);

        Mat flipped;// = Mat(240, 320, CV_8UC3);
        flipImageBasic(currentImage, flipped);
        //imshow("Flipped", flipped);

        //BGR to Gray Scale
        Mat blackWhite = blackAndWhite(currentImage);
        imshow("Black and White", blackWhite);
        IplImage* image = cvCreateImage(cvSize(currentImage.cols, currentImage.rows), 8, 3);
        IplImage ipltemp = currentImage;
        cvCopy(&ipltemp, image);

        //BGR to YIQ
        //Mat yiqImage(convertImageRGBtoYIQ(image));
        //imshow("YIQOther", yiqImage);
        Mat yiqOurImage = bgr2yiq(currentImage);
        imshow("Our YIQ", yiqOurImage);

        //BGR to HSV
        Mat hsv;
        cvtColor(currentImage, hsv, CV_BGR2HSV);
        imshow("HSV", hsv);

        // Histogram
        vector<Mat> bgr_planes;
        split( currentImage, bgr_planes );
        int histSize = 256; //from 0 to 255
        /// Set the ranges ( for B,G,R) )
        float range[] = { 0, 256 } ; //the upper boundary is exclusive
        const float* histRange = { range };
        bool uniform = true; bool accumulate = false;
        Mat b_hist, g_hist, r_hist;
        calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
        // Draw the histograms for R, G and B
        int hist_w = 256; int hist_h = 240;
        int bin_w = cvRound( (double) hist_w/histSize );
        int barHeight = 50;
        Mat histImageC1( hist_h+barHeight, hist_w, CV_8UC3, Scalar( 0,0,0) );
        Mat histImageC2( hist_h+barHeight, hist_w, CV_8UC3, Scalar( 0,0,0) );
        Mat histImageC3( hist_h+barHeight, hist_w, CV_8UC3, Scalar( 0,0,0) );
        normalize(b_hist, b_hist, 0, histImageC1.rows, NORM_MINMAX, -1, Mat() );
        normalize(g_hist, g_hist, 0, histImageC2.rows, NORM_MINMAX, -1, Mat() );
        normalize(r_hist, r_hist, 0, histImageC3.rows, NORM_MINMAX, -1, Mat() );
        /// Draw for each channel
        for( int i = 1; i < histSize; i++ )
        {
            line( histImageC1, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                             Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                             Scalar( 255, 0, 0), 2, 8, 0  );
            line( histImageC2, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                             Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                             Scalar( 0, 255, 0), 2, 8, 0  );
            line( histImageC3, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                             Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                             Scalar( 0, 0, 255), 2, 8, 0  );
        }
        // draw intensity bars
        int space = 10;
        Scalar white(255,255,255);
        for (int j=0;j<barHeight;j++) {
            for (int i=0;i<256;i++) {
                Scalar histC1Color = (i==vC1) ? white: Scalar( bin_w*(i-1), 0, 0);
                Scalar histC2Color = (i==vC2) ? white: Scalar( 0, bin_w*(i-1), 0);
                Scalar histC3Color = (i==vC3) ? white: Scalar( 0, 0, bin_w*(i-1));
                // blue
                line( histImageC1, Point( bin_w*(i-1), space+hist_h+j ) ,
                                 Point( bin_w*(i), space+hist_h+j ),
                                 histC1Color, 2, 8, 0  );
                // green
                line( histImageC2, Point( bin_w*(i-1), space+hist_h+j ) ,
                                 Point( bin_w*(i), space+hist_h+j ),
                                 histC2Color, 2, 8, 0  );
                // red
                line( histImageC3, Point( bin_w*(i-1), space+hist_h+j ) ,
                                 Point( bin_w*(i), space+hist_h+j ),
                                 histC3Color, 2, 8, 0  );
            }
        }
        // put text to histograms
        ostringstream histTextStream;
        histTextStream<<"Intensity: "<<C1Px;
        putText(histImageC1, histTextStream.str(), cvPoint(5,15), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
        histTextStream.str(string());
        histTextStream<<"Intensity: "<<C2Px;
        putText(histImageC2, histTextStream.str(), cvPoint(5,15), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
        histTextStream.str(string());
        histTextStream<<"Intensity: "<<C3Px;
        putText(histImageC3, histTextStream.str(), cvPoint(5,15), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
        histTextStream.str(string());
        // show histograms
        imshow("C1", histImageC1 );
        imshow("C2", histImageC2 );
        imshow("C3", histImageC3 );

        char key = waitKey(5);
        switch (key) {
            case 'a': yaw = -20000.0; break;
            case 'd': yaw = 20000.0; break;
            case 'w': height = -20000.0; break;
            case 's': height = 20000.0; break;
            // case 'q': heli->takeoff(); break;
            // case 'e': heli->land(); break;
            // case 'z': heli->switchCamera(0); break;
            // case 'x': heli->switchCamera(1); break;
            // case 'c': heli->switchCamera(2); break;
            // case 'v': heli->switchCamera(3); break;
            case 'j': roll = -20000.0; break;
            case 'l': roll = 20000.0; break;
            case 'i': pitch = -20000.0; break;
            case 'k': pitch = 20000.0; break;
            case 'h': hover = (hover + 1) % 2; break;
	    case 'f':
		currentImage.copyTo(frozenImageBGR);
		imshow("Frozen Image", frozenImageBGR);

		//Congela una imagen en el modelo HSV
        	cvtColor(frozenImageBGR, frozenImageHSV, CV_BGR2HSV);
		imshow("Frozen Image in HSV", frozenImageHSV);

		//Congela una imagen en el modelo HSV
		frozenImageYIQ=bgr2yiq(frozenImageBGR);
		imshow("Frozen Image in YIQ", frozenImageYIQ);

		break;

            case 27: stop = true; break;
            default: pitch = roll = yaw = height = 0.0;
        }

        // if (joypadTakeOff) {
        //     heli->takeoff();
        // }
        // if (joypadLand) {
        //     heli->land();
        // }
        //hover = joypadHover ? 1 : 0;

        //setting the drone angles
        if (joypadRoll != 0 || joypadPitch != 0 || joypadVerticalSpeed != 0 || joypadYaw != 0)
        {
            // heli->setAngles(joypadPitch, joypadRoll, joypadYaw, joypadVerticalSpeed, hover);
            navigatedWithJoystick = true;
        }
        else
        {
            // heli->setAngles(pitch, roll, yaw, height, hover);
            navigatedWithJoystick = false;
        }
	
		//image is captured
		// heli->renewImage(image);

		// Copy to OpenCV Mat
		// rawToMat(currentImage, image);
        

        usleep(15000);
	}
	
	// heli->land();
    SDL_JoystickClose(m_joystick);
    // delete heli;
	//delete image;
	return 0;
}
