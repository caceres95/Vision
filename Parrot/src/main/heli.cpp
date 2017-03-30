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
#include <time.h>       /* time */
#include <map>
#include <fstream>

#include <opencv/cv.h>
#include <errno.h>
#include <math.h>
#include <opencv/highgui.h>
#include <string>


using namespace std;
using namespace cv;

#include <sstream>

string IntToString (int a)
{
    ostringstream temp;
    temp<<a;
    return temp.str();
}
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

int Px;
int Py;
int vC1, vC2, vC3;
int thresh1=0, thresh2=0, thresh3=0;

Mat imagenClick;

//Variable donde se almacenara la imagen congelada
Mat frozenImageBGR;
Mat frozenImageYIQ;
Mat frozenImageHSV;
//Matriz donde se guardara la imagen en blanco y negro
Mat binarizedImage;
Mat segmentedImg;



Mat selectedImage;
int selected = 1;
string canales = "RGB";

// Matriz para convertir a YIQ
double yiqMat[3][3] = {
    {0.114, 0.587, 0.299},
    {-0.332, -0.274, 0.596},
    {0.312, -0.523, 0.211}
};

// segmentation code
#define PI 3.14159265

class Pix{
public:
    long long int x, y;
    int val;
    int color;
};

class Blob{
public:

    vector<Pix> elements;
    long long int area() {return (long long int) elements.size();}
    int color;
    long long int m00, m01, m02, m10, m20, m11;
    double x_centroid, y_centroid;
    double M00, M02, M20, M11;
    double n20, n02, n11;
    double phi1, phi2;
    double theta;
};

void stat_moments(Blob &obj){
    obj.m00 = obj.area();
    obj.m10 = 0;
    obj.m01 = 0;
    obj.m20 = 0;
    obj.m02 = 0;
    obj.m11 = 0;
    for (long long int i = 0; i < obj.elements.size(); i++){
        obj.m10 += obj.elements[i].x;
        obj.m01 += obj.elements[i].y;
        obj.m20 += (obj.elements[i].x * obj.elements[i].x);
        obj.m02 += (obj.elements[i].y * obj.elements[i].y);
        obj.m11 += (obj.elements[i].x * obj.elements[i].y); 
    }
    obj.x_centroid = (double) obj.m10/obj.m00;
    obj.y_centroid = (double) obj.m01/obj.m00;
}

void central_moments(Blob &obj){
    obj.M00 = obj.m00;
    obj.M02 = obj.m02 - (obj.y_centroid*obj.m01);
    obj.M20 = obj.m20 - (obj.x_centroid*obj.m10);
    obj.M11 = obj.m11 - (obj.x_centroid*obj.m01);
}

void invariant_moments(Blob &obj){
    obj.n20 = obj.M20/(obj.M00*obj.M00);
    obj.n02 = obj.M02/(obj.M00*obj.M00);
    obj.n11 = obj.M11;

    obj.phi1 = obj.n20 + obj.n02;
    obj.phi2 = (obj.n20 - obj.n02)*(obj.n20 - obj.n02) + (4*obj.n11*obj.n11);

    obj.theta = 0.5*atan2(2*obj.M11, obj.M20 - obj.M02);        
}

void mergeRegions(vector<Blob> &region_vec, int index_1, int index_2, Mat &blobTemp){
    Blob *masterBlob, *slaveBlob;
    Pix masterPix, slavePix;

    masterBlob = &region_vec[index_2];
    slaveBlob = &region_vec[index_1];

    masterPix = masterBlob->elements.back();

    for(int i = 0; i < slaveBlob->elements.size(); i++){
        slavePix = slaveBlob->elements[i];
        masterBlob->elements.push_back(slavePix);
        blobTemp.at<ushort>(slavePix.y,slavePix.x) = (ushort) masterPix.color;
    }
    slaveBlob->elements.clear();
    slaveBlob->color = 0;
}

void blobColoring (Mat &sourceImage){

    Mat colorImg(sourceImage.rows, sourceImage.cols, CV_8UC3, Scalar::all(0));
    Blob blob, *ptr2blob;
    Pix pc, pi, ps;
    Mat blobTemp(sourceImage.rows, sourceImage.cols, CV_16UC1, Scalar::all(0));

    vector<Blob> regions, segments;

    int color; 
    int numofReg = 0;

    cout << "Size img: " << sourceImage.rows << " x " << sourceImage.cols << endl;
    cout << "No. Pixels: " << sourceImage.rows*sourceImage.cols << endl;

    for (long long int y = 1; y < sourceImage.rows; y++){

        for (long long int x = 1; x < sourceImage.cols; x++){

            pc.x = x; pc.y = y;
            pc.val = (int) sourceImage.at<uchar>(y,x);

            pi.x = x-1; pi.y = y;
            pi.color = (int) blobTemp.at<ushort>(y,x-1);

            ps.x = x; ps.y = y-1;
            ps.color = (int) blobTemp.at<ushort>(y-1,x);

            if(pc.val == 0) {}
            else{
                if(pi.color == 0 && ps.color == 0){
                    color = (int) regions.size() + 1;
                    pc.color = color;
                    blob.elements.push_back(pc);
                    blob.color = color;
                    blobTemp.at<ushort>(y,x) = (ushort) pc.color;
                    regions.push_back(blob);
                    blob.elements.clear();
                }
                else if (pi.color != 0 && ps.color == 0){
                    pc.color = pi.color;
                    ptr2blob = &regions[pc.color-1];
                    (*ptr2blob).elements.push_back(pc);
                    blobTemp.at<ushort>(y,x) = (ushort) pc.color;
                }
                else if (pi.color == 0 && ps.color != 0){
                    pc.color = ps.color;
                    ptr2blob = &regions[pc.color-1];
                    (*ptr2blob).elements.push_back(pc);
                    blobTemp.at<ushort>(y,x) = (ushort) pc.color;
                }
                else if (pi.color != 0 && ps.color != 0){
                        pc.color = ps.color;
                        ptr2blob = &regions[pc.color-1];
                        (*ptr2blob).elements.push_back(pc);
                        blobTemp.at<ushort>(y,x) = (ushort) pc.color;

                    if (pi.color != ps.color){
                        mergeRegions(regions, pi.color-1, ps.color-1, blobTemp);
                    }
                }
            }
        }
    }

    for (int i = 0; i < regions.size(); i++){
        if (regions[i].color != 0){
            segments.push_back(regions[i]);
        }
    }

    for (int i = 0; i < segments.size(); i++){
        segments[i].color = i;  
        uchar b = (uchar) rand() % 256;
        uchar g = (uchar) rand() % 256;
        uchar r = (uchar) rand() % 256;

        for (int j = 0; j < segments[i].elements.size(); j++){
            Vec3b & color = colorImg.at<Vec3b>(segments[i].elements[j].y,segments[i].elements[j].x);
            color[0] = b;
            color[1] = g;
            color[2] = r;
        }
    }

    cout << "Number of regions: " << segments.size() << endl;
    for (int i=0; i<segments.size(); i++){
        stat_moments(segments[i]);
        central_moments(segments[i]);
        invariant_moments(segments[i]);

        cout << setprecision(2) << fixed;
        cout << "Area: " << i << " " << segments[i].area() << " ";
        cout << "m00: " << segments[i].m00 << " ";
        cout << "m01: " << segments[i].m01 << " ";
        cout << "m10: " << segments[i].m10 << " ";
        cout << "m02: " << segments[i].m02 << " ";
        cout << "m20: " << segments[i].m20 << " ";
        cout << "m11: " << segments[i].m11 << " ";
        cout << "X Centroid: " << segments[i].x_centroid << " ";
        cout << "Y Centroid " << segments[i].y_centroid << endl;
        cout << "M00: " << segments[i].M00 << " ";
        cout << "M02: " << segments[i].M02 << " ";
        cout << "M20: " << segments[i].M20 << " ";
        cout << "M11: " << segments[i].M11 << endl;
        cout << "n02: " << segments[i].n02 << " ";
        cout << "n20: " << segments[i].n20 << " ";
        cout << "n11: " << segments[i].n11 << endl;
        cout << "phi1: " << segments[i].phi1 << " ";
        cout << "phi2: " << segments[i].phi2 << " ";
        cout << "theta: " << segments[i].theta * 180/PI << endl << endl;
        // cout << segments[i].phi1 << "," << segments[i].phi2 << endl;

    }

    imshow( "Color image", colorImg );
}

void bgr2yiq(const Mat &sourceImage, Mat &destinationImage) {
    if (destinationImage.empty())
        destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
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
            destination = (uchar*) selectedImage.ptr<uchar>(Py);
            vC1=destination[Px * 3];
            vC2=destination[Px*3+1];
            vC3=destination[Px*3+2];
            points.push_back(Point(x, y));
            break;
        case CV_EVENT_MOUSEMOVE: //Desplazamiento de flecha
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
    }
}
void on_trackbar( int, void* ){}

void filterColorFromImage(const Mat &sourceImage, Mat &destinationImage) {
    if (destinationImage.empty())
        destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
    Vec3b white(255, 255, 255);
    Vec3b black(0, 0, 0);
    for (int y = 0; y < sourceImage.rows; ++y)
        for (int x = 0; x < sourceImage.cols; ++x) {
            if (
                sourceImage.at<Vec3b>(y, x)[0] >= (vC1-thresh1) && sourceImage.at<Vec3b>(y, x)[0] <= (vC1+thresh1) &&
                sourceImage.at<Vec3b>(y, x)[1] >= (vC2-thresh2) && sourceImage.at<Vec3b>(y, x)[1] <= (vC2+thresh2) &&
                sourceImage.at<Vec3b>(y, x)[2] >= (vC3-thresh3) && sourceImage.at<Vec3b>(y, x)[2] <= (vC3+thresh3)
                )
            {
                destinationImage.at<Vec3b>(y, x) = white;
            }
            else
            {
                destinationImage.at<Vec3b>(y, x) = black;
            }
        }
}

//Retorna un numero random
int randomNumber(int min, int max) //range : [min, max)
{
   static bool first = true;
   if ( first ) 
   {  
      srand(time(NULL)); //seeding for the first time only!
      first = false;
   }
   return min + rand() % (max - min);
}

/*
	SEGMENTACION
	Esta funcion recibe una imagen binarizada y retorna por referencia una imagen segmentada,
	la imagen de salida estara coloreada segun su region, ademas esta funcion genera una tabla
	con los identificadores de cada segmento


*/


void segment(const Mat &binarizedImage, Mat &segmentedImage)
{
	//Si la imagen de destino ests vacia, se crea una nueva con las caracteristicas
	//de la imagen binarizada
	if (segmentedImage.empty())
	segmentedImage = Mat(binarizedImage.rows, binarizedImage.cols, binarizedImage.type());
	
	//Esta matriz contendra los ID's de cada segmento
	unsigned int idImage[binarizedImage.rows][binarizedImage.cols];

	
	/*

    Vec3b white(255, 255, 255);
    Vec3b black(0, 0, 0);

    for (int y = 0; y < binarizedImage.rows; ++y)
        for (int x = 0; x < binarizedImage.cols; ++x){

        	if(binarizedImage.at<Vec3b>(y,x)==white){
        		segmentedImage.at<Vec3b>(y, x) = black;
        	}
        	else {
        		segmentedImage.at<Vec3b>(y, x) = white;
        	}
        }
    */
	//Nuestra tala identificadora de regiones
	map<unsigned int,Vec3b> idTable;
	//idTable.insert(make_pair("1", Vec3b(1,2,3)));

    Vec3b white(255, 255, 255);
    Vec3b black(0, 0, 0);
	
	int i,j,k; //Variables auxiliares
	unsigned int id=0; //Identificador de cada region
	Vec3b regionColor(0,0,0); //Esta variable servira para agregar un color random a la region
	unsigned char red,green,blue;

	//Comenzamos con la region 0, la cual sera la que no tenga ningun color
	idTable.insert(make_pair(id, regionColor));

	id=1; 

	for (int i = 0; i < binarizedImage.rows; i++){
    	for (int j = 0; j < binarizedImage.cols; j++){
    		idImage[i][j]=0;
    	}
	}



	for (int i = 0; i < binarizedImage.rows; i++)
    	for (int j = 1; j < binarizedImage.cols; j++){

    		//Para este algoritmo se presentan varios caso, primero el algoritmo encuentra un uno
    		if(binarizedImage.at<Vec3b>(i,j)[0] == 255 && binarizedImage.at<Vec3b>(i,j)[1] == 255 && binarizedImage.at<Vec3b>(i,j)[2] == 255){

    			//Acabamos de encontrar una semilla
    			if(binarizedImage.at<Vec3b>(i-1,j)[0] == 0 && binarizedImage.at<Vec3b>(i,j-1)[0] == 0){
    				
    				//Guardamos el id para posteriormente colorearlo
    				idImage[i][j] = id;
    				
    				//El elemento id tendra un valor de region color
    				
    				//Generamos un color aleatorio para colorear la region
    				regionColor.val[0]=(unsigned char) randomNumber(0,255);
    				regionColor.val[1]=(unsigned char) randomNumber(0,255);
    				regionColor.val[2]=(unsigned char) randomNumber(0,255);

    				idTable.insert(make_pair(idImage[i][j], regionColor));

    				//Incrementamos el id para la proxima region
    				id++;


    			}

    			//Propagacion descendiente
    			else if (binarizedImage.at<Vec3b>(i-1,j)[0] == 255 && binarizedImage.at<Vec3b>(i,j-1)[0]  == 0){

    				idImage[i][j]=idImage[i-1][j];
    
    			}

    			//Propagacion lateral
    			else if (binarizedImage.at<Vec3b>(i-1,j)[0] == 0 && binarizedImage.at<Vec3b>(i,j-1)[0] == 255){

    				idImage[i][j]=idImage[i][j-1];
 
    			}

    			//Propagacion indistinta
    			else if (binarizedImage.at<Vec3b>(i-1,j)[0] == 255 && binarizedImage.at<Vec3b>(i,j-1)[0] == 255){
    			  	
    				idImage[i][j]=idImage[i][j-1];

    				/*
    				//Ahora el color del pixel superior va a ser igual al color del pixel actual,
    				//porque somos la misma region
    				idTable[idImage[i-1][j]].val[0]=idTable[idImage[i][j-1]].val[0];
    				idTable[idImage[i-1][j]].val[1]=idTable[idImage[i][j-1]].val[1];
    				idTable[idImage[i-1][j]].val[2]=idTable[idImage[i][j-1]].val[2];

    				*/

					//Cambiamos por borrar ese ID de la table

    				idTable.erase (idImage[i-1][j]);

    				regionColor.val[0]=idTable[idImage[i][j-1]].val[0];
    				regionColor.val[1]=idTable[idImage[i][j-1]].val[1];
    				regionColor.val[2]=idTable[idImage[i][j-1]].val[2];

    				idTable.insert(make_pair(idImage[i-1][j], regionColor));
    				
/*

    				map<unsigned int,Vec3b>::iterator it;

    				Vec3b aux;

    				aux.val[0]=idTable[idImage[i-1][j]].val[0];
    				aux.val[1]=idTable[idImage[i-1][j]].val[1];
    				aux.val[2]=idTable[idImage[i-1][j]].val[2];

    				it=idTable.find(idImage[i-1][j]);
  					idTable.erase (it); 

  					idTable.insert(make_pair(idImage[i-1][j], aux));*/

    			}
    

    		}
    	}	

    	//Ahora coloreamos la imagen con la tabla de ID y la matriz de IDs que generamos

    	for (int i = 0; i < binarizedImage.rows; i++)
    		for (int j = 0; j < binarizedImage.cols; j++){

    			segmentedImage.at<Vec3b>(i, j)[0] = idTable[idImage[i][j]].val[0];
    			segmentedImage.at<Vec3b>(i, j)[1] = idTable[idImage[i][j]].val[1];
    			segmentedImage.at<Vec3b>(i, j)[2] = idTable[idImage[i][j]].val[2];

    		}

	//BackUp
	/*

	for (int i = 0; i < binarizedImage.rows; ++i)
    	for (int j = 0; j < binarizedImage.cols; ++j){

    		//Para este algoritmo se presentan varios caso, primero el algoritmo encuentra un uno
    		if(binarizedImage.at<Vec3b>(i,j)==white){

    			//Acabamos de encontrar una semilla
    			if(binarizedImage.at<Vec3b>(i-1,j)==black && binarizedImage.at<Vec3b>(i,j-1)==black){
    				
    				segmentedImage.at<Vec3b>(i, j) = regionColor;
    				
    				//El elemento id tendra un valor de region color
    				idTable.insert(make_pair(id, regionColor));
    				//Generamos un color aleatorio para colorear la region
    				regionColor.val[0]=(unsigned char) randomNumber(0,255);
    				regionColor.val[1]=(unsigned char) randomNumber(0,255);
    				regionColor.val[2]=(unsigned char) randomNumber(0,255);

    				//Incrementamos el id para la proxima region
    				id++;


    			}

    			//Propagacion descendiente
    			else if (binarizedImage.at<Vec3b>(i-1,j)==white && binarizedImage.at<Vec3b>(i,j-1)==black){

    				segmentedImage.at<Vec3b>(i, j)[0] = segmentedImage.at<Vec3b>(i-1, j)[0];
    				segmentedImage.at<Vec3b>(i, j)[1] = segmentedImage.at<Vec3b>(i-1, j)[1];
    				segmentedImage.at<Vec3b>(i, j)[2] = segmentedImage.at<Vec3b>(i-1, j)[2];
    			}

    			//Propagacion lateral
    			else if (binarizedImage.at<Vec3b>(i-1,j)==black && binarizedImage.at<Vec3b>(i,j-1)==white){
    	
    				segmentedImage.at<Vec3b>(i, j)[0] = segmentedImage.at<Vec3b>(i, j-1)[0];
    				segmentedImage.at<Vec3b>(i, j)[1] = segmentedImage.at<Vec3b>(i, j-1)[1];
    				segmentedImage.at<Vec3b>(i, j)[2] = segmentedImage.at<Vec3b>(i, j-1)[2];
    			}

    			//Propagacion indistinta
    			else if (binarizedImage.at<Vec3b>(i-1,j)==white && binarizedImage.at<Vec3b>(i,j-1)==white){
    			  	
    				segmentedImage.at<Vec3b>(i, j)[0] = segmentedImage.at<Vec3b>(i, j-1)[0];
    				segmentedImage.at<Vec3b>(i, j)[1] = segmentedImage.at<Vec3b>(i, j-1)[1];
    				segmentedImage.at<Vec3b>(i, j)[2] = segmentedImage.at<Vec3b>(i, j-1)[2];

    				//Iteramos sobre todas las regiones registradas para ver quien tiene ese color
    				map<unsigned int, Vec3b>::iterator it = idTable.begin();

    				while(it != idTable.end())
    				{
        				//std::cout<<it->first<<" :: "<<it->second<<std::endl;
        				//Si ese ID tiene el mismo color que la imagen de este momento
        				it++;
    				}


    			}



    

    		}
    	}
    	*/
    //IMPRMIR ID MATRIX



}

void segment2( Mat &binarizedImage, Mat &segmentedImage)
{
	//Si la imagen de destino ests vacia, se crea una nueva con las caracteristicas
	//de la imagen binarizada

    ofstream outputFile("program3data.txt");
	if (segmentedImage.empty())
	segmentedImage = Mat(binarizedImage.rows, binarizedImage.cols, binarizedImage.type());
	
	//Esta matriz contendra los ID's de cada segmento
	unsigned int idImage[binarizedImage.rows][binarizedImage.cols];
	
	unsigned char idTableV0[80000];
	unsigned char idTableV1[80000];
	unsigned char idTableV2[80000];
	
	//Nuestra tala identificadora de regiones
	


	//idTable.insert(make_pair("1", Vec3b(1,2,3)));

    Vec3b white(255, 255, 255);
    Vec3b black(0, 0, 0);
	
	int i,j; //Variables auxiliares
    unsigned k;
	unsigned int id=0; //Identificador de cada region
	Vec3b regionColor(0,0,0); //Esta variable servira para agregar un color random a la region
	unsigned char red,green,blue;

	//Comenzamos con la region 0, la cual sera la que no tenga ningun color
	//idTable.insert(make_pair(id, regionColor));

	idTableV0[id]=regionColor.val[0];
	idTableV1[id]=regionColor.val[1];
	idTableV2[id]=regionColor.val[2];

	id=1; 

	for (int i = 0; i < binarizedImage.rows; i++){
    	for (int j = 0; j < binarizedImage.cols; j++){
    		idImage[i][j]=0;
    	}
	}

	//Antes de iniciar tenemos que hacer un marco a binarized image de color negro para que no halla cosas raras
	for (int i = 0; i < binarizedImage.rows; i++)
	{
		binarizedImage.at<Vec3b>(i,0)[0]=0;
		binarizedImage.at<Vec3b>(i,0)[1]=0;
		binarizedImage.at<Vec3b>(i,0)[2]=0;

		binarizedImage.at<Vec3b>(i,binarizedImage.cols-1)[0]=0;
		binarizedImage.at<Vec3b>(i,binarizedImage.cols-1)[1]=0;
		binarizedImage.at<Vec3b>(i,binarizedImage.cols-1)[2]=0;

	}

	for (int j = 0; j < binarizedImage.cols; j++)
	{
		binarizedImage.at<Vec3b>(0,j)[0]=0;
		binarizedImage.at<Vec3b>(0,j)[1]=0;
		binarizedImage.at<Vec3b>(0,j)[2]=0;

		binarizedImage.at<Vec3b>(binarizedImage.rows-1,j)[0]=0;
		binarizedImage.at<Vec3b>(binarizedImage.rows-1,j)[1]=0;
		binarizedImage.at<Vec3b>(binarizedImage.rows-1,j)[2]=0;

	}


	for (int i = 1; i < binarizedImage.rows-1; i++)
    	for (int j = 1; j < binarizedImage.cols-1; j++){

    		//Para este algoritmo se presentan varios caso, primero el algoritmo encuentra un uno
    		if(binarizedImage.at<Vec3b>(i,j)[0] == 255 && binarizedImage.at<Vec3b>(i,j)[1] == 255 && binarizedImage.at<Vec3b>(i,j)[2] == 255){

    			//Acabamos de encontrar una semilla
    			if(binarizedImage.at<Vec3b>(i-1,j)[0] == 0 && binarizedImage.at<Vec3b>(i,j-1)[0] == 0){
    				
    				//Guardamos el id para posteriormente colorearlo
    				idImage[i][j] = id;


    				
    				//El elemento id tendra un valor de region color
    				
    				//Generamos un color aleatorio para colorear la region
    				regionColor.val[0]=(unsigned char) randomNumber(0,255);
    				regionColor.val[1]=(unsigned char) randomNumber(0,255);
    				regionColor.val[2]=(unsigned char) randomNumber(0,255);

    				//idTable.insert(make_pair(idImage[i][j], regionColor));
    				
    				idTableV0[idImage[i][j]]=regionColor.val[0];
					idTableV1[idImage[i][j]]=regionColor.val[1];
					idTableV2[idImage[i][j]]=regionColor.val[2];

    				//Incrementamos el id para la proxima region
    				id++;


    			}

    			//Propagacion descendiente
    			else if (binarizedImage.at<Vec3b>(i-1,j)[0] == 255 && binarizedImage.at<Vec3b>(i,j-1)[0]  == 0){

    				idImage[i][j]=idImage[i-1][j];
    
    			}

    			//Propagacion lateral
    			else if (binarizedImage.at<Vec3b>(i-1,j)[0] == 0 && binarizedImage.at<Vec3b>(i,j-1)[0] == 255){

    				idImage[i][j]=idImage[i][j-1];
 
    			}

    			//Propagacion indistinta
    			else if (binarizedImage.at<Vec3b>(i-1,j) != black && binarizedImage.at<Vec3b>(i,j-1) != black){

    			  	
                    outputFile <<"\nAnalizando ["<<IntToString(i)<<"]["<<IntToString(j)<<"]\nPixel idImage["<<IntToString(i-1)<<"]["<<IntToString(j)<<"]="<<IntToString(idImage[i-1][j])<<" es True y pixel ["<<IntToString(i)<<"]["<<IntToString(j-1)<<"]="<<IntToString(idImage[i][j-1])<<" es True\n";
    				//Propagacon lateral idImage[i][j]=idImage[i][j-1];
                    idImage[i][j]=idImage[i-1][j];


    				/*
    				//Ahora el color del pixel superior va a ser igual al color del pixel actual,
    				//porque somos la misma region
    				idTable[idImage[i-1][j]].val[0]=idTable[idImage[i][j-1]].val[0];
    				idTable[idImage[i-1][j]].val[1]=idTable[idImage[i][j-1]].val[1];
    				idTable[idImage[i-1][j]].val[2]=idTable[idImage[i][j-1]].val[2];

    				*/

					//Cambiamos por borrar ese ID de la table


                    outputFile <<"El color actual IDTable del pixel ["<<IntToString(i)<<"]["<<IntToString(j-1)<<"] lateral es ="<<IntToString(idTableV0[idImage[i][j-1]])<<"/"<<IntToString(idTableV1[idImage[i][j-1]])<<"/"<<IntToString(idTableV2[idImage[i][j-1]])<<"\n";
    				
                    //Todos los segmentos que tengan el color del pixel superior deben ser cambiados y poner el color del pixel lateral

                    for(k=0; k<id+1 ; k++){

                        if(idTableV0[k]==idTableV0[idImage[i][j-1]] && idTableV1[k]==idTableV1[idImage[i][j-1]] && idTableV2[k]==idTableV2[idImage[i][j-1]])
                        {
                            idTableV0[k]=idTableV0[idImage[i-1][j]];
                            idTableV1[k]=idTableV1[idImage[i-1][j]];
                            idTableV2[k]=idTableV2[idImage[i-1][j]];
                        }
                    }
                    /*
                    idTableV0[idImage[i][j-1]]=idTableV0[idImage[i-1][j]];
					idTableV1[idImage[i][j-1]]=idTableV1[idImage[i-1][j]];
					idTableV2[idImage[i][j-1]]=idTableV2[idImage[i-1][j]];*/
                    outputFile <<"El color actual IDTable del pixel ["<<IntToString(i-1)<<"]["<<IntToString(j)<<"] superior es ="<<IntToString(idTableV0[idImage[i-1][j]])<<"/"<<IntToString(idTableV1[idImage[i-1][j]])<<"/"<<IntToString(idTableV2[idImage[i-1][j]])<<"\n";
                    outputFile <<"El color nuevo IDTable del pixel lateral es ="<<IntToString(idTableV0[idImage[i][j-1]])<<"/"<<IntToString(idTableV1[idImage[i][j-1]])<<"/"<<IntToString(idTableV2[idImage[i][j-1]])<<"\n";

    				
/*

    				map<unsigned int,Vec3b>::iterator it;

    				Vec3b aux;

    				aux.val[0]=idTable[idImage[i-1][j]].val[0];
    				aux.val[1]=idTable[idImage[i-1][j]].val[1];
    				aux.val[2]=idTable[idImage[i-1][j]].val[2];

    				it=idTable.find(idImage[i-1][j]);
  					idTable.erase (it); 

  					idTable.insert(make_pair(idImage[i-1][j], aux));*/

    			}
    

    		}
    	}	

    	//Ahora coloreamos la imagen con la tabla de ID y la matriz de IDs que generamos

        

        for (int i = 0; i < binarizedImage.rows; i++){
            for (int j = 0; j < binarizedImage.cols; j++){

                outputFile << IntToString(idImage[i][j])<<"\t";
            }

            outputFile << "\n";
        }


        for (int i = 0; i <= id; i++){
        

                outputFile << "\nID: "<<IntToString(i)<<" "<<IntToString(idTableV0[i])<<" "<<IntToString(idTableV1[i])<<" "<<IntToString(idTableV2[i])<<" ";

            

            outputFile << "\n";
        }



    	for (int i = 0; i < binarizedImage.rows; i++)
    		for (int j = 0; j < binarizedImage.cols; j++){

    			segmentedImage.at<Vec3b>(i, j)[0] = idTableV0[idImage[i][j]];
    			segmentedImage.at<Vec3b>(i, j)[1] = idTableV1[idImage[i][j]];
    			segmentedImage.at<Vec3b>(i, j)[2] = idTableV2[idImage[i][j]];

    		}

	//BackUp
	/*

	for (int i = 0; i < binarizedImage.rows; ++i)
    	for (int j = 0; j < binarizedImage.cols; ++j){

    		//Para este algoritmo se presentan varios caso, primero el algoritmo encuentra un uno
    		if(binarizedImage.at<Vec3b>(i,j)==white){

    			//Acabamos de encontrar una semilla
    			if(binarizedImage.at<Vec3b>(i-1,j)==black && binarizedImage.at<Vec3b>(i,j-1)==black){
    				
    				segmentedImage.at<Vec3b>(i, j) = regionColor;
    				
    				//El elemento id tendra un valor de region color
    				idTable.insert(make_pair(id, regionColor));
    				//Generamos un color aleatorio para colorear la region
    				regionColor.val[0]=(unsigned char) randomNumber(0,255);
    				regionColor.val[1]=(unsigned char) randomNumber(0,255);
    				regionColor.val[2]=(unsigned char) randomNumber(0,255);

    				//Incrementamos el id para la proxima region
    				id++;


    			}

    			//Propagacion descendiente
    			else if (binarizedImage.at<Vec3b>(i-1,j)==white && binarizedImage.at<Vec3b>(i,j-1)==black){

    				segmentedImage.at<Vec3b>(i, j)[0] = segmentedImage.at<Vec3b>(i-1, j)[0];
    				segmentedImage.at<Vec3b>(i, j)[1] = segmentedImage.at<Vec3b>(i-1, j)[1];
    				segmentedImage.at<Vec3b>(i, j)[2] = segmentedImage.at<Vec3b>(i-1, j)[2];
    			}

    			//Propagacion lateral
    			else if (binarizedImage.at<Vec3b>(i-1,j)==black && binarizedImage.at<Vec3b>(i,j-1)==white){
    	
    				segmentedImage.at<Vec3b>(i, j)[0] = segmentedImage.at<Vec3b>(i, j-1)[0];
    				segmentedImage.at<Vec3b>(i, j)[1] = segmentedImage.at<Vec3b>(i, j-1)[1];
    				segmentedImage.at<Vec3b>(i, j)[2] = segmentedImage.at<Vec3b>(i, j-1)[2];
    			}

    			//Propagacion indistinta
    			else if (binarizedImage.at<Vec3b>(i-1,j)==white && binarizedImage.at<Vec3b>(i,j-1)==white){
    			  	
    				segmentedImage.at<Vec3b>(i, j)[0] = segmentedImage.at<Vec3b>(i, j-1)[0];
    				segmentedImage.at<Vec3b>(i, j)[1] = segmentedImage.at<Vec3b>(i, j-1)[1];
    				segmentedImage.at<Vec3b>(i, j)[2] = segmentedImage.at<Vec3b>(i, j-1)[2];

    				//Iteramos sobre todas las regiones registradas para ver quien tiene ese color
    				map<unsigned int, Vec3b>::iterator it = idTable.begin();

    				while(it != idTable.end())
    				{
        				//std::cout<<it->first<<" :: "<<it->second<<std::endl;
        				//Si ese ID tiene el mismo color que la imagen de este momento
        				it++;
    				}


    			}



    

    		}
    	}
    	*/
    //IMPRMIR ID MATRIX
    //...


//outputFile << IntToString(idImage[0][0])<<"\t"<<IntToString(idTableV0[1]);
//... 



}


int main(int argc,char* argv[])
{

	Mat imageTest;
    imageTest = imread("test.png", CV_LOAD_IMAGE_COLOR);   // Read the file
    Mat imageTestSeg;

    if(! imageTest.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Display window" );// Create a window for display.
    segment2(imageTest,imageTestSeg);


    imshow( "Display window", imageTestSeg );


    imwrite( "Gray_Image.bmp", imageTestSeg );

	Vec3b aux(111,222,255);
	map<unsigned int,Vec3b> idTable;
	
	idTable.insert(make_pair(0, aux));
	aux.val[0]=11;
	aux.val[1]=22;
	aux.val[2]=33;

	idTable.insert(make_pair(1, aux));

		aux.val[0]=44;
	aux.val[1]=55;
	aux.val[2]=66;


	idTable.insert(make_pair(2, aux));

		aux.val[0]=77;
	aux.val[1]=88;
	aux.val[2]=99;

	idTable.insert(make_pair(3, aux));


	//Experimento
	//Declaramos matriz 3 x 3
	unsigned int matriz[2][2];
	matriz[0][0]=2;
	matriz[0][1]=3;
	matriz[1][0]=4;
	matriz[1][1]=0;

	idTable[matriz[0][0]].val[1]=idTable[matriz[1][1]].val[2];


    // VideoCapture cap(0); // open the default camera
    // if(!cap.isOpened())  // check if we succeeded
    //     return -1;
    // establishing connection with the quadcopter
    heli = new CHeli();
    
    // this class holds the image from the drone 
    image = new CRawImage(320,240);
    
    // Initial values for control   
    pitch = roll = yaw = height = 0.0;
    joypadPitch = joypadRoll = joypadYaw = joypadVerticalSpeed = 0.0;

    // Destination OpenCV Mat   
    Mat currentImage = Mat(240, 320, CV_8UC3);
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
    namedWindow("Controls", WINDOW_NORMAL);
    createTrackbar( "Threshold 1", "Controls", &thresh1, 100, on_trackbar );
    createTrackbar( "Threshold 2", "Controls", &thresh2, 100, on_trackbar );
    createTrackbar( "Threshold 3", "Controls", &thresh3, 100, on_trackbar );

    cap >> currentImage;

    selectedImage = currentImage;
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

        //Vec3b aux;

        // prints the drone telemetric data, helidata struct contains drone angles, speeds and battery status
        printf("===================== Parrot Basic Example =====================\n\n");
        fprintf(stdout,"First val1 %d Secod Val %d, Third Val %d \n",idTable[matriz[0][0]].val[0],idTable[matriz[0][0]].val[1],idTable[matriz[0][0]].val[2]);
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
        cout<<"Pos X: "<<Px<<" Pos Y: "<<Py<<" Valor "<<canales<<": ("<<vC3<<","<<vC2<<","<<vC1<<")"<<endl;

        // cap >> currentImage;


        resize(currentImage, currentImage, Size(320, 240), 0, 0, cv::INTER_CUBIC);
        // imshow("ParrotCam", currentImage);
        currentImage.copyTo(imagenClick);
        // put Text
        ostringstream textStream;
        textStream<<"X: "<<Px<<" Y: "<<Py<<" "<<canales<<": ("<<vC3<<","<<vC2<<","<<vC1<<")";
        //Pone texto en la Mat imageClick y el stream textStream lo pone en la posision
        putText(imagenClick, textStream.str(), cvPoint(5,15), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(0,0,0), 1, CV_AA);
        // drawPolygonWithPoints();

        if (points.size()) circle(imagenClick, (Point)points[points.size() -1], 5, Scalar(0,0,255), CV_FILLED);
        imshow("Click", imagenClick);

        //BGR to YIQ
        Mat yiqOurImage; bgr2yiq(currentImage, yiqOurImage);

        // imshow("YIQ1", yiqOurImage);

        //BGR to HSV
        Mat hsv; cvtColor(currentImage, hsv, CV_BGR2HSV);
        // imshow("HSV", hsv);

        switch(selected) {
            case 1: selectedImage = currentImage; canales="RGB"; break;
            case 2: selectedImage = yiqOurImage; canales="YIQ"; break;
            case 3: selectedImage = hsv; canales="HSV"; break;
        }
        // Histogram
        vector<Mat> bgr_planes;
        split( selectedImage, bgr_planes );
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
        Scalar gray(128, 128, 128);
        for (int j=0;j<barHeight;j++) {
            for (int i=0;i<256;i++) {
                Scalar histC1Color = (i==vC1) ? white: (i==(vC1-thresh1)||i==(vC1+thresh1)) ? gray: Scalar( bin_w*(i-1), 0, 0);
                Scalar histC2Color = (i==vC2) ? white: (i==(vC2-thresh2)||i==(vC2+thresh2)) ? gray: Scalar( 0, bin_w*(i-1), 0);
                Scalar histC3Color = (i==vC3) ? white: (i==(vC3-thresh3)||i==(vC3+thresh3)) ? gray: Scalar( 0, 0, bin_w*(i-1));
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
        histTextStream<<canales[2]<<": "<<vC1;
        if (thresh1 > 0 && (vC1-thresh1) > 0) histTextStream<<" "<<canales[2]<<"Min"<<": "<<vC1-thresh1;
        if (thresh1 > 0 && (vC1+thresh1) < 256) histTextStream<<" "<<canales[2]<<"Max"<<": "<<vC1+thresh1;
        putText(histImageC1, histTextStream.str(), cvPoint(5,15), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
        histTextStream.str(string());
        histTextStream<<canales[1]<<": "<<vC2;
        if (thresh2 > 0 && (vC2-thresh2) > 0) histTextStream<<" "<<canales[1]<<"Min"<<": "<<vC2-thresh2;
        if (thresh2 > 0 && (vC2+thresh2) < 256) histTextStream<<" "<<canales[1]<<"Max"<<": "<<vC2+thresh2;
        putText(histImageC2, histTextStream.str(), cvPoint(5,15), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
        histTextStream.str(string());
        histTextStream<<canales[0]<<": "<<vC3;
        if (thresh3 > 0 && (vC3-thresh3) > 0) histTextStream<<" "<<canales[0]<<"Min"<<": "<<vC3-thresh3;
        if (thresh3 > 0 && (vC3+thresh3) < 256) histTextStream<<" "<<canales[0]<<"Max"<<": "<<vC3+thresh3;
        putText(histImageC3, histTextStream.str(), cvPoint(5,15), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
        histTextStream.str(string());
        // show histograms
        imshow("C1", histImageC1 );
        imshow("C2", histImageC2 );
        imshow("C3", histImageC3 );

        // Blur image
        blur(selectedImage,selectedImage,Size(10,10)); 
        // Filter image
        Mat filteredImage; filterColorFromImage(selectedImage, filteredImage);
        imshow("Filtered Image", filteredImage);
                //Probamos segmentacion
        segment2(filteredImage,segmentedImg);
        imshow("SEGMENTACION",segmentedImg);

        char key = waitKey(5);
        switch (key) {
            case 'a': yaw = -20000.0; break;
            case 'd': yaw = 20000.0; break;
            case 'w': height = -20000.0; break;
            case 's': height = 20000.0; break;
            case 'q': heli->takeoff(); break;
            case 'e': heli->land(); break;
            case 'z': heli->switchCamera(0); break;
            case 'x': heli->switchCamera(1); break;
            case 'c': heli->switchCamera(2); break;
            case 'v': heli->switchCamera(3); break;
            case 'j': roll = -20000.0; break;
            case 'l': roll = 20000.0; break;
            case 'i': pitch = -20000.0; break;
            case 'k': pitch = 20000.0; break;
            case 'h': hover = (hover + 1) % 2; break;

            case '1': selected=1; break;
            case '2': selected=2; break;
            case '3': selected=3; break;

            case 27: stop = true; break;
            default: pitch = roll = yaw = height = 0.0;
        }
 
        if (joypadTakeOff) {
            heli->takeoff();
        }
        if (joypadLand) {
            heli->land();
        }
        hover = joypadHover ? 1 : 0;

        //setting the drone angles
        if (joypadRoll != 0 || joypadPitch != 0 || joypadVerticalSpeed != 0 || joypadYaw != 0)
        {
            heli->setAngles(joypadPitch, joypadRoll, joypadYaw, joypadVerticalSpeed, hover);
            navigatedWithJoystick = true;
        }
        else
        {
            heli->setAngles(pitch, roll, yaw, height, hover);
            navigatedWithJoystick = false;
        }
    
        // image is captured
        heli->renewImage(image);

        // Copy to OpenCV Mat
        rawToMat(currentImage, image);
        

        usleep(15000);
    }
    
    heli->land();
    SDL_JoystickClose(m_joystick);
    delete heli;
    delete image;
    return 0;
}
