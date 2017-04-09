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
#include <algorithm>
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

#define PI 3.14159265

struct caracterizacion{
    //Estructura con todas los momentos estadisticos que puede tener una figura
    Vec3b color;
    unsigned int area;
    string whatitis;
    //MOMENTOS ORDINARIOS
    unsigned long long m00;
    unsigned long long m10;
    unsigned long long m20;
    unsigned long long m30;
    unsigned long long m01;
    unsigned long long m02;
    unsigned long long m03;
    unsigned long long m11;
    unsigned long long m12;
    unsigned long long m21;

    //MOMENTOS CENTRALIZADOS
    unsigned long long u00;
    unsigned long long u10;
    unsigned long long u01;
    double u20;
    double u02;
    double u11;
    double u30;
    double u03;
    double u12;
    double u21;

    //MOMENTOS NORMALIZADOS
    double n02;
    double n03;
    double n11;
    double n12;
    double n20;
    double n21;
    double n30;

    double phi1;
    double phi2;
    double phi3;
    double phi4;

    double theta;

    //PROMEDIOS
    double xPromedio;
    double yPromedio;


};

//Esta estructra servira para almacenar el color de una region y sus momentos caracteristicos
struct region {
  Vec3b color;
  struct caracterizacion caracteristicas;
} ;

string IntToString (unsigned int a)
{
    ostringstream temp;
    temp<<a;
    return temp.str();
}



string DoubleToString(double a)
{
    ostringstream os;
    os<<a;
    return os.str();

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
int vC1=85, vC2=115, vC3=152;
int thresh1=22, thresh2=20, thresh3=36;

Mat imagenClick;

//Variable donde se almacenara la imagen congelada
Mat frozenImageBGR;
Mat frozenImageYIQ;
Mat frozenImageHSV;
//Matriz donde se guardara la imagen en blanco y negro
Mat binarizedImage;
Mat segmentedImg;



Mat selectedImage;
int selected = 2;
string canales = "RGB";

vector<struct caracterizacion> figuresGlobVar;

// Matriz para convertir a YIQ
double yiqMat[3][3] = {
    {0.114, 0.587, 0.299},
    {-0.332, -0.274, 0.596},
    {0.312, -0.523, 0.211}
};

// segmentation code
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


void segment(Mat &binarizedImage, Mat &segmentedImage)
{
   

    //Variables usadas en este algoritmo
    int i, j, y, x; //Para los ciclos
    unsigned int id, k, areaTemp; //Para la idenficacion(id) y color(k) de los segmentos
    //Si la imagen de destino esta vacia, se inicializa
    Vec3b white(255, 255, 255);
    Vec3b black(0, 0, 0);
    Vec3b regionColor;
    Vec3b Pi,Ps, Pc; //Para identificar los tres pixeles analizadores
    ofstream outputFile("LUT.txt");

    if (segmentedImage.empty())
    segmentedImage = Mat(binarizedImage.rows, binarizedImage.cols, binarizedImage.type());

    //Inicializamos la matriz color toda en color negro
    for (i=0; i<binarizedImage.rows; i++)
    {
        for (j=0; j<binarizedImage.cols; j++)
        {
            segmentedImage.at<Vec3b>(i, j)=black;
        }
    }

    k=1;
    id=1;

    //
    //Nuestra tabla identificadora de regiones
    /*unsigned int m10;
    unsigned int m20;
    unsigned int m30;
    unsigned int m01;
    unsigned int m02;
    unsigned int m03;
    unsigned int m11;
    unsigned int m12;
    unsigned int m21;
    LUT

    ID  K(Color)    Area
    1   1           A=A1+A2
    2   2->1        A2
    .   .           .
    */

    map<unsigned int,struct region> LUT;
    map<unsigned int,struct region> FinalLUT;

    struct region regionTemp;
    unsigned int idImage[binarizedImage.rows][binarizedImage.cols];
    unsigned int LUTSize;

    for (i=0; i<binarizedImage.rows-1; i++)
    {
        for (j=0; j<binarizedImage.cols-1; j++)
        {
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

    //Comenzamos nuestro analisis pixel por pixel sobre la imagen
     //Inicializamos la matriz color toda en color negro
    for (y=1; y<binarizedImage.rows-1; y++)
    {
        for (x=1; x<binarizedImage.cols-1; x++)
        {
            if(binarizedImage.at<Vec3b>(y,x)==black)
            {
                continue;
            }

            else //La imagen orginal tiene un 1
            {
                Pi=binarizedImage.at<Vec3b>(y,x-1);
                Ps=binarizedImage.at<Vec3b>(y-1,x);
                Pc=binarizedImage.at<Vec3b>(y,x);

                if(Ps==white && Pi == black)
                {
                    //Propagacion descendiente
                    idImage[y][x]=idImage[y-1][x];

                }
                else if(Ps==black && Pi == white)
                {
                    //Propagacion lateral
                    idImage[y][x]=idImage[y][x-1];
                }

                else if(Ps==white && Pi == white)
                {
                    //Propagacion indistinta, tenemos que detectar conflicto
                    if(LUT[idImage[y-1][x]].color != LUT[idImage[y][x-1]].color)
                    {

                        
                        //Region color contendra el color del pixel superior
                        regionColor=LUT[idImage[y-1][x]].color;

                        //Borrar dos lineas en caso de error
                        LUT[idImage[y][x-1]].caracteristicas.area+=LUT[idImage[y-1][x]].caracteristicas.area;
                        LUT[idImage[y-1][x]].caracteristicas.area=0;
                        //Guardamos su tamaño
                        LUTSize=(unsigned int) LUT.size();


                        //Iteramos sobre la LTU
                        for (k=1; k<=LUTSize; k++)
                        {
                            //Quien tenga el color del pixel superior sera cambiado por el color del pixel lateral
                            if(LUT[k].color==regionColor)
                            {
                                areaTemp=LUT[k].caracteristicas.area;
                                LUT.erase(k);
                                   
                                regionTemp.color=LUT[idImage[y][x-1]].color;
                                LUT[idImage[y][x-1]].caracteristicas.area+=areaTemp;
                                regionTemp.caracteristicas.area=0;
                                LUT.insert(make_pair(k, regionTemp));

                            }
                        }
                    }

                    //Propagacion lateral
                    idImage[y][x]=idImage[y][x-1];
                }

                else if(Ps==black && Pi == black)
                {

                    //Creamos un color aleatorio
                    regionColor.val[0]=(unsigned char) randomNumber(0,255);
                    regionColor.val[1]=(unsigned char) randomNumber(0,255);
                    regionColor.val[2]=(unsigned char) randomNumber(0,255);

                    //Inicializamos una nueva region
                    regionTemp.color=regionColor;
                    regionTemp.caracteristicas.area=0;

                    idImage[y][x]=id;

                    LUT.insert(make_pair(id, regionTemp));

                    id=id+1;

                }

                //Aumentamos area
                LUT[idImage[y][x]].caracteristicas.area++;


            }
        }

    }   


    //Coloreamos la imagen en base a los valores de la LUT
    for (i=1; i<binarizedImage.rows-1; i++)
    {
        for (j=1; j<binarizedImage.cols-1; j++)
        {
            segmentedImage.at<Vec3b>(i, j)=LUT[idImage[i][j]].color;

        }
    }

    LUTSize=(unsigned int) LUT.size();
    // Almacenamos tabla
    for( k=1; k<=LUTSize; k++)
    {
        outputFile << "\nID: "<<IntToString(k)<<" Color: "<<IntToString(LUT[k].color[0])<<" "<<IntToString(LUT[k].color[1])<<" "<<IntToString(LUT[k].color[2])<<" Area: "<<IntToString(LUT[k].caracteristicas.area)<<"\n";
    }



}

//Esta funcion retorna true si ya existe un elemento
bool exists(Vec3b color, map<unsigned int, struct caracterizacion> figures) {
  // somehow I should find whether my MAP has a car
  // with the name provided

    unsigned int LUTSize, k;
    LUTSize=(unsigned int) figures.size();

    if(LUTSize==0)
    {
        return false;
    }
   
    for (k=0; k<=LUTSize; k++)
    {
        if(figures[k].color==color)
        {
            return true;
        }

    }

    return false;
 

}

unsigned int getIdByColor(Vec3b color,  map<unsigned int, struct caracterizacion> figures)
{
    unsigned int LUTSize, k;
    LUTSize=(unsigned int) figures.size();

    if(LUTSize==0)
    {
        return 0;
    }
   
    for (k=0; k<=LUTSize; k++)
    {
        if(figures[k].color==color)
        {
            return k;
        }

    }

    return 0;
}


//Obtencion de momentos estadisticos
void momentos(Mat &segmentedImage)
{
    unsigned  id,k,figuresSize;
    unsigned long long i, j,x,y;
    map<unsigned int,struct caracterizacion> figures;
    Vec3b black(0,0,0);
    id=0;
    struct caracterizacion caracteristicas;
    ofstream outputFile("figures.txt");

        //Coloreamos la imagen en base a los valores de la LUT
    for (x=0; x<segmentedImage.cols; x++)
    {
        for (y=0; y<segmentedImage.rows; y++)
        {
            if(segmentedImage.at<Vec3b>(y, x)!=black)
            {
                //Existe este color en la tabla de figuras?
                if(!exists(segmentedImage.at<Vec3b>(y, x),figures))
                {
                    //No existe, crea un nuevo id
                    caracteristicas.color=segmentedImage.at<Vec3b>(y, x);
                    caracteristicas.area=0;
                    caracteristicas.m00=0;
                    caracteristicas.m10=0;
                    caracteristicas.m20=0;
                    caracteristicas.m30=0;
                    caracteristicas.m01=0;
                    caracteristicas.m02=0;
                    caracteristicas.m03=0;
                    caracteristicas.m11=0;
                    caracteristicas.m12=0;
                    caracteristicas.m21=0;

                    figures.insert(make_pair(id, caracteristicas));
                    id++;
                }

                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].area++;
                /*
                AGREGAR SUMATORIAS EN ESTE CAMPO
                Y AAGREGAR MOMENTO EN STRUCT CARACTERIZACION
                */
                /*SE COMIENZAN A OBTENER MOMENTOS ORDINARIOS*/

                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m00++; /* m00= [sum x sum y] 1 */
                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m10+=x; /* m00= [sum x sum y] x */
                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m20+=pow(x,2); /* m00= [sum x sum y] x² */
                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m30+=pow(x,3); /* m00= [sum x sum y] x³ */
                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m01+=y; /* m00= [sum x sum y] y */
                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m02+=pow(y,2); /* m00= [sum x sum y] y² */
                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m03+=pow(y,3); /* m00= [sum x sum y] y³ */
                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m11+=x*y; /* m00= [sum x sum y] x*y */
                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m12+=x*pow(y,2); /* m00= [sum x sum y] x*y² */
                figures[getIdByColor(segmentedImage.at<Vec3b>(y, x), figures)].m21+=pow(x,2)*y; /* m00= [sum x sum y] x²*y */

            }

        }
    }

    //OBTENEMOS MOMENTOS CENTRALIZADOS (Para estos ya no necesitamos iterar la figura)
    figuresGlobVar.clear();
    figuresSize=figures.size();
    for( k=0; k<figuresSize; k++)
    {
        //OBTENEMOS PROMEDIOS
        figures[k].xPromedio=((double)figures[k].m10)/((double)figures[k].m00);
        figures[k].yPromedio=((double)figures[k].m01)/((double)figures[k].m00);

        
        //Primer Orden
        figures[k].u00=figures[k].m00;
        figures[k].u10=0;
        figures[k].u01=0;

        //Segundo Orden
        figures[k].u20=(double)figures[k].m20-figures[k].xPromedio*(double)figures[k].m10;
        figures[k].u02=(double)figures[k].m02-figures[k].yPromedio*(double)figures[k].m01;
        figures[k].u11=(double)figures[k].m11-figures[k].yPromedio*(double)figures[k].m10;
        
        //Tercer Orden
        figures[k].u30=(double)figures[k].m30-3*figures[k].xPromedio*(double)figures[k].m20+2*pow(figures[k].xPromedio,2)*(double)figures[k].m10;
        figures[k].u03=(double)figures[k].m03-3*figures[k].yPromedio*(double)figures[k].m02+2*pow(figures[k].yPromedio,2)*(double)figures[k].m01;

        figures[k].u12=(double)figures[k].m12-2*figures[k].yPromedio*(double)figures[k].m11-figures[k].xPromedio*(double)figures[k].m02+2*pow(figures[k].yPromedio,2)*(double)figures[k].m10;
        figures[k].u21=(double)figures[k].m21-2*figures[k].xPromedio*(double)figures[k].m11-figures[k].yPromedio*(double)figures[k].m20+2*pow(figures[k].xPromedio,2)*(double)figures[k].m01;

        //Momentos Invariantes
        figures[k].n02=figures[k].u02/(pow((double)figures[k].m00,2.0));
        figures[k].n03=figures[k].u03/(pow((double)figures[k].m00,((double)3/(double)2)+1.0));
        figures[k].n11=figures[k].u11/(pow((double)figures[k].m00,((double)2/(double)2)+1.0));
        figures[k].n12=figures[k].u12/(pow((double)figures[k].m00,((double)3/(double)2)+1.0));
        figures[k].n20=figures[k].u20/(pow((double)figures[k].m00,((double)2/(double)2)+1.0));
        figures[k].n21=figures[k].u21/(pow((double)figures[k].m00,((double)3/(double)2)+1.0));
        figures[k].n30=figures[k].u30/(pow((double)figures[k].m00,((double)3/(double)2)+1.0));

        //MOMENTOS de HU
        figures[k].phi1=figures[k].n20+figures[k].n02;
        figures[k].phi2=pow(figures[k].n20-figures[k].n02,2)+4*pow(figures[k].n11,2);
        figures[k].phi3=pow(figures[k].n30-3*figures[k].n12,2)+pow(3*figures[k].n21-figures[k].n03,2);
        figures[k].phi4=pow(figures[k].n30+figures[k].n12,2)+pow(figures[k].n21+figures[k].n03,2);

        figures[k].theta=0.5*atan2(2.0*figures[k].u11,figures[k].u20-figures[k].u02);

        figuresGlobVar.push_back(figures[k]);



    }

    int length = 50;
    figuresSize=figures.size();
    for( k=0; k<figuresSize; k++)
    {
        outputFile << "\nID: "<<IntToString(k)<<" | Color: "<<IntToString(figures[k].color[0])<<" "<<IntToString(figures[k].color[1])<<" "<<IntToString(figures[k].color[2])<<" | Area: "<<IntToString(figures[k].area)<<" ";
        outputFile<<"| m00: "<<IntToString(figures[k].m00)<<" | m10: "<<IntToString(figures[k].m10)<<" | m20: "<<IntToString(figures[k].m20)<<" | m30: "<<IntToString(figures[k].m30);
        outputFile<<" | m01: "<<IntToString(figures[k].m01)<<" | m02: "<<IntToString(figures[k].m02)<<" | m03: "<<IntToString(figures[k].m03);
        outputFile<<" | m11: "<<IntToString(figures[k].m11)<<" | m12: "<<IntToString(figures[k].m12)<<" | m21: "<<IntToString(figures[k].m21)<<" | XProm: "<<DoubleToString(figures[k].xPromedio)<<" | YProm: "<<DoubleToString(figures[k].yPromedio)<<" ";
        outputFile<<" | u10: "<<IntToString(figures[k].u10)<<" | u01: "<<IntToString(figures[k].u01)<<" | u20: "<<DoubleToString(figures[k].u20);
        outputFile<<" | u02: "<<DoubleToString(figures[k].u02)<<" | u11: "<<DoubleToString(figures[k].u11)<<" | u30: "<<DoubleToString(figures[k].u30);
        outputFile<<" | u03: "<<DoubleToString(figures[k].u03)<<" | u12: "<<DoubleToString(figures[k].u12)<<" | u21: "<<DoubleToString(figures[k].u21);
        outputFile<<" | n02: "<<DoubleToString(figures[k].n02)<<" | n03: "<<DoubleToString(figures[k].n03)<<" | n11: "<<DoubleToString(figures[k].n11);
        outputFile<<" | n12: "<<DoubleToString(figures[k].n12)<<" | n20: "<<DoubleToString(figures[k].n20)<<" | n21: "<<DoubleToString(figures[k].n21);
        outputFile<<" | n30: "<<DoubleToString(figures[k].n30)<<" | phi1: "<<DoubleToString(figures[k].phi1)<<" | phi2: "<<DoubleToString(figures[k].phi2);
        outputFile<<" | phi3: "<<DoubleToString(figures[k].phi3)<<" | phi4: "<<DoubleToString(figures[k].phi4)<<" | theta: "<<DoubleToString(figures[k].theta);
        outputFile<<" | Degrees: "<<DoubleToString(figures[k].theta*180 / 3.14159265);
        outputFile<<" | XP: "<<IntToString(figures[k].xPromedio+.5)<<" | YP: "<<IntToString(figures[k].yPromedio+.5)<<endl<<endl;

        // For training!
        // cout << DoubleToString(figures[k].phi1)<<" "<<DoubleToString(figures[k].phi2) << endl;
        //

        // Dibujamos sobre "segmentedImage" datos relevantes
        // centroide
        circle (segmentedImage, Point(figures[k].xPromedio+.5,figures[k].yPromedio+.5),4,Scalar(255,0,0),CV_FILLED);
        // angulo compuesto de 
        // dos lineas una horizontal y otra con el angulo al final
        // y un segmento de circulo para senalar el angulo
        line (
            segmentedImage, 
            Point(
                figures[k].xPromedio+.5, 
                figures[k].yPromedio+.5
                ), // Centroide
            Point(
                figures[k].xPromedio+.5 + length*cos(figures[k].theta), 
                figures[k].yPromedio+.5
                ), // Centroide + distancia a la derecha en X
            Scalar( 255, 0, 0), 2, 8, 0  
            );
        line (
            segmentedImage,
            Point(
                figures[k].xPromedio+.5,
                figures[k].yPromedio+.5
                ), // Centroide
            Point(
                figures[k].xPromedio+.5 + length*cos(figures[k].theta), // x 
                figures[k].yPromedio+.5 + length*sin(figures[k].theta) // y
                ),
                Scalar( 255, 0, 0), 2, 8, 0  
            );
        ellipse( segmentedImage, 
            Point(
                figures[k].xPromedio+.5,
                figures[k].yPromedio+.5 
                ),
            Size( length/2, length/2 ), 0, 0, figures[k].theta*180 / PI,
            Scalar( 0, 255, 0 ), 1, 8 );
        // Se pone un texto mencionando el angulo en grados
        // ostringstream textStream;
        // textStream << "Rotated ";
        // putText(segmentedImage, textStream.str(), cvPoint(figures[k].xPromedio+.5,figures[k].yPromedio+.5), 
        //     FONT_HERSHEY_COMPLEX_SMALL, 0.50, cvScalar(255,255,255), 1, CV_AA);
        // textStream.str("");
        // textStream << fixed;
        // textStream << setprecision(1);
        // textStream << (-1)*figures[k].theta*180 / PI;
        // textStream <<" Degrees";
        // //Pone texto en la Mat imageClick y el stream textStream lo pone en la posision
        // putText(segmentedImage, textStream.str(), cvPoint(figures[k].xPromedio+.5,figures[k].yPromedio+.5+10), 
        //     FONT_HERSHEY_COMPLEX_SMALL, 0.50, cvScalar(255,255,255), 1, CV_AA);

        /*
            //MOMENTOS NORMALIZADOS
    double n02;
    double n03;
    double n11;
    double n12;
    double n20;
    double n21;
    double n30;
    */
    }


}

// carlos training
// double phi1X=0.234635125, phi2X=0.010914375, phi1DevX=0.0173943456, phi2DevX=0.0022282768;
// double phi1I=0.2757821111, phi2I=0.0279318389, phi1DevI=0.0058238707, phi2DevI=0.0023386929;
// double phi1O=0.2207848824, phi2O=0.0062462229, phi1DevO=0.010904511, phi2DevO=0.001624447;
// double phi1L=0.325014, phi2L=0.0550844737, phi1DevL=0.0173370089, phi2DevL=0.0074505507;

// homeros training
double phi1X=0.3291002434, phi2X=0.0253875885, phi1DevX=0.0288278764, phi2DevX=0.0039292151;
double phi1I=0.4447836087, phi2I=0.1189788907, phi1DevI=0.0933866007, phi2DevI=0.0359672862;
double phi1O=0.3228794214, phi2O=0.0094902365, phi1DevO=0.0459546901, phi2DevO=0.0090612347;
double phi1L=0.5555926979, phi2L=0.1814852988, phi1DevL=0.0224679117, phi2DevL=0.0193219872;
double phi1R=0.2489313303, phi2R=0.0023523036, phi1DevR=0.0242349705, phi2DevR=0.0040109567;
double phi1Deadmau5=0.1995033381, phi2Deadmau5=0.003130226, phi1DevDeadmau5=0.0025950912, phi2DevDeadmau5=0.0005943853;

bool isX(double phi1, double phi2) {
    return phi1 >= (phi1X-phi1DevX) && phi1 <= (phi1X+phi1DevX) &&
            phi2 >= (phi2X-phi2DevX) && phi2 <= (phi2X+phi2DevX);
}

bool isI(double phi1, double phi2) {
    return phi1 >= (phi1I-phi1DevI) && phi1 <= (phi1I+phi1DevI) &&
            phi2 >= (phi2I-phi2DevI) && phi2 <= (phi2I+phi2DevI);
}

bool isO(double phi1, double phi2) {
    return phi1 >= (phi1O-phi1DevO) && phi1 <= (phi1O+phi1DevO) &&
            phi2 >= (phi2O-phi2DevO) && phi2 <= (phi2O+phi2DevO);
}

bool isL(double phi1, double phi2) {
    return phi1 >= (phi1L-phi1DevL) && phi1 <= (phi1L+phi1DevL) &&
            phi2 >= (phi2L-phi2DevL) && phi2 <= (phi2L+phi2DevL);
}

bool isR(double phi1, double phi2) {
    return phi1 >= (phi1R-phi1DevR) && phi1 <= (phi1R+phi1DevR) &&
            phi2 >= (phi2R-phi2DevR) && phi2 <= (phi2R+phi2DevR);
}

bool isDeadmau5(double phi1, double phi2) {
    return phi1 >= (phi1Deadmau5-phi1DevDeadmau5) && phi1 <= (phi1Deadmau5+phi1DevDeadmau5) &&
            phi2 >= (phi2Deadmau5-phi2DevDeadmau5) && phi2 <= (phi2Deadmau5+phi2DevDeadmau5);
}

double getDistance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1-x2,2)+pow(y1-y2,2));
}

double getMinFromArray(double *array, int size) {
    int k;
    double smallest = (double)array[0];
    for (k=1;k<size;k++) {
        smallest = min(smallest, (double)array[k]);
    }
    return smallest;
}

string rounded(double value, int precision) {
    ostringstream os;
    os << setprecision(precision) << fixed;
    os << value;
    return os.str();
}

void decision() {

}

void classification() {
    string rutina="";
    // ofstream output("reconocimiento.txt");
    int k;
    for(k=0;k<figuresGlobVar.size();k++) {
        double phi1=figuresGlobVar[k].phi1;
        double phi2=figuresGlobVar[k].phi2;
        double dX=getDistance(phi1, phi2, phi1X, phi2X);
        double dO=getDistance(phi1, phi2, phi1O, phi2O);
        double dI=getDistance(phi1, phi2, phi1I, phi2I);
        double dL=getDistance(phi1, phi2, phi1L, phi2L);
        double dR=getDistance(phi1, phi2, phi1R, phi2R);
        double dDeadmau5=getDistance(phi1, phi2, phi1Deadmau5, phi2Deadmau5);
        int size = 6;
        double distances[size];
        distances[0]=dX;
        distances[1]=dO;
        distances[2]=dI;
        distances[3]=dL;
        distances[4]=dR;
        distances[5]=dDeadmau5;

        if (isX(phi1, phi2) && getMinFromArray(distances, size) == dX) {
            figuresGlobVar[k].whatitis="X";
            // rutina 1
            // cout << "X" << endl;
            // rutina="rutina1 para X con angulo de " + rounded((-1)*figuresGlobVar[k].theta*180/PI, 1) + " grados";
            // output << rutina << endl;
        }
        else if (isI(phi1, phi2) && getMinFromArray(distances, size) == dI) {
            figuresGlobVar[k].whatitis="I";
            // cout << "I" << endl;
            // rutina="rutina1 para I con angulo de " + rounded((-1)*figuresGlobVar[k].theta*180/PI, 1) + " grados";
            // output << rutina << endl;
        }
        else if (isO(phi1, phi2) && getMinFromArray(distances, size) == dO) {
            figuresGlobVar[k].whatitis="O";
            // cout << "O" << endl;
            // rutina="rutina1 para O con angulo de " + rounded((-1)*figuresGlobVar[k].theta*180/PI, 1) + " grados";
            // output << rutina << endl;
        }
        else if (isL(phi1, phi2) && getMinFromArray(distances, size) == dL) {
            figuresGlobVar[k].whatitis="L";
            // cout << "L" << endl;
            // rutina="rutina1 para L con angulo de " + rounded((-1)*figuresGlobVar[k].theta*180/PI, 1) + " grados";
            // output << rutina << endl;
        }
        else if (isR(phi1, phi2) && getMinFromArray(distances, size) == dR) {
            figuresGlobVar[k].whatitis="R";
            // cout << "L" << endl;
            // rutina="rutina1 para R con angulo de " + rounded((-1)*figuresGlobVar[k].theta*180/PI, 1) + " grados";
            // output << rutina << endl;
        }
        else if (isDeadmau5(phi1, phi2) && getMinFromArray(distances, size) == dDeadmau5) {
            figuresGlobVar[k].whatitis="Deadmau5";
            // cout << "L" << endl;
            // rutina="rutina1 para Deadmau5 con angulo de " + rounded((-1)*figuresGlobVar[k].theta*180/PI, 1) + " grados";
            // output << rutina << endl;
        }
        else {
            figuresGlobVar[k].whatitis="Unknown";
            // cout << "desconocido" << endl;
            // rutina="objeto desconocido";
            // output << rutina << endl;
        }
    }
}

void createWindows() {
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
}

void histograms() {
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
}

void phisPlot() {
    Mat phis = Mat(selectedImage.rows, selectedImage.cols, selectedImage.type());
    Vec3b black(0, 0, 0);
    int i,j,k;
    //Inicializamos la matriz color toda en color negro
    for (i=0; i<selectedImage.rows; i++)
    {
        for (j=0; j<selectedImage.cols; j++)
        {
            phis.at<Vec3b>(i, j)=black;
        }
    }
    for(k=0;k<figuresGlobVar.size();k++) {
        Scalar color(figuresGlobVar[k].color);
        circle (phis, Point((int)(figuresGlobVar[k].phi1*selectedImage.cols),(selectedImage.rows)-(int)(figuresGlobVar[k].phi2*selectedImage.rows)),5,color,CV_FILLED);
    }
    int y=10;
    for(k=0;k<figuresGlobVar.size();k++, y+=10) {
        Scalar color(figuresGlobVar[k].color);
        circle (phis, Point(10, y),5,color,CV_FILLED);
        ostringstream textStream;
        textStream<<"("<<rounded(figuresGlobVar[k].phi1, 6)<<", "<<rounded(figuresGlobVar[k].phi2, 6)<<")"<<" "<<figuresGlobVar[k].whatitis;
        //Pone texto en la Mat imageClick y el stream textStream lo pone en la posision
        putText(phis, textStream.str(), Point(40, y), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
    }
    imshow("Phis (phi1, phi2)", phis);
}

int main(int argc,char* argv[])
{

    /*
**********************************

     ATENCION EQUIPO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

     La imagen binarizada se introduce en la funcion segment(ImagenBinarizada, ImagenSegmentada)

     Despues la imagen segmentada se introduce en la funcion momentos(ImagenSegmentada, figures)

     La funcion momentos recibe ademas como parametros un mapa, este mapa contendra los momentos de cada figura
     
     Este programa produce un archivo de texto llamado "figures.txt", por favor abranlo para que vean como esta estructurado todo



*/


    /* ESTE MAP CONTIENE EL ID, COLOR, Y MOMENTOS ESTADISTICOS DE CADA REGION

    */





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

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
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


    /* ventanas puestas por default
    // en las posiciones en pixeles
    // (10,10) (380, 10) (700, 10) (1020, 10)
    //         (380, 300) (700, 300) (1020, 300)
    */

    // createWindows();
    namedWindow("Click");
    setMouseCallback("Click", mouseCoordinatesExampleCallback);
    moveWindow("Click", 10, 10);
    // moveWindow("C1", 380, 300);
    // moveWindow("C2", 700, 300);
    // moveWindow("C3", 1020, 300);
    // moveWindow("Controls", 1020, 10);

    namedWindow("Filtered Image");
    namedWindow("SEGMENTACION");
    moveWindow("Filtered Image", 380, 10);
    moveWindow("SEGMENTACION", 700, 10);

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

        Vec3b aux;

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

        cap >> currentImage;


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

        // Histogram
        // histograms();

        //BGR to YIQ
        Mat yiqOurImage; bgr2yiq(currentImage, yiqOurImage);

        // imshow("YIQ1", yiqOurImage);

        //BGR to HSV
        Mat hsv;// cvtColor(currentImage, hsv, CV_BGR2HSV);
        // imshow("HSV", hsv);

        switch(selected) {
            case 1: selectedImage = currentImage; canales="RGB"; break;
            case 2: selectedImage = yiqOurImage; canales="YIQ"; break;
            case 3: selectedImage = hsv; canales="HSV"; break;
        }

        // Blur image
        blur(selectedImage,selectedImage,Size(3,3)); 
        // Filter image
        Mat filteredImage; filterColorFromImage(selectedImage, filteredImage);
        imshow("Filtered Image", filteredImage);
        segment(filteredImage,segmentedImg);
        momentos(segmentedImg);
        imshow("SEGMENTACION",segmentedImg);
        classification();
        // draw phis
        phisPlot();
        // take decision
        decision();

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
            case 'b': 
                segment(filteredImage,segmentedImg);
                momentos(segmentedImg);
                imshow("SEGMENTACION",segmentedImg);
                classification();
                decision();
            break;

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
        // heli->renewImage(image);

        // // Copy to OpenCV Mat
        // rawToMat(currentImage, image);
        

        usleep(15000);
    }
    
    heli->land();
    SDL_JoystickClose(m_joystick);
    delete heli;
    delete image;
    return 0;
}
