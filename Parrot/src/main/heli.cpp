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
#include <queue>

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
bool navigatedWithJoystick, joypadTakeOff, joypadLand, joypadHover, joypadScan;

int Px;
int Py;
int vC1=85, vC2=115, vC3=152;
int thresh1=22, thresh2=20, thresh3=36;

//Variables globales de figuras
double ang[2];
string let1 = "";
string let2 = "";


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
string canales = "YIQ";

map<unsigned int,struct caracterizacion> globalFigures;

// Matriz para convertir a YIQ
double yiqMat[3][3] = {
    {0.114, 0.587, 0.299},
    {-0.332, -0.274, 0.596},
    {0.312, -0.523, 0.211}
};

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
      srand(time(NULL)); //seed for the first time only!
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
                        LUT[idImage[y][x-1]].caracteristicas.m00+=LUT[idImage[y-1][x]].caracteristicas.m00;
                        LUT[idImage[y][x-1]].caracteristicas.m10+=LUT[idImage[y-1][x]].caracteristicas.m10;
                        LUT[idImage[y][x-1]].caracteristicas.m20+=LUT[idImage[y-1][x]].caracteristicas.m20;
                        LUT[idImage[y][x-1]].caracteristicas.m30+=LUT[idImage[y-1][x]].caracteristicas.m30;
                        LUT[idImage[y][x-1]].caracteristicas.m01+=LUT[idImage[y-1][x]].caracteristicas.m01;
                        LUT[idImage[y][x-1]].caracteristicas.m02+=LUT[idImage[y-1][x]].caracteristicas.m02;
                        LUT[idImage[y][x-1]].caracteristicas.m03+=LUT[idImage[y-1][x]].caracteristicas.m03;
                        LUT[idImage[y][x-1]].caracteristicas.m11+=LUT[idImage[y-1][x]].caracteristicas.m11;
                        LUT[idImage[y][x-1]].caracteristicas.m12+=LUT[idImage[y-1][x]].caracteristicas.m12;
                        LUT[idImage[y][x-1]].caracteristicas.m21+=LUT[idImage[y-1][x]].caracteristicas.m21;
                        LUT[idImage[y-1][x]].caracteristicas.area=0;
                        LUT[idImage[y-1][x]].caracteristicas.m00=0;
                        LUT[idImage[y-1][x]].caracteristicas.m10=0;
                        LUT[idImage[y-1][x]].caracteristicas.m20=0;
                        LUT[idImage[y-1][x]].caracteristicas.m30=0;
                        LUT[idImage[y-1][x]].caracteristicas.m01=0;
                        LUT[idImage[y-1][x]].caracteristicas.m02=0;
                        LUT[idImage[y-1][x]].caracteristicas.m03=0;
                        LUT[idImage[y-1][x]].caracteristicas.m11=0;
                        LUT[idImage[y-1][x]].caracteristicas.m12=0;
                        LUT[idImage[y-1][x]].caracteristicas.m21=0;
                        //Guardamos su tamaño
                        LUTSize=(unsigned int) LUT.size();


                        //Iteramos sobre la LTU
                        for (k=1; k<=LUTSize; k++)
                        {
                            //Quien tenga el color del pixel superior sera cambiado por el color del pixel lateral
                            if(LUT[k].color==regionColor)
                            {   
                                regionTemp.color=LUT[idImage[y][x-1]].color;
                                LUT[idImage[y][x-1]].caracteristicas.area+=LUT[k].caracteristicas.area;
                                LUT[idImage[y][x-1]].caracteristicas.m00+=LUT[k].caracteristicas.m00;
                                LUT[idImage[y][x-1]].caracteristicas.m10+=LUT[k].caracteristicas.m10;
                                LUT[idImage[y][x-1]].caracteristicas.m20+=LUT[k].caracteristicas.m20;
                                LUT[idImage[y][x-1]].caracteristicas.m30+=LUT[k].caracteristicas.m30;
                                LUT[idImage[y][x-1]].caracteristicas.m01+=LUT[k].caracteristicas.m01;
                                LUT[idImage[y][x-1]].caracteristicas.m02+=LUT[k].caracteristicas.m02;
                                LUT[idImage[y][x-1]].caracteristicas.m03+=LUT[k].caracteristicas.m03;
                                LUT[idImage[y][x-1]].caracteristicas.m11+=LUT[k].caracteristicas.m11;
                                LUT[idImage[y][x-1]].caracteristicas.m12+=LUT[k].caracteristicas.m12;
                                LUT[idImage[y][x-1]].caracteristicas.m21+=LUT[k].caracteristicas.m21;
                                regionTemp.caracteristicas.area=0;
                                regionTemp.caracteristicas.m00=0;
                                regionTemp.caracteristicas.m10=0;
                                regionTemp.caracteristicas.m20=0;
                                regionTemp.caracteristicas.m30=0;
                                regionTemp.caracteristicas.m01=0;
                                regionTemp.caracteristicas.m02=0;
                                regionTemp.caracteristicas.m03=0;
                                regionTemp.caracteristicas.m11=0;
                                regionTemp.caracteristicas.m12=0;
                                regionTemp.caracteristicas.m21=0;
                                LUT.erase(k);
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
                    regionTemp.caracteristicas.m00=0;
                    regionTemp.caracteristicas.m10=0;
                    regionTemp.caracteristicas.m20=0;
                    regionTemp.caracteristicas.m30=0;
                    regionTemp.caracteristicas.m01=0;
                    regionTemp.caracteristicas.m02=0;
                    regionTemp.caracteristicas.m03=0;
                    regionTemp.caracteristicas.m11=0;
                    regionTemp.caracteristicas.m12=0;
                    regionTemp.caracteristicas.m21=0;
                    idImage[y][x]=id;

                    LUT.insert(make_pair(id, regionTemp));

                    id=id+1;

                }

                //Aumentamos area
                LUT[idImage[y][x]].caracteristicas.area++;
                LUT[idImage[y][x]].caracteristicas.m00++; /* m00= [sum x sum y] 1 */
                LUT[idImage[y][x]].caracteristicas.m10+=x; /* m00= [sum x sum y] x */
                LUT[idImage[y][x]].caracteristicas.m20+=pow(x,2); /* m00= [sum x sum y] x² */
                LUT[idImage[y][x]].caracteristicas.m30+=pow(x,3); /* m00= [sum x sum y] x³ */
                LUT[idImage[y][x]].caracteristicas.m01+=y; /* m00= [sum x sum y] y */
                LUT[idImage[y][x]].caracteristicas.m02+=pow(y,2); /* m00= [sum x sum y] y² */
                LUT[idImage[y][x]].caracteristicas.m03+=pow(y,3); /* m00= [sum x sum y] y³ */
                LUT[idImage[y][x]].caracteristicas.m11+=x*y; /* m00= [sum x sum y] x*y */
                LUT[idImage[y][x]].caracteristicas.m12+=x*pow(y,2); /* m00= [sum x sum y] x*y² */
                LUT[idImage[y][x]].caracteristicas.m21+=pow(x,2)*y; /* m00= [sum x sum y] x²*y */

            }
        }

    }   


    // //Coloreamos la imagen en base a los valores de la LUT
    for (i=1; i<binarizedImage.rows-1; i++)
    {
        for (j=1; j<binarizedImage.cols-1; j++)
        {
            segmentedImage.at<Vec3b>(i, j)=LUT[idImage[i][j]].color;

        }
    }

    globalFigures.clear();
    LUTSize=(unsigned int) LUT.size();
    struct caracterizacion caracteristicas;
    // patch by removing duplicates
    map<unsigned int, struct caracterizacion> tempPatch;
    vector<unsigned int> indexes;
    unsigned int colorIndex;
    for( k=1; k<=LUTSize; k++)
    {
        colorIndex=(unsigned int)((LUT[k].color[0]+LUT[k].color[1]+LUT[k].color[2])/3*100);
        if (colorIndex) {
            if ( tempPatch.find(colorIndex) == tempPatch.end() ) {
                caracteristicas.color=LUT[k].color;
                caracteristicas.area=LUT[k].caracteristicas.area;
                caracteristicas.m00=LUT[k].caracteristicas.m00;
                caracteristicas.m10=LUT[k].caracteristicas.m10;
                caracteristicas.m20=LUT[k].caracteristicas.m20;
                caracteristicas.m30=LUT[k].caracteristicas.m30;
                caracteristicas.m01=LUT[k].caracteristicas.m01;
                caracteristicas.m02=LUT[k].caracteristicas.m02;
                caracteristicas.m03=LUT[k].caracteristicas.m03;
                caracteristicas.m11=LUT[k].caracteristicas.m11;
                caracteristicas.m12=LUT[k].caracteristicas.m12;
                caracteristicas.m21=LUT[k].caracteristicas.m21;
                // not found
                indexes.push_back(colorIndex);
                tempPatch.insert(make_pair(colorIndex, caracteristicas));
            } else {
                 // found
                tempPatch[colorIndex].area+=LUT[k].caracteristicas.area;
                tempPatch[colorIndex].m00+=LUT[k].caracteristicas.m00;
                tempPatch[colorIndex].m10+=LUT[k].caracteristicas.m10;
                tempPatch[colorIndex].m20+=LUT[k].caracteristicas.m20;
                tempPatch[colorIndex].m30+=LUT[k].caracteristicas.m30;
                tempPatch[colorIndex].m01+=LUT[k].caracteristicas.m01;
                tempPatch[colorIndex].m02+=LUT[k].caracteristicas.m02;
                tempPatch[colorIndex].m03+=LUT[k].caracteristicas.m03;
                tempPatch[colorIndex].m11+=LUT[k].caracteristicas.m11;
                tempPatch[colorIndex].m12+=LUT[k].caracteristicas.m12;
                tempPatch[colorIndex].m21+=LUT[k].caracteristicas.m21;
            }
        }
    }

    // Almacenamos tabla  
    for( k=0; k<indexes.size(); k++)
    {
        caracteristicas.color=tempPatch[indexes[k]].color;
        caracteristicas.area=tempPatch[indexes[k]].area;
        caracteristicas.m00=tempPatch[indexes[k]].m00;
        caracteristicas.m10=tempPatch[indexes[k]].m10;
        caracteristicas.m20=tempPatch[indexes[k]].m20;
        caracteristicas.m30=tempPatch[indexes[k]].m30;
        caracteristicas.m01=tempPatch[indexes[k]].m01;
        caracteristicas.m02=tempPatch[indexes[k]].m02;
        caracteristicas.m03=tempPatch[indexes[k]].m03;
        caracteristicas.m11=tempPatch[indexes[k]].m11;
        caracteristicas.m12=tempPatch[indexes[k]].m12;
        caracteristicas.m21=tempPatch[indexes[k]].m21;
        globalFigures.insert(make_pair(k, caracteristicas));
       
    }
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

void giraIzq() {
cout<<"Gira Izquierda"<<endl;
//hover
//heli->setAngles(pitch, roll, yaw, height, hover);
heli->setAngles(0.0, -10000.0, 0.0, 0.0, 0.0);
usleep(500000);
}

void giraDer() {
cout<<"Gira Derecha"<<endl;
//hover
//heli->setAngles(pitch, roll, yaw, height, hover);
heli->setAngles(0.0, 10000.0, 0.0, 0.0, 0.0);
usleep(500000);
}

void avanza() {
cout<<"Avanza"<<endl;
heli->setAngles(-10000, 0.0, 0.0, 0.0, 0.0);
usleep(500000);
}

void retrocede() {
cout<<"Retrocede"<<endl;
heli->setAngles(10000, 0.0, 0.0, 0.0, 0.0);
usleep(500000);
}

void sube() {
cout<<"Sube"<<endl;
//hover
//heli->setAngles(pitch, roll, yaw, height, hover);
heli->setAngles(0.0, 0.0, 0, 10000, 0.0);
usleep(500000);
}

void baja() {
cout<<"Baja"<<endl;
//hover
//heli->setAngles(pitch, roll, yaw, height, hover);
heli->setAngles(0.0, 0.0, 0, -10000, 0.0);
usleep(500000);
}


//Obtencion de momentos estadisticos
void momentos(Mat &segmentedImage)
{
    unsigned  id,k,figuresSize;
    unsigned long long i, j,x,y;
    Vec3b black(0,0,0);
    id=0;
    struct caracterizacion caracteristicas;

    //OBTENEMOS MOMENTOS CENTRALIZADOS (Para estos ya no necesitamos iterar la figura)
    figuresSize=globalFigures.size();
    for( k=0; k<figuresSize; k++)
    {
        //OBTENEMOS PROMEDIOS
        globalFigures[k].xPromedio=((double)globalFigures[k].m10)/((double)globalFigures[k].m00);
        globalFigures[k].yPromedio=((double)globalFigures[k].m01)/((double)globalFigures[k].m00);

        
        //Primer Orden
        globalFigures[k].u00=globalFigures[k].m00;
        globalFigures[k].u10=0;
        globalFigures[k].u01=0;

        //Segundo Orden
        globalFigures[k].u20=(double)globalFigures[k].m20-globalFigures[k].xPromedio*(double)globalFigures[k].m10;
        globalFigures[k].u02=(double)globalFigures[k].m02-globalFigures[k].yPromedio*(double)globalFigures[k].m01;
        globalFigures[k].u11=(double)globalFigures[k].m11-globalFigures[k].yPromedio*(double)globalFigures[k].m10;
        
        //Tercer Orden
        globalFigures[k].u30=(double)globalFigures[k].m30-3*globalFigures[k].xPromedio*(double)globalFigures[k].m20+2*pow(globalFigures[k].xPromedio,2)*(double)globalFigures[k].m10;
        globalFigures[k].u03=(double)globalFigures[k].m03-3*globalFigures[k].yPromedio*(double)globalFigures[k].m02+2*pow(globalFigures[k].yPromedio,2)*(double)globalFigures[k].m01;

        globalFigures[k].u12=(double)globalFigures[k].m12-2*globalFigures[k].yPromedio*(double)globalFigures[k].m11-globalFigures[k].xPromedio*(double)globalFigures[k].m02+2*pow(globalFigures[k].yPromedio,2)*(double)globalFigures[k].m10;
        globalFigures[k].u21=(double)globalFigures[k].m21-2*globalFigures[k].xPromedio*(double)globalFigures[k].m11-globalFigures[k].yPromedio*(double)globalFigures[k].m20+2*pow(globalFigures[k].xPromedio,2)*(double)globalFigures[k].m01;

        //Momentos Invariantes
        globalFigures[k].n02=globalFigures[k].u02/(pow((double)globalFigures[k].m00,2.0));
        globalFigures[k].n03=globalFigures[k].u03/(pow((double)globalFigures[k].m00,((double)3/(double)2)+1.0));
        globalFigures[k].n11=globalFigures[k].u11/(pow((double)globalFigures[k].m00,((double)2/(double)2)+1.0));
        globalFigures[k].n12=globalFigures[k].u12/(pow((double)globalFigures[k].m00,((double)3/(double)2)+1.0));
        globalFigures[k].n20=globalFigures[k].u20/(pow((double)globalFigures[k].m00,((double)2/(double)2)+1.0));
        globalFigures[k].n21=globalFigures[k].u21/(pow((double)globalFigures[k].m00,((double)3/(double)2)+1.0));
        globalFigures[k].n30=globalFigures[k].u30/(pow((double)globalFigures[k].m00,((double)3/(double)2)+1.0));

        //MOMENTOS de HU
        globalFigures[k].phi1=globalFigures[k].n20+globalFigures[k].n02;
        globalFigures[k].phi2=pow(globalFigures[k].n20-globalFigures[k].n02,2)+4*pow(globalFigures[k].n11,2);
        globalFigures[k].phi3=pow(globalFigures[k].n30-3*globalFigures[k].n12,2)+pow(3*globalFigures[k].n21-globalFigures[k].n03,2);
        globalFigures[k].phi4=pow(globalFigures[k].n30+globalFigures[k].n12,2)+pow(globalFigures[k].n21+globalFigures[k].n03,2);

        globalFigures[k].theta=0.5*atan2(2.0*globalFigures[k].u11,globalFigures[k].u20-globalFigures[k].u02);

    }

    int length = 50;
    figuresSize=globalFigures.size();
    for( k=0; k<figuresSize; k++)
    {
        // For training!
        //cout << DoubleToString(globalFigures[k].phi1)<<" "<<DoubleToString(globalFigures[k].phi2) << endl;
        //

        // Dibujamos sobre "segmentedImage" datos relevantes
        // centroide
        circle (segmentedImage, Point(globalFigures[k].xPromedio+.5,globalFigures[k].yPromedio+.5),4,Scalar(255,0,0),CV_FILLED);
        // angulo compuesto de 
        // dos lineas una horizontal y otra con el angulo al final
        // y un segmento de circulo para senalar el angulo
        line (
            segmentedImage, 
            Point(
                globalFigures[k].xPromedio+.5, 
                globalFigures[k].yPromedio+.5
                ), // Centroide
            Point(
                globalFigures[k].xPromedio+.5 + length*cos(globalFigures[k].theta), 
                globalFigures[k].yPromedio+.5
                ), // Centroide + distancia a la derecha en X
            Scalar( 255, 0, 0), 2, 8, 0  
            );
        line (
            segmentedImage,
            Point(
                globalFigures[k].xPromedio+.5,
                globalFigures[k].yPromedio+.5
                ), // Centroide
            Point(
                globalFigures[k].xPromedio+.5 + length*cos(globalFigures[k].theta), // x 
                globalFigures[k].yPromedio+.5 + length*sin(globalFigures[k].theta) // y
                ),
                Scalar( 255, 0, 0), 2, 8, 0  
            );
        ellipse( segmentedImage, 
            Point(
                globalFigures[k].xPromedio+.5,
                globalFigures[k].yPromedio+.5 
                ),
            Size( length/2, length/2 ), 0, 0, globalFigures[k].theta*180 / PI,
            Scalar( 0, 255, 0 ), 1, 8 );
    }

}

// carlos training
// double phi1X=0.234635125, phi2X=0.010914375, phi1DevX=0.0173943456, phi2DevX=0.0022282768;
// double phi1I=0.2757821111, phi2I=0.0279318389, phi1DevI=0.0058238707, phi2DevI=0.0023386929;
// double phi1O=0.2207848824, phi2O=0.0062462229, phi1DevO=0.010904511, phi2DevO=0.001624447;
// double phi1L=0.325014, phi2L=0.0550844737, phi1DevL=0.0173370089, phi2DevL=0.0074505507;

// homeros training

#define trainedPhisSize 4
string trainedObjects[trainedPhisSize] = {"X", "I", "L", "R"};

// ORDER -->  {PHI1_AVERAGE, PHI1_STANDARD_DEVIATION, PHI2_AVERAGE, PHI2_STANDARD_DEVIATION}
double trainedPhis[trainedPhisSize][4] = {
    {0.3661988504, 0.0414660219, 0.0323226694, 0.0070028227}, // X
    {0.440089257, 0.0295243932, 0.0877471495, 0.0277660379}, // I
    //{0.3648078675, 0.0100242852, 0.0098792798, 0.0010190414}, // O
    {0.5499775581, 0.0225243932, 0.1812142186, 0.0277660379}, // L
    {0.2763952578, 0.0123850278, 0.0016686626, 0.0020864559}, // R
    //{0.1995033381, 0.0025950912, 0.003130226, 0.0005943853}, // Deadmau5
};
int trainedPhisColors[trainedPhisSize][3] = {
    {0, 245, 0}, // X
    {34, 21, 132}, // I
    //{132, 140, 77}, // O
    {191, 113, 24}, // L
    {0, 245, 245}, // R
    //{173, 46, 143}, // Deadmau5
};

// Checks whether (testPhi1, testPhi2) intersects in [range (phi1Avg+-phi1StdDev) and range (phi2Avg+-phi2StdDev)]
bool intersects(double testPhi1Avg, double testPhi2Avg, double phi1Avg, double phi2Avg, double phi1StdDev, double phi2StdDev) {
    return testPhi1Avg >= (phi1Avg-phi1StdDev) && testPhi1Avg <= (phi1Avg+phi1StdDev) &&
            testPhi2Avg >= (phi2Avg-phi2StdDev) && testPhi2Avg <= (phi2Avg+phi2StdDev);
}

double getDistance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1-x2,2)+pow(y1-y2,2));
}

double getMinFromList(vector<double> list) {
    int k;
    double smallest = (double)list[0];
    for (k=1;k<list.size();k++) {
        smallest = min(smallest, (double)list[k]);
    }
    return smallest;
}

double getMaxFromList(vector<double> list) {
    int k;
    double biggest = (double)list[0];
    for (k=1;k<list.size();k++) {
        biggest = max(biggest, (double)list[k]);
    }
    return biggest;
}

string rounded(double value, int precision) {
    ostringstream os;
    os << setprecision(precision) << fixed;
    os << value;
    return os.str();
}

int itsPosIs(double phi1Avg, double phi2Avg) {
    int index;
    for (index=0;index<trainedPhisSize;index++) {
        if (intersects(phi1Avg, phi2Avg, trainedPhis[index][0], trainedPhis[index][2], trainedPhis[index][1], trainedPhis[index][3]))
            return index;
    }
    return -1;
}

string itsNameIs(double phi1Avg, double phi2Avg, vector<double> distances) {
    double minDistance=getMinFromList(distances);
    int index;
    for (index=0;index<trainedPhisSize;index++) {
        if (minDistance == distances[index] && intersects(phi1Avg, phi2Avg, trainedPhis[index][0], trainedPhis[index][2], trainedPhis[index][1], trainedPhis[index][3])) {
            return trainedObjects[index];
        }
    }
    return "Unknown";
}

void classification() {
    // ofstream output("reconocimiento.txt");

    int k;
    for(k=0;k<globalFigures.size();k++) {
        double phi1=globalFigures[k].phi1;
        double phi2=globalFigures[k].phi2;
        vector<double> distances;
        int index;
        for (index=0;index<trainedPhisSize;index++) {
            distances.push_back(getDistance(phi1, phi2, trainedPhis[index][0], trainedPhis[index][2]));
        }
        globalFigures[k].whatitis=itsNameIs(phi1, phi2, distances);
        if(k == 0)
        {
            let1 = globalFigures[k].whatitis;    
        }
        else 
        {
            let2 = globalFigures[k].whatitis;    
        }

        ang[k]= globalFigures[k].theta * 180/PI;
        // guardar whatitis   ----->  globalFigures[k].whatitis
        // guardar theta ---------> globalFigures[k].theta
    }
}

//AQUI ME QUEDEEEEEEEEEEEEEEEE_ continue
void decision() {
    if(let1 != "" && let2 != "")
    {
        bool largo = FALSE;
        bool corto = FALSE;
        double angulo = 0;

        if(let1 == "I" || let2 == "L")
        {
            largo = TRUE;

            if(let1 == "I" || let1 == "L")
            {
                angulo = ang[0];
            }

            else if(let2 == "I" || let2 == "L")
            {
                angulo = ang[1];
            }
        }

        cout << "Figura 1: " << let1 << "\t Angulo: " << ang[0] << endl;
        cout << "Figura 2: " << let2 << "\t Angulo: " << ang[1] << endl;
    }
    else
    {
        cout << "The pair of letters was not detected" << endl;
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

void phisPlot(double multiplier, double pointSize) {
    Mat phis = Mat(selectedImage.rows*multiplier, selectedImage.cols*multiplier, selectedImage.type());
    Vec3b black(0, 0, 0);
    int i,j,k;
    int offset=5*multiplier;
    //Inicializamos la matriz color toda en color negro
    for (i=0; i<phis.rows; i++)
    {
        for (j=0; j<phis.cols; j++)
        {
            phis.at<Vec3b>(i, j)=black;
        }
    }
    for(k=0;k<globalFigures.size();k++) {
        Scalar color(globalFigures[k].color);
        circle (phis, Point((int)(globalFigures[k].phi1*phis.cols),(phis.rows-offset)-(int)(globalFigures[k].phi2*phis.rows)),pointSize,color,CV_FILLED);
    }
    int y=10;
    for(k=0;k<globalFigures.size();k++, y+=10) {
        Scalar color(globalFigures[k].color);
        circle (phis, Point(20, y-5),5,color,CV_FILLED);
        ostringstream textStream;
        textStream<<"("<<rounded(globalFigures[k].phi1, 6)<<", "<<rounded(globalFigures[k].phi2, 6)<<")"<<" "<<globalFigures[k].whatitis;
        //Pone texto en la Mat imageClick y el stream textStream lo pone en la posision
        putText(phis, textStream.str(), Point(40, y), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
        if (globalFigures[k].whatitis!="Unknown") {
            int position=itsPosIs(globalFigures[k].phi1, globalFigures[k].phi2);
            Scalar objectColor(trainedPhisColors[position][0], trainedPhisColors[position][1], trainedPhisColors[position][2]);
            circle (phis, Point(phis.cols-40, y-5),5,objectColor,CV_FILLED);
        }
    }
    /*
    code for showing phis areas
    */
    int index;
    for(index=0;index<trainedPhisSize;index++) {

        // show area by using ellipse
        ellipse(
            phis, 
            Point(
                trainedPhis[index][0] * phis.cols,
                (phis.rows-offset) - trainedPhis[index][2] * phis.rows
                ),
            Size( (trainedPhis[index][1] * phis.cols) , (trainedPhis[index][3] * phis.rows) ),
            0, 0, 360,
            Scalar(trainedPhisColors[index][0], trainedPhisColors[index][1], trainedPhisColors[index][2]), 
            1, 8 );

        // show center by using cross marker made of 4 lines
        line(phis, 
            Point(
                trainedPhis[index][0] * phis.cols,
                (phis.rows-offset) - trainedPhis[index][2] * phis.rows
                ),
            Point(
                (trainedPhis[index][0]-trainedPhis[index][1]) * phis.cols,
                (phis.rows-offset) - (trainedPhis[index][2]) * phis.rows
                ),
            Scalar(trainedPhisColors[index][0], trainedPhisColors[index][1], trainedPhisColors[index][2]),
            2, 8, 0  );
        line(phis, 
            Point(
                trainedPhis[index][0] * phis.cols,
                (phis.rows-offset) - trainedPhis[index][2] * phis.rows
                ),
            Point(
                (trainedPhis[index][0]+trainedPhis[index][1]) * phis.cols,
                (phis.rows-offset) - (trainedPhis[index][2]) * phis.rows
                ),
            Scalar(trainedPhisColors[index][0], trainedPhisColors[index][1], trainedPhisColors[index][2]),
            2, 8, 0  );
        line(phis, 
            Point(
                trainedPhis[index][0] * phis.cols,
                (phis.rows-offset) - trainedPhis[index][2] * phis.rows
                ),
            Point(
                (trainedPhis[index][0]) * phis.cols,
                (phis.rows-offset) - (trainedPhis[index][2]-trainedPhis[index][3]) * phis.rows
                ),
            Scalar(trainedPhisColors[index][0], trainedPhisColors[index][1], trainedPhisColors[index][2]),
            2, 8, 0  );
        line(phis, 
            Point(
                trainedPhis[index][0] * phis.cols,
                (phis.rows-offset) - trainedPhis[index][2] * phis.rows
                ),
            Point(
                (trainedPhis[index][0]) * phis.cols,
                (phis.rows-offset) - (trainedPhis[index][2]+trainedPhis[index][3]) * phis.rows
                ),
            Scalar(trainedPhisColors[index][0], trainedPhisColors[index][1], trainedPhisColors[index][2]),
            2, 8, 0  );

        
        // put text to indicate what each area represent
        putText(phis, trainedObjects[index],
            Point(
                (trainedPhis[index][0]+trainedPhis[index][1]) * phis.cols+offset,
                (phis.rows-offset) - trainedPhis[index][2] * phis.rows
                ), 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);

    }
    imshow("Phis (phi1, phi2)", phis);
}
String base="/home/vision/Desktop/";
String filename="conobs.png";
String window_name="Display window";
Mat stage;
Mat tempStage;
Point obstacle1;
Point obstacle2;
Point robot;
bool moveObstable1=false;
bool moveObstable2=false;
bool moveRobot=false;
int obstacleRadius = 20;
int robotRadius = 1;
int maxRadius=70;
int pointRadius=10;
Scalar obstacleColor=Scalar(0,0,0);
int maxValue=65535;
Mat gota_aceite_espacio;
Point finalPoint;
int leftOrRight=maxRadius/2;
int initialDir=0;
Scalar startColor=Scalar(0,255,0);
Scalar pathColor=Scalar(255,0,255);
Scalar endColor=Scalar(0,0,255);

int oposite(int direction) {
    int opositeDirection=-1;
    switch(direction) {
        case 0:
            opositeDirection=2;
            break;
        case 1:
            opositeDirection=3;
            break;
        case 2:
            opositeDirection=0;
            break;
        case 3:
            opositeDirection=1;
            break;
    }
    return opositeDirection;
}

void findPath(Mat &dst, Mat &src, Point start, Point end, int direction) {
    Point current=start;
    int currentValue;
    currentValue = src.at<Vec3w>(current.y, current.x)[0];
    circle(dst, start, 5, startColor, -1);
    putText(dst, "Start Point", start, 
            FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(0,0,0), 1, CV_AA);
    vector<Point>distances; // right top left bottom
    
    while(currentValue) {
        vector<double>values;
        vector<Point>coordinates; // right top left bottom
        int right=0;
        int top=1;
        int left=2;
        int bottom=3;
        coordinates.push_back(Point(current.x+1, current.y));
        coordinates.push_back(Point(current.x, current.y-1));
        coordinates.push_back(Point(current.x-1, current.y));
        coordinates.push_back(Point(current.x, current.y+1));
        
        values.push_back(src.at<Vec3w>(coordinates[right].y, coordinates[right].x)[0]);
        values.push_back(src.at<Vec3w>(coordinates[top].y, coordinates[top].x)[0]);
        values.push_back(src.at<Vec3w>(coordinates[left].y, coordinates[left].x)[0]);
        values.push_back(src.at<Vec3w>(coordinates[bottom].y, coordinates[bottom].x)[0]);
        int lower=getMinFromList(values);
        int lowerIndex=0;
        int pathchoices=0;
        if (lower == values[0]) {
            lowerIndex=0;
            pathchoices++;
        }
        if (lower == values[1]) {
            lowerIndex=1;
            pathchoices++;
        }
        if (lower == values[2]) {
            lowerIndex=2;
            pathchoices++;
        }
        if (lower == values[3]) {
            lowerIndex=3;
            pathchoices++;
        }
        if (pathchoices==1)
            direction=lowerIndex;



        circle(dst, coordinates[direction], 1, pathColor, -1);
        // if(values[direction] > currentValue) {
        //     direction=oposite(direction);
        // }
        currentValue=values[direction];
        current=coordinates[direction];
    }
    circle(dst, end, 5, endColor, -1);
    putText(dst, "End Point", end, 
                FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(0,0,0), 1, CV_AA);
}

void gotaDeAceite(Mat &dst, Mat &src, Point semilla) {
    int goal=0;
    int acum=0;
    std::queue<Point> puntos_gota_de_aceite;
    std::queue<int> counts;
    puntos_gota_de_aceite.push(semilla);
    dst.at<Vec3w>(semilla.y, semilla.x) = Vec3w(acum, acum, acum);
    counts.push(1);
    int count=0;
    count+=counts.front();
    counts.pop();
    while (!puntos_gota_de_aceite.empty()) {
        int children=0;
        Point elemento=puntos_gota_de_aceite.front();
        puntos_gota_de_aceite.pop();

        Point right = Point(elemento.x+1, elemento.y);
        Point top  = Point(elemento.x, elemento.y-1);
        Point left  = Point(elemento.x-1, elemento.y);
        Point bottom  = Point(elemento.x, elemento.y+1);
        if (src.at<Vec3b>(right.y, right.x)[0] && dst.at<Vec3w>(right.y, right.x)[0] == maxValue){
            puntos_gota_de_aceite.push(right);
            dst.at<Vec3w>(right.y, right.x) = Vec3w(acum+1, acum+1, acum+1);
            children++;
        }
        if (src.at<Vec3b>(top.y, top.x)[0] && dst.at<Vec3w>(top.y, top.x)[0] == maxValue){
            puntos_gota_de_aceite.push(top);
            dst.at<Vec3w>(top.y, top.x) = Vec3w(acum+1, acum+1, acum+1);
            children++;
        }
        if (src.at<Vec3b>(left.y, left.x)[0] && dst.at<Vec3w>(left.y, left.x)[0] == maxValue){
            puntos_gota_de_aceite.push(left);
            dst.at<Vec3w>(left.y, left.x) = Vec3w(acum+1, acum+1, acum+1);
            children++;
        }
        if (src.at<Vec3b>(bottom.y, bottom.x)[0] && dst.at<Vec3w>(bottom.y, bottom.x)[0] == maxValue){
            puntos_gota_de_aceite.push(bottom);
            dst.at<Vec3w>(bottom.y, bottom.x) = Vec3w(acum+1, acum+1, acum+1);
            children++;
        }
        counts.push(children);
        count--;
        if (!count) {
            acum++;
            while(!counts.empty()) {
                count+=counts.front();
                counts.pop();
            }
        }
    }
    // cout << acum << endl;
}

Point topLeft, topRight, bottomRight, bottomLeft;
void stageSpace(Mat &image) {
    
    line(image, Point(topLeft.x+robotRadius, topLeft.y+2*robotRadius), Point(bottomLeft.x+robotRadius, bottomLeft.y-2*robotRadius), obstacleColor);
    line(image, Point(topLeft.x+2*robotRadius, topLeft.y+robotRadius), Point(topRight.x-2*robotRadius, topRight.y+robotRadius), obstacleColor);
    line(image, Point(bottomLeft.x+2*robotRadius, bottomLeft.y-robotRadius), Point(bottomRight.x-2*robotRadius, bottomRight.y-robotRadius), obstacleColor);
    line(image, Point(topRight.x-robotRadius, topRight.y+2*robotRadius), Point(bottomRight.x-robotRadius, bottomRight.y-2*robotRadius), obstacleColor);

    ellipse(image, Point(topLeft.x+2*robotRadius, topLeft.y+2*robotRadius), Size(robotRadius, robotRadius), 0, 180, 270, obstacleColor);
    ellipse(image, Point(topRight.x-2*robotRadius, topRight.y+2*robotRadius), Size(robotRadius, robotRadius), 0, 270, 360, obstacleColor);
    ellipse(image, Point(bottomRight.x-2*robotRadius, bottomRight.y-2*robotRadius), Size(robotRadius, robotRadius), 0, 0, 90, obstacleColor);
    ellipse(image, Point(bottomLeft.x+2*robotRadius, bottomLeft.y-2*robotRadius), Size(robotRadius, robotRadius), 0, 90, 180, obstacleColor);
}


void obstacles(Mat &image) {
    circle(image, obstacle1, obstacleRadius+robotRadius, obstacleColor, -1);
    circle(image, obstacle2, obstacleRadius+robotRadius, obstacleColor, -1);
}

void obstaclesBorder(Mat &image) {
    circle(image, obstacle1, obstacleRadius+robotRadius, obstacleColor, 1);
    circle(image, obstacle2, obstacleRadius+robotRadius, obstacleColor, 1);
}

void view_refresh() {
    tempStage.setTo(Scalar(255, 255, 255));
    stageSpace(tempStage);
    obstacles(tempStage);
    stage = imread(base+filename, CV_LOAD_IMAGE_COLOR);   // Read the file
    stageSpace(stage);
    obstaclesBorder(stage);
}


void on_radius_change( int, void* ){
    view_refresh();
    imshow( window_name, stage );
}

void on_left_right_selection( int, void* ){
    int max=maxRadius;
    if (leftOrRight > max/2) {
        initialDir=0;

    }
    else {
        initialDir=2;
    }
}

bool insideCircle(int x, int y, Point &center, int radius) {
    int center_x=center.x, center_y=center.y;
    return (pow((x - center_x),2) + pow((y - center_y),2) < pow(radius,2));
}

void mouseHandler(int event, int x, int y, int flags, void *param)
{
    switch(event) {
    case CV_EVENT_LBUTTONDOWN:      //left button press
        if (insideCircle(x, y, obstacle1, obstacleRadius+robotRadius)) {
            moveObstable1=!moveObstable1;
            view_refresh();
        }
        else if (insideCircle(x, y, obstacle2, obstacleRadius+robotRadius)) {
            moveObstable2=!moveObstable2;
            view_refresh();
        }
        else if (insideCircle(x,y,robot, obstacleRadius+robotRadius)){
            moveRobot=!moveRobot;
            view_refresh();

        }
        else {
            finalPoint.x=x;
            finalPoint.y=y;
            view_refresh();
            gota_aceite_espacio.setTo(Vec3w(maxValue, maxValue, maxValue));
            gotaDeAceite(gota_aceite_espacio, tempStage, Point(finalPoint.x, finalPoint.y));
            circle(stage, finalPoint, 5, endColor, -1);
            putText(stage, "End Point", finalPoint, 
                FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(0,0,0), 1, CV_AA);
        }
        imshow( window_name, stage );
        break;
    case CV_EVENT_RBUTTONDOWN: // right button press
        view_refresh();
        findPath(stage, gota_aceite_espacio, Point(x, y), finalPoint, initialDir);
        imshow( window_name, stage );
        break;
    case CV_EVENT_MOUSEMOVE:
        if (moveObstable1) {
            obstacle1.x=x;
            obstacle1.y=y;
        }
        
        if (moveObstable2) {
            obstacle2.x=x;
            obstacle2.y=y;

        }
        if (moveRobot) {
            robot.x=x;
            robot.y=y;
        }
        /* draw a rectangle*/
        break;

        
    }
}

int main(int argc,char* argv[])
{
    stage = imread(base+filename, CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! stage.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    topLeft=Point(0, 40);
    topRight=Point(stage.cols, 40);
    bottomRight=Point(stage.cols, stage.rows);
    bottomLeft=Point(0, stage.rows);

    namedWindow( window_name, WINDOW_AUTOSIZE );// Create a window for display.
    createTrackbar( "Robot Radius", window_name, &robotRadius, maxRadius, on_radius_change );
    createTrackbar( "Left Or Right", window_name, &leftOrRight, maxRadius, on_left_right_selection );
    setMouseCallback( window_name, mouseHandler);
    // x = stage.cols / 2
    obstacle1=Point(362,280);
    obstacle2=Point(362,538);
    finalPoint=Point(stage.cols/2, stage.rows/2);
    // robot=Point(50,50);
    // circle(stage, robot, robotRadius, Scalar(255,0,0), -1);
    tempStage = Mat(stage.rows, stage.cols, CV_8UC3, Scalar(255, 255, 255));
    gota_aceite_espacio=Mat(stage.rows, stage.cols, CV_16UC3, Scalar(maxValue, maxValue, maxValue));
    view_refresh();
    gotaDeAceite(gota_aceite_espacio, tempStage, Point(finalPoint.x, finalPoint.y));
    imshow( window_name, stage );   
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}