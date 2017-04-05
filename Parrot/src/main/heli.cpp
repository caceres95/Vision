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

//Esta estructra servira para almacenar el color de una region y sus momentos caracteristicos
struct region {
  Vec3b color;
  unsigned int area;


} ;

struct caracterizacion{
    //Estructura con todas los momentos estadisticos que puede tener una figura
    Vec3b color;
    unsigned int area;
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
    int i, j; //Para los ciclos
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
    for (i=1; i<binarizedImage.rows-1; i++)
    {
        for (j=1; j<binarizedImage.cols-1; j++)
        {
            if(binarizedImage.at<Vec3b>(i,j)==black)
            {
                continue;
            }

            else //La imagen orginal tiene un 1
            {
                Pi=binarizedImage.at<Vec3b>(i,j-1);
                Ps=binarizedImage.at<Vec3b>(i-1,j);
                Pc=binarizedImage.at<Vec3b>(i,j);

                if(Ps==white && Pi == black)
                {
                    //Propagacion descendiente
                    idImage[i][j]=idImage[i-1][j];

                }
                else if(Ps==black && Pi == white)
                {
                    //Propagacion lateral
                    idImage[i][j]=idImage[i][j-1];
                }

                else if(Ps==white && Pi == white)
                {
                    //Propagacion indistinta, tenemos que detectar conflicto
                    if(LUT[idImage[i-1][j]].color != LUT[idImage[i][j-1]].color)
                    {

                        
                        //Region color contendra el color del pixel superior
                        regionColor=LUT[idImage[i-1][j]].color;

                        //Borrar dos lineas en caso de error
                        LUT[idImage[i][j-1]].area+=LUT[idImage[i-1][j]].area;
                        LUT[idImage[i-1][j]].area=0;
                        //Guardamos su tamaño
                        LUTSize=(unsigned int) LUT.size();


                        //Iteramos sobre la LTU
                        for (k=1; k<=LUTSize; k++)
                        {
                            //Quien tenga el color del pixel superior sera cambiado por el color del pixel lateral
                            if(LUT[k].color==regionColor)
                            {
                                areaTemp=LUT[k].area;
                                LUT.erase(k);
                                   
                                regionTemp.color=LUT[idImage[i][j-1]].color;
                                LUT[idImage[i][j-1]].area+=areaTemp;
                                regionTemp.area=0;
                                LUT.insert(make_pair(k, regionTemp));

                            }
                        }
                    }

                    //Propagacion lateral
                    idImage[i][j]=idImage[i][j-1];
                }

                else if(Ps==black && Pi == black)
                {

                    //Creamos un color aleatorio
                    regionColor.val[0]=(unsigned char) randomNumber(0,255);
                    regionColor.val[1]=(unsigned char) randomNumber(0,255);
                    regionColor.val[2]=(unsigned char) randomNumber(0,255);

                    //Inicializamos una nueva region
                    regionTemp.color=regionColor;
                    regionTemp.area=0;

                    idImage[i][j]=id;

                    LUT.insert(make_pair(id, regionTemp));

                    id=id+1;

                }

                //Aumentamos area
                LUT[idImage[i][j]].area++;


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
    //Almacenamos tabla
    for( k=1; k<=LUTSize; k++)
    {
        outputFile << "\nID: "<<IntToString(k)<<" Color: "<<IntToString(LUT[k].color[0])<<" "<<IntToString(LUT[k].color[1])<<" "<<IntToString(LUT[k].color[2])<<" Area: "<<IntToString(LUT[k].area)<<"\n";
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




    }

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

        circle (segmentedImage, Point(figures[k].xPromedio+.5,figures[k].yPromedio+.5),4,Scalar(255,0,0),CV_FILLED);


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

    //cap >> currentImage;

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
        segment(filteredImage,segmentedImg);
        momentos(segmentedImg);
        
        //momentos(segmentedImg);
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
