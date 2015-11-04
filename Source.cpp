// 2015_ARLISS SNUSAT
// 3d matching code
// 2015.08.18
// OpenCV SURF edited 2015.08.20
// OpenGL edited 2015.08.23
// Processing edited 2015.08.24
// Han-Byul Kim

// OpenCV FLANN, OpenGL glut, glew included

// opencv docx surf code

#define _CRT_SECURE_NO_DEPRECATE
#define wid 680
#define hei 480

#define RADPERDEG 0.0174533

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>

// OpenCV Lib
#include <cv.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/flann/miniflann.hpp"

// OpenGL Lib
#include <GL/glut.h>
#include <GL/GL.h>
#include <GL/GLU.h>

using namespace cv;

void readme();
void renderscene();	// GL Function
void DataProcess2TO3(std::vector<Point2f> o, std::vector<Point2f> s);
float Rounding(float x, int digit){ return (floor((x)* pow(float(10), digit) + 0.5f) / pow(float(10), digit)); }
void Edge_Map();
void Things_3D();
void SpecialKey(int key, int x, int y);
void txtshow();
void init();

//void drawAxes(GLdouble length);
//void Arrow(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2, GLdouble D);

std::vector<Point2f> obj_corners(4);
std::vector<Point2f> scene_corners(4);

// the keypoint's coordinate <x, y, z value> -- will be processed
std::vector<Point3f> obj_3;
std::vector<Point3f> scene_3;

// sample image(Because of canny)
IplImage* sample_obj;
IplImage* sample_scene;

// mapping value x10
int map_obj[640][480];
int map_scene[640][480];

// moving parameter
int updown = 0;
int leftright = 0;
int yud = 0;
int focus[3];
float theta = 0;

int main(int argc, char** argv)
{
	if (argc != 3){
		readme(); return -1;
	}

	Mat img_object = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_scene = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

	// -- getting sample(Because of canny)
	sample_obj = cvLoadImage(argv[1], 1);
	sample_scene = cvLoadImage(argv[2], 1);

	if (!img_object.data || !img_scene.data){
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	//-- Step 1: Detect the keypoints using SURF Detector****************************************************
	int minHessian = 20; //Hessian critical value basic 200

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;	// keypoints(cv::KeyPoint)

	detector.detect(img_object, keypoints_object);	// 1
	detector.detect(img_scene, keypoints_scene);	// 2

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(img_object, keypoints_object, descriptors_object);
	extractor.compute(img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher	// matching keypoints
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++){
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )		// good matches (the real keypoints)*************************
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++){
		if (matches[i].distance < 3 * min_dist){
			good_matches.push_back(matches[i]);
		}
	}
	
	Mat img_matches;
	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;			// the keypoint's coordinate <x, y value>
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++){
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(img_object.cols, 0);
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
	obj_corners[3] = cvPoint(0, img_object.rows);
	
	perspectiveTransform(obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

	// -- param process (Point2f to Point3f)
	DataProcess2TO3(obj, scene);

	//-- Show detected matches
	imshow("Sample(Good Matches & Object detection)", img_matches);
	
	// -- param analysis
	std::ofstream fs("data.txt");

	std::cout << " 0 : " << scene_corners[0] << " 1 : " << scene_corners[1] << " 2 : " << scene_corners[2] << " 3 : " << scene_corners[3] << std::endl;
	std::cout << " 0 : " << obj_corners[0] << " 1 : " << obj_corners[1] << " 2 : " << obj_corners[2] << " 3 : " << obj_corners[3] << std::endl;

	for (int i = 0; i < good_matches.size(); i++){
		obj_3[i].x = Rounding(obj_3[i].x, 1);
		obj_3[i].y = Rounding(obj_3[i].y, 1);
		scene_3[i].x = Rounding(scene_3[i].x, 1);
		scene_3[i].y = Rounding(scene_3[i].y, 1);

		fs << "obj [" << i << "] : " << obj_3[i];
		fs << " scene [" << i << "] : " << scene_3[i] << std::endl;
	}

	std::ofstream axi3("3dcoor.txt");
	// -- Finding Edge
	//Edge_Map();

	Things_3D();

	for (int i = 0; i < obj_3.size(); i++){
		obj_3[i].z /= 10;
		obj_3[i].z = Rounding(obj_3[i].z, 1);
		axi3 << "obj" << i << obj_3[i] << std::endl;
	}


	// --OpenGL GLUT_DEPTH | GLUT_DOUBLE | 
	glutInitDisplayMode(GLUT_DEPTH | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(1300, 1030);
	glutCreateWindow("glsample");
	glutSpecialFunc(SpecialKey);
	glutDisplayFunc(renderscene);
	glutMainLoop();
		
	// -- closing
	cvReleaseData(&argv);
	cvDestroyWindow("Sample(Good Matches & Object detection)");
	cvDestroyWindow("canny");

	return 0;
}

void readme(){
	std::cout << " SNUSAT / 2015_ARLISS / Vision Software " << std::endl;
	std::cout << " Made_by_Han-Byul Kim / 2015 " << std::endl;
	std::cout << " Usage: ./Execution file <img1> <img2>" << std::endl;
	std::cout << " Execute this system to file auto input algorithm." << std::endl;
}

void renderscene(){
	init();
	// --drawing axis
	glClear(GL_COLOR_BUFFER_BIT);
	
	glBegin(GL_LINES);
	for (int i = -800; i <= 800; i++){
		glColor3f(1, 0, 0); glVertex3f(-2000, 0, i * 2.5); glVertex3f(2000, 0, i * 2.5);
		glColor3f(0, 0, 1); glVertex3f(i * 2.5, 0, -2000); glVertex3f(i * 2.5, 0, 2000);
	}
	glColor3f(255, 255, 0); glVertex3f(0, -2000, 0); glVertex3f(0, 2000, 0);
	
	//txtshow();

	// -- drawing sphere in 0,0,0
	Mat start;
	glPushMatrix();
	glTranslated(0, 0, 0);
	glColor3f(0.5, 0.5, 0.5);
	glutSolidSphere(0.3, 50, 50);
	glPopMatrix();
	
	/*for (int i = 0; i < obj_corners.size(); i++){
		Mat object_coor(obj_corners[i]);
		glPushMatrix();
		glColor3f(1.0, 1.0, 1.0);
		glutSolidSphere(1, 50, 50);
		glPopMatrix();
	}*/

	for (int i = 0; i < obj_3.size(); i++){
		glPushMatrix();
		glTranslated(obj_3[i].x, obj_3[i].y, obj_3[i].z);
		glColor3f(1, 1, 1);
		glutSolidSphere(0.3, 50, 50);
		glPopMatrix();
	}

	glLoadIdentity();

	glEnd();
	glFlush();
}


void init() {

	// Set the current clear color to black and the current drawing color to
	// white.
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glColor3f(1.0, 1.0, 1.0);

	// Set the camera lens to have a 60 degree (vertical) field of view, an
	// aspect ratio of 4/3, and have everything closer than 1 unit to the
	// camera and greater than 40 units distant clipped away.
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, 4.0 / 3.0, 1, 40);

	// Position camera at (4, 6, 5) looking at (0, 0, 0) with the vector
	// <0, 1, 0> pointing upward.

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	int x, z;
	int x_f, z_f;
	x = 10 + leftright + (focus[0] * 4 / sqrt(77));
	z = 12 + updown + (focus[2] * 5 / sqrt(77));
	x_f = 0 + leftright;
	z_f = 0 + updown;
	gluLookAt(x*cos(theta) + z*(-sin(theta)), 15 + (focus[1] * 6 / sqrt(77)), x*sin(theta) + z*cos(theta),
		x_f*cos(theta) + z_f*(-sin(theta)), 0+yud, x_f*sin(theta) + z_f*cos(theta),
			  0, 1, 0);
}

void SpecialKey(int key, int x, int y){
	switch (key)
	{
	case GLUT_KEY_RIGHT:
		leftright++;
		break;
	case GLUT_KEY_LEFT:
		leftright--;
		break;
	case GLUT_KEY_UP:
		updown--;
		break;
	case GLUT_KEY_DOWN:
		updown++;
		break;
	case GLUT_KEY_END:
		focus[0]++;
		focus[1]++;
		focus[2]++;
		break;
	case GLUT_KEY_HOME:
		focus[0]--;
		focus[1]--;
		focus[2]--;
		break;
	case GLUT_KEY_PAGE_DOWN:
		focus[1]--;
		break;
	case GLUT_KEY_PAGE_UP:
		focus[1]++;
		break;
	case GLUT_KEY_F1:
		theta -= 3.1415 / 18;
		break;
	case GLUT_KEY_F2:
		theta += 3.1415 / 18;
		break;
	case GLUT_KEY_F11:
		yud++;
		yud++;
		break;
	case GLUT_KEY_F12:
		yud--;
		yud--;
		break;
	}
	glutPostRedisplay();
}

void DataProcess2TO3(std::vector<Point2f> o, std::vector<Point2f> s){
	// -- Point2f -> Point3f
	Point3f temp;

	for (int i = 0; i < o.size(); i++){
		temp.x = o[i].x/10;
		temp.y = o[i].y/10;
		temp.z = 0;
		obj_3.push_back(temp);
		temp.x = s[i].x/10;
		temp.y = s[i].y/10;
		temp.z = 0;
		scene_3.push_back(temp);
	}
}

void Edge_Map(){
	std::ofstream omap("map_obj.txt");
	std::ofstream smap("map_scene.txt");

	// -- getting edge (cvCanny)
	IplImage* canny_obj;
	IplImage* canny_scene;
	IplImage* gray_obj;
	IplImage* gray_scene;

	gray_obj = cvCreateImage(cvGetSize(sample_obj), 8, 1);
	gray_scene = cvCreateImage(cvGetSize(sample_scene), 8, 1);
	canny_obj = cvCreateImage(cvGetSize(sample_obj), 8, 1);
	canny_scene = cvCreateImage(cvGetSize(sample_scene), 8, 1);
	cvNamedWindow("canny_obj");
	cvNamedWindow("canny_scene");
	cvCvtColor(sample_obj, gray_obj, CV_BGR2GRAY);
	cvCvtColor(sample_scene, gray_scene, CV_BGR2GRAY);
	cvCanny(gray_obj, canny_obj, 100, 200);		// changing threshold can make the edge apparent
	cvCanny(gray_scene, canny_scene, 200, 0);

	// --Showing
	cvShowImage("canny_obj", canny_obj);
	cvShowImage("canny_scene", canny_scene);

	// --making edge map
	// obj
	for (int y = 0; y < hei; y++){
		for (int x = 0; x < wid; x++){
			int index = x + y*canny_obj->widthStep;		// to avoid abort() err.(argument out of range)
			unsigned char value = canny_obj->imageData[index];
			if (value >= 200){
				map_obj[x][y] = 1;
				omap << map_obj[x][y];
			}
			else{
				map_obj[x][y] = 0;
				omap << map_obj[x][y];
			}
		}
		omap << std::endl;
	}
	// scene
	for (int y = 0; y < hei; y++){
		for (int x = 0; x < wid; x++){
			int index = x + y*canny_scene->widthStep;		// to avoid abort() err.(argument out of range)
			unsigned char value = canny_scene->imageData[index];
			if (value >= 200){
				map_scene[x][y] = 1;
				smap << map_scene[x][y];
			}
			else{
				map_scene[x][y] = 0;
				smap << map_scene[x][y];
			}
		}
		smap << std::endl;
	}
	
	// --calculating Z
	// obj : left, scene : right
	/*std::vector<float> Z;		 // Z = B*f / (XL - XR)
	float B = 58.6;		// need to initialize TBD(58.6mm)
	float f = 22.35 - 3.45;		// 22.35-3.45(mm)

	for (int i = 0; i < obj_3.size(); i++){
		Z.push_back((B*f) / (obj_3[i].x - scene_3[i].x));
		obj_3[i].z = Z[i];
	}*/

	//closing
	//cvReleaseImage(&canny);
	//cvDestroyWindow("canny");
}

void Things_3D(){
	std::vector<float> Z;		 // Z = B*f / (XL - XR)
	float B = 58.6;		// need to initialize TBD(58.6mm)
	float f = 22.35 - 3.45;		// 22.35-3.45(mm)

	for (int i = 0; i < obj_3.size(); i++){
		Z.push_back((B*f) / (obj_3[i].x - scene_3[i].x));
		obj_3[i].z = Z[i];
	}
}

void txtshow(){
	static float dis = 0, ddis = 0, elev = 0, delev = 0, azim = 0, dazim = 0;


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	gluOrtho2D(0.0, 1000, 0.0, 800);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(1.0f, 0.0f, 0.0f);//needs to be called before RasterPos
	glRasterPos2i(10, 10);
	std::string s = "SNUSAT";
	void * font = GLUT_BITMAP_9_BY_15;

	for (std::string::iterator i = s.begin(); i != s.end(); ++i)
	{
		char c = *i;
		glutBitmapCharacter(font, c);
	}
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glEnable(GL_TEXTURE_2D);

	glutSwapBuffers();
	glutPostRedisplay();
}

/*
#include <windows.h>
#include <GL/glut.h>
#include <stdlib.h> 

static void resize(int width, int height)
{
	const float ar = (float)width / (float)height;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-ar, ar, -1.0, 1.0, 2.0, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

static void display(void)
{

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColor3d(1, 0, 0);

	glPushMatrix();
	glTranslated(0.0, 3, -6);
	glutSolidSphere(1, 50, 50);
	glPopMatrix();

	glPushMatrix();
	glTranslated(0.0, -1.2, -6);
	glutWireSphere(1, 16, 16);
	glPopMatrix();

	glutSwapBuffers();
}

const GLfloat light_ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[] = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[] = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };


int main(int argc, char *argv[])
{
	glutInit(&argc, argv);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(10, 10);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("Programming Techniques - 3D Spheres");

	glutReshapeFunc(resize);
	glutDisplayFunc(display);

	glClearColor(1, 1, 1, 1);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

	glutMainLoop();

	return EXIT_SUCCESS;
}*/