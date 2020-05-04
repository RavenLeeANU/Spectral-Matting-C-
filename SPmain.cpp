#include "SpectralMatting.h"

using namespace std;

int main()
{
	double m_scaleSize = 4;
	IplImage* img = cvLoadImage("C:/Users/54053/Desktop/InterBox/face.jpg");
	cvSaveImage("C:/Users/54053/Desktop/InterBox/img.jpg", img);
	CvSize reCvsize = cvSize(cvCeil(img->width / m_scaleSize), cvCeil(img->height / m_scaleSize));
	IplImage* resizeImg = cvCreateImage(reCvsize, img->depth, img->nChannels);
	cvResize(img, resizeImg, CV_INTER_AREA);
	IplImage*	tempImg = cvCreateImage(cvGetSize(resizeImg), IPL_DEPTH_64F, 3);
	IplImage*	resizeImgF = cvCreateImage(cvGetSize(resizeImg), IPL_DEPTH_64F, 3);
	cvConvert(resizeImg, tempImg);
	cvScale(tempImg, resizeImgF, 1 / 255.0);

	SpectralMatting sp;
	IplImage* res = cvCreateImage(cvSize(resizeImg->width, resizeImg->height), resizeImg->depth, resizeImg->nChannels);
	cvZero(res);

	vector<CvMat*> *sm = new vector < CvMat* >;
	cvSaveImage("C:/Users/54053/Desktop/InterBox/resizeImgF.jpg", resizeImgF);
	sp.processImg(resizeImgF, sm);
	sp.VisualizeResult(*sm, res);

	cvSaveImage("C:/Users/54053/Desktop/InterBox/final.jpg", res);

	return 0;
}