#include "SpectralMatting.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

const int m_windowRadius = 1;
//kmeans 
const int	 m_kmeansIter = 400;
const double m_kmeansBias = 1e-10;
const double m_kmeansAttempt = 10;

//迭代优化相关
const int	 m_maxIter = 10;
const double m_spMat = 0.8;
const double m_iterEps = 1e-10;
const double m_w1 = 0.3;
const double m_w0 = 0.3;
const double m_sparsityParam = 0.8;

double m_epsilon = 1e-7;

const int m_compCnt = 30;


double m_boxFilt[9] = { 1 / 9.0, 1 / 9.0, 1 / 9.0,
						1 / 9.0, 1 / 9.0, 1 / 9.0,
						1 / 9.0, 1 / 9.0, 1 / 9.0 };

SpectralMatting::SpectralMatting()
{
}

SpectralMatting::~SpectralMatting()
{
}

void SpectralMatting::processImg(IplImage* pResizeImgF, std::vector<CvMat*> *pSmLayers)
{
	int width = pResizeImgF->width;
	int height = pResizeImgF->height;
	int N = width * height;

	CvMat *pEigenValue = cvCreateMat(m_compCnt, 1, CV_64F);
	CvMat *pEigenVector = cvCreateMat(N, m_compCnt, CV_64F);
	CvMat *pWMat = cvCreateMat(N, N, CV_64F);

	MattingAffinity(pResizeImgF, pEigenVector, pEigenValue, pWMat);
	SoftSegmentsFromEigs(pEigenVector, pEigenValue, pWMat, pSmLayers, width, height);

}

void SpectralMatting::LocalRGBnormalDistributions(IplImage *pResizeImg, int windowRadius, double epsilon, IplImage* pMeanImg, CvMat* pCovarMat[])
{
	int h = pResizeImg->height;
	int w = pResizeImg->width;
	int c = pResizeImg->nChannels;

	int N = h * w;
	int WindowSize = 2 * windowRadius + 1;

	IplImage* imgSplit[3];

	imgSplit[0] = cvCreateImage(cvSize(w, h), pResizeImg->depth, 1);
	imgSplit[1] = cvCreateImage(cvSize(w, h), pResizeImg->depth, 1);
	imgSplit[2] = cvCreateImage(cvSize(w, h), pResizeImg->depth, 1);

	cvSplit(pResizeImg, imgSplit[0], imgSplit[1], imgSplit[2], NULL);

	CvMat kernel = cvMat(3, 3, CV_64F, m_boxFilt);

	cvFilter2D(pResizeImg, pMeanImg, &kernel);

	IplImage *imgSplitMean[3];
	imgSplitMean[0] = cvCreateImage(cvSize(w, h), pResizeImg->depth, 1);
	imgSplitMean[1] = cvCreateImage(cvSize(w, h), pResizeImg->depth, 1);
	imgSplitMean[2] = cvCreateImage(cvSize(w, h), pResizeImg->depth, 1);
	cvSplit(pMeanImg, imgSplitMean[0], imgSplitMean[1], imgSplitMean[2], NULL);

	//i=row,j=col
	for (int i = 0; i < 3; i++)
	{
		for (int j = i; j < 3; j++)
		{
			IplImage *mulImg = cvCreateImage(cvSize(w, h), pResizeImg->depth, 1);
			IplImage *mulImgMean = cvCreateImage(cvSize(w, h), pResizeImg->depth, 1);
			IplImage *temp = cvCreateImage(cvSize(w, h), pResizeImg->depth, 1);

			cvMul(imgSplit[i], imgSplit[j], mulImg);
			cvFilter2D(mulImg, temp, &kernel);

			cvMul(imgSplitMean[i], imgSplitMean[j], mulImgMean);
			cvSub(temp, mulImgMean, mulImg);

			//Convert the temp into 1d vector
			CvMat* mat = cvCreateMat(1, w * h, CV_64F);
			cvZero(mat);


			for (int x = 0; x < temp->height; x++)
			{
				double *ptr = (double*)(mulImg->imageData + x * mulImg->widthStep);

				for (int y = 0; y < mulImg->width; y++)
				{
					int row = x + mulImg->height * y;
					//int row = x * mulImg->width + y;
					double tvalue = ptr[y];
					CV_MAT_ELEM(*mat, double, 0, row) = tvalue;
				}
			}

			//add eps
			if (i == j)
			{
				CvMat* matEps = cvCreateMat(1, w * h, CV_64F);
				cvAddS(mat, cvScalar(epsilon), matEps);
				pCovarMat[i * 3 + j] = matEps;
			}
			else
			{
				pCovarMat[i * 3 + j] = mat;
			}
			cvReleaseImage(&temp);
			cvReleaseImage(&mulImgMean);
			cvReleaseImage(&mulImg);
		}
	}

	//fill the matrix
	for (int i = 1; i < 3; i++)
	{
		for (int j = 0; j < i; j++)
		{
			pCovarMat[i * 3 + j] = pCovarMat[j * 3 + i];
		}
	}
}

void SpectralMatting::Im2col(CvMat *pIndices, int windowSize, CvMat *pNeighInd)
{
	int skip = windowSize - 1;
	for (int indW = 0; indW < pIndices->width - skip; indW++)
	{
		for (int indH = 0; indH < pIndices->height - skip; indH++)
		{
			//calculate kernel
			for (int kerWidth = 0; kerWidth < windowSize; kerWidth++)
			{
				for (int kerHeight = 0; kerHeight < windowSize; kerHeight++)
				{
					int elem = CV_MAT_ELEM(*pIndices, ushort, indH + kerHeight, indW + kerWidth);
					CV_MAT_ELEM(*pNeighInd, ushort,
						indH + (pIndices->height - skip) * indW,
						kerHeight + windowSize * kerWidth) = elem;
				}
			}
		}
	}
}

void SpectralMatting::MattingAffinity(IplImage* pResizeImg, CvMat *pEigenVector, CvMat *pEigenValue, CvMat *pWMat)
{

	const int windowSize = 2 * m_windowRadius + 1;
	const int neighSize = windowSize * windowSize;

	int h = pResizeImg->height;
	int w = pResizeImg->width;
	int c = pResizeImg->nChannels;

	int N = h * w;

	int padding = 1;
	int neighInd_w, neighInd_h;

	neighInd_w = (w - 2 * padding) * (h - 2 * padding);
	neighInd_h = windowSize * windowSize;

	int pixCnt = neighInd_w;

	m_epsilon = m_epsilon / neighSize;

	//covariance matrix
	CvMat*		covarMat[neighSize];
	IplImage*	meanImg = cvCreateImage(cvSize(w, h), pResizeImg->depth, pResizeImg->nChannels);
	CvMat*		neighInd = cvCreateMat(neighInd_w, neighInd_h, CV_16U);

	//the indices of pixels 
	CvMat*		indices = cvCreateMat(h, w, CV_16U);
	CvMat*		inInd = cvCreateMat(neighInd_w, 1, CV_16U);

	//reshape img & meanImg
	CvMat*		imgVec = cvCreateMat(N, c, CV_64F);
	CvMat*		imgMeanVec = cvCreateMat(N, c, CV_64F);

	//Laplacian matrix and pixel indices
	CvMat*		flowRows = cvCreateMat(neighSize * neighSize, pixCnt, CV_16U);
	CvMat*		flowCols = cvCreateMat(neighSize * neighSize, pixCnt, CV_16U);
	CvMat*		flows = cvCreateMat(neighSize * neighSize, pixCnt, CV_64F);

	cvZero(neighInd);

	LocalRGBnormalDistributions(pResizeImg, m_windowRadius, m_epsilon, meanImg, covarMat);

	for (int i = 0; i < w; i++)
		for (int j = 0; j < h; j++)
			CV_MAT_ELEM(*indices, ushort, j, i) = i * h + j;

	//img2col
	Im2col(indices, windowSize, neighInd);

	//get central index of the window
	for (int i = 0; i < neighInd_w; i++)
	{
		ushort inValue = CV_MAT_ELEM(*neighInd, ushort, i, (neighSize - 1) / 2);
		CV_MAT_ELEM(*inInd, ushort, i, 0) = inValue;
	}

	for (int x = 0; x < pResizeImg->height; x++)
	{
		double *ptr = (double*)(pResizeImg->imageData + x * pResizeImg->widthStep);
		double *ptrMean = (double*)(meanImg->imageData + x * meanImg->widthStep);
		for (int y = 0; y < pResizeImg->width; y++)
		{
			int cnt = x + pResizeImg->height * y;
			CV_MAT_ELEM(*imgVec, double, cnt, 0) = ptr[3 * y];
			CV_MAT_ELEM(*imgVec, double, cnt, 1) = ptr[3 * y + 1];
			CV_MAT_ELEM(*imgVec, double, cnt, 2) = ptr[3 * y + 2];

			CV_MAT_ELEM(*imgMeanVec, double, cnt, 0) = ptrMean[3 * y];
			CV_MAT_ELEM(*imgMeanVec, double, cnt, 1) = ptrMean[3 * y + 1];
			CV_MAT_ELEM(*imgMeanVec, double, cnt, 2) = ptrMean[3 * y + 2];
		}
	}

	//compute Laplacian Matrix
	CvMat *neighs = cvCreateMat(neighSize, 1, CV_16U);
	CvMat *shiftWinColors = cvCreateMat(neighSize, c, CV_64F);
	CvMat *shiftWinColorsT = cvCreateMat(c, neighSize, CV_64F);

	//block = covarMat(:, :, inInd(i))
	CvMat *block = cvCreateMat(windowSize, windowSize, CV_64F);
	CvMat *blockInv = cvCreateMat(windowSize, windowSize, CV_64F);

	CvMat *shiftTemp = cvCreateMat(windowSize, neighSize, CV_64F);
	CvMat *flowTemp = cvCreateMat(neighSize, neighSize, CV_64F);

	for (int i = 0; i < inInd->rows; i++)
	{
		ushort ind = CV_MAT_ELEM(*inInd, ushort, i, 0);

		cvZero(block);

		for (int j = 0; j < neighs->rows; j++)
			CV_MAT_ELEM(*neighs, ushort, j, 0) = CV_MAT_ELEM(*neighInd, ushort, i, j);

		for (int y = 0; y < windowSize; y++)
		{
			for (int x = 0; x < neighSize; x++)
			{
				ushort inds = CV_MAT_ELEM(*neighs, ushort, x, 0);
				double imgV = CV_MAT_ELEM(*imgVec, double, inds, y);
				double imgMeanV = CV_MAT_ELEM(*imgMeanVec, double, ind, y);
				CV_MAT_ELEM(*shiftWinColors, double, x, y) = imgV - imgMeanV;
			}
		}

		//transpose
		cvTranspose(shiftWinColors, shiftWinColorsT);

		for (int b = 0; b < block->width; b++)
		{
			for (int c = 0; c < block->height; c++)
			{
				double temp = CV_MAT_ELEM(*covarMat[b * block->height + c], double, 0, ind);
				CV_MAT_ELEM(*block, double, c, b) = temp;
			}
		}

		cvInvert(block, blockInv);

		//(covarMat(:, :, inInd(i)) \ shiftedWinColors')
		cvMatMul(blockInv, shiftWinColorsT, shiftTemp);
		cvMatMul(shiftWinColors, shiftTemp, flowTemp);

		//store flowtemp into flow
		for (int flowsWidth = 0; flowsWidth < neighSize; flowsWidth++)
		{
			for (int flowsHeight = 0; flowsHeight < neighSize; flowsHeight++)
			{
				double res = (CV_MAT_ELEM(*flowTemp, double, flowsWidth, flowsHeight) + 1) / neighSize;
				CV_MAT_ELEM(*flows, double, flowsWidth * neighSize + flowsHeight, i) = res;
			}
		}

		//flowRows and cols
		for (int flowsWidth = 0; flowsWidth < neighSize; flowsWidth++)
		{
			ushort nval = CV_MAT_ELEM(*neighs, ushort, flowsWidth, 0);
			for (int flowsHeight = 0; flowsHeight < neighSize; flowsHeight++)
			{
				CV_MAT_ELEM(*flowRows, ushort, flowsWidth * neighSize + flowsHeight, i) = nval;
				CV_MAT_ELEM(*flowCols, ushort, flowsHeight * neighSize + flowsWidth, i) = nval;
			}
		}
	}

	cvReleaseMat(&neighs);
	cvReleaseMat(&shiftWinColors);
	cvReleaseMat(&shiftWinColorsT);
	cvReleaseMat(&block);
	cvReleaseMat(&blockInv);
	cvReleaseMat(&shiftTemp);
	cvReleaseMat(&flowTemp);


	//create sparse matrix
	for (int j = 0; j < flowRows->cols; j++)
	{
		for (int k = 0; k < flowRows->rows; k++)
		{
			ushort indX = CV_MAT_ELEM(*flowRows, ushort, k, j);
			ushort indY = CV_MAT_ELEM(*flowCols, ushort, k, j);

			double vtemp = CV_MAT_ELEM(*flows, double, k, j);
			CV_MAT_ELEM(*pWMat, double, indX, indY) += vtemp;
		}
	}

	CvMat *WT = cvCreateMat(N, N, CV_64F);
	CvMat *WSum = cvCreateMat(N, N, CV_64F);
	CvMat *WDiv = cvCreateMat(N, N, CV_64F);

	cvSet(WDiv, cvScalar(0.5));
	cvTranspose(pWMat, WT);
	cvAdd(pWMat, WT, WSum);
	cvMul(WSum, WDiv, pWMat);

	cvReleaseMat(&WT);
	cvReleaseMat(&WSum);
	cvReleaseMat(&WDiv);

	//L = D - A
	CvMat *diagW = cvCreateMat(pWMat->rows, 1, CV_64F);
	CvMat *eigenValueP = cvCreateMat(N, 1, CV_64F);
	CvMat *eigenVectorP = cvCreateMat(N, N, CV_64F);

	//sum the cols of W;
	for (int i = 0; i < pWMat->rows; i++)
	{
		double sumV = 0;
		for (int j = 0; j < pWMat->cols; j++)
		{
			sumV += CV_MAT_ELEM(*pWMat, double, i, j);
		}
		CV_MAT_ELEM(*diagW, double, i, 0) = sumV;
	}

	//update W
	for (int i = 0; i < pWMat->rows; i++)
	{
		for (int j = 0; j < pWMat->cols; j++)
		{
			double factor = 0;
			if (i == j)
			{
				factor = CV_MAT_ELEM(*diagW, double, i, 0);
			}
			CV_MAT_ELEM(*pWMat, double, i, j) = factor - CV_MAT_ELEM(*pWMat, double, i, j);
		}
	}

	cvEigenVV(pWMat, eigenVectorP, eigenValueP);

	//select minimal eigen of total eigens
	int totalSelect = pEigenVector->width;

	vector<EigenV> eigenSort;
	for (int i = 0; i < N; i++)
	{
		EigenV viPair;
		viPair.eigenIndex = i;
		viPair.eigenValue = CV_MAT_ELEM(*eigenValueP, double, i, 0);
		eigenSort.push_back(viPair);
	}
	std::sort(eigenSort.begin(), eigenSort.end(), EigCmp);

	//select n vector of total;
	for (int i = 0; i < pEigenVector->width; i++)
	{
		int indexP = eigenSort.at(i).eigenIndex;
		for (int j = 0; j < pEigenVector->height; j++)
		{
			CV_MAT_ELEM(*pEigenVector, double, j, i) = CV_MAT_ELEM(*eigenVectorP, double, indexP, j);
		}
		CV_MAT_ELEM(*pEigenValue, double, i, 0) = CV_MAT_ELEM(*eigenValueP, double, indexP, 0);
	}

	cvReleaseMat(&diagW);
	cvReleaseMat(&eigenVectorP);
	cvReleaseMat(&eigenValueP);
	cvReleaseMat(&imgVec);
	cvReleaseMat(&imgMeanVec);
	cvReleaseMat(&neighInd);
	cvReleaseMat(&indices);
	cvReleaseMat(&inInd);
	cvReleaseMat(&flows);
	cvReleaseMat(&flowCols);
	cvReleaseMat(&flowRows);
	cvReleaseImage(&meanImg);
	//covariance matrix

	cvReleaseMat(&covarMat[0]);
	cvReleaseMat(&covarMat[1]);
	cvReleaseMat(&covarMat[2]);
	cvReleaseMat(&covarMat[4]);
	cvReleaseMat(&covarMat[5]);
	cvReleaseMat(&covarMat[8]);

}

void SpectralMatting::SoftSegmentsFromEigs(CvMat *pEigenvectors, CvMat *pEigenvalues, CvMat *pLaplacian, vector<CvMat*> *pSmLayers, int width, int height)
{
	int start = 1;

	int compCnt = pEigenvectors->cols;
	int eigValCnt = pEigenvectors->cols;

	int w = pEigenvectors->width;
	int h = pEigenvectors->height;

	int clusterNum = eigValCnt;
	int initEigCnt = ceil(0.5 * eigValCnt);

	int removeIter = (int)ceil(m_maxIter / 5.0);
	int removeIterCycle = (int)ceil(m_maxIter / 5.0);

	CvMat*	clusters = cvCreateMat(pEigenvectors->rows, pEigenvectors->cols, CV_32FC1);
	cvConvert(pEigenvectors, clusters);

	CvMat*	initEigWeight = cvCreateMat(initEigCnt, initEigCnt, CV_32FC1);
	cvZero(initEigWeight);

	CvMat*	eigVecsCut = cvCreateMat(pEigenvectors->rows, initEigCnt, CV_32FC1);
	CvMat*	initEigVecs = cvCreateMat(pEigenvectors->rows, initEigCnt, CV_32FC1);
	CvMat*	initialSegments = cvCreateMat(pEigenvectors->rows, 1, CV_32SC1);
	CvMat*	centers = cvCreateMat(pEigenvectors->rows, clusterNum, CV_64F);

	for (int i = 0; i < pEigenvalues->rows; i++)
		CV_MAT_ELEM(*pEigenvalues, double, i, 0) = abs(CV_MAT_ELEM(*pEigenvalues, double, i, 0));

	for (int i = start; i < initEigCnt + start; i++)
	{
		double powV = pow(1.0 / abs(CV_MAT_ELEM(*pEigenvalues, double, i, 0)), 0.5);
		CV_MAT_ELEM(*initEigWeight, float, i - start, i - start) = (float)powV;
		for (int j = 0; j < eigVecsCut->rows; j++)
			CV_MAT_ELEM(*eigVecsCut, float, j, i - start) = CV_MAT_ELEM(*pEigenvectors, double, j, i);
	}

	cvMatMul(eigVecsCut, initEigWeight, initEigVecs);

	for (int i = 0; i < centers->cols; i++)
		for (int j = 0; j < centers->rows; j++)
			CV_MAT_ELEM(*centers, float, j, i) = CV_MAT_ELEM(*pEigenvectors, double, j % pEigenvectors->rows / clusterNum, i);

	CvTermCriteria term = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, m_kmeansIter, m_kmeansBias);
	//cvKMeans2(clusters,clusterNum,initialSegments,term);
	//cv::kmeans(clusters,clusterNum,cv::Mat(initialSegments),term,10, 0,NULL);

	cv::Mat matc(initEigVecs);
	cv::Mat mati(initialSegments);
	cv::Mat matr(centers);

	kmeans(matc, clusterNum, mati, term, m_kmeansAttempt, cv::KMEANS_PP_CENTERS, matr);
	//kmeans(Mat(clusters), clusterNum, Mat(initialSegments), term, 10, KMEANS_PP_CENTERS, Mat(centers));

	MatrixXd softSegmentE(h, w);
	softSegmentE.setZero();

	for (int i = 0; i < softSegmentE.rows(); i++)
		for (int clus = 0; clus < clusterNum; clus++)
			if (clus == CV_MAT_ELEM(*initialSegments, int, i, 0))
				softSegmentE(i, clus) = 1.0f;

	////----------------print result--------------------------
	/*for (int i = 0; i < softSegmentE.cols(); i++)
	{
		IplImage *test = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		cvZero(test);

		for (int j = 0; j < test->height; j++)
		{
			uchar* uptr = (uchar*)(test->imageData + test->widthStep * j);
			for (int m = 0; m < test->width; m++)
			{
				uchar valueN = (uchar)softSegmentE(m * test->height + j, i) * 255;
				uptr[m] = valueN;
			}
		}

		USES_CONVERSION;

		CString str;
		str.Format(_T("C:/Users/54053/Desktop/InterBox/kmeans-%d.jpg"), i);
		cvSaveImage(T2A(str.GetBuffer(0)), test);
		str.ReleaseBuffer();
		cvReleaseImage(&test);
	}*/
	//------------------------------------------------------

	//Iteration Start
	MatrixXd LaplacianE(pLaplacian->rows, pLaplacian->cols); LaplacianE.setZero();
	MatrixXd eig_vectorsE(softSegmentE.rows(), eigValCnt); eig_vectorsE.setZero();
	MatrixXd eig_valuesE(eigValCnt, eigValCnt); eig_valuesE.setZero();
	MatrixXd e0E(softSegmentE.rows(), softSegmentE.cols()); e0E.setZero();
	MatrixXd e1E(softSegmentE.rows(), softSegmentE.cols()); e1E.setZero();

	//cv2eigen(pLaplacian, LaplacianE);
	for (int i = 0; i < pLaplacian->rows; i++)
		for (int j = 0; j < pLaplacian->cols; j++)
			LaplacianE(i, j) = CV_MAT_ELEM(*pLaplacian, double, i, j);

	//cv2eigen(pEigenvalues, eig_valuesE);
	for (int j = 0; j < eig_valuesE.cols(); j++)
		eig_valuesE(j, j) = CV_MAT_ELEM(*pEigenvalues, double, j, 0);

	//cv2eigen(pEigenvectors, eig_vectorsE);
	for (int i = 0; i < eig_vectorsE.rows(); i++)
		for (int j = 0; j < eig_vectorsE.cols(); j++)
			eig_vectorsE(i, j) = CV_MAT_ELEM(*pEigenvectors, double, i, j);

	e0E.setZero();
	e1E.setZero();

	for (int i = 0; i < softSegmentE.rows(); i++)
	{
		for (int j = 0; j < softSegmentE.cols(); j++)
		{
			//w0 * sparsityParam * max(abs(softSegments-1), thr_e) 
			double param1 = pow(m_w1, m_sparsityParam);
			double param0 = pow(m_w0, m_sparsityParam);

			double maxV1 = pow(max(abs(softSegmentE(i, j) - 1), m_iterEps), m_spMat - 2);
			double maxV0 = pow(max(abs(softSegmentE(i, j)), m_iterEps), m_spMat - 2);

			e1E(i, j) = param1 * maxV1;
			e0E(i, j) = param0 * maxV0;
		}
	}

	//Compute matting component with sparsity prior
	for (int iter = 0; iter < m_maxIter; iter++)
	{
		//construct equals
		int tSize = (compCnt - 1) * eigValCnt;

		MatrixXd tAE(tSize, tSize);
		MatrixXd tbE(tSize, 1);
		tAE.setZero();
		tbE.setZero();

		int k;
		for (k = 0; k < compCnt - 1; k++)
		{
			MatrixXd weighted_eigsE(e0E.rows(), eigValCnt);
			weighted_eigsE.setZero();
			weighted_eigsE.colwise() += e1E.col(k) + e0E.col(k);
			weighted_eigsE = weighted_eigsE.array() * eig_vectorsE.array();

			tAE.block(k * eigValCnt, k * eigValCnt, eigValCnt, eigValCnt) = eig_vectorsE.transpose() * weighted_eigsE + eig_valuesE;
			tbE.block(k * eigValCnt, 0, eigValCnt, 1) = eig_vectorsE.transpose() * e1E.col(k);
		}

		//add num k
		k = compCnt - 1;

		MatrixXd weighted_eigsE(e0E.rows(), eigValCnt);
		weighted_eigsE.setZero();
		weighted_eigsE.colwise() += e1E.col(k) + e0E.col(k);
		weighted_eigsE.array() *= eig_vectorsE.array();

		// calculate ttA,ttB 
		MatrixXd ttAE(eig_vectorsE.cols(), weighted_eigsE.cols());
		ttAE.setZero();
		VectorXd ttbE(eig_vectorsE.cols());
		ttbE.setZero();
		MatrixXd sumE(eig_vectorsE.cols(), eig_vectorsE.rows());
		sumE.setZero();
		sumE = eig_vectorsE.transpose() * LaplacianE;

		ttAE = eig_vectorsE.transpose() * weighted_eigsE + eig_valuesE;
		ttbE = eig_vectorsE.transpose() * e0E.col(k) + sumE.rowwise().sum();

		for (int i = 0; i < compCnt - 1; i++)
		{
			for (int j = 0; j < compCnt - 1; j++)
			{
				tAE.block(i * eigValCnt, j * eigValCnt, eigValCnt, eigValCnt) += ttAE;
			}
			tbE.block(i * eigValCnt, 0, eigValCnt, 1) += ttbE;
		}

		//MatrixXd y = tAE.llt().solve(tbE);
		MatrixXd y(tbE.rows(), tbE.cols());
		y.setZero();
		y = tAE.ldlt().solve(tbE);

		MatrixXd yShapeE(eigValCnt, compCnt - 1);
		yShapeE.setZero();

		for (int i = 0; i < tbE.rows(); i++)
			yShapeE(i % yShapeE.rows(), i / yShapeE.rows()) = y(i, 0);

		softSegmentE.block(0, 0, softSegmentE.rows(), compCnt - 1) = eig_vectorsE.block(0, 0, eig_vectorsE.rows(), eigValCnt) * yShapeE;

		MatrixXd one(softSegmentE.rows(), 1);
		one.fill(1);

		softSegmentE.col(compCnt - 1) = one - softSegmentE.block(0, 0, softSegmentE.rows(), compCnt - 1).rowwise().sum();

		//remove matting components 
		if (iter > removeIter)
		{
			//find every channels max value larger than
			vector<int> nzii;

			for (int i = 0; i < softSegmentE.cols(); i++)
			{
				if (abs(softSegmentE.col(i).maxCoeff()) > 0.1)
					nzii.push_back(i);
			}

			//remove 
			compCnt = nzii.size();
			MatrixXd softSegmentNE(softSegmentE.rows(), compCnt);
			softSegmentNE.setZero();

			for (int i = 0; i < softSegmentNE.cols(); i++)
				softSegmentNE.col(i) = softSegmentE.col(nzii[i]);

			softSegmentE = softSegmentNE;
			removeIter += removeIterCycle;

		}

		//Recompute the derivates of sparsity penalites
		MatrixXd e0E1(softSegmentE.rows(), softSegmentE.cols());
		MatrixXd e1E1(softSegmentE.rows(), softSegmentE.cols());

		e0E1.setZero();
		e1E1.setZero();
		for (int i = 0; i < softSegmentE.rows(); i++)
		{
			for (int j = 0; j < softSegmentE.cols(); j++)
			{
				//w0 * sparsityParam * max(abs(softSegments-1), thr_e) 
				double param1 = pow(m_w1, m_sparsityParam);
				double param0 = pow(m_w0, m_sparsityParam);

				double maxV1 = pow(max(abs(softSegmentE(i, j) - 1), m_iterEps), m_spMat - 2);
				double maxV0 = pow(max(abs(softSegmentE(i, j)), m_iterEps), m_spMat - 2);

				e1E1(i, j) = param1 * maxV1;
				e0E1(i, j) = param0 * maxV0;
			}
		}

		e0E = e0E1;
		e1E = e1E1;
	}

	//reshape the soft as sm
	for (int i = 0; i < softSegmentE.cols(); i++)
	{
		IplImage *test = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		CvMat *smElem = cvCreateMat(height, width, CV_32F);
		for (int j = 0; j < height; j++)
		{
			for (int m = 0; m < width; m++)
			{
				uchar* uptr = (uchar*)(test->imageData + test->widthStep * j);
				CV_MAT_ELEM(*smElem, float, j, m) = softSegmentE(j + height * m, i);
				uptr[m] = (uchar)(max(softSegmentE(j + height * m, i), 0.0) * 255.0);
			}
		}
		pSmLayers->push_back(smElem);
	}

	cvReleaseMat(&clusters);
	cvReleaseMat(&initEigWeight);
	cvReleaseMat(&eigVecsCut);
	cvReleaseMat(&initEigVecs);
	cvReleaseMat(&initialSegments);
	cvReleaseMat(&centers);
}


void SpectralMatting::VisualizeResult(vector<CvMat*> pSmLayers, IplImage* pResult)
{
	for (int i = 0; i < pSmLayers.size(); i++)
	{
		//generate random color
		int rgb[3];
		for (int j = 0; j < 3;j++)
		{
			rgb[j] = rand() % 256;
		}

		for (int j = 0; j < pResult->height;j++)
		{
			uchar *ptr = (uchar*)(pResult->imageData + j * pResult->widthStep);
			for (int k = 0; k < pResult->width; k++)
			{
				for (int c = 0; c < pResult->nChannels;c++)
				{
					double temp = (double)CV_MAT_ELEM(*pSmLayers[i], float, j, k);
					ptr[3 * k + c] += (uchar)min(max(temp, 0.0) * rgb[c],255.0);
					//cout << k << "," << c <<","<< 3 * k + c<<endl;
					//ptr[3 * k + c] += (uchar)rgb[c];
				}
			}
		}

	}

}

