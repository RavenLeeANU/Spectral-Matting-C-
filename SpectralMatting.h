#include "../Include/OpenCV/OpenCV2.4.11/opencv.hpp"

typedef struct EigenV
{
	double eigenValue;
	int eigenIndex;
}
EigenV;

inline bool EigCmp(EigenV point1, EigenV point2)
{
	return abs(point1.eigenValue) < abs(point2.eigenValue);
}

class SpectralMatting
{
public:
	SpectralMatting();
	~SpectralMatting();
	void	processImg(IplImage* pResizeImgF, std::vector<CvMat*> *pSmLayers);
	void	VisualizeResult(std::vector<CvMat*> pSmLayers, IplImage* pResult);

private:

	void	LocalRGBnormalDistributions(IplImage *pResizeImg, int windowRadius, double epsilon, IplImage* pMeanImg, CvMat* pCovarMat[]);
	void	Im2col(CvMat *pIndices, int windowSize, CvMat *pNeighInd);
	void	MattingAffinity(IplImage* pResizeImg, CvMat *pEigenVector, CvMat *pEigenValue, CvMat *pWMat);
	void	SoftSegmentsFromEigs(CvMat *pEigenvectors, CvMat *pEigenvalues, CvMat *pLaplacian, std::vector<CvMat*> *pSmLayers, int width, int height);
	

};