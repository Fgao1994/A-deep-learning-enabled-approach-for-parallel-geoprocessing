#include "stdio.h"
#include "gdal_priv.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "mpi.h"
#include <math.h>
#include <algorithm>
#include <cstdio>
#include "SerialPeakPointExtract1.h"

using namespace std;

void serialPeakPointExtract(const char* inPath, const char* outPath);

/*int main(int argc, char* argv[]) {
	const char* inPath = "/home/fgao/LoadBalance/PeakPoint/data/sichuan_1.tif";
	const char* outPath = "/home/fgao/LoadBalance/PeakPoint/data/3_result.tif";
	serialPeakPointExtract(inPath, outPath);
	return 0;
}*/

void serialPeakPointExtract(const char* inPath, const char* outPath)
{
	double timeBegin = clock();
	int nImgSizeX, nImgSizeY;
	int row, col;

	float* pImgData;
	pImgData = NULL;

	double cellSize;
	double nodata;
	double zConvFactor = 1;

	GDALDataset* poDataset;
	GDALAllRegister();
	poDataset = (GDALDataset*)GDALOpen(inPath, GA_ReadOnly);
	if (poDataset == NULL) {
		std::cout << "nothing" << std::endl;
	}
	double goeInformation[6];
	poDataset->GetGeoTransform(goeInformation);
	const char* gdalProjection = poDataset->GetProjectionRef();
	nImgSizeX = poDataset->GetRasterXSize();
	nImgSizeY = poDataset->GetRasterYSize();
	std::cout << "x:" << nImgSizeX << " y:" << nImgSizeY << std::endl;
	cellSize = goeInformation[1];
	pImgData = new float[nImgSizeX * nImgSizeY];
	GDALRasterBand* pInRasterBand1 = poDataset->GetRasterBand(1);
	CPLErr error;
	error = pInRasterBand1->RasterIO(GF_Read, 0, 0, nImgSizeX, nImgSizeY, pImgData, nImgSizeX, nImgSizeY, GDT_Float32, 0, 0);
	nodata = pInRasterBand1->GetNoDataValue(NULL);
	if (error == CE_Failure) {
		std::cout << "failure" << std::endl;
		GDALDestroyDriverManager();
	}

	vector <PixelCoordinates> peakPoints = peakPointExtract(pImgData, nImgSizeX, nImgSizeY, nodata);
	vector <PixelCoordinates> perilouPoints = perilousPointExtract(peakPoints, pImgData, nImgSizeX, nImgSizeY, cellSize, nodata);
	double regionRisk = perilousAssess(perilouPoints, pImgData, nImgSizeX, nImgSizeY, cellSize, nodata);
	cout << "The risk of the region is " << regionRisk << endl;
	double timeEnd = clock();
	double timeCost = (double)(timeEnd - timeBegin)/ CLOCKS_PER_SEC;
	cout << "sum time cost: " << timeCost << "s" << endl;
}

vector<PixelCoordinates> peakPointExtract(float* image, int cols, int rows, double nodata) 
{
	std::vector<PixelCoordinates> peakPoints;
	peakPoints.clear();
	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++)
		{
			bool isPeak = true;
			int k = 0;
			float window[9];
			for (int j = y - 1; j < y + 2; j++)
				for (int i = x - 1; i < x + 2; i++)
					if (image[j * cols + i] == nodata) {
						window[k++] = image[y * cols + x];
					}
					else {
						window[k++] = image[j * cols + i];
					}
			for (int j = 0; j < 9; j++)
			{
				if (j != 4 && window[j] >= window[4])
					isPeak = false;
			}
			if (isPeak) {
				peakPoints.push_back(PixelCoordinates(x, y));
			}
		}
	return peakPoints;
}

vector<PixelCoordinates> perilousPointExtract(vector<PixelCoordinates> peakPoints, float* image, int cols, int rows, double cellSize, double nodata) {
	double zConvFactor = 1.0;
	std::vector<PixelCoordinates> perilousPoints;
	perilousPoints.clear();
	int row, col; //行列坐标
	double z;
	double neighbors[8];
	float progress = 0;
	int dy[] = { -1, 0, 1, 1, 1, 0, -1, -1 };
	int dx[] = { 1, 1, 1, 0, -1, -1, -1, 0 };
	double radToDeg = 180 / M_PI;
	double curv;
	int perilousPeakNum = 0;
	std::vector<PixelCoordinates>::iterator it;
	for (it = peakPoints.begin(); it != peakPoints.end(); it++) {
		col = it->coordx;
		row = it->coordy;
		z = image[row * cols + col];
		if (z != nodata) {
			z = z * zConvFactor;
			for (int i = 0; i < 8; i++) {
				neighbors[i] = image[(row + dy[i]) * cols + col + dx[i]];
				if (neighbors[i] != nodata) neighbors[i] = neighbors[i] * zConvFactor;
				else neighbors[i] = z;
			}
			curv = getProfCurvature(neighbors, z, cellSize);
			if (curv > 0.5) {
				perilousPoints.push_back(*it);
				perilousPeakNum += 1;
			}
		}
	}
	return perilousPoints;
}

double perilousAssess(vector<PixelCoordinates> perilousPoints, float* image, int cols, int rows, double cellSize, double nodata)
{
	double z, z0;
	double zConvFactor = 1.0;
	int row;
	int col;
	int dy[] = { -1, 0, 1, 1, 1, 0, -1, -1 };
	int dx[] = { 1, 1, 1, 0, -1, -1, -1, 0 };
	double neighbors[8];
	double Profcurvature;
	int times = 0;
	double regionRisk = 0.0;
	
	if (perilousPoints.empty()) {
		//std::cout << "未提取出险峰点" << std::endl;
		return 0.0;
	}
	else 
	{
		std::vector<PixelCoordinates>::iterator it;
		for (it = perilousPoints.begin(); it != perilousPoints.end(); it++)
		{
			col = it->coordx;
			row = it->coordy;
			z0 = z = image[row * cols + col];
			if (z != nodata) {
				z = z * zConvFactor;
				for (int i = 0; i < 8; i++) {
					neighbors[i] = image[(row + dy[i]) * cols + col + dx[i]];
					if (neighbors[i] != nodata) neighbors[i] = neighbors[i] * zConvFactor;
					else neighbors[i] = z;
				}
			}
			double profCurvature = getProfCurvature(neighbors, z, cellSize);
			for (; profCurvature > 0.5; z = z - 0.00001) {//计算高度下降值
				profCurvature = getProfCurvature(neighbors, z, cellSize);
			}
			double pointRisk = z0 - z;
			regionRisk += pointRisk;
		}
		return regionRisk;
	}
}

double getProfCurvature(double* neighbors, double z, double cellSize) {
	double gridResTimes2 = cellSize * 2;
	double gridResSquared = cellSize * cellSize;
	double fourTimesGridResSquared = gridResSquared * 4;
	double radToDeg = 180 / M_PI;
	double DegTorad = M_PI / 180;
	double D, E, F, G, H;
	
	D = ((neighbors[5] + neighbors[1]) / 2 - z) / gridResSquared;
	E = ((neighbors[7] + neighbors[3]) / 2 - z) / gridResSquared;
	F = (neighbors[0] + neighbors[4] - neighbors[6] - neighbors[2]) / fourTimesGridResSquared;
	G = (neighbors[1] - neighbors[5]) / gridResTimes2;
	H = (neighbors[7] - neighbors[3]) / gridResTimes2;
	if (G == 0 && H == 0) {
		return 0;
	}
	else {
		return (-2 * (D * G * G + E * H * H + F * G * H) / (G * G + H * H) * 100); 
	}
}

double getRegionPerilousPointValue(vector<PixelCoordinates> perilousPoints, float* image, int cols, int rows, double cellSize, double nodata)
{
	double z, z0;
	double zConvFactor = 1.0;
	int row;
	int col;
	int dy[] = { -1, 0, 1, 1, 1, 0, -1, -1 };
	int dx[] = { 1, 1, 1, 0, -1, -1, -1, 0 };
	double neighbors[8];
	int times = 0;
	double regionPerilousPointValue = 0.0;

	if (perilousPoints.empty()) {
		return 0.0;
	}
	else
	{
		std::vector<PixelCoordinates>::iterator it;
		for (it = perilousPoints.begin(); it != perilousPoints.end(); it++)
		{
			col = it->coordx;
			row = it->coordy;
			z0 = z = image[row * cols + col];
			if (z != nodata) {
				z = z * zConvFactor;
				regionPerilousPointValue += z;
			}

		}
		return regionPerilousPointValue;
	}
}

double getRegionProfCurvature(vector<PixelCoordinates> perilousPoints, float* image, int cols, int rows, double cellSize, double nodata)
{
	double z, z0;
	double zConvFactor = 1.0;
	int row;
	int col;
	int dy[] = { -1, 0, 1, 1, 1, 0, -1, -1 };
	int dx[] = { 1, 1, 1, 0, -1, -1, -1, 0 };
	double neighbors[8];
	int times = 0;
	double regionProfcurvature = 0.0;

	if (perilousPoints.empty()) {
		return 0.0;
	}
	else
	{
		std::vector<PixelCoordinates>::iterator it;
		for (it = perilousPoints.begin(); it != perilousPoints.end(); it++)
		{
			col = it->coordx;
			row = it->coordy;
			z0 = z = image[row * cols + col];
			if (z != nodata) {
				z = z * zConvFactor;
				for (int i = 0; i < 8; i++) {
					neighbors[i] = image[(row + dy[i]) * cols + col + dx[i]];
					if (neighbors[i] != nodata) neighbors[i] = neighbors[i] * zConvFactor;
					else neighbors[i] = z;
				}
			}
			double profCurvature = getProfCurvature(neighbors, z, cellSize);
			regionProfcurvature += profCurvature;
		}
		return regionProfcurvature;
	}
}



