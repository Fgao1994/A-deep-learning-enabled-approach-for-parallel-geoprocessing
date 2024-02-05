#include "stdio.h"
#include "gdal_priv.h"
#include <iostream>
#include "mpi.h"
#include <math.h>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include "SerialPeakPointExtract1.h"
#include "Util.h"

using namespace std;

void parallelPeakPointSampleGeneration(const char* inPath, char* outLabelDir, char* outImageDir, int gridSize);
void sampleGenerationForML(char* inSamplePath, char* imageDir, int gridSize, char* outSamplePath);

int main(int argc, char* argv[]) {
	parallelPeakPointSampleGeneration(argv[1], argv[2], argv[3], 256);
	return 0;
}

void sampleGenerationForML(char* inSamplePath, char* imageDir, int gridSize, char* outSamplePath)
{
	ifstream sample(inSamplePath);
	ofstream foutSample(outSamplePath, std::ios::out);
	string line;
	int nImgSizeX, nImgSizeY;
	float* gridData;
	double cellSize;
	double nodata;
	int count = 0;
	while (getline(sample, line))
	{
		cout << count << endl;
		count += 1;
		vector<string> v;
		string separater = "\t";
		splitString(line, v, separater);
		string imagePath = imageDir;
		imagePath = imagePath + v[0] + ".tif";
		const char* outputImagePath = imagePath.data();
		GDALDataset* poDataset = NULL;
		double goeInformation[6];
		const char* gdalProjection = NULL;

		GDALAllRegister();
		poDataset = (GDALDataset*)GDALOpen(outputImagePath, GA_ReadOnly);
		if (poDataset == NULL)
		{
			std::cout << "nothing" << std::endl;
		}

		poDataset->GetGeoTransform(goeInformation);
		cellSize = goeInformation[1];
		nImgSizeX = poDataset->GetRasterXSize();
		nImgSizeY = poDataset->GetRasterYSize();
		gridData = new float[nImgSizeX * nImgSizeY];
		GDALRasterBand* pInRasterBand1 = poDataset->GetRasterBand(1);
		CPLErr error;
		error = pInRasterBand1->RasterIO(GF_Read, 0, 0, nImgSizeX, nImgSizeY, gridData, nImgSizeX, nImgSizeY, GDT_Float32, 0, 0);
		nodata = pInRasterBand1->GetNoDataValue(NULL);
		gdalProjection = poDataset->GetProjectionRef();
		if (error == CE_Failure)
		{
			std::cout << "failure" << std::endl;
			GDALDestroyDriverManager();
		}
		vector <PixelCoordinates> peakPoints = peakPointExtract(gridData, gridSize, gridSize, nodata);
		vector <PixelCoordinates> perilouPoints = perilousPointExtract(peakPoints, gridData, gridSize, gridSize, cellSize, nodata);
		double regionPerilousPointValue = getRegionPerilousPointValue(perilouPoints, gridData, gridSize, gridSize, cellSize, nodata);
		double regionProfCurvature = getRegionProfCurvature(perilouPoints, gridData, gridSize, gridSize, cellSize, nodata);
		foutSample << v[0] << "\t" << v[1] << "\t" << v[2] << "\t" << v[3] << "\t" << regionPerilousPointValue << "\t" << regionProfCurvature << "\t" << v[4] << std::endl;
	}
	foutSample.close();

}

void parallelPeakPointSampleGeneration(const char* inPath, char* outLabelDir, char* outImageDir, int gridSize)
{
	cout << "Parallel task is running..." << endl;
	double begin = clock();
	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int interval;
	int nImgSizeX, nImgSizeY;
	float* pImgData;
	pImgData = NULL;
	double cellSize;
	double nodata;
	GDALDataset* poDataset = NULL;
	double goeInformation[6];
	const char* gdalProjection = NULL;

	GDALAllRegister();
	poDataset = (GDALDataset*)GDALOpen(inPath, GA_ReadOnly);
	if (poDataset == NULL)
	{
		std::cout << "nothing" << std::endl;
	}

	poDataset->GetGeoTransform(goeInformation);
	cellSize = goeInformation[1];
	nImgSizeX = poDataset->GetRasterXSize();
	nImgSizeY = poDataset->GetRasterYSize();
	pImgData = new float[nImgSizeX * nImgSizeY];
	GDALRasterBand* pInRasterBand1 = poDataset->GetRasterBand(1);
	CPLErr error;
	error = pInRasterBand1->RasterIO(GF_Read, 0, 0, nImgSizeX, nImgSizeY, pImgData, nImgSizeX, nImgSizeY, GDT_Float32, 0, 0);
	nodata = pInRasterBand1->GetNoDataValue(NULL);
	gdalProjection = poDataset->GetProjectionRef();
	if (error == CE_Failure)
	{
		std::cout << "failure" << std::endl;
		GDALDestroyDriverManager();
	}

	int xDim = (int)(nImgSizeX / gridSize);
	int yDim = (int)(nImgSizeY / gridSize);
	vector<PixelCoordinates> gridCornerPoints;
	for (int y = 0; y < yDim * gridSize; y += gridSize)
	{
		for (int x = 0; x < xDim * gridSize; x += gridSize)
		{
			gridCornerPoints.push_back(PixelCoordinates(x, y));
		}
	}

	interval = gridCornerPoints.size() / size;

	int* split_node = new int[size + 1];
	for (int i = 0; i < size; i++) split_node[i] = i * interval;
	split_node[size] = gridCornerPoints.size();
	if (!rank)
	{
		for (int i = 0; i < size + 1; i++) std::cout << split_node[i] << "  ";
		std::cout << endl;
	}

	float* gridData;
	gridData = new float[gridSize * gridSize];
	double rankRegionRisk = 0;
	int gridNum = 0;
	std::stringstream ssRank;
	ssRank << rank;
	string outLabelPath = outLabelDir;
	outLabelPath = outLabelPath + "sample_" + ssRank.str() + ".txt";
	std::ofstream foutSample(outLabelPath.c_str(), std::ios::out);
	for (int k = split_node[rank]; k < split_node[rank + 1]; k++)
	{
		int x_corner_fix = gridCornerPoints[k].coordx;
		int y_corner_fix = gridCornerPoints[k].coordy;

		for (int j = 0; j < gridSize; j++) {
			for (int i = 0; i < gridSize; i++)
			{
				gridData[i + j * gridSize] = pImgData[(x_corner_fix + i) + (y_corner_fix + j) * nImgSizeX];
			}
		}
		double taskBegin = clock();
		vector <PixelCoordinates> peakPoints = peakPointExtract(gridData, gridSize, gridSize, nodata);
		vector <PixelCoordinates> perilouPoints = perilousPointExtract(peakPoints, gridData, gridSize, gridSize, cellSize, nodata);
		double regionRisk = perilousAssess(perilouPoints, gridData, gridSize, gridSize, cellSize, nodata);
		double taskEnd = clock();
		double taskCost = (double)(taskEnd - taskBegin) / CLOCKS_PER_SEC;

		int focalRange;
		if (taskCost < 10)
			focalRange = 1;
		else if (taskCost < 20)
			focalRange = 6;
		else if (taskCost < 40)
			focalRange = 10;
		else
			focalRange = 20;
		int startFocalRange = 0 - focalRange;
		int endFocalRange = focalRange;

		for (int p = startFocalRange; p <= endFocalRange; p++) {
			int y_corner = y_corner_fix + p;
			if (y_corner < 0 || y_corner > nImgSizeY - gridSize) continue;		
			for (int q = startFocalRange; q <= endFocalRange; q++) {			
				int x_corner = x_corner_fix + q;
				if (x_corner < 0 || x_corner > nImgSizeX - gridSize) continue;
				for (int j = 0; j < gridSize; j++) {
					for (int i = 0; i < gridSize; i++)
					{
						gridData[i + j * gridSize] = pImgData[(x_corner + i) + (y_corner + j) * nImgSizeX];
					}
				}
				double taskBegin = clock();
				vector <PixelCoordinates> peakPoints = peakPointExtract(gridData, gridSize, gridSize, nodata);
				vector <PixelCoordinates> perilouPoints = perilousPointExtract(peakPoints, gridData, gridSize, gridSize, cellSize, nodata);
				double regionRisk = perilousAssess(perilouPoints, gridData, gridSize, gridSize, cellSize, nodata);
				double taskEnd = clock();
				double taskCost = (double)(taskEnd - taskBegin) / CLOCKS_PER_SEC;

				double regionPerilousPointValue = getRegionPerilousPointValue(perilouPoints, gridData, gridSize, gridSize, cellSize, nodata);
				double regionProfCurvature = getRegionProfCurvature(perilouPoints, gridData, gridSize, gridSize, cellSize, nodata);

				std::stringstream ssGridNum;
				ssGridNum << gridNum;

				string index = ssRank.str() + "_" + ssGridNum.str() + "_dataset1";
				int peakPointNum = peakPoints.size();
				int perilousPointNum = perilouPoints.size();
				foutSample << index << "\t" << peakPointNum << "\t" << perilousPointNum << "\t" << doubleToInt(regionRisk) << "\t" << taskCost << std::endl;
				//foutSample << index << "\t" << peakPointNum << "\t" << perilousPointNum << "\t" << doubleToInt(regionRisk) << "\t" << regionPerilousPointValue << "\t" << regionProfCurvature << "\t" << taskCost << std::endl;

				string outputImagePathStr = outImageDir;
				outputImagePathStr = outputImagePathStr + index + ".tif";
				GDALDataset * poDstDS;
				const char* pszFormat = "GTiff";
				GDALDriver* poDriver;
				poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
				if (poDriver == NULL) exit(1);
				const char* outputImagePath = outputImagePathStr.data();
				poDstDS = poDriver->Create(outputImagePath, gridSize, gridSize, 1, GDT_Float32, NULL);
				poDstDS->SetGeoTransform(goeInformation);
				poDstDS->SetProjection(gdalProjection);
				poDstDS->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, gridSize, gridSize,
					gridData, gridSize, gridSize, GDT_Float32, 0, 0);
				poDstDS->GetRasterBand(1)->SetNoDataValue(nodata);
				GDALClose((GDALDatasetH)poDstDS);

				gridNum += 1;
				cout << index << endl;
			}
		}	
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank)
	{
		double end = clock();
		double cost = (double)(end - begin) / CLOCKS_PER_SEC;
		cout << "sum time cost: " << cost << "s" << endl;
	}
	MPI_Finalize();
}
