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
#include <assert.h>
#include <time.h> 

using namespace std;

/**
* Assign data based on ML grid computational intensity,when performing intersection.
* Using vector<vector<int>> getPartitionBoundary(double* gatherCI, long cellNum, int size)
*/
void parallelPeakPointExtractMachineLearning1(const char* inPath, const char* gridFeatureDir, const char* outPath, int gridSize);

int main(int argc, char* argv[]) {
	const char* inPath = "/home/fgao/LoadBalance/PeakPoint/data/sichuan_1.tif";
        const char* gridFeatureDir = "/home/fgao/LoadBalance/PeakPoint/data/rank/";
        const char* outPath = "/home/fgao/LoadBalance/PeakPoint/data/3_result.tif";
	parallelPeakPointExtractMachineLearning1(inPath, gridFeatureDir, outPath, 256);
	return 0;
}

void parallelPeakPointExtractMachineLearning1(const char* inPath, const char* gridFeatureDir, const char* outPath, int gridSize)
{
	cout << "Parallel task is running..." << endl;
	time_t begin, end;
	time(&begin);

	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status* status = new MPI_Status;

	double rankCost;
	time_t rankBegin, rankEnd;
	time(&rankBegin);

	double rankRegionRisk = 0;
	int interval;
	int nImgSizeX, nImgSizeY;
	float* pImgData;
	pImgData = NULL;
	double cellSize;
	double nodata;

	int* partitionBounds = new int[size];
	double* partitionComputeIntensity = new double[size - 1];
	memset(partitionComputeIntensity, 0.0, (size - 1) * sizeof(double));

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

	interval = gridCornerPoints.size() / (size - 1);

	int* splitNode = new int[size];
	for (int i = 0; i < size - 1; i++) splitNode[i] = i * interval;
	splitNode[size - 1] = gridCornerPoints.size();
	if (!rank)
	{
		cout << "partition bounds for CI prediction is ";
		for (int i = 0; i < size; i++) cout << splitNode[i] << "  ";
		cout << endl;
	}

	if (rank)
	{
		float* gridData;
		gridData = new float[gridSize * gridSize];
		std::stringstream rankStr;
		rankStr << rank;
		string inputRankDirStr = (string)gridFeatureDir + rankStr.str();
		ifstream f(inputRankDirStr.c_str());
		if (!f.good())
		{
			string command = "mkdir " + inputRankDirStr;
			system(command.c_str());
		}
		ofstream foutFeatures((inputRankDirStr + "/features.txt").c_str(), std::ios::out);
		for (int k = splitNode[rank - 1]; k < splitNode[rank]; k++)
		{
			int x_corner = gridCornerPoints[k].coordx;
			int y_corner = gridCornerPoints[k].coordy;
			for (int j = 0; j < gridSize; j++) {
				for (int i = 0; i < gridSize; i++)
				{
					gridData[i + j * gridSize] = pImgData[(x_corner + i) + (y_corner + j) * nImgSizeX];
				}
			}

			vector <PixelCoordinates> peakPoints = peakPointExtract(gridData, gridSize, gridSize, nodata);
			vector <PixelCoordinates> perilouPoints = perilousPointExtract(peakPoints, gridData, gridSize, gridSize, cellSize, nodata);
			double regionPerilousPointValue = getRegionPerilousPointValue(perilouPoints, gridData, gridSize, gridSize, cellSize, nodata);
			double regionProfCurvature = getRegionProfCurvature(perilouPoints, gridData, gridSize, gridSize, cellSize, nodata);
			foutFeatures << perilouPoints.size() << "\t" << regionPerilousPointValue << "\t" << regionProfCurvature << endl;
			//foutFeatures << peakPoints.size() << "\t" << perilouPoints.size() << "\t" << regionPerilousPointValue << "\t" << regionProfCurvature << endl;

		}
		foutFeatures.close();
		const char* function = "randomforest";
		const char* modelPath = "/home/fgao/LoadBalance/PeakPoint/python/ml_env/ml/result_model/ml_rfr.pk";
		const char* inputPath = (inputRankDirStr + "/features.txt").c_str();
		const char* outputCIPath = (inputRankDirStr + "/rfr_predict.txt").c_str();
		callMlModel(function, modelPath, (inputRankDirStr + "/features.txt").c_str(), (inputRankDirStr + "/rfr_predict.txt").c_str());

		ifstream computeIntensityFile(outputCIPath);
		string temp;
		double* rankGridCI = new double[splitNode[rank] - splitNode[rank - 1]];
		int i = 0;
		while (getline(computeIntensityFile, temp))
		{
			stringstream ss;
			ss << temp;
			ss >> rankGridCI[i];
			i++;
		}
		assert(i == (splitNode[rank] - splitNode[rank - 1]));

		//slave processes send computation intensity of grid to master process
		int startGridRange = splitNode[rank - 1]; int endGridRange = splitNode[rank];
		int gridRange = endGridRange - startGridRange;
		MPI_Send(rankGridCI, gridRange, MPI_DOUBLE, 0, rank + 888, MPI_COMM_WORLD);

		//slave processes receive grids and perform peak point extraction
		int rankGridNum;
		MPI_Recv(&rankGridNum, 1, MPI_INT, 0, rank + 999, MPI_COMM_WORLD, status);
		int* container = new int[rankGridNum];
		MPI_Recv(container, rankGridNum, MPI_INT, 0, rank + 1000, MPI_COMM_WORLD, status);

		rankRegionRisk = 0;
		double* rankGridRealTimeCost = new double[rankGridNum];
		for (int k = 0; k < rankGridNum; k++)
		{
			int index = container[k];
			int x_corner = gridCornerPoints[index].coordx;
			int y_corner = gridCornerPoints[index].coordy;
			for (int j = 0; j < gridSize; j++) {
				for (int i = 0; i < gridSize; i++)
				{
					gridData[i + j * gridSize] = pImgData[(x_corner + i) + (y_corner + j) * nImgSizeX];
				}
			}
			//time_t gridRealTimeBegin, gridRealTimeEnd;
			//time(&gridRealTimeBegin);

			double gridRealTimeBegin = clock();

			vector <PixelCoordinates> peakPoints = peakPointExtract(gridData, gridSize, gridSize, nodata);
			vector <PixelCoordinates> perilouPoints = perilousPointExtract(peakPoints, gridData, gridSize, gridSize, cellSize, nodata);
			double regionRisk = perilousAssess(perilouPoints, gridData, gridSize, gridSize, cellSize, nodata);

			double gridRealTimeEnd = clock();
			double gridRealTimeCost = (double)(gridRealTimeEnd - gridRealTimeBegin) / CLOCKS_PER_SEC;

			//time(&gridRealTimeEnd);
			//double gridRealTimeCost = difftime(gridRealTimeEnd, gridRealTimeBegin);

			rankGridRealTimeCost[k] = gridRealTimeCost;

			rankRegionRisk += regionRisk;
		}

		// slave processes send the result(i.e. rankRegionRisk and rankGridRealTimeCost) to master process
		MPI_Send(&rankRegionRisk, 1, MPI_DOUBLE, 0, rank + 1111, MPI_COMM_WORLD);
		MPI_Send(rankGridRealTimeCost, rankGridNum, MPI_DOUBLE, 0, rank + 2222, MPI_COMM_WORLD);
	}

	if (!rank)
	{
		//master process receives computation intensity of grid
		vector<double> gridComputeIntensity;
		for (int i = 0; i < size - 1; i++)
		{
			int startGridRange = splitNode[i]; int endGridRange = splitNode[i + 1];
			int gridRange = endGridRange - startGridRange;
			double* rankGridCI = new double[gridRange];
			MPI_Recv(rankGridCI, gridRange, MPI_DOUBLE, i + 1, i + 1 + 888, MPI_COMM_WORLD, status);
			for (int j = 0; j < gridRange; gridComputeIntensity.push_back(rankGridCI[j]), j++);
		}

		//master process determines partition bounds for peak point extraction based on computational intensity
		//and sends the partition bounds to slave processes
		vector< vector<int> > vcontainers = getPartitionBoundary(gridComputeIntensity, (size - 1));
		for (int i = 0; i < size - 1; i++)
		{
			vector <int> vcontainer = vcontainers[i];
			int rankGridNum = vcontainer.size();
			MPI_Send(&rankGridNum, 1, MPI_INT, i + 1, i + 1 + 999, MPI_COMM_WORLD);

			int* container = new int[rankGridNum];
			memcpy(container, &vcontainer[0], vcontainer.size() * sizeof(vcontainer[0]));
			MPI_Send(container, rankGridNum, MPI_INT, i + 1, i + 1 + 1000, MPI_COMM_WORLD);
		}

		//master process receives rankRegionRisk and rankGridRealTimeCost
		double regionRisk = 0;
		vector<double> gridRealTimeCost;

		for (int i = 0; i < size - 1; i++)
		{
			MPI_Recv(&rankRegionRisk, 1, MPI_DOUBLE, i + 1, i + 1 + 1111, MPI_COMM_WORLD, status);
			regionRisk += rankRegionRisk;

			int rankGridNum = vcontainers[i].size();
			double* rankGridRealTimeCost = new double[rankGridNum];
			MPI_Recv(rankGridRealTimeCost, rankGridNum, MPI_DOUBLE, i + 1, i + 1 + 2222, MPI_COMM_WORLD, status);
			double rankRealTimeCost = 0.0;
			for (int j = 0; j < rankGridNum; gridRealTimeCost.push_back(rankGridRealTimeCost[j]), j++) {
				rankRealTimeCost += rankGridRealTimeCost[j];
			}
			cout << "The 3-tage real time cost of rank " << i + 1 << " is " << rankRealTimeCost << endl;
		}
		assert(gridComputeIntensity.size() == gridRealTimeCost.size());

		vector<double> assignedGridComputeIntensity;
		for (int i = 0; i < vcontainers.size(); i++)
		{
			vector <int> vcontainer = vcontainers[i];
			for (int j = 0; j < vcontainer.size(); j++)
			{
				assignedGridComputeIntensity.push_back(gridComputeIntensity[vcontainer[j]]);
			}
		}
		assert(assignedGridComputeIntensity.size() == gridRealTimeCost.size());
		for (int i = 0; i < assignedGridComputeIntensity.size(); i++) cout << i << ": {predict=" << assignedGridComputeIntensity[i] << ", actual=" << gridRealTimeCost[i] << "}" << endl;
		cout << "The risk of the region is " << regionRisk << endl;
	}

	time(&rankEnd);
	rankCost = difftime(rankEnd, rankBegin);

	double* processCost = new double[size];
	MPI_Gather(&rankCost, 1, MPI_DOUBLE, processCost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);


	if (!rank)
	{
		time(&end);
		double cost = difftime(end, begin);
		cout << "sum time cost: " << cost << "s" << endl;

		double sum = 0.0, mean, mean2 = 0.0, var, stdDev;
		for (size_t i = 1; i < size; i++) {
			cout << "process " << i << " time cost: " << processCost[i] << "s" << endl;
			sum += processCost[i];
		}

		mean = sum / (size - 1);
		for (size_t i = 1; i < size; i++) {
			mean2 += (processCost[i] - mean) * (processCost[i] - mean);
		}
		var = mean2 / (size - 1);
		stdDev = sqrt(var);
		std::cout << "load balance metric: " << stdDev << std::endl;
	}
	MPI_Finalize();
}
