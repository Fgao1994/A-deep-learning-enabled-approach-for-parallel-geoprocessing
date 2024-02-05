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

void parallelPeakPointExtractDeepLearning1(const char* inPath, const char* gridImageDir, const char* outPath, int gridSize);

int main(int argc, char* argv[]) {
	const char* inPath = "/home/fgao/LoadBalance/PeakPoint/data/sichuan_1.tif";
	const char* gridImageDir = "/home/fgao/LoadBalance/PeakPoint/data/rank/";
	const char* outPath = "/home/fgao/LoadBalance/PeakPoint/data/3_result.tif";
	parallelPeakPointExtractDeepLearning1(inPath, gridImageDir, outPath, 256);
	return 0;
}

void parallelPeakPointExtractDeepLearning1(const char* inPath, const char* gridImageDir, const char* outPath, int gridSize)
{
	cout << "Parallel task is running..." << endl;
	time_t begin, end;
	time(&begin);

	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status* status = new MPI_Status;
	
	time_t rankBegin, rankEnd;
	double rankCost;
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
	splitNode[size-1] = gridCornerPoints.size();
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
		string inputRankDirStr = (string)gridImageDir + rankStr.str();
		ifstream f(inputRankDirStr.c_str());
		if (!f.good())
		{
			string command = "mkdir " + inputRankDirStr;
			system(command.c_str());
		}

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
			GDALAllRegister();
			GDALDataset* poDstDS;
			const char* pszFormat = "GTiff";
			GDALDriver* poDriver;
			poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
			if (poDriver == NULL)exit(1);

			std::stringstream indexStr;
			indexStr << k;
			const char* gridImagePath = ((string)gridImageDir + rankStr.str() + "/" + indexStr.str() + ".tif").c_str();
			poDstDS = poDriver->Create(gridImagePath, gridSize, gridSize, 1, GDT_Float32, NULL);
			poDstDS->SetGeoTransform(goeInformation);
			poDstDS->SetProjection(gdalProjection);
			CPLErr error = poDstDS->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, gridSize, gridSize,
				gridData, gridSize, gridSize, GDT_Float32, 0, 0);
			poDstDS->GetRasterBand(1)->SetNoDataValue(nodata);
			GDALClose((GDALDatasetH)poDstDS);
			indexStr.clear();
		}
		
		double* flag = new double[1];
		flag[0] = 1;
		MPI_Send(flag, 1, MPI_DOUBLE, 0, rank + 888, MPI_COMM_WORLD);

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

		// slave processes send the result(i.e. rankRegionRisk) to master process
		MPI_Send(&rankRegionRisk, 1, MPI_DOUBLE, 0, rank + 1111, MPI_COMM_WORLD);
		MPI_Send(rankGridRealTimeCost, rankGridNum, MPI_DOUBLE, 0, rank + 2222, MPI_COMM_WORLD);
	}
	if (!rank)
	{
		//master process receives that slave processes have calculated features of grid
		for (int i = 0; i < size - 1; i++)
		{
			double* flag = new double[1];
			MPI_Recv(flag, 1, MPI_DOUBLE, i + 1, i + 1 + 888, MPI_COMM_WORLD, status);
		}	

		//master process begins to calculate computation intensity of grid
		time_t predictBegin, predictEnd;
		time(&predictBegin);
		vector<double> gridComputeIntensity;
		for (int i = 0; i < size - 1; i++)
		{
			time_t processPredictBegin, processPredictEnd;
			time(&processPredictBegin);
			std::stringstream ssRank;
			string strRank;
			ssRank << (i + 1);
			ssRank >> strRank;
			string inputRankDirStr = (string)gridImageDir + strRank;

			const char* script = "/home/fgao/LoadBalance/Intersection/python/ml_env/dl/peak_point/model_call.py";
			const char* function = "resnet";
			const char* modelPath = "/home/fgao/LoadBalance/Intersection/python/ml_env/dl/peak_point/result_model/resnet4.pth";
			const char* inputRankDir = inputRankDirStr.c_str();
			const char* outputCIPath = (inputRankDirStr + "/predictCI.txt").c_str();

			string gpuIndexStr;
			gpuIndexStr = "cuda:0";
			gpuIndexStr = "N";
			const char* gpuIndex = gpuIndexStr.c_str();

			//callCnnModel(script, function, modelPath, inputRankDir, outputCIPath);
			callCnnModel(script, function, modelPath, inputRankDir, outputCIPath, gpuIndex);

			time(&processPredictEnd);
			double processPredictCost = difftime(processPredictEnd, processPredictBegin);
			cout << "process " << i + 1 << " prediction time cost is " << processPredictCost << "s" << endl;	
		}
		for (int i = 0; i < size - 1; i++)
		{
			std::stringstream ssRank;
			string strRank;
			ssRank << (i + 1);
			ssRank >> strRank;
			string inputRankDirStr = (string)gridImageDir + strRank;
			const char* outputCIPath = (inputRankDirStr + "/predictCI.txt").c_str();
			ifstream computeIntesityFile(outputCIPath);
			string temp;
			while (getline(computeIntesityFile, temp))
			{
				double computeIntensity;
				vector<string> pair;
				string separater = ",";
				splitString(temp, pair, separater);
				stringstream ss;
				ss << pair[1];
				ss >> computeIntensity;
				gridComputeIntensity.push_back(computeIntensity);
			}
			computeIntesityFile.close();
		}
		time(&predictEnd);
		double predictCost = difftime(predictEnd, predictBegin);
		cout << "Prediction sum time cost is " << predictCost << endl;

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

		//master process receives rankRegionRisk and sum
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
			cout << "The 3-stage real time cost of rank " << i + 1 << " is " << rankRealTimeCost << endl;
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

	double* rankRegionRisks = new double[size];
	MPI_Gather(&rankRegionRisk, 1, MPI_DOUBLE, rankRegionRisks, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
