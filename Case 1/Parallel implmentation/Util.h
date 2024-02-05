#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include<string>
using namespace std;

#ifndef UTIL_H
#define UTIL_H

template<class T>
extern int getLength(const T& arr);

extern void splitString(string& s, vector<string>& v, string& c);

extern int dMinIndex(vector<double> arr);

extern vector< vector<int> > getPartitionBoundary(vector<double> cellCI, int size);

extern void getPartitionBoundary(double* gatherCI, int size, long cellNum, int* bounds, double* partitionCI);

extern int doubleToInt(double dValue);

extern void callMlModel(const char* funcName, const char* modelPath, const char* inputPath, const char* outputPath);

extern void callCnnModel(const char* funcName, const char* modelPath, const char* inputDir, const char* outputPath);

extern void callCnnModel(const char* script, const char* funcName, const char* modelPath, const char* inputDir, const char* outputPath);

extern void callCnnModel(const char* script, const char* funcName, const char* modelPath, const char* inputDir, const char* outputPath, const char* gpuIndex);

typedef struct GeoGrid
{
	int gridID;
	vector<int> polygonsFID1;
	vector<int> polygonsFID2;
	double computeIntensity;
	~GeoGrid()
	{
		vector<int>().swap(polygonsFID1);
		vector<int>().swap(polygonsFID2);
	}
}GeoGrid;

#endif
