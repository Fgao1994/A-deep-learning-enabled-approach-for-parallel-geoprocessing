#include "stdio.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include "gdal_priv.h"
#include "ogr_spatialref.h"
#include "ogrsf_frmts.h"
#include <Python.h>
#include <algorithm>

#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308

using namespace std;

template<class T>
int getLength(const T& arr) {
	return sizeof(arr) / sizeof(arr[0]);
}

void splitString(string& s, vector<string>& v, string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

int dMinIndex(vector<double> arr) 
{
	double min = arr[0];
	int index = 0;
	for (int i = 0; i < arr.size(); i++)
	{
		if (arr[i] < min) 
		{
			min = arr[i];
			index = i;
		}
	}
	return index;
}

vector< vector<int> > getPartitionBoundary(vector<double> cellCI, int size)
{
	vector<double> orderCellCI = cellCI;
	sort(orderCellCI.begin(), orderCellCI.end());
	int cellNum = cellCI.size();

	vector< vector<int> > containers;
	for (int i = 0; i < size; i++) 
	{
		vector<int> container;
		containers.push_back(container);
	}
	vector<double> containerCI;

	for (int i = 0; i < size; i++) 
	{
		containerCI.push_back(orderCellCI[cellNum - i - 1]);
		int index;
		for (int j = 0; j < cellNum; j++)
		{
			if (cellCI[j] == orderCellCI[cellNum - i - 1])
			{
				int index = j;
				containers[i].push_back(index);
				break;
			}			
		}
		
	}
	int minIndex = dMinIndex(containerCI);

	for (int i = size; i < cellNum; i++)
	{
		double value = orderCellCI[cellNum - i - 1];
		containerCI[minIndex] += value;
		for (int j = 0; j < cellNum; j++)
		{
			if (cellCI[j] == value)
			{
				int index = j;
				bool flag = true;
				for (int k = 0; k < size; k++)
				{
					int containerGridNum = containers[k].size();
					for (int p = 0; p < containerGridNum; p++)
					{
						if (containers[k][p] == index)
						{
							flag = false;
							break;
						}
							
					}
					if (!flag) break;
				}
				if (flag) {
					containers[minIndex].push_back(index);
					break;
				}				
			}
		}
		minIndex = dMinIndex(containerCI);
	}
	
	cout << "partition computational intensity is ";
	for (int i = 0; i < size; i++) cout << containerCI[i] << "  ";
	cout << endl;
	return containers;
}

void getPartitionBoundary(double* gatherCI, int size, long cellNum, int* bounds, double* partitionCI)
{
	double sumCI = 0.0;
	for (int i = 0; i < cellNum; i++)
	{
		sumCI += gatherCI[i];
	}
	double str = DBL_MAX;
	for (int count = -20; count < 20; count++)
	{
		int partition_para = size + count;
		double averageCI = sumCI / partition_para;
		int* bounds_tmp = new int[size + 1];
		double* partitionCI_tmp = new double[size];
		for (int i = 0; i < size; i++)
			partitionCI_tmp[i] = 0.0;
		int split = 0; int tmp = 0;
		for (int i = 0; i < size; i++)
		{
			bounds_tmp[i] = split;
			for (int j = tmp; j < cellNum; j++)
			{
				if (partitionCI_tmp[i] <= averageCI)
				{
					tmp++; split++;
					partitionCI_tmp[i] += gatherCI[j];
				}
				else if (i == size - 1)
				{
					tmp++; split++;
					partitionCI_tmp[i] += gatherCI[j];
				}
				else
				{
					break;
				}

			}
		}
		bounds_tmp[size] = split;
		double ci_min = DBL_MAX;
		double ci_max = DBL_MIN;
		for (int i = 0; i < size; i++)
		{
			if (partitionCI_tmp[i] < ci_min)ci_min = partitionCI_tmp[i];
			if (partitionCI_tmp[i] > ci_max)ci_max = partitionCI_tmp[i];
		}
		if ((ci_max - ci_min) / ci_max < str)
		{
			str = (ci_max - ci_min) / ci_max;
			for (int i = 0; i < size + 1; i++)
				bounds[i] = bounds_tmp[i];
			for (int i = 0; i < size; i++)
				partitionCI[i] = partitionCI_tmp[i];
		}
	}
	cout << "partition bounds is ";
	for (int i = 0; i < size+1; i++) cout << bounds[i] << "  ";
	cout << endl;
	cout << "partition computational intensity is ";
	for (int i = 0; i < size; i++) cout << partitionCI[i] << "  ";
	cout << endl;
}

int doubleToInt(double dValue)
{
	if (dValue < 0.0)
		return static_cast<int>(dValue - 0.5);
	else
		return static_cast<int>(dValue + 0.5);
	return 0;
}

void callMlModel(const char* funcName, const char* modelPath, const char* inputPath, const char* outputPath)
{
	string command = "python /home/fgao/LoadBalance/PeakPoint/python/ml_env/ml/inference/model_call.py "
		+ (string)funcName + " "
		+ (string)modelPath + " "
		+ (string)inputPath + " "
		+ (string)outputPath;
	cout << command << endl;
	system(command.c_str());
}

void callCnnModel(const char* funcName, const char* modelPath, const char* inputDir, const char* outputPath)
{
	string command = "python /home/fgao/LoadBalance/PeakPoint/python/ml_env/dl/peak_point/model_call.py "
		+ (string)funcName + " "
		+ (string)modelPath + " "
		+ (string)inputDir + " "
		+ (string)outputPath;
	cout << command << endl;
	system(command.c_str());
}

void callCnnModel(const char* script, const char* funcName, const char* modelPath, const char* inputDir, const char* outputPath)
{
	string command = "python "
		+ (string)script + " "
		+ (string)funcName + " "
		+ (string)modelPath + " "
		+ (string)inputDir + " "
		+ (string)outputPath;
		//+ "> /dev/null";

	system(command.c_str());
}

void callCnnModel(const char* script, const char* funcName, const char* modelPath, const char* inputDir, const char* outputPath, const char* gpuIndex)
{
	string command = "python "
		+ (string)script + " "
		+ (string)funcName + " "
		+ (string)modelPath + " "
		+ (string)inputDir + " "
		+ (string)outputPath + " "
		+ (string)gpuIndex;

	system(command.c_str());
}

/*int main(int argc, char* argv[])
{
	const char* function = "randomforest";        
	const char* modelPath = "/home/fgao/LoadBalance/PeakPoint/python/ml_env/ml/result_model/ml_rfr.pk";
        const char* inputPath = "/home/fgao/LoadBalance/PeakPoint/model/test10/sample_ml_features.txt";
        const char* outputCIPath = "/home/fgao/LoadBalance/PeakPoint/model/test10/sample_ml_label1.txt";
        callMlModel(function, modelPath, inputPath, outputCIPath);
	
	//const char* function = "resnet";
	//const char* modelPath = "/home/fgao/LoadBalance/PeakPoint/python/ml_env/dl/peak_point/result_model/resnet.pth";
	//const char* inputDir = "/home/fgao/LoadBalance/PeakPoint/model/test2/sample_image";
	//const char* outputCIPath = "/home/fgao/LoadBalance/PeakPoint/python/ml_env/dl/peak_point/result_model/resnet_predict.txt";	
	//callCnnModel(function, modelPath, inputDir, outputCIPath);
}*/
