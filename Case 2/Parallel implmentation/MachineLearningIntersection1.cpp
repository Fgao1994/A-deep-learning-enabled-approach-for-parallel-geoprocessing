#include "stdio.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "gdal_priv.h"
#include "ogr_spatialref.h"
#include "ogrsf_frmts.h"
#include "mpi.h"
#include "Util.h"

void parallelIntersectionMachineLearning(const char* inPath1, const char* inPath2, char* outPath, char* gridFeatureDir, int gridDimX, int gridDimY);

int main(int argc, char* argv[])
{
        int gridDimX, gridDimY;
        sscanf(argv[5], "%d", &gridDimX);
        sscanf(argv[6], "%d", &gridDimY);
        parallelIntersectionMachineLearning(argv[1], argv[2], argv[3], argv[4], gridDimX, gridDimY);

        return 0;
}

void parallelIntersectionMachineLearning(const char* inPath1, const char* inPath2, char* outPath, char* gridFeatureDir, int gridDimX, int gridDimY)
{
	cout << "Parallel task is running..." << endl;

	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Datatype* MPI_Envelope = new MPI_Datatype;
	MPI_Status* status = new MPI_Status;
	MPI_Type_contiguous(4, MPI_DOUBLE, MPI_Envelope);
	MPI_Type_commit(MPI_Envelope);

	GDALAllRegister();
	OGRRegisterAll();
	CPLSetConfigOption("SHAPE_ENCODING", "");

	time_t start, end;
	time(&start);

	time_t rankStart, rankEnd;
	time(&rankStart);

	int* partitionBounds = new int[size];
	double* partitionComputeIntensity = new double[size - 1];
	memset(partitionComputeIntensity, 0.0, (size - 1) * sizeof(double));

	OGREnvelope* extent = new OGREnvelope();
	int sumRecord1 = 0;
	int sumRecord2 = 0;

	GDALDataset* poDS1, * poDS2;
	poDS1 = (GDALDataset*)GDALOpenEx(inPath1, GDAL_OF_VECTOR, NULL, NULL, NULL);
	poDS2 = (GDALDataset*)GDALOpenEx(inPath2, GDAL_OF_VECTOR, NULL, NULL, NULL);
	if (poDS1 == NULL || poDS2 == NULL)
	{
		printf("Open failed.\n%s");
		return;
	}
	OGRLayer* poLayer1, * poLayer2;
	poLayer1 = poDS1->GetLayer(0);
	poLayer2 = poDS2->GetLayer(0);
	OGRSpatialReference* poSpatialRef = poLayer1->GetSpatialRef();

	if (!rank)
	{
		OGREnvelope* extent1 = new OGREnvelope();
		OGREnvelope* extent2 = new OGREnvelope();

		sumRecord1 = poLayer1->GetFeatureCount();

		OGRErr error1 = poLayer1->GetExtent(extent1);

		sumRecord2 = poLayer2->GetFeatureCount();
		OGRErr error2 = poLayer2->GetExtent(extent2);

		cout << extent1->MaxX << "," << extent1->MinX << "," << extent1->MaxY << "," << extent1->MinY << endl;
		cout << extent2->MaxX << "," << extent2->MinX << "," << extent2->MaxY << "," << extent2->MinY << endl;

		double extentMinX = (extent1->MinX < extent2->MinX) ? extent1->MinX : extent2->MinX;
		double extentMinY = (extent1->MinY < extent2->MinY) ? extent1->MinY : extent2->MinY;
		double extentMaxX = (extent1->MaxX > extent2->MaxX) ? extent1->MaxX : extent2->MaxX;
		double extentMaxY = (extent1->MaxY > extent2->MaxY) ? extent1->MaxY : extent2->MaxY;
		extent->MinX = extentMinX; extent->MaxX = extentMaxX;
		extent->MinY = extentMinY; extent->MaxY = extentMaxY;
	}
	MPI_Bcast(&sumRecord1, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sumRecord2, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(extent, 1, *MPI_Envelope, 0, MPI_COMM_WORLD);
	
	cout << "Slave processes are attaching polygons to grid..." << endl;
	if (rank)
	{
		int averageRecord1 = sumRecord1 / (size - 1);
		int averageRecord2 = sumRecord2 / (size - 1);
		double extentMinX = extent->MinX; double extentMaxX = extent->MaxX;
		double extentMinY = extent->MinY; double extentMaxY = extent->MaxY;

		double xSize = (extentMaxX - extentMinX) / (double)gridDimX;
		double ySize = (extentMaxY - extentMinY) / (double)gridDimY;

		int startRange1 = (rank - 1) * averageRecord1;
		int endRange1 = (rank - 1 + 1) * averageRecord1;
		int startRange2 = (rank - 1) * averageRecord2;
		int endRange2 = (rank - 1 + 1) * averageRecord2;
		if (rank == size - 1) { endRange1 = sumRecord1;  endRange2 = sumRecord2; }
		cout << rank << " in first dataset:<" << startRange1 << "," << endRange1 << ">" << ", second dataset:<" << startRange2 << "," << endRange2 << ">" << endl;

		GeoGrid** geoGrid = new GeoGrid * [gridDimX * gridDimY];
		for (int i = 0; i < gridDimX * gridDimY; i++)
			geoGrid[i] = new GeoGrid;

		OGRFeature* poFeature;
		for (int FID = startRange1; FID < endRange1; FID++)
		{
			poFeature = poLayer1->GetFeature(FID);
			OGRGeometry* poGeo = poFeature->GetGeometryRef();
			OGRPolygon* polygon = (OGRPolygon*)poGeo;
			if (polygon == NULL)
				continue;
			if (!polygon->IsValid())
				continue;
			OGREnvelope* envelopePolygon = new OGREnvelope();
			polygon->getEnvelope(envelopePolygon);
			for (int i = 0; i < gridDimX; i++)
			{
				double gridMinX = extentMinX + i * xSize;
				double gridMaxX = extentMinX + (i + 1) * xSize;
				for (int j = 0; j < gridDimY; j++)
				{
					double gridMinY = extentMinY + j * ySize;
					double gridMaxY = extentMinY + (j + 1) * ySize;
					OGREnvelope* envelope = new OGREnvelope();
					envelope->MinX = gridMinX; envelope->MaxX = gridMaxX;
					envelope->MinY = gridMinY; envelope->MaxY = gridMaxY;
					if (envelope->Intersects(*envelopePolygon))
						geoGrid[i * gridDimY + j]->polygonsFID1.push_back(FID);

					delete envelope;
					envelope = NULL;
				}
			}
			delete envelopePolygon;
			envelopePolygon = NULL;
		}
		for (int FID = startRange2; FID < endRange2; FID++)
		{
			poFeature = poLayer2->GetFeature(FID);
			OGRGeometry* poGeo = poFeature->GetGeometryRef();
			OGRPolygon* polygon = (OGRPolygon*)poGeo;
			if (polygon == NULL)
				continue;
			if (!polygon->IsValid())
				continue;
			OGREnvelope* envelopePolygon = new OGREnvelope();
			polygon->getEnvelope(envelopePolygon);
			for (int i = 0; i < gridDimX; i++)
			{
				double gridMinX = extentMinX + i * xSize;
				double gridMaxX = extentMinX + (i + 1) * xSize;
				for (int j = 0; j < gridDimY; j++)
				{
					double gridMinY = extentMinY + j * ySize;
					double gridMaxY = extentMinY + (j + 1) * ySize;
					OGREnvelope* envelope = new OGREnvelope();
					envelope->MinX = gridMinX; envelope->MaxX = gridMaxX;
					envelope->MinY = gridMinY; envelope->MaxY = gridMaxY;
					if (envelope->Intersects(*envelopePolygon))
						geoGrid[i * gridDimY + j]->polygonsFID2.push_back(FID);

					delete envelope;
					envelope = NULL;
				}
			}
			delete envelopePolygon;
			envelopePolygon = NULL;
		}

		int* rankGridPolygonsNum1 = new int[gridDimX * gridDimY];
		vector<int> rankGridPolygonsFID1;
		int* rankGridPolygonsNum2 = new int[gridDimX * gridDimY];
		vector<int> rankGridPolygonsFID2;
		for (int i = 0; i < gridDimX; i++)
		{
			for (int j = 0; j < gridDimY; j++)
			{
				rankGridPolygonsNum1[i * gridDimY + j] = geoGrid[i * gridDimY + j]->polygonsFID1.size();
				rankGridPolygonsNum2[i * gridDimY + j] = geoGrid[i * gridDimY + j]->polygonsFID2.size();
				rankGridPolygonsFID1.insert(rankGridPolygonsFID1.end(), geoGrid[i * gridDimY + j]->polygonsFID1.begin(), geoGrid[i * gridDimY + j]->polygonsFID1.end());
				rankGridPolygonsFID2.insert(rankGridPolygonsFID2.end(), geoGrid[i * gridDimY + j]->polygonsFID2.begin(), geoGrid[i * gridDimY + j]->polygonsFID2.end());
			}
		}

		MPI_Send(rankGridPolygonsNum1, gridDimX * gridDimY, MPI_INT, 0, rank + 101, MPI_COMM_WORLD);
		MPI_Send(rankGridPolygonsNum2, gridDimX * gridDimY, MPI_INT, 0, rank + 102, MPI_COMM_WORLD);
		MPI_Send(&rankGridPolygonsFID1[0], rankGridPolygonsFID1.size(), MPI_INT, 0, rank + 103, MPI_COMM_WORLD);
		MPI_Send(&rankGridPolygonsFID2[0], rankGridPolygonsFID2.size(), MPI_INT, 0, rank + 104, MPI_COMM_WORLD);

		delete[] rankGridPolygonsNum1;
		delete[] rankGridPolygonsNum2;
		vector<int>().swap(rankGridPolygonsFID1);
		vector<int>().swap(rankGridPolygonsFID2);
		for (int i = 0; i < gridDimX * gridDimY; i++)
		{
			vector<int>().swap(geoGrid[i]->polygonsFID1);
			vector<int>().swap(geoGrid[i]->polygonsFID2);
		}
	}
	if (!rank)
	{
		int** rankGridPolygonsNumReceive1 = new int* [size - 1];
		int** rankGridPolygonsNumReceive2 = new int* [size - 1];
		int** rankGridPolygonsFIDReceive1 = new int* [size - 1];
		int** rankGridPolygonsFIDReceive2 = new int* [size - 1];
		for (int process = 0; process < size - 1; process++)
		{
			rankGridPolygonsNumReceive1[process] = new int[gridDimX * gridDimY];
			rankGridPolygonsNumReceive2[process] = new int[gridDimX * gridDimY];
			MPI_Recv(rankGridPolygonsNumReceive1[process], gridDimX * gridDimY, MPI_INT, process + 1, process + 1 + 101, MPI_COMM_WORLD, status);
			MPI_Recv(rankGridPolygonsNumReceive2[process], gridDimX * gridDimY, MPI_INT, process + 1, process + 1 + 102, MPI_COMM_WORLD, status);
			int rankPolygonSum1 = 0, rankPolygonSum2 = 0;
			for (int i = 0; i < gridDimX * gridDimY; i++)
			{
				rankPolygonSum1 += rankGridPolygonsNumReceive1[process][i];
				rankPolygonSum2 += rankGridPolygonsNumReceive2[process][i];
			}
			rankGridPolygonsFIDReceive1[process] = new int[rankPolygonSum1];
			rankGridPolygonsFIDReceive2[process] = new int[rankPolygonSum2];
			MPI_Recv(rankGridPolygonsFIDReceive1[process], rankPolygonSum1, MPI_INT, process + 1, process + 1 + 103, MPI_COMM_WORLD, status);
			MPI_Recv(rankGridPolygonsFIDReceive2[process], rankPolygonSum2, MPI_INT, process + 1, process + 1 + 104, MPI_COMM_WORLD, status);

			cout << "Master process has received attaching polygons from slave process " << process + 1 << endl;
		}
		GeoGrid** geoGridJoin = new GeoGrid * [gridDimX * gridDimY];
		for (int i = 0; i < gridDimX * gridDimY; i++)
			geoGridJoin[i] = new GeoGrid;
		for (int process = 0; process < size - 1; process++)
		{
			int j1 = 0, j2 = 0;
			int offset1 = 0, offset2 = 0;
			for (int i = 0; i < gridDimX * gridDimY; i++)
			{
				for (; j1 < offset1 + rankGridPolygonsNumReceive1[process][i]; j1++)
				{
					geoGridJoin[i]->polygonsFID1.push_back(rankGridPolygonsFIDReceive1[process][j1]);
				}
				offset1 = j1;

				for (; j2 < offset2 + rankGridPolygonsNumReceive2[process][i]; j2++)
				{
					geoGridJoin[i]->polygonsFID2.push_back(rankGridPolygonsFIDReceive2[process][j2]);
				}
				offset2 = j2;
			}
		}
		vector<GeoGrid*> geoGridJoinAndRepartition(geoGridJoin, geoGridJoin + gridDimX * gridDimY);
		double* computeIntensityCalFeatureCI = new double[geoGridJoinAndRepartition.size()];
		for (int i = 0; i < geoGridJoinAndRepartition.size(); i++) computeIntensityCalFeatureCI[i] = geoGridJoinAndRepartition[i]->polygonsFID1.size() * geoGridJoinAndRepartition[i]->polygonsFID2.size();;
		getPartitionBoundary(computeIntensityCalFeatureCI, (size - 1), geoGridJoinAndRepartition.size(), partitionBounds, partitionComputeIntensity);
		for (int i = 0; i < size - 1; i++)
		{
			int startGridRange = partitionBounds[i]; int endGridRange = partitionBounds[i + 1];
			int gridRange = partitionBounds[i + 1] - partitionBounds[i];
			int* rankGridPolygonsNumCI1 = new int[gridRange];
			int* rankGridPolygonsNumCI2 = new int[gridRange];
			vector<int> rankGridPolygonsFIDCI1;
			vector<int> rankGridPolygonsFIDCI2;

			for (int j = startGridRange; j < endGridRange; j++)
			{
				rankGridPolygonsNumCI1[j - startGridRange] = geoGridJoinAndRepartition[j]->polygonsFID1.size();
				rankGridPolygonsNumCI2[j - startGridRange] = geoGridJoinAndRepartition[j]->polygonsFID2.size();
				rankGridPolygonsFIDCI1.insert(rankGridPolygonsFIDCI1.end(), geoGridJoinAndRepartition[j]->polygonsFID1.begin(), geoGridJoinAndRepartition[j]->polygonsFID1.end());
				rankGridPolygonsFIDCI2.insert(rankGridPolygonsFIDCI2.end(), geoGridJoinAndRepartition[j]->polygonsFID2.begin(), geoGridJoinAndRepartition[j]->polygonsFID2.end());
			}
			MPI_Send(partitionBounds, size, MPI_INT, i + 1, i + 1 + 999, MPI_COMM_WORLD);
			MPI_Send(rankGridPolygonsNumCI1, gridRange, MPI_INT, i + 1, i + 1 + 105, MPI_COMM_WORLD);
			MPI_Send(rankGridPolygonsNumCI2, gridRange, MPI_INT, i + 1, i + 1 + 106, MPI_COMM_WORLD);
			MPI_Send(&rankGridPolygonsFIDCI1[0], rankGridPolygonsFIDCI1.size(), MPI_INT, i + 1, i + 1 + 107, MPI_COMM_WORLD);
			MPI_Send(&rankGridPolygonsFIDCI2[0], rankGridPolygonsFIDCI2.size(), MPI_INT, i + 1, i + 1 + 108, MPI_COMM_WORLD);
		}

		cout << "Slave proceses are calculating features and computational intensity of each grid..." << endl;

		vector<double> gridComputeIntensity;
		for (int i = 0; i < size - 1; i++)
		{
			int startGridRange = partitionBounds[i]; int endGridRange = partitionBounds[i + 1];
			int gridRange = endGridRange - startGridRange;
			double* rankGridCI = new double[gridRange];
			MPI_Recv(rankGridCI, gridRange, MPI_DOUBLE, i + 1, i + 1 + 888, MPI_COMM_WORLD, status);
			for (int j = 0; j < gridRange; gridComputeIntensity.push_back(rankGridCI[j]), j++);
			cout << "Master process has received grid computational intensity from slave process " << i + 1 << " successfully!" << endl;
		}

		double* gridComputeIntensityArr = new double[gridComputeIntensity.size()];
		for (int i = 0; i < gridComputeIntensity.size(); i++) gridComputeIntensityArr[i] = gridComputeIntensity[i];
		getPartitionBoundary(gridComputeIntensityArr, (size - 1), gridComputeIntensity.size(), partitionBounds, partitionComputeIntensity);
		cout << "Slave processes are performing intersection analysis..." << endl;
		for (int i = 0; i < size - 1; i++)
		{
			int startGridRange = partitionBounds[i]; int endGridRange = partitionBounds[i + 1];
			int gridRange = partitionBounds[i + 1] - partitionBounds[i];
			int* rankGridPolygonsNum1 = new int[gridRange];
			int* rankGridPolygonsNum2 = new int[gridRange];
			vector<int> rankGridPolygonsFID1;
			vector<int> rankGridPolygonsFID2;

			for (int j = startGridRange; j < endGridRange; j++)
			{
				rankGridPolygonsNum1[j - startGridRange] = geoGridJoinAndRepartition[j]->polygonsFID1.size();
				rankGridPolygonsNum2[j - startGridRange] = geoGridJoinAndRepartition[j]->polygonsFID2.size();
				rankGridPolygonsFID1.insert(rankGridPolygonsFID1.end(), geoGridJoinAndRepartition[j]->polygonsFID1.begin(), geoGridJoinAndRepartition[j]->polygonsFID1.end());
				rankGridPolygonsFID2.insert(rankGridPolygonsFID2.end(), geoGridJoinAndRepartition[j]->polygonsFID2.begin(), geoGridJoinAndRepartition[j]->polygonsFID2.end());
			}
			MPI_Send(partitionBounds, size, MPI_INT, i + 1, i + 1 + 1000, MPI_COMM_WORLD);
			MPI_Send(rankGridPolygonsNum1, gridRange, MPI_INT, i + 1, i + 1 + 111, MPI_COMM_WORLD);
			MPI_Send(rankGridPolygonsNum2, gridRange, MPI_INT, i + 1, i + 1 + 222, MPI_COMM_WORLD);
			MPI_Send(&rankGridPolygonsFID1[0], rankGridPolygonsFID1.size(), MPI_INT, i + 1, i + 1 + 333, MPI_COMM_WORLD);
			MPI_Send(&rankGridPolygonsFID2[0], rankGridPolygonsFID2.size(), MPI_INT, i + 1, i + 1 + 444, MPI_COMM_WORLD);
		}
	}
	if (rank)
	{
		MPI_Recv(partitionBounds, size, MPI_INT, 0, rank + 999, MPI_COMM_WORLD, status);
		int startGridRange = partitionBounds[rank - 1]; int endGridRange = partitionBounds[rank - 1 + 1];
		int gridRange = endGridRange - startGridRange;
		int* rankGridPolygonsNumReceiveCI1 = new int[gridRange];
		int* rankGridPolygonsNumReceiveCI2 = new int[gridRange];
		MPI_Recv(rankGridPolygonsNumReceiveCI1, gridRange, MPI_INT, 0, rank + 105, MPI_COMM_WORLD, status);
		MPI_Recv(rankGridPolygonsNumReceiveCI2, gridRange, MPI_INT, 0, rank + 106, MPI_COMM_WORLD, status);

		int rankPolygonSumCI1 = 0, rankPolygonSumCI2 = 0;
		for (int i = 0; i < gridRange; i++)
		{
			rankPolygonSumCI1 += rankGridPolygonsNumReceiveCI1[i];
			rankPolygonSumCI2 += rankGridPolygonsNumReceiveCI2[i];
		}
		int* rankGridPolygonsFIDReceiveCI1 = new int[rankPolygonSumCI1];
		int* rankGridPolygonsFIDReceiveCI2 = new int[rankPolygonSumCI2];
		MPI_Recv(rankGridPolygonsFIDReceiveCI1, rankPolygonSumCI1, MPI_INT, 0, rank + 107, MPI_COMM_WORLD, status);
		MPI_Recv(rankGridPolygonsFIDReceiveCI2, rankPolygonSumCI2, MPI_INT, 0, rank + 108, MPI_COMM_WORLD, status);

		cout << rank << ": the data from master thread " << "was received for calculate computational intensity" << endl;

		time_t processCICalculationStart, processCICalculationEnd;
		time(&processCICalculationStart);
		double* computeIntensity = new double[gridRange];
		GeoGrid** geoGridCI = new GeoGrid * [gridRange];
		for (int i = 0; i < gridRange; geoGridCI[i] = new GeoGrid, i++);

		std::stringstream ssRank;
		string strRank;
		ssRank << rank;
		ssRank >> strRank;
		string inputRankDirStr = (string)gridFeatureDir + strRank;
		ifstream f(inputRankDirStr.c_str());
		if (!f.good())
		{
			string command = "mkdir " + inputRankDirStr;
			system(command.c_str());
		}
		ofstream foutFeatures((inputRankDirStr + "/features.txt").c_str(), std::ios::out);

		int j1 = 0, j2 = 0;
		int offset1 = 0, offset2 = 0;
		for (int i = 0; i < gridRange; i++)
		{
			for (; j1 < offset1 + rankGridPolygonsNumReceiveCI1[i]; j1++)
			{
				geoGridCI[i]->polygonsFID1.push_back(rankGridPolygonsFIDReceiveCI1[j1]);
			}
			offset1 = j1;

			for (; j2 < offset2 + rankGridPolygonsNumReceiveCI2[i]; j2++)
			{
				geoGridCI[i]->polygonsFID2.push_back(rankGridPolygonsFIDReceiveCI2[j2]);
			}
			offset2 = j2;
		}

		double extentMinX = extent->MinX; double extentMaxX = extent->MaxX;
		double extentMinY = extent->MinY; double extentMaxY = extent->MaxY;
		double xSize = (extentMaxX - extentMinX) / (double)gridDimX;
		double ySize = (extentMaxY - extentMinY) / (double)gridDimY;

		for (int i = 0; i < gridRange; i++)
		{
			OGRFeature* poFeature1, * poFeature2;

			int polygonsNum1 = geoGridCI[i]->polygonsFID1.size();
			int polygonsNum2 = geoGridCI[i]->polygonsFID2.size();
			int verticesNum1 = 0, verticesNum2 = 0;
			double varianceLayer1 = 0.0, varianceLayer2 = 0.0;
			double avgDist;

			double xSumLayer1 = 0, ySumLayer1 = 0;
			double xSumLayer2 = 0, ySumLayer2 = 0;

			int x_coordinate = (startGridRange + i) / gridDimX;
			int y_coordinate = (startGridRange + i) % gridDimX;
			double gridMinX = extentMinX + x_coordinate * xSize;
			double gridMaxX = extentMinX + (x_coordinate + 1) * xSize;
			double gridMinY = extentMinY + y_coordinate * ySize;
			double gridMaxY = extentMinY + (y_coordinate + 1) * ySize;
			int referencePoint = 0;

			bool* check = new bool[geoGridCI[i]->polygonsFID1.size() * geoGridCI[i]->polygonsFID2.size()];
			for (int k = 0; k < geoGridCI[i]->polygonsFID1.size() * geoGridCI[i]->polygonsFID2.size(); k++) check[k] = 1;

			for (int j1 = 0; j1 < geoGridCI[i]->polygonsFID1.size(); j1++)
			{
				int FID1 = geoGridCI[i]->polygonsFID1[j1];
				OGRFeature* poFeature1 = poLayer1->GetFeature(FID1);
				OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
				OGREnvelope* envelopePolygon1 = new OGREnvelope();
				polygon1->getEnvelope(envelopePolygon1);

				delete envelopePolygon1;
				envelopePolygon1 = NULL;
				OGRFeature::DestroyFeature(poFeature1);
			}
			for (int j1 = 0; j1 < geoGridCI[i]->polygonsFID1.size(); j1++)
			{
				int FID1 = geoGridCI[i]->polygonsFID1[j1];
				poFeature1 = poLayer1->GetFeature(FID1);

				OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
				verticesNum1 += polygon1->getExteriorRing()->getNumPoints();
				for (int t = 0; t < polygon1->getNumInteriorRings(); t++)
					verticesNum1 += polygon1->getInteriorRing(t)->getNumPoints();
				OGRPoint* poPoint = new OGRPoint; polygon1->Centroid(poPoint);
				xSumLayer1 += poPoint->getX(); ySumLayer1 += poPoint->getY();
				OGRFeature::DestroyFeature(poFeature1);
			}
			double xAverage1 = xSumLayer1 / polygonsNum1;
			double yAverage1 = ySumLayer1 / polygonsNum1;
			for (int j1 = 0; j1 < geoGridCI[i]->polygonsFID1.size(); j1++)
			{
				int FID1 = geoGridCI[i]->polygonsFID1[j1];
				poFeature1 = poLayer1->GetFeature(FID1);
				OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
				OGRPoint* poPoint = new OGRPoint; polygon1->Centroid(poPoint);
				varianceLayer1 = varianceLayer1 + (poPoint->getX() - xAverage1) * (poPoint->getX() - xAverage1) +
					(poPoint->getY() - yAverage1) * (poPoint->getY() - yAverage1);
				OGRFeature::DestroyFeature(poFeature1);
			}
			varianceLayer1 = varianceLayer1 / polygonsNum1;

			for (int j2 = 0; j2 < geoGridCI[i]->polygonsFID2.size(); j2++)
			{
				int FID2 = geoGridCI[i]->polygonsFID2[j2];
				poFeature2 = poLayer2->GetFeature(FID2);

				OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
				verticesNum2 += polygon2->getExteriorRing()->getNumPoints();
				for (int t = 0; t < polygon2->getNumInteriorRings(); t++)
					verticesNum2 += polygon2->getInteriorRing(t)->getNumPoints();
				OGRPoint* poPoint = new OGRPoint; polygon2->Centroid(poPoint);
				xSumLayer2 += poPoint->getX(); ySumLayer2 += poPoint->getY();
				OGRFeature::DestroyFeature(poFeature2);
			}
			double xAverage2 = xSumLayer2 / polygonsNum2;
			double yAverage2 = ySumLayer2 / polygonsNum2;
			for (int j2 = 0; j2 < geoGridCI[i]->polygonsFID2.size(); j2++)
			{
				int FID2 = geoGridCI[i]->polygonsFID2[j2];
				poFeature2 = poLayer2->GetFeature(FID2);
				OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
				OGRPoint* poPoint = new OGRPoint; polygon2->Centroid(poPoint);
				varianceLayer2 = varianceLayer2 + (poPoint->getX() - xAverage2) * (poPoint->getX() - xAverage2) +
					(poPoint->getY() - yAverage2) * (poPoint->getY() - yAverage2);
				OGRFeature::DestroyFeature(poFeature2);
			}
			varianceLayer2 = varianceLayer2 / polygonsNum2;
			avgDist = (xAverage1 - xAverage2) * (xAverage1 - xAverage2) + (yAverage1 - yAverage2) * (yAverage1 - yAverage2);

			if (polygonsNum1 == 0)varianceLayer1 = 0; if (polygonsNum2 == 0)varianceLayer2 = 0;
			if (polygonsNum1 == 0 || polygonsNum2 == 0)avgDist = 100000;

			for (int j1 = 0; j1 < geoGridCI[i]->polygonsFID1.size(); j1++)
			{
				int FID1 = geoGridCI[i]->polygonsFID1[j1];
				poFeature1 = poLayer1->GetFeature(FID1);
				OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
				OGREnvelope* envelopePolygon1 = new OGREnvelope();
				polygon1->getEnvelope(envelopePolygon1);

				for (int j2 = 0; j2 < geoGridCI[i]->polygonsFID2.size(); j2++)
				{
					int FID2 = geoGridCI[i]->polygonsFID2[j2];
					poFeature2 = poLayer2->GetFeature(FID2);
					OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
					OGREnvelope* envelopePolygon2 = new OGREnvelope();
					polygon2->getEnvelope(envelopePolygon2);

					if (envelopePolygon2->Intersects(*envelopePolygon1))
					{
						double x_rp = (envelopePolygon1->MinX > envelopePolygon2->MinX) ? envelopePolygon1->MinX : envelopePolygon2->MinX;
						double y_rp = (envelopePolygon1->MinY > envelopePolygon2->MinY) ? envelopePolygon1->MinY : envelopePolygon2->MinY;
						if (x_rp < gridMaxX && x_rp > gridMinX&& y_rp < gridMaxY && y_rp > gridMinY) referencePoint++;
					}

					delete envelopePolygon2;
					envelopePolygon2 = NULL;
					OGRFeature::DestroyFeature(poFeature2);
				}

				delete envelopePolygon1;
				envelopePolygon1 = NULL;
				OGRFeature::DestroyFeature(poFeature1);
			}

			foutFeatures << polygonsNum1 << "\t" << polygonsNum2 << "\t" << verticesNum1 << "\t" << verticesNum2 << "\t" << varianceLayer1 << "\t" << varianceLayer2 << "\t" << referencePoint << endl;
		}
		foutFeatures.close();

		const char* script = "/home/fgao/LoadBalance/Intersection/python/ml_env/ml/inference/model_call.py";
		const char* function = "randomforest";
		const char* modelPath = "/home/fgao/LoadBalance/Intersection/python/ml_env/ml/result_model/ml_rfr1.pk";
		const char* inputPath = (inputRankDirStr + "/features.txt").c_str();
		const char* outputCIPath = (inputRankDirStr + "/rfr_predict.txt").c_str();
		callMlModel(script, function, modelPath, (inputRankDirStr + "/features.txt").c_str(), (inputRankDirStr + "/rfr_predict.txt").c_str());
		
		ifstream computeIntesityFile(outputCIPath);
		string temp;
		int i = 0;
		while (getline(computeIntesityFile, temp))
		{
			stringstream ss;
			ss << temp;
			ss >> computeIntensity[i];
			i++;
		}

		time(&processCICalculationEnd);
		double processCICalculationCost = difftime(processCICalculationEnd, processCICalculationStart);
		cout << "process " << rank << ": CICalulation time cost is " << processCICalculationCost << endl;

		MPI_Send(computeIntensity, gridRange, MPI_DOUBLE, 0, rank + 888, MPI_COMM_WORLD);
		MPI_Recv(partitionBounds, size, MPI_INT, 0, rank + 1000, MPI_COMM_WORLD, status);
		startGridRange = partitionBounds[rank - 1]; endGridRange = partitionBounds[rank - 1 + 1];
		gridRange = endGridRange - startGridRange;
		int* rankGridPolygonsNumReceive1 = new int[gridRange];
		int* rankGridPolygonsNumReceive2 = new int[gridRange];
		MPI_Recv(rankGridPolygonsNumReceive1, gridRange, MPI_INT, 0, rank + 111, MPI_COMM_WORLD, status);
		MPI_Recv(rankGridPolygonsNumReceive2, gridRange, MPI_INT, 0, rank + 222, MPI_COMM_WORLD, status);

		int rankPolygonSum1 = 0, rankPolygonSum2 = 0;

		for (int i = 0; i < gridRange; i++)
		{
			rankPolygonSum1 += rankGridPolygonsNumReceive1[i];
			rankPolygonSum2 += rankGridPolygonsNumReceive2[i];
		}

		int* rankGridPolygonsFIDReceive1 = new int[rankPolygonSum1];
		int* rankGridPolygonsFIDReceive2 = new int[rankPolygonSum2];
		MPI_Recv(rankGridPolygonsFIDReceive1, rankPolygonSum1, MPI_INT, 0, rank + 333, MPI_COMM_WORLD, status);
		MPI_Recv(rankGridPolygonsFIDReceive2, rankPolygonSum2, MPI_INT, 0, rank + 444, MPI_COMM_WORLD, status);

		cout << "process " << rank << " has received data from master process for intersection" << startGridRange << "-" << endGridRange << endl;

		GeoGrid** geoGridProc = new GeoGrid * [gridRange];
		for (int i = 0; i < gridRange; geoGridProc[i] = new GeoGrid, i++);
		j1 = 0; j2 = 0;
		offset1 = 0; offset2 = 0;
		for (int i = 0; i < gridRange; i++)
		{
			for (; j1 < offset1 + rankGridPolygonsNumReceive1[i]; j1++)
			{
				geoGridProc[i]->polygonsFID1.push_back(rankGridPolygonsFIDReceive1[j1]);
			}
			offset1 = j1;

			for (; j2 < offset2 + rankGridPolygonsNumReceive2[i]; j2++)
			{
				geoGridProc[i]->polygonsFID2.push_back(rankGridPolygonsFIDReceive2[j2]);
			}
			offset2 = j2;
		}

		string outShpPath = (string)outPath + "result_" + strRank + ".shp";

		GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
		if (poDriver == NULL)
			return;
		GDALDataset* poDstDS = poDriver->Create(outShpPath.c_str(), 0, 0, 0, GDT_Unknown, NULL);
		OGRLayer* poLayer = poDstDS->CreateLayer("Result", poSpatialRef, wkbMultiPolygon, NULL);
		OGRFeatureDefn* poDefn = poLayer->GetLayerDefn();
		OGRFeature* poFeature = OGRFeature::CreateFeature(poDefn);

		time_t processIntersectionBegin, processIntersectionEnd;
		time(&processIntersectionBegin);
		for (int i = 0; i < gridRange; i++)
		{
			int x_coordinate = (startGridRange + i) / gridDimX;
			int y_coordinate = (startGridRange + i) % gridDimX;
			double gridMinX = extentMinX + x_coordinate * xSize;
			double gridMaxX = extentMinX + (x_coordinate + 1) * xSize;
			double gridMinY = extentMinY + y_coordinate * ySize;
			double gridMaxY = extentMinY + (y_coordinate + 1) * ySize;
			for (int j1 = 0; j1 < geoGridProc[i]->polygonsFID1.size(); j1++)
			{
				int FID1 = geoGridProc[i]->polygonsFID1[j1];
				OGRFeature* poFeature1 = poLayer1->GetFeature(FID1);
				OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
				OGREnvelope* envelopePolygon1 = new OGREnvelope();
				polygon1->getEnvelope(envelopePolygon1);

				for (int j2 = 0; j2 < geoGridProc[i]->polygonsFID2.size(); j2++)
				{
					int FID2 = geoGridProc[i]->polygonsFID2[j2];
					OGRFeature* poFeature2 = poLayer2->GetFeature(FID2);
					OGRPolygon* polygon2 = (OGRPolygon*)poFeature2->GetGeometryRef();
					OGREnvelope* envelopePolygon2 = new OGREnvelope();
					polygon2->getEnvelope(envelopePolygon2);

					if (envelopePolygon2->Intersects(*envelopePolygon1))
					{
						double x_rp = (envelopePolygon1->MinX > envelopePolygon2->MinX) ? envelopePolygon1->MinX : envelopePolygon2->MinX;
						double y_rp = (envelopePolygon1->MinY > envelopePolygon2->MinY) ? envelopePolygon1->MinY : envelopePolygon2->MinY;
						if (x_rp < gridMaxX && x_rp > gridMinX&& y_rp < gridMaxY && y_rp > gridMinY)
						{
							OGRGeometry* pIst = polygon2->Intersection(polygon1);
							OGRwkbGeometryType type = pIst->getGeometryType();
							if (pIst != NULL && (type == 3 || type == 6))
							{
								poFeature->SetGeometry(pIst);
								OGRErr error = poLayer->CreateFeature(poFeature);

							}
						}
					}
					delete envelopePolygon2;
					envelopePolygon2 = NULL;
					OGRFeature::DestroyFeature(poFeature2);
				}
				delete envelopePolygon1;
				envelopePolygon1 = NULL;
				OGRFeature::DestroyFeature(poFeature1);
			}
		}
		time(&processIntersectionEnd);
		cout << "process " << rank << ": intersection time cost is " << difftime(processIntersectionEnd, processIntersectionBegin) << endl;
		OGRFeature::DestroyFeature(poFeature);
		GDALClose(poDstDS);
	}

	time(&rankEnd);
	double rankCost = difftime(rankEnd, rankStart);
	double* processCost = new double[size];
	MPI_Gather(&rankCost, 1, MPI_DOUBLE, processCost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (!rank)
	{
		time(&end);
		double cost = difftime(end, start);
		std::cout << "sum time cost: " << cost << "s" << std::endl;
		double sum = 0.0, mean, mean2 = 0.0, var, stdDev;
		for (size_t i = 1; i < size; i++) {
			sum += processCost[i];
			std::cout << "process " << i << " time cost: " << processCost[i] << "s" << std::endl;
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
