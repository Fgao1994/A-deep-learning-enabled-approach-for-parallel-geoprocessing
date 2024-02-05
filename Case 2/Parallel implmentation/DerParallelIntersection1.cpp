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

void parallelIntersectionDer(const char* inPath1, const char* inPath2, char* outPath, int gridDimX, int gridDimY);

int main(int argc, char* argv[])
{
        int gridDimX, gridDimY;
        sscanf(argv[4], "%d", &gridDimX);
        sscanf(argv[5], "%d", &gridDimY);
        parallelIntersectionDer(argv[1], argv[2], argv[3], gridDimX, gridDimY);

        return 0;
}

void parallelIntersectionDer(const char* inPath1, const char* inPath2, char* outPath, int gridDimX, int gridDimY)
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

	double start = clock();
	double rankCost, rankStart, rankEnd;
	rankStart = clock();

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

		double* gridComputeIntensityArr = new double[geoGridJoinAndRepartition.size()];
		for (int i = 0; i < geoGridJoinAndRepartition.size(); i++) gridComputeIntensityArr[i] = 1;
		getPartitionBoundary(gridComputeIntensityArr, (size - 1), geoGridJoinAndRepartition.size(), partitionBounds, partitionComputeIntensity);

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
		MPI_Recv(partitionBounds, size, MPI_INT, 0, rank + 1000, MPI_COMM_WORLD, status);
		int startGridRange = partitionBounds[rank - 1]; int endGridRange = partitionBounds[rank - 1 + 1];
		int gridRange = endGridRange - startGridRange;
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

		cout << "process " << rank << " has received data from master process for intersection" << endl;

		GeoGrid** geoGridProc = new GeoGrid * [gridRange];
		for (int i = 0; i < gridRange; geoGridProc[i] = new GeoGrid, i++);
		int j1 = 0, j2 = 0;
		int offset1 = 0, offset2 = 0;
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

		std::stringstream ssRank;
		string strRank;
		ssRank << rank;
		ssRank >> strRank;
		string outShpPath = (string)outPath + "result_" + strRank + ".shp";

		GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
		if (poDriver == NULL)
			return;
		GDALDataset* poDstDS = poDriver->Create(outShpPath.c_str(), 0, 0, 0, GDT_Unknown, NULL);
		OGRLayer* poLayer = poDstDS->CreateLayer("Result", poSpatialRef, wkbMultiPolygon, NULL);
		OGRFeatureDefn* poDefn = poLayer->GetLayerDefn();
		OGRFeature* poFeature = OGRFeature::CreateFeature(poDefn);
		double extentMinX = extent->MinX; double extentMaxX = extent->MaxX;
		double extentMinY = extent->MinY; double extentMaxY = extent->MaxY;
		double xSize = (extentMaxX - extentMinX) / (double)gridDimX;
		double ySize = (extentMaxY - extentMinY) / (double)gridDimY;

		double processIntersectionBegin = clock();
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
		double processIntersectionEnd = clock();
		cout << "process " << rank << ": intersection time cost is " << processIntersectionEnd - processIntersectionBegin << endl;
		OGRFeature::DestroyFeature(poFeature);
		GDALClose(poDstDS);
	}
	rankEnd = clock();

	rankCost = (double)(rankEnd - rankStart) / CLOCKS_PER_SEC;

	double* processCost = new double[size];
	MPI_Gather(&rankCost, 1, MPI_DOUBLE, processCost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (!rank)
	{
		double end = clock();
		double cost = (double)(end - start) / CLOCKS_PER_SEC;
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
