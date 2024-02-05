#include "stdio.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "gdal_priv.h"
#include "ogr_spatialref.h"
#include "ogrsf_frmts.h"
#include "Util.h"

using namespace std;

void serialIntersection(const char* inPath1, const char* inPath2, const char* outPath, int gridDimX, int gridDimY);

int main(int argc, char* argv[])
{
        int gridDimX, gridDimY;
        sscanf(argv[4], "%d", &gridDimX);
        sscanf(argv[5], "%d", &gridDimY);
        serialIntersection(argv[1], argv[2], argv[3], gridDimX, gridDimY);

        return 0;
}

void serialIntersection(const char* inPath1, const char* inPath2, const char* outPath, int gridDimX, int gridDimY)
{
	cout << "Serial task is running..." << endl;
	double begin = clock();

	GDALAllRegister();
	OGRRegisterAll();

	int sumRecord1 = 0;
	int sumRecord2 = 0;
	OGREnvelope* extent = new OGREnvelope();

	GDALDataset* poDS1, * poDS2;
	CPLSetConfigOption("SHAPE_ENCODING", "");
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

	double xSize = (extentMaxX - extentMinX) / (double)gridDimX;
	double ySize = (extentMaxY - extentMinY) / (double)gridDimY;

	GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
	if (poDriver == NULL) return;
	GDALDataset* poDstDS = poDriver->Create(outPath, 0, 0, 0, GDT_Unknown, NULL);
	OGRLayer* poLayer = poDstDS->CreateLayer("Result", poSpatialRef, wkbMultiPolygon, NULL);
	OGRFeatureDefn* poDefn = poLayer->GetLayerDefn();
	OGRFeature* poDstFeature = OGRFeature::CreateFeature(poDefn);

	GeoGrid** geoGrid = new GeoGrid * [gridDimX * gridDimY];
	for (int i = 0; i < gridDimX * gridDimY; i++)
		geoGrid[i] = new GeoGrid;

	OGRFeature* poFeature;
	for (int FID = 0; FID < sumRecord1; FID++)
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
		OGRFeature::DestroyFeature(poFeature);
	}
	for (int FID = 0; FID < sumRecord2; FID++)
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
		OGRFeature::DestroyFeature(poFeature);
	}

	OGRFeature* poFeature1, * poFeature2;
	for (int i = 0; i < gridDimX * gridDimY; i++)
	{
		int x_coordinate = i / gridDimX;
		int y_coordinate = i % gridDimX;
		double gridMinX = extentMinX + x_coordinate * xSize;
		double gridMaxX = extentMinX + (x_coordinate + 1) * xSize;
		double gridMinY = extentMinY + y_coordinate * ySize;
		double gridMaxY = extentMinY + (y_coordinate + 1) * ySize;
		for (int j1 = 0; j1 < geoGrid[i]->polygonsFID1.size(); j1++)
		{
			int FID1 = geoGrid[i]->polygonsFID1[j1];
			poFeature1 = poLayer1->GetFeature(FID1);
			OGRPolygon* polygon1 = (OGRPolygon*)poFeature1->GetGeometryRef();
			OGREnvelope* envelopePolygon1 = new OGREnvelope();
			polygon1->getEnvelope(envelopePolygon1);

			for (int j2 = 0; j2 < geoGrid[i]->polygonsFID2.size(); j2++)
			{
				int FID2 = geoGrid[i]->polygonsFID2[j2];
				poFeature2 = poLayer2->GetFeature(FID2);
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
							poDstFeature->SetGeometry(pIst);
							OGRErr error = poLayer->CreateFeature(poDstFeature);

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

	OGRFeature::DestroyFeature(poDstFeature);
	GDALClose(poDstDS);

	double end = clock();
	double cost = (double)(end - begin) / CLOCKS_PER_SEC;
	std::cout << "sum time cost: " << cost << "s" << std::endl;
}
