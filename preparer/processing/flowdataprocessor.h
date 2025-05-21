#ifndef FLOWDATAPROCESSOR_H
#define FLOWDATAPROCESSOR_H

#include <string>
#include <Eigen/Core>
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkLookupTable.h>
#include <vtkCellDataToPointData.h>

#include "vtkFLUENTCFFCustomReader.h"


class FlowDataProcessor
{
public:
    FlowDataProcessor();
    std::vector<Eigen::Vector2f> velocity_field;
    vtkNew<vtkActor> actor;

    void LoadFluentResult(const std::string fileNameDAT, const std::string fileNameCAS, int width, int height);
    void ApplyTransform(double scale, double offsetX, double offsetY);


    void Rasterize();
    void ApplyDiffusion(std::vector<int> &path_indices, int iterations, float alpha);
    void SaveAsHDF5(std::string ProjectDirectory);

private:
    int width, height;
    std::string inputFileName;
    vtkNew<vtkFLUENTCFFCustomReader> reader;

    vtkNew<vtkTransform> transform;
    vtkNew<vtkTransformFilter> transformFilter;

    vtkNew<vtkDataSetMapper> mapper;
    vtkNew<vtkLookupTable> lut;
    vtkNew<vtkLookupTable> lut_custom;

};

#endif // FLOWDATAPROCESSOR_H
