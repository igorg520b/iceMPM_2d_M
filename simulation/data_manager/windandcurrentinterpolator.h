 #ifndef WINDINTERPOLATOR_H
#define WINDINTERPOLATOR_H

#include <string_view>
#include <string>
#include <utility>
#include <vector>
#include <H5Cpp.h>
#include <Eigen/Core>

#include "parameters_sim.h"


class WindAndCurrentInterpolator
{
public:
    explicit WindAndCurrentInterpolator(SimParams& params);

    SimParams &prms;
    std::vector<t_GridReal> current_flow_data;  // currently it's a static flow

    void OpenCustomHDF5(std::string fileName);

    bool SetTime(double t);     // set current time; return whether data needs to be copied to GPU
private:
    H5::H5File file;
    void ReadStaticFlowData();
};


#endif // WINDINTERPOLATOR_H
