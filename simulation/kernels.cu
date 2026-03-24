#include "parameters_sim.h"
#include "gpu_partition.h"
#include "kernel_declarations.cuh"
#include "helper_math.cuh"


using namespace Eigen;

constexpr double coeff1 = 1.4142135623730950; // sqrt((6-d)/2.);
constexpr double coeff1_inv = 0.7071067811865475;

// flags writte into point's utility_data
// (Defined in SimParams)

// error indicator and flags
__device__ uint32_t gpu_error_indicator;
constexpr uint32_t error_code_point_pos_nan = 0x0001;           // point's position is NaN
constexpr uint32_t error_code_point_vel_nan = 0x0002;           // point's velocity is NaN
constexpr uint32_t error_code_point_jump_cells = 0x0004;    // point is flying too fast
constexpr uint32_t error_code_point_left_area = 0x0008;     // point is outside of bounds
constexpr uint32_t error_code_point_left_global = 0x0010;     // point is outside of global bounds
constexpr uint32_t error_code_point_Bp_nan = 0x0020;
constexpr uint32_t error_code_point_Fe_nan = 0x0040;

constexpr uint32_t error_code_grid_p2g_nan_vel = 0x0100;    // during P2G writing NaN velocity into grid
constexpr uint32_t error_code_grid_p2g_nan_mass = 0x0200;   // during P2G writing NaN velocity into grid
constexpr uint32_t error_code_grid_nan = 0x0400;            // velocity on the grid is NaN (during grid update)


__constant__ SimParams gprms;


__global__ void partition_kernel_p2g(const PartitionParams pparams)
{
    const size_t pt_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    const double &h = gprms.cellsize;


    const int &gridY = gprms.GridYTotal;
    const unsigned &halo = gprms.GridHaloSize;
    const size_t &pitch = pparams.pitch_pts;
    const size_t &gridX_offset = pparams.gridX_offset;
    double* const &bpts = pparams.buffer_pts;
    double* const &bgrid = pparams.buffer_grid;

    const double utility_double = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_utility_data];
    const long long utility_data = __double_as_longlong(utility_double);
    if(utility_data & SimParams::status_disabled) return; // point is disabled

    // pull point data from SOA
    Eigen::Vector2d pos, velocity;
    Eigen::Matrix2d Cp, Fe, PFt;
    Eigen::Matrix2d stress_contribution = Eigen::Matrix2d::Zero();

    const double thickness = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_thickness];
    const double particle_mass = gprms.ParticleMass * thickness;

    for(int i=0; i<SimParams::dim; i++)
    {
        velocity[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::velx+i)];
        pos[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::posx+i)];

        for(int j=0; j<SimParams::dim; j++)
        {
            Fe(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];
            Cp(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Bp00 + i*SimParams::dim + j)];
        }
    }
    double Jp_inv = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_Jp_inv];

    // PFt is 1st Piola-Kirchhoff Stress times F-transposed
    PFt = KirchhoffStress_Wolper(Fe);
    stress_contribution = -(gprms.dt_vol_Dpinv*Jp_inv*thickness)*PFt;
    stress_contribution += Cp*particle_mass;    // this is part of the linear term from the velocity approximateion

    Eigen::Vector2i cell_i  = getIntegerCellIndex(bpts[pt_idx + pitch*SimParams::PtArrIdx::integer_cell_idx]);

    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);

    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            const double Wip = ww[i+1][0]*ww[j+1][1];
            const Eigen::Vector2d dpos((i-pos[0])*h, (j-pos[1])*h);

            // index of the cell takes into accout the partition's offset of the gird fragment
            const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;

            const double incM = Wip*particle_mass;
            atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_mass*pparams.pitch_grid + idx_gridnode], incM);

            const Eigen::Vector2d incV = Wip*(velocity*particle_mass + stress_contribution*dpos);

            // distribute values to the grid (mass and momentum)
            atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_px*pparams.pitch_grid + idx_gridnode], incV[0]);
            atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_py*pparams.pitch_grid + idx_gridnode], incV[1]);
#ifdef ENABLE_NAN_CHECKS
            // sanity checks
            if(isnan(incV[0]) || isnan(incV[1])) gpu_error_indicator |= error_code_grid_p2g_nan_vel;
            if(isnan(incM)) gpu_error_indicator |= error_code_grid_p2g_nan_mass;
#endif
        }

    // check if a point is out of bounds of the local grid partition
    const int lboundX = 1 + (int)pparams.gridX_offset - (int)gprms.GridHaloSize;
    const int hboundX = pparams.partition_gridX + pparams.gridX_offset - 2 + gprms.GridHaloSize;

#ifdef ENABLE_NAN_CHECKS
    // global bounds
    if(cell_i[0] < 1 || cell_i[1] < 1 || cell_i[0] > (gprms.GridXTotal-2) || cell_i[1] > gridY-2)
        gpu_error_indicator |= error_code_point_left_global;
    else if(cell_i[0] < lboundX || cell_i[0] > hboundX)
        gpu_error_indicator |= error_code_point_left_area;
#endif
}


__global__ void partition_kernel_update_nodes(const PartitionParams pparams,
                                              const double simulation_time, const double current_alpha, const double current_alpha_wind)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nNodes = (pparams.partition_gridX + 2*gprms.GridHaloSize) * gprms.GridYTotal;
    if(idx >= nNodes) return;

    //const int &gridY = gprms.GridYTotal;
    const size_t &pitch_grid = pparams.pitch_grid;
    double* const &bgrid = pparams.buffer_grid;
    const size_t &pitch_grid_forcing = pparams.pitch_grid_forcing;
    float* const &bgrid_forcing = pparams.buffer_grid_forcing;

    //const double &cellsize = gprms.cellsize;
    const double &dt = gprms.InitialTimeStep;               // time step

    const double mass = bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_mass*pitch_grid + idx];
    if(mass == 0) return;

    double vx = bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_px*pitch_grid + idx];
    double vy = bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_py*pitch_grid + idx];
    
    // Normalize momentum to get velocity
    const Vector2i gi((int)idx/gprms.GridYTotal+(int)pparams.gridX_offset-(int)gprms.GridHaloSize, idx%gprms.GridYTotal);   // integer x-y index of the grid node
    const Eigen::Vector2d gnpos = gi.cast<double>()*gprms.cellsize;    // position of the grid node in the whole grid

    Eigen::Vector2d momentum(vx, vy);  // at this point it is momentum
    Eigen::Vector2d velocity = momentum/mass;

    uint8_t is_modeled_area = pparams.buffer_grid_regions[idx];
    if(is_modeled_area != SimParams::ModelledAreaIndicator)
    {
        velocity.setZero();
        // Force accumulation removed
        // bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fx*pitch_grid + idx] += momentum[0];
        // bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fy*pitch_grid + idx] += momentum[1];
    }
    else
    {
        // obtain horizontal velocity from 2 time frames for drag calculation
        Eigen::Vector2d v_frame0(bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_current_vx_frame0*pitch_grid_forcing + idx],
                                 bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_current_vy_frame0*pitch_grid_forcing + idx]);

        Eigen::Vector2d v_frame1(bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_current_vx_frame1*pitch_grid_forcing + idx],
                                 bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_current_vy_frame1*pitch_grid_forcing + idx]);

        Eigen::Vector2d v_w = (1.0 - current_alpha) * v_frame0 + current_alpha * v_frame1;

        // obtain wind velocity from GPU global memory
        Eigen::Vector2d v_frame0_wind(bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_wind_vx_frame0*pitch_grid_forcing + idx],
                                      bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_wind_vy_frame0*pitch_grid_forcing + idx]);

        Eigen::Vector2d v_frame1_wind(bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_wind_vx_frame1*pitch_grid_forcing + idx],
                                      bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_wind_vy_frame1*pitch_grid_forcing + idx]);

        Eigen::Vector2d v_wind = (1.0 - current_alpha_wind) * v_frame0_wind + current_alpha_wind * v_frame1_wind;


        // effect of the water drag on horizontal velocity
        // const double kL = gprms.waterDragEffectiveLinear * dt; // linear param REMOVED
        const double kQp = gprms.waterDragEffectiveQuadratic * dt; // quadratic

        Eigen::Vector2d U_rel = (v_w - velocity);  // relative velocity
        const double U_rel_mag = U_rel.norm();  // magnitude

        double k = kQp*U_rel_mag;
        k = min(k, 0.1);   // k cannot exceed 0.1

        // effect of the wind drag
        const double drag_coeff = gprms.windDragEffectiveQuadratic;
        Eigen::Vector2d U_rel_wind = (v_wind - velocity);  // relative velocity
        const double U_rel_mag_wind = U_rel_wind.norm();  // magnitude
        double k_wind = drag_coeff * dt * U_rel_mag_wind; // quadratic
        k_wind = min(k_wind, 0.1);   // k cannot exceed 0.1

        velocity += (k*U_rel);
        if(gprms.UseWindData) velocity += k_wind*U_rel_wind;
    }

    // write the updated grid velocity back to memory
    if(velocity.squaredNorm() > gprms.vmax*gprms.vmax*0.5) velocity.setZero();

    bgrid[SimParams::grid_idx_px*pitch_grid + idx] = velocity[0];
    bgrid[SimParams::grid_idx_py*pitch_grid + idx] = velocity[1];

#ifdef ENABLE_NAN_CHECKS
    if(isnan(velocity[0]) || isnan(velocity[1])) gpu_error_indicator |= error_code_grid_nan;
#endif

}

__global__ void partition_kernel_g2p(const PartitionParams pparams, const bool recordPQ, const int step)
{
    const size_t pt_idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    const unsigned &halo = gprms.GridHaloSize;
    const size_t &pitch_pts = pparams.pitch_pts;
    const size_t &pitch_grid = pparams.pitch_grid;
    const size_t &gridX_offset = pparams.gridX_offset;
    double* const &bpts = pparams.buffer_pts;
    double* const &bgrid = pparams.buffer_grid;

    const double &h_inv = gprms.cellsize_inv;
    const double &dt = gprms.InitialTimeStep;
    const double &mu = gprms.mu;
    const double &kappa = gprms.kappa;
    const int &gridY = gprms.GridYTotal;

    // skip if a point is disabled
    const double utility_double_g2p = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data];
    unsigned long long utility_data = __double_as_longlong(utility_double_g2p);
    if(utility_data & SimParams::status_disabled) return; // point is disabled
    unsigned long long utility_original = utility_data;

    Eigen::Vector2d pos;
    Eigen::Vector2d p_velocity; p_velocity.setZero();
    double Je_tr, p_tr, q_tr;

    Eigen::Matrix2d Fe;         // deformation gradient
    Eigen::Matrix2d p_Bp; p_Bp.setZero();

    // pull point data from SOA
    const double initial_thickness = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_thickness];
    double Jp_inv = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Jp_inv];
    for(int i=0; i<SimParams::dim; i++)
    {
        pos[i] = bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::posx+i)];
        for(int j=0; j<SimParams::dim; j++)
        {
            Fe(i,j) = bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];
        }
    }

    Eigen::Vector2i cell_i  = getIntegerCellIndex(bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);
    //const double PSI_prev = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_strain_energy] ;

    // optimized method of computing the quadratic weight function without conditional operators
    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);
    // pull velocity from the grid
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            Eigen::Vector2d dpos = Eigen::Vector2d(i, j) - pos;
            double weight = ww[i+1][0]*ww[j+1][1];

            // grid node index within the 3x3 loop
            const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;

            Eigen::Vector2d node_velocity;  // normal in-plane velocity
            node_velocity[0] = bgrid[SimParams::grid_idx_px*pitch_grid + idx_gridnode];
            node_velocity[1] = bgrid[SimParams::grid_idx_py*pitch_grid + idx_gridnode];
            p_velocity += weight * node_velocity;
            p_Bp += (4.*h_inv*weight) * (node_velocity*dpos.transpose());
        }

    // Advection and update of the deformation gradient
    bool cell_updated = false; // record if the point moved into another cell

    pos += p_velocity * (dt*h_inv); // position is in local cell coordinates [-0.5 to 0.5]

#ifdef ENABLE_NAN_CHECKS
    // check if there is an error
    if(isnan(p_velocity[0]) || isnan(p_velocity[1])) gpu_error_indicator |= error_code_point_vel_nan;
    if(isnan(pos[0]) || isnan(pos[1])) gpu_error_indicator |= error_code_point_pos_nan;
    if(isnan(p_Bp(0,0)) || isnan(p_Bp(1,0)) || isnan(p_Bp(0,1)) || isnan(p_Bp(1,1))) gpu_error_indicator |= error_code_point_Bp_nan;
#endif

    // encode the position of the point as coordinates + cell index
    // if a point moves to the next cell, account for the change
    if(pos.x() > 0.5) { pos.x() -= 1.0; cell_i.x()++; cell_updated = true; }
    else if(pos.x() < -0.5) { pos.x() += 1.0; cell_i.x()--; cell_updated = true; }
    if(pos.y() > 0.5) { pos.y() -= 1.0; cell_i.y()++; cell_updated = true; }
    else if(pos.y() < -0.5) { pos.y() += 1.0; cell_i.y()--; cell_updated = true; }

    // this allows the points to leave the simulation area and become disabled
    if(cell_updated)
    {
        if(cell_i.x() <= 1 || cell_i.x() >= gprms.GridXTotal-2 || cell_i.y() <= 1 || cell_i.y() >= gridY-2)
        {
            utility_data |= SimParams::status_disabled;
            atomicAdd(pparams.disabled_points_count, 1);
        }
    }

    Fe = (Eigen::Matrix2d::Identity() + dt*p_Bp) * Fe;     // Bp plays the role of the gradient of the velocity vector
    ComputePQ(Je_tr, p_tr, q_tr, Fe);    // computes P, Q

    // for testing - compute strain energy
//    const double PSI = StrainEnergyDensity(Fe);


    if(!(utility_data & SimParams::status_crushed))
    {
        CheckIfPointIsInsideFailureSurface(utility_data, p_tr, q_tr);
    }

    Eigen::Matrix2d U, V;
    Eigen::Vector2d vSigma, vSigmaSquared, v_s_hat_tr;

    const bool perform_glen_step = (step % SimParams::glen_flow_every_N_step == 0) && (gprms.GlenA != 0);
    const bool is_damaged = ((utility_data & SimParams::status_crushed) || (utility_data & SimParams::status_cracked));
    if(perform_glen_step || is_damaged)
        ComputeSVD(Fe, U, vSigma, V, vSigmaSquared, v_s_hat_tr, kappa, mu, Je_tr);

    // Glen's Flow Rule
    double glen_flow_change = 0;
    if(perform_glen_step) Glen_Nye_flow_law(dt*SimParams::glen_flow_every_N_step, q_tr, vSigmaSquared, U, V, v_s_hat_tr, Fe, glen_flow_change);

    if(is_damaged)
    {
//        Plastic_Project_to_Fracture_Surface(utility_data, initial_thickness, p_tr, q_tr, Je_tr, U, V, vSigmaSquared, v_s_hat_tr, Fe, Jp_inv);
        Wolper_Drucker_Prager(utility_data, initial_thickness, p_tr, q_tr, Je_tr, U, V, vSigmaSquared, v_s_hat_tr, Fe, Jp_inv);
    }

    // distribute the values of p back into GPU memory
    if(perform_glen_step) bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_glen_flow] += glen_flow_change;

    for(int i=0; i<SimParams::dim; i++)
    {
        bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::posx+i)] = pos[i];
        bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::velx+i)] = p_velocity[i];
        for(int j=0; j<SimParams::dim; j++)
        {
            bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)] = Fe(i,j);
            bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::Bp00 + i*SimParams::dim + j)] = p_Bp(i,j);
        }
    }

    bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Jp_inv] = Jp_inv;

    if(cell_updated)
    {
        long long cell = ((long long)cell_i[1] << 32) | (long long)cell_i[0];
        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx] = __longlong_as_double(cell);
    }

    //bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_strain_energy] = PSI;

    // upon request, PQ are recorded for visualization
//    if(recordPQ)
//    {
//        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_P] = p_tr;
//        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Q] = q_tr;
//    }

    // save crushed/disabled status (preserves upper 32 bits with color info)
    if(utility_data != utility_original)
        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data] = __longlong_as_double(utility_data);

#ifdef ENABLE_NAN_CHECKS
    if(isnan(Fe(0,0)) || isnan(Fe(1,0)) || isnan(Fe(0,1)) || isnan(Fe(1,1))) gpu_error_indicator |= error_code_point_Fe_nan;
    // ensure the coordinates are valid
    if(pos.x() > 0.5 || pos.x() < -0.5 || pos.y() > 0.5 || pos.y() < -0.5)
        gpu_error_indicator |= error_code_point_jump_cells;
#endif
}




// ======================================== END OF P2G/UDPATE/G2P KERNELS

__global__ void partition_kernel_render_results(const PartitionParams pparams, int group)
{
    // from the point data, populate the gird arrays:
    // grid_idx_vis_r/g/b, grid_idx_vis_Jpinv, grid_idx_vis_P/Q,
    // grid_idx_vis_strain_EqvGreenLagrange, grid_idx_vis_strain_vonMises

    const size_t pt_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    const unsigned &halo = gprms.GridHaloSize;
    const int &gridY = gprms.GridYTotal;
    const size_t &gridX_offset = pparams.gridX_offset;
    const size_t &pitch = pparams.pitch_pts;
    const size_t &pitch_g = pparams.pitch_grid;

    double* const &bpts = pparams.buffer_pts;
    float* const &bgrid = (float*)pparams.buffer_grid;  // for rendering, we treat the grid buffer as 'float'

    const double utility_double = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_utility_data];
    unsigned long long utility = __double_as_longlong(utility_double);

    if(utility & SimParams::status_disabled) return; // point is disabled

    Eigen::Vector2d pos;
    for(int i=0; i<SimParams::dim; i++)
        pos[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::posx+i)];
    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);

    Eigen::Vector2i cell_i = getIntegerCellIndex(bpts[pt_idx + pitch*SimParams::PtArrIdx::integer_cell_idx]);
    const float thickness = (float)bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_thickness];
    const float particle_mass = (float)gprms.ParticleMass * thickness;

    if(group == 0)
    {
        Eigen::Vector2f velocity;
        for(int i=0; i<SimParams::dim; i++) velocity[i] = (float)bpts[pt_idx + pitch*(SimParams::PtArrIdx::velx+i)];

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;

                // if(idx_gridnode >= (size_t)gridY*(pparams.partition_gridX+2*halo)) gpu_error_indicator |= error_code_point_left_area;


                const float incM = Wip*particle_mass;
                const Eigen::Vector2f incV = incM*velocity;

                // distribute values to the grid
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_mass*pitch_g + idx_gridnode], incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_px*pitch_g + idx_gridnode], incV.x());
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_py*pitch_g + idx_gridnode], incV.y());
            }
    }
    else if(group == 1)
    {
        // group 1: Color
        // Extract RGB (R: 24-31, G: 32-39, B: 40-47)
        uint8_t r = (utility >> 24) & 0xFF;
        uint8_t g = (utility >> 32) & 0xFF;
        uint8_t b = (utility >> 40) & 0xFF;

        const float rR = (double)r / 255.0;
        const float rG = (double)g / 255.0;
        const float rB = (double)b / 255.0;

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;

                const float incM = Wip*particle_mass;
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_r*pitch_g + idx_gridnode], rR*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_g*pitch_g + idx_gridnode], rG*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_b*pitch_g + idx_gridnode], rB*incM);
            }
    }

    else if(group == 2)
    {
        // P and Q are scalars (pressure and deviatoric stress measure)
        const float Jp_inv = (float)bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_Jp_inv];

        double Je_tr, p_tr, q_tr;
        Eigen::Matrix2d Fe;
        for(int i=0; i<SimParams::dim; i++)
            for(int j=0; j<SimParams::dim; j++)
                Fe(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];
        ComputePQ(Je_tr, p_tr, q_tr, Fe);    // computes P, Q

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;

                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_Jpinv*pitch_g + idx_gridnode], Jp_inv*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_P*pitch_g + idx_gridnode], (float)p_tr*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_Q*pitch_g + idx_gridnode], (float)q_tr*incM);
            }
    }

    else if(group == 3)
    {
        Eigen::Matrix2d Fe;
        for(int i=0; i<SimParams::dim; i++)
            for(int j=0; j<SimParams::dim; j++)
                Fe(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];

        Eigen::Matrix2d E = 0.5f*(Fe.transpose()*Fe-Eigen::Matrix2d::Identity()); // GreenLagrangeStrainTensor
        Eigen::Matrix2d E_dev = dev(E);
        const float str_vonMises = (float)(std::sqrt((2.0f / 3.0f) * (E_dev.array() * E_dev.array()).sum()));
        const float str_EqvGreenLagrange = (float)std::sqrt(E.squaredNorm());

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_pts_density*pitch_g + idx_gridnode], Wip);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_strain_EqvGreenLagrange*pitch_g + idx_gridnode], str_EqvGreenLagrange*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_strain_vonMises*pitch_g + idx_gridnode], str_vonMises*incM);
            }
    }

    else if(group == 4)
    {
        const float val_crushed = (utility & SimParams::status_crushed) ? 1.0f : 0.0f;
        const float val_cracked = (utility & SimParams::status_cracked) ? 1.0f : 0.0f;
        const float thickness = (float)bpts[pt_idx + pitch * SimParams::PtArrIdx::idx_thickness];

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_thickness*pitch_g + idx_gridnode], thickness*incM);
                // Determine status from utility flags read at kernel start
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_crushed*pitch_g + idx_gridnode], val_crushed*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_cracked*pitch_g + idx_gridnode], val_cracked*incM);
            }
    }

    else if(group == 5)
    {
        const float val_tension = (utility & SimParams::fracture_tension) ? 1.0f : 0.0f;
        const float val_shear = (utility & SimParams::fracture_compression_shear) ? 1.0f : 0.0f;
        const float val_crush = (utility & SimParams::fracture_crush) ? 1.0f : 0.0f;

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;
                // Determine fracture status
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fracture_tension*pitch_g + idx_gridnode], val_tension*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fracture_shear*pitch_g + idx_gridnode], val_shear*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fracture_crush*pitch_g + idx_gridnode], val_crush*incM);
            }
    }

    else if(group == 6)
    {
        const float glen_flow = (float)bpts[pt_idx + pitch * SimParams::PtArrIdx::idx_glen_flow];
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;

                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_glen_flow*pitch_g + idx_gridnode], glen_flow*incM);
            }
    }
}


/*
__global__ void partition_kernel_summarize_forces(const PartitionParams pparams)
{
    // forces that were recorded (accumulated) in grid_idx_fx/fy are now summarized by region
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nNodes = (pparams.partition_gridX + 2*gprms.GridHaloSize) * gprms.GridYTotal;
    if(idx >= nNodes) return;

    //const int &gridY = gprms.GridYTotal;
    const size_t &pitch_grid = pparams.pitch_grid;
    double* const &bgrid = pparams.buffer_grid;

    double fx = bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fx*pitch_grid + idx];
    double fy = bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fy*pitch_grid + idx];

    uint8_t area_idx = pparams.buffer_grid_regions[idx];
    if(area_idx < SimParams::MAX_REGIONS && (fx != 0 || fy != 0))
    {
        atomicAdd(&pparams.grid_forces_summary_per_region[area_idx*2+0], fx);
        atomicAdd(&pparams.grid_forces_summary_per_region[area_idx*2+1], fy);
    }
}
*/




// ======================================== KERNELS RELATED TO MULTI-GPU IMPLEMENTATION


__global__ void partition_kernel_receive_subgrid(const PartitionParams pparams,
                                                 const size_t transfer_buffer_idx,
                                                 const size_t receive_offset,
                                                 const size_t receive_width)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t receive_elem_count = gprms.GridYTotal * receive_width;
    if(idx >= receive_elem_count) return;

    for(int i=0; i<SimParams::grid_arrays_to_clear; i++)
    {
        const size_t elem_idx = idx + i*pparams.pitch_grid + receive_offset*gprms.GridYTotal;
        const size_t buffer_idx = idx + i*pparams.transfer_buffer_width*gprms.GridYTotal;
        pparams.buffer_grid[elem_idx] += pparams.halo_transfer_buffer[transfer_buffer_idx][buffer_idx];
    }
}


__global__ void partition_kernel_check_if_transfer_needed(const PartitionParams pparams)
{
    const size_t pt_idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    double* const &bpts = pparams.buffer_pts;
    const int threshold = gprms.HaloDiffusionThreshold;
    const size_t &pitch_pts = pparams.pitch_pts;
    const size_t &gridX_offset = pparams.gridX_offset;
    const size_t &gridX = pparams.partition_gridX;

    // skip if a point is disabled
    const double utility_double_pt = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data];
    const unsigned long long utility_data = __double_as_longlong(utility_double_pt);
    if(utility_data & SimParams::status_disabled) return; // point is disabled

    Eigen::Vector2i cell_i  = getIntegerCellIndex(bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);
    const int cx = cell_i.x() - (int)gridX_offset; // x-index of the cell of the point

    if(cx < ( -threshold))
    {
        const int diffusion = (int)(-cx);
        atomicMax(&pparams.pud->diffusion_distance_into_halo, diffusion);
    }
    else if(cx >= (threshold + gridX))
    {
        const int diffusion = (int)(cx-gridX);
        atomicMax(&pparams.pud->diffusion_distance_into_halo, diffusion);
    }
}



__global__ void partition_kernel_point_transfer(const PartitionParams pparams)
{
    const size_t pt_idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    double* const &bpts = pparams.buffer_pts;
    const size_t &pitch_pts = pparams.pitch_pts;
    const size_t &gridX_offset = pparams.gridX_offset;
    const size_t &gridX = pparams.partition_gridX;

    // skip if a point is disabled
    const double utility_double_g2p = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data];
    unsigned long long utility_data = __double_as_longlong(utility_double_g2p);
    if(utility_data & SimParams::status_disabled) return; // point is disabled

    Eigen::Vector2i cell_i  = getIntegerCellIndex(bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);
    const int cx = cell_i.x() - (int)gridX_offset; // x-index of the cell in the current partition
    int transfer_threshold = 0;

    auto transferPoint = [&](int bufferIndex, unsigned int* counter) {
        unsigned flyIdx = atomicAdd(counter, 1);
        if (flyIdx < pparams.point_transfer_buffer_capacity)
        {
            double *ptb = pparams.point_transfer_buffer[bufferIndex];
            for(int i=0;i<SimParams::PtArrIdx::nPtsArrays;i++)
                ptb[i + flyIdx*SimParams::PtArrIdx::nPtsArrays] = bpts[pt_idx + pitch_pts*i];

            atomicAdd(pparams.disabled_points_count, 1);
            utility_data |= SimParams::status_disabled;
            bpts[pt_idx + pitch_pts * SimParams::PtArrIdx::idx_utility_data] = __longlong_as_double(utility_data);
        }
    };

    if (cx < -transfer_threshold)
    {
        transferPoint(0, &pparams.pud->transfer_to_left);
    }
    else if (cx >= (transfer_threshold + gridX))
    {
        transferPoint(1, &pparams.pud->transfer_to_right);
    }
}


__global__ void partition_kernel_receive_points(const PartitionParams pparams, const unsigned nPts,
                                                const unsigned bufferIdx)
{
    const size_t pt_idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= nPts) return;

    double* const &bpts = pparams.buffer_pts;
    const size_t &pitch_pts = pparams.pitch_pts;
    double* const transfer_buffer = pparams.point_transfer_buffer[bufferIdx];
    size_t idx_in_soa = pparams.count_pts + pt_idx;
    if(idx_in_soa >= pitch_pts) { gpu_error_indicator = 0xfffe; return; } // no space for incoming points

    // copy point data
    for(int i=0;i<SimParams::PtArrIdx::nPtsArrays;i++)
    {
        bpts[idx_in_soa + i*pitch_pts] = transfer_buffer[i + SimParams::PtArrIdx::nPtsArrays*pt_idx];
    }
}


// =========================================  DEVICE FUNCTIONS



__device__ void CalculateWeightCoeffs(const Eigen::Vector2d &pos, Eigen::Array2d ww[3])
{
    // optimized method of computing the quadratic (!) weight function (no conditional operators)
    Eigen::Array2d arr_v0 = 0.5 - pos.array();
    Eigen::Array2d arr_v1 = pos.array();
    Eigen::Array2d arr_v2 = pos.array() + 0.5;
    ww[0] = 0.5*arr_v0*arr_v0;
    ww[1] = 0.75-arr_v1*arr_v1;
    ww[2] = 0.5*arr_v2*arr_v2;
}


__device__ void ComputePQ(double &Je_tr, double &p_tr, double &q_tr,
    const Eigen::Matrix2d &F)
{
    const double kappa = gprms.kappa;
    const double &mu = gprms.mu;

    Je_tr = F.determinant();
    p_tr = -(kappa/2.) * (Je_tr*Je_tr - 1.);
    q_tr = coeff1*mu*(1./Je_tr)*dev(F*F.transpose()).norm();
}


__device__ void CheckIfPointIsInsideFailureSurface(unsigned long long &utility_data,
                            const double &p, const double &q)
{
    const double pmax = gprms.IceCompressiveStrength;
    const double pmin = -gprms.IceTensileStrength;

    const double qmax = gprms.IceShearStrength;
    const double pmin2 = -gprms.IceTensileStrength2;

    const double beta = gprms.IceTensileStrength/gprms.IceCompressiveStrength;
    const double M_sq = (4.*qmax*qmax*(1.+2.*beta))/((pmax-pmin)*(pmax-pmin));

    if(p < pmin2)
    {
        // cracked by exceeding tensile threshold
        utility_data |= SimParams::status_cracked;
        utility_data |= SimParams::fracture_tension;
        return;
    }
    else if(p < 0)
    {
        /*
        // can be cracked in tension/shear (if outside failure envelope)
        double q0 = 2*sqrt(-pmax*pmin)*qmax/(pmax-pmin);
        double k = -q0/pmin2;
        double q_limit = k*(p-pmin2);
        if(q > q_limit)
        {
            utility_data |= SimParams::status_cracked;
            utility_data |= SimParams::fracture_tension;
        }
*/
        double y = (1.+2.*beta)*q*q + M_sq*(p+beta*pmax) * (p-pmax);
        if(y > 0)
        {
            utility_data |= SimParams::status_cracked;
            utility_data |= SimParams::fracture_tension;
        }


        return;
    }
    else
    {
        // does not exceed raw compressive failure - can be cracked in shear
        double y = (1.+2.*beta)*q*q + M_sq*(p+beta*pmax) * (p-pmax);
        if(y > 0)
        {
            if(p <= gprms.IceCompressiveThreshold)
            {
                utility_data |= SimParams::status_cracked;
                utility_data |= SimParams::fracture_compression_shear;
            }
            else
            {
                utility_data |= SimParams::status_crushed;
                utility_data |= SimParams::fracture_crush;
            }
        }
    }
}


/*
__device__ bool CheckIfPointIsOutsideFailureSurface(const double &p, const double &q)
{
    const double pmax = gprms.IceCompressiveStrength;
    const double pmin = -gprms.IceTensileStrength;
    const double qmax = gprms.IceShearStrength;

    const double beta = gprms.IceTensileStrength/gprms.IceCompressiveStrength;
    const double M_sq = (4.*qmax*qmax*(1.+2.*beta))/((pmax-pmin)*(pmax-pmin));

    double y = (1.+2.*beta)*q*q + M_sq*(p+beta*pmax) * (p-pmax);
    return (y > 0);
}
*/


__device__ void Wolper_Drucker_Prager(const unsigned long long &utility_data, const double &initial_thickness,
                                      const double &p_tr, const double &q_tr, const double &Je_tr,
                                      const Eigen::Matrix2d &U, const Eigen::Matrix2d &V, const Eigen::Vector2d &vSigmaSquared, const Eigen::Vector2d &v_s_hat_tr,
                                      Eigen::Matrix2d &Fe, double &Jp_inv)
{
    const double &mu = gprms.mu;
    const double kappa = gprms.kappa;

    double DP_threshold_p = gprms.DP_threshold_p;
    const double pmin = -gprms.IceTensileStrength;

    const double &pmax = gprms.IceCompressiveStrength;
    const double &qmax = gprms.IceShearStrengthFractured;

    const double tan_phi = tan(gprms.DP_phi*SimParams::pi/180);

    double q_yield = 0;
    double q_n_1 = 0, p_n_1 = 0;
    constexpr double Jp_inv_threshold = 0.05;

    if(p_tr < DP_threshold_p)
    {
        // tension
        if(Jp_inv < Jp_inv_threshold)
        {
            double sqrt_Je_new = sqrt(Je_tr);
            Eigen::Vector2d vSigma_new(sqrt_Je_new,sqrt_Je_new); //= Vector2d::Constant(1.)*sqrt(Je_new);  //Matrix2d::Identity() * pow(Je_new, 1./(double)d);
            Fe = U*vSigma_new.asDiagonal()*V.transpose();
        }
        else
        {
            // stretching in tension - no resistance
            Eigen::Vector2d vSigma_new(1.0,1.0);
            Fe = U*vSigma_new.asDiagonal()*V.transpose();
            Jp_inv /= Je_tr;
        }
    }
    else
    {
        // determine q_yeld from the combination of DP / elliptic yield surface, whichever is lower
        if(p_tr < pmax)
        {
            double q_from_failure_surface = 2*sqrt((pmax-p_tr)*(p_tr-pmin))*qmax/(pmax-pmin);  // elliptic
            double q_from_dp = max((double)0, (p_tr-DP_threshold_p)*tan_phi); // linear Drucker-Prager
            q_yield = min(q_from_failure_surface, q_from_dp);
        }
        else
        {
            // such hight pressures should not happen - everythigng is liquified
            q_yield = 0;
        }

        if(q_tr > q_yield)
        {
            // plasticity will be applied

            // estimate the new P based on the ridge height
            if(p_tr < 0 && Jp_inv < Jp_inv_threshold)
            {
                p_n_1 = p_tr;
                // otherwise p_n_1 = 0
            }
            else if(p_tr >= 0)
            {
                if(utility_data & SimParams::status_crushed)
                {
                    const double p_ridge_max = gprms.RidgeFormationCoeff * SimParams::g * gprms.IceDensity * initial_thickness * (Jp_inv);
                    p_n_1 = min(p_tr, p_ridge_max);
                }
                else
                {
                    if(Jp_inv >= 1.0) p_n_1 = p_tr;
                }
            }

            // re-evaluate q (to find the new "projected" value)
            if(p_n_1 < pmax)
            {
                const double q_from_dp = max((double)0, (p_n_1-DP_threshold_p)*tan_phi);
                const double q_from_failure_surface = 2*sqrt((pmax-p_n_1)*(p_n_1-pmin))*qmax/(pmax-pmin);
                q_n_1 = min(q_from_failure_surface, q_from_dp);
            }
            else
            {
                q_n_1 = 0;
            }

            // given p_n_1 and q_n_1, compute the new Fe
            const double Je_new = sqrt(-2*p_n_1/kappa + 1);
            double s_hat_n_1_norm = q_n_1*coeff1_inv;
            //Matrix2d B_hat_E_new = s_hat_n_1_norm*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2d::Identity()*(SigmaSquared.trace()/d);

            const Eigen::Vector2d vB_hat_E_new = s_hat_n_1_norm*(Je_tr/mu)*v_s_hat_tr.normalized() +
                                                 Eigen::Vector2d::Constant(1)*(Je_new);

            const Eigen::Vector2d vSigma_new = vB_hat_E_new.array().sqrt().matrix();
            Fe = U*vSigma_new.asDiagonal()*V.transpose();
            Jp_inv *= Je_new/Je_tr;
        }
    }

#ifdef ENABLE_NAN_CHECKS
    // check if something went wrong
    if(isnan(Fe(0,0)) || isnan(Fe(1,0)) || isnan(Fe(0,1)) || isnan(Fe(1,1)))
    {
        gpu_error_indicator |= error_code_point_Fe_nan;
    }
#endif
}



/*
__device__ void Plastic_Project_to_Fracture_Surface(const unsigned long long &utility_data, const double &initial_thickness,
                                                    const double &p_tr, const double &q_tr, const double &Je_tr,
                                                    const Eigen::Matrix2d &U, const Eigen::Matrix2d &V, const Eigen::Vector2d &vSigmaSquared, const Eigen::Vector2d &v_s_hat_tr,
                                                    Eigen::Matrix2d &Fe, double &Jp_inv)
{
    const double &mu = gprms.mu;
    const double kappa = gprms.kappa;

    double DP_threshold_p = gprms.DP_threshold_p;
    const double pmin = -gprms.IceTensileStrength;

    const double &pmax = gprms.IceCompressiveStrength;
    const double &qmax = gprms.IceShearStrengthFractured;

    const double tan_phi = tan(gprms.DP_phi*SimParams::pi/180);

    double q_yield = 0;
    double q_n_1 = 0, p_n_1 = 0;
    constexpr double Jp_inv_threshold = 0.05;

    if(p_tr < pmin)
    {
        // tension
        if(Jp_inv < Jp_inv_threshold)
        {
            double sqrt_Je_new = sqrt(Je_tr);
            Eigen::Vector2d vSigma_new(sqrt_Je_new,sqrt_Je_new); //= Vector2d::Constant(1.)*sqrt(Je_new);  //Matrix2d::Identity() * pow(Je_new, 1./(double)d);
            Fe = U*vSigma_new.asDiagonal()*V.transpose();
        }
        else
        {
            // stretching in tension - no resistance
            Eigen::Vector2d vSigma_new(1.0,1.0);
            Fe = U*vSigma_new.asDiagonal()*V.transpose();
            Jp_inv /= Je_tr;
        }
    }
    else
    {
        // determine q_yeld from the combination of DP / elliptic yield surface, whichever is lower
        if(p_tr < pmax)
        {
            double q_from_failure_surface = 2*sqrt((pmax-p_tr)*(p_tr-pmin))*qmax/(pmax-pmin);  // elliptic
            q_yield = q_from_failure_surface;
        }
        else
        {
            // such hight pressures should not happen - everythigng is liquified
            q_yield = 0;
        }

        if(q_tr > q_yield)
        {
            // plasticity will be applied

            // estimate the new P based on the ridge height
            if(p_tr < 0 && Jp_inv < Jp_inv_threshold)
            {
                p_n_1 = p_tr;
                // otherwise p_n_1 = 0
            }
            else if(p_tr >= 0)
            {
                if(utility_data & SimParams::status_crushed)
                {
                    const double p_ridge_max = gprms.RidgeFormationCoeff * SimParams::g * gprms.IceDensity * initial_thickness * (Jp_inv);
                    p_n_1 = min(p_tr, p_ridge_max);
                }
                else
                {
                    if(Jp_inv >= 1.0) p_n_1 = p_tr;
                }
            }

            // re-evaluate q (to find the new "projected" value)
            if(p_n_1 < pmax)
            {
                const double q_from_dp = max((double)0, (p_n_1-DP_threshold_p)*tan_phi);
                const double q_from_failure_surface = 2*sqrt((pmax-p_n_1)*(p_n_1-pmin))*qmax/(pmax-pmin);
                q_n_1 = min(q_from_failure_surface, q_from_dp);
            }
            else
            {
                q_n_1 = 0;
            }

            // given p_n_1 and q_n_1, compute the new Fe
            const double Je_new = sqrt(-2*p_n_1/kappa + 1);
            double s_hat_n_1_norm = q_n_1*coeff1_inv;
            //Matrix2d B_hat_E_new = s_hat_n_1_norm*(pow(Je_tr,2./d)/mu)*s_hat_tr.normalized() + Matrix2d::Identity()*(SigmaSquared.trace()/d);

            const Eigen::Vector2d vB_hat_E_new = s_hat_n_1_norm*(Je_tr/mu)*v_s_hat_tr.normalized() +
                                                 Eigen::Vector2d::Constant(1)*(Je_new);

            const Eigen::Vector2d vSigma_new = vB_hat_E_new.array().sqrt().matrix();
            Fe = U*vSigma_new.asDiagonal()*V.transpose();
            Jp_inv *= Je_new/Je_tr;
        }
    }

#ifdef ENABLE_NAN_CHECKS
    // check if something went wrong
    if(isnan(Fe(0,0)) || isnan(Fe(1,0)) || isnan(Fe(0,1)) || isnan(Fe(1,1)))
    {
        gpu_error_indicator |= error_code_point_Fe_nan;
    }
#endif
}

*/


__device__ void svd2x2(const Eigen::Matrix2d &mA, Eigen::Matrix2d &mU, Eigen::Vector2d &mS, Eigen::Matrix2d &mV)
{
    double U[4], V[4], S[2];

    GivensRotation<double> gv(0, 1);
    GivensRotation<double> gu(0, 1);
    singular_value_decomposition(mA.data(), gu, S, gv);
    gu.template fill<2, double>(U);
    gv.template fill<2, double>(V);

    mU << U[0],U[1],U[2],U[3];
    mS << S[0],S[1];
    mV << V[0],V[1],V[2],V[3];
}

__device__ void ComputeSVD(const Eigen::Matrix2d &Fe, Eigen::Matrix2d &U, Eigen::Vector2d &vSigma, Eigen::Matrix2d &V,
                            Eigen::Vector2d &vSigmaSquared, Eigen::Vector2d &v_s_hat_tr,
                            const double &kappa, const double &mu, const double &Je_tr)
{
    svd2x2(Fe, U, vSigma, V);
    vSigmaSquared = vSigma.array().square().matrix();
    v_s_hat_tr = mu/Je_tr * dev_d(vSigmaSquared); //mu * pow(Je_tr,-2./d)* dev(SigmaSquared);
}



// deviatoric part of a diagonal matrix
__device__ Eigen::Vector2d dev_d(Eigen::Vector2d Adiag)
{
    return Adiag - Adiag.sum()/2.*Eigen::Vector2d::Constant(1.);
}

__device__ Eigen::Matrix2d dev(Eigen::Matrix2d A)
{
    return A - A.trace()/2*Eigen::Matrix2d::Identity();
}


__device__ Eigen::Matrix2d KirchhoffStress_Wolper(const Eigen::Matrix2d &F)
{
    const double kappa = gprms.kappa;
    const double &mu = gprms.mu;

    // Kirchhoff stress as per Wolper (2019)
    double Je = F.determinant();
    Eigen::Matrix2d b = F*F.transpose();
    Eigen::Matrix2d PFt = mu*(1./Je)*dev(b) + kappa*0.5*(Je*Je-1.)*Eigen::Matrix2d::Identity();
    return PFt;
}


__device__ double StrainEnergyDensity(const Eigen::Matrix2d &F)
{
    const double kappa = gprms.kappa;
    const double &mu = gprms.mu;

    // Strain energy density as per Wolper (2019)
    const double trace = (F.transpose()*F).trace();
    double J = F.determinant();
    double term1 = mu*(trace/J - SimParams::dim);
    double term2 = kappa*((J*J-1.)*0.5 - log(J));
    const double result = 0.5*(term1+term2);
    return result;
}


__device__ void Glen_Nye_flow_law(const double dt, double &q_tr,
                                  Eigen::Vector2d &vSigmaSquared,
                                  const Eigen::Matrix2d &U,
                                  const Eigen::Matrix2d &V,
                                  Eigen::Vector2d &v_s_hat_tr,
                                  Eigen::Matrix2d &Fe, double &track_change)
{
    const double &mu = gprms.mu;

    const double Je_tr = Fe.determinant();
    double epsilon_dot_dt = gprms.GlenA * dt * pow(q_tr,3);      // Glen's Law

    double q_n_1 = max(q_tr - mu*epsilon_dot_dt, 0.0);

    double s_hat_n_1_norm = q_n_1*coeff1_inv;
    Eigen::Vector2d vB_hat_E_new = s_hat_n_1_norm*(Je_tr/mu)*v_s_hat_tr.normalized() +
                                   Eigen::Vector2d::Constant(1)*(vSigmaSquared.sum()/SimParams::dim);
    Eigen::Vector2d vSigma_new = vB_hat_E_new.array().sqrt().matrix();
    Fe = U*vSigma_new.asDiagonal()*V.transpose();

    // Update state variables for consistency with new Fe
    double ratio = (q_tr > 1e-12) ? (q_n_1 / q_tr) : 0.0;
    q_tr = q_n_1;
    v_s_hat_tr *= ratio; // deviatoric stress vector scales linearly with q
    vSigmaSquared = vB_hat_E_new; // vB_hat_E_new holds the new squared eigenvalues

    track_change = epsilon_dot_dt;
}



// ============================================================================
// HELPER FUNCTIONS
// ============================================================================


__device__ Eigen::Vector2i getIntegerCellIndex(double raw_value)
{
    const long long cell = __double_as_longlong(raw_value);
    Eigen::Vector2i cell_i((int)(cell & 0xffffffff), (int)(cell >> 32));
    return cell_i;
}
