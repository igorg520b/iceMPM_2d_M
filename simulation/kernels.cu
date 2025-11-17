#include "parameters_sim.h"
#include "gpu_partition.h"
#include "kernel_declarations.cuh"
#include "helper_math.cuh"


using namespace Eigen;

constexpr double coeff1 = 1.4142135623730950; // sqrt((6-d)/2.);
constexpr double coeff1_inv = 0.7071067811865475;

// flags writte into point's utility_data
constexpr uint32_t status_crushed = 0x10000;
constexpr uint32_t status_disabled = 0x20000;

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

    const uint32_t utility_data = *reinterpret_cast<const uint32_t*>(&bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_utility_data]);
    if(utility_data & status_disabled) return; // point is disabled

    // pull point data from SOA
    Eigen::Vector2d pos, velocity;
    Eigen::Matrix2d Cp, Fe;

    for(int i=0; i<SimParams::dim; i++)
    {
        pos[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::posx+i)];
        velocity[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::velx+i)];
        for(int j=0; j<SimParams::dim; j++)
        {
            Fe(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];
            Cp(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Bp00 + i*SimParams::dim + j)];
        }
    }
    const double Jp_inv = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_Jp_inv];
    const double thickness = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_thickness];
    const double particle_mass = gprms.ParticleMass * thickness;

    const uint32_t cell = *reinterpret_cast<const uint32_t*>(&bpts[pt_idx + pitch*SimParams::PtArrIdx::integer_cell_idx]);
    Eigen::Vector2i cell_i((int)(cell & 0xffff), (int)(cell >> 16));

    // perform computation
    const Eigen::Matrix2d PFt = KirchhoffStress_Wolper(Fe);
//    Eigen::Matrix2d subterm2 = particle_mass*Bp - (gprms.dt_vol_Dpinv)*PFt;

    // version that accounts for surface density change
    Eigen::Matrix2d stress_contribution = -(gprms.dt_vol_Dpinv*Jp_inv*thickness)*PFt;
    stress_contribution += Cp*particle_mass;    // this is part of the linear term from the velocity approximateion

    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);

    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            const double Wip = ww[i+1][0]*ww[j+1][1];
            const double incM = Wip*particle_mass;
            const Eigen::Vector2d dpos((i-pos[0])*h, (j-pos[1])*h);
            Eigen::Vector2d velocity_at_node = velocity;
            const Eigen::Vector2d incV = Wip*(velocity_at_node*particle_mass + stress_contribution*dpos);

            // index of the cell takes into accout the partition's offset of the gird fragment
            const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset)*gridY + gridY*halo;

            // distribute values to the grid (mass and momentum)
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_mass*pparams.pitch_grid + idx_gridnode], (double)incM);
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_px*pparams.pitch_grid + idx_gridnode], (double)incV[0]);
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_py*pparams.pitch_grid + idx_gridnode], (double)incV[1]);

            if(isnan(incV[0]) || isnan(incV[1])) gpu_error_indicator |= error_code_grid_p2g_nan_vel;
            if(isnan(incM)) gpu_error_indicator |= error_code_grid_p2g_nan_mass;
        }

    // check if a point is out of bounds of the local grid partition
    const int lboundX = 1 + (int)pparams.gridX_offset - (int)gprms.GridHaloSize;
    const int hboundX = pparams.partition_gridX + pparams.gridX_offset - 2 + gprms.GridHaloSize;

    // global bounds
    if(cell_i[0] < 1 || cell_i[1] < 1 || cell_i[0] > (gprms.GridXTotal-2) || cell_i[1] > gridY-2)
    {
        gpu_error_indicator |= error_code_point_left_global;
    }
    else if(cell_i[0] < lboundX || cell_i[0] > hboundX)
    {
        gpu_error_indicator |= error_code_point_left_area;
    }
}




__global__ void partition_kernel_update_nodes(const PartitionParams pparams,
                                              const double simulation_time, const double current_alpha)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nNodes = (pparams.partition_gridX + 2*gprms.GridHaloSize) * gprms.GridYTotal;
    if(idx >= nNodes) return;

    //const int &gridY = gprms.GridYTotal;
    const size_t &pitch_grid = pparams.pitch_grid;
    double* const &bgrid = pparams.buffer_grid;

    //const double &cellsize = gprms.cellsize;
    const double &dt = gprms.InitialTimeStep;               // time step

    const double mass = bgrid[SimParams::GridArrayIndex::grid_idx_mass*pitch_grid + idx];
    if(mass == 0) return;

    double vx = bgrid[SimParams::GridArrayIndex::grid_idx_px*pitch_grid + idx];
    double vy = bgrid[SimParams::GridArrayIndex::grid_idx_py*pitch_grid + idx];

    const Vector2i gi((int)idx/gprms.GridYTotal+(int)pparams.gridX_offset-(int)gprms.GridHaloSize, idx%gprms.GridYTotal);   // integer x-y index of the grid node
    const Eigen::Vector2d gnpos = gi.cast<double>()*gprms.cellsize;    // position of the grid node in the whole grid

    Eigen::Vector2d momentum(vx, vy);  // at this point it is momentum
    Eigen::Vector2d velocity = momentum/mass;

    uint8_t is_modeled_area = pparams.buffer_grid_regions[idx];
    if(is_modeled_area != SimParams::MAX_REGIONS)
    {
        velocity.setZero();
        bgrid[SimParams::grid_idx_fx*pitch_grid + idx] += momentum[0];
        bgrid[SimParams::grid_idx_fy*pitch_grid + idx] += momentum[1];
    }
    else
    {
        //grid_water_current - interpolate between two frames
        Eigen::Vector2d v_frame0(bgrid[SimParams::grid_idx_current_vx_frame0*pitch_grid + idx],
                                 bgrid[SimParams::grid_idx_current_vy_frame0*pitch_grid + idx]);

        Eigen::Vector2d v_frame1(bgrid[SimParams::grid_idx_current_vx_frame1*pitch_grid + idx],
                                 bgrid[SimParams::grid_idx_current_vy_frame1*pitch_grid + idx]);

        Eigen::Vector2d v_w = (1.0 - current_alpha) * v_frame0 + current_alpha * v_frame1;

        const double kL = gprms.waterDragEffectiveLinear * dt; // linear param
        const double kQp = gprms.waterDragEffectiveQuadratic * dt; // quadratic

        Eigen::Vector2d U_rel = (v_w - velocity);  // relative velocity
        const double U_rel_mag = U_rel.norm();  // magnitude

        double k = kL + kQp*U_rel_mag;
        k = min(k, 0.1);   // k cannot exceed 0.1

        velocity += k*U_rel;
    }

    // write the updated grid velocity back to memory
    if(velocity.squaredNorm() > gprms.vmax*gprms.vmax*0.5) velocity.setZero();
    bgrid[SimParams::grid_idx_px*pitch_grid + idx] = velocity[0];
    bgrid[SimParams::grid_idx_py*pitch_grid + idx] = velocity[1];

    if(isnan(velocity[0]) || isnan(velocity[1])) gpu_error_indicator |= error_code_grid_nan;
}


__global__ void partition_kernel_render_results(const PartitionParams pparams)
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
    double* const &bgrid = pparams.buffer_grid;

    const uint32_t utility_data = *reinterpret_cast<const uint32_t*>(&bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_utility_data]);
    if(utility_data & status_disabled) return; // point is disabled

    Eigen::Vector2d pos, velocity;
    Eigen::Matrix2d Fe;

    // pull point data from SOA
    for(int i=0; i<SimParams::dim; i++)
    {
        pos[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::posx+i)];
        velocity[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::velx+i)];
        for(int j=0; j<SimParams::dim; j++)
        {
            Fe(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];
        }
    }
    const double Jp_inv = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_Jp_inv];

    const double thickness = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_thickness];

    const double particle_mass = gprms.ParticleMass * thickness;

    const double rR = bpts[pt_idx + pitch*(SimParams::PtArrIdx::idx_pt_color_RGB+0)];
    const double rG = bpts[pt_idx + pitch*(SimParams::PtArrIdx::idx_pt_color_RGB+1)];
    const double rB = bpts[pt_idx + pitch*(SimParams::PtArrIdx::idx_pt_color_RGB+2)];
    const double P = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_P];
    const double Q = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_Q];

    const uint32_t cell = *reinterpret_cast<const uint32_t*>(&bpts[pt_idx + pitch*SimParams::PtArrIdx::integer_cell_idx]);
    Eigen::Vector2i cell_i((int)(cell & 0xffff), (int)(cell >> 16));

    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);

    Eigen::Matrix2d E = 0.5f*(Fe.transpose()*Fe-Eigen::Matrix2d::Identity()); // GreenLagrangeStrainTensor
    Eigen::Matrix2d E_dev = dev(E);
    const double str_vonMises = std::sqrt((2.0f / 3.0f) * (E_dev.array() * E_dev.array()).sum());
    const double str_EqvGreenLagrange = std::sqrt(E.squaredNorm());

    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            const double Wip = ww[i+1][0]*ww[j+1][1];

            // index of the cell takes into accout the partition's offset of the gird fragment
            const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset)*gridY + gridY*halo;


            const double incM = Wip*particle_mass;
            const Eigen::Vector2d incV = incM*velocity;

            // distribute values to the grid
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_mass*pitch_g + idx_gridnode], (double)incM);
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_px*pitch_g + idx_gridnode], (double)incV[0]);
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_py*pitch_g + idx_gridnode], (double)incV[1]);

            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_vis_r*pitch_g + idx_gridnode], (double)(rR*incM));
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_vis_g*pitch_g + idx_gridnode], (double)(rG*incM));
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_vis_b*pitch_g + idx_gridnode], (double)(rB*incM));

            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_vis_Jpinv*pitch_g + idx_gridnode], (double)(Jp_inv*incM));
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_vis_P*pitch_g + idx_gridnode], (double)(P*incM));
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_vis_Q*pitch_g + idx_gridnode], (double)(Q*incM));

            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange*pitch_g + idx_gridnode], (double)(str_EqvGreenLagrange*incM));
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_vis_strain_vonMises*pitch_g + idx_gridnode], (double)(str_vonMises*incM));
            atomicAdd(&bgrid[SimParams::GridArrayIndex::grid_idx_vis_pts_density*pitch_g + idx_gridnode], (double)Wip);
        }
}

__global__ void partition_kernel_summarize_forces(const PartitionParams pparams)
{
    // forces that were recorded (accumulated) in grid_idx_fx/fy are now summarized by region
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nNodes = (pparams.partition_gridX + 2*gprms.GridHaloSize) * gprms.GridYTotal;
    if(idx >= nNodes) return;

    //const int &gridY = gprms.GridYTotal;
    const size_t &pitch_grid = pparams.pitch_grid;
    double* const &bgrid = pparams.buffer_grid;

    double fx = bgrid[SimParams::grid_idx_fx*pitch_grid + idx];
    double fy = bgrid[SimParams::grid_idx_fy*pitch_grid + idx];

    uint8_t area_idx = pparams.buffer_grid_regions[idx];
    if(area_idx < SimParams::MAX_REGIONS && (fx != 0 || fy != 0))
    {
        atomicAdd(&pparams.grid_forces_summary_per_region[area_idx*2+0], fx);
        atomicAdd(&pparams.grid_forces_summary_per_region[area_idx*2+1], fy);
    }
}



__global__ void partition_kernel_normalize_render(const PartitionParams pparams)
{
    // forces that were recorded (accumulated) in grid_idx_fx/fy are now summarized by region
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nNodes = (pparams.partition_gridX + 2*gprms.GridHaloSize) * gprms.GridYTotal;
    if(idx >= nNodes) return;

    //const int &gridY = gprms.GridYTotal;
    const size_t &pitch_grid = pparams.pitch_grid;
    double* const &bgrid = pparams.buffer_grid;

    // normalize grid data
    const double mass = bgrid[SimParams::GridArrayIndex::grid_idx_mass*pitch_grid + idx];
    if(mass == 0) return;

    //  double pt_density = bgrid[SimParams::grid_idx_vis_pts_density*pitch_grid + idx];

    bgrid[SimParams::GridArrayIndex::grid_idx_px*pitch_grid + idx] /= mass;
    bgrid[SimParams::GridArrayIndex::grid_idx_py*pitch_grid + idx] /= mass;

    bgrid[SimParams::GridArrayIndex::grid_idx_vis_r*pitch_grid + idx] /= mass;
    bgrid[SimParams::GridArrayIndex::grid_idx_vis_g*pitch_grid + idx] /= mass;
    bgrid[SimParams::GridArrayIndex::grid_idx_vis_b*pitch_grid + idx] /= mass;

    bgrid[SimParams::GridArrayIndex::grid_idx_vis_Jpinv*pitch_grid + idx] /= mass;
    bgrid[SimParams::GridArrayIndex::grid_idx_vis_P*pitch_grid + idx] /= mass;
    bgrid[SimParams::GridArrayIndex::grid_idx_vis_Q*pitch_grid + idx] /= mass;
    bgrid[SimParams::GridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange*pitch_grid + idx] /= mass;
    bgrid[SimParams::GridArrayIndex::grid_idx_vis_strain_vonMises*pitch_grid + idx] /= mass;

    bgrid[SimParams::GridArrayIndex::grid_idx_mass*pitch_grid + idx] /= (gprms.cellsize*gprms.cellsize); // make it mass per area
}


__global__ void partition_kernel_g2p(const PartitionParams pparams, const bool recordPQ)
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
    uint32_t utility_data = *reinterpret_cast<uint32_t*>(&bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data]);
    if(utility_data & status_disabled) return; // point is disabled

    Eigen::Vector2d pos;
    Eigen::Matrix2d Fe;

    // pull point data from SOA
    for(int i=0; i<SimParams::dim; i++)
    {
        pos[i] = bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::posx+i)];
        for(int j=0; j<SimParams::dim; j++)
        {
            Fe(i,j) = bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];
        }
    }

    const double initial_thickness = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_thickness];
    double Jp_inv = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Jp_inv];

    uint32_t cell = *reinterpret_cast<const uint32_t*>(&bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);
    // coords of grid node for point
    Eigen::Vector2i cell_i((int)(cell & 0xffff), (int)(cell >> 16));


    // optimized method of computing the quadratic weight function without conditional operators
    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);

    Eigen::Vector2d p_velocity;
    Eigen::Matrix2d p_Bp;
    p_velocity.setZero();
    p_Bp.setZero();

    // pull velocity from the grid
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            Eigen::Vector2d dpos = Eigen::Vector2d(i, j) - pos;
            double weight = ww[i+1][0]*ww[j+1][1];

            // grid node index within the 3x3 loop
            const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset)*gridY + gridY*halo;

            Eigen::Vector2d node_velocity;
            node_velocity[0] = (double)bgrid[SimParams::grid_idx_px*pitch_grid + idx_gridnode];
            node_velocity[1] = (double)bgrid[SimParams::grid_idx_py*pitch_grid + idx_gridnode];
            p_velocity += weight * node_velocity;
            p_Bp += (4.*h_inv*weight) * (node_velocity*dpos.transpose());
        }

    // Advection and update of the deformation gradient
    pos += p_velocity * (dt*h_inv); // position is in local cell coordinates [-0.5 to 0.5]

    // check if there is an error
    if(isnan(p_velocity[0]) || isnan(p_velocity[1])) gpu_error_indicator |= error_code_point_vel_nan;
    if(isnan(pos[0]) || isnan(pos[1])) gpu_error_indicator |= error_code_point_pos_nan;
    if(isnan(p_Bp(0,0)) || isnan(p_Bp(1,0)) || isnan(p_Bp(0,1)) || isnan(p_Bp(1,1))) gpu_error_indicator |= error_code_point_Bp_nan;


    // encode the position of the point as coordinates + cell index
    // if a point moves to the next cell, account for the change
    bool cell_updated = false;
    if(pos.x() > 0.5) { pos.x() -= 1.0; cell_i.x()++; cell_updated = true; }
    else if(pos.x() < -0.5) { pos.x() += 1.0; cell_i.x()--; cell_updated = true; }
    if(pos.y() > 0.5) { pos.y() -= 1.0; cell_i.y()++; cell_updated = true; }
    else if(pos.y() < -0.5) { pos.y() += 1.0; cell_i.y()--; cell_updated = true; }

    // this allows the points to leave the simulation area and become disabled
    if(cell_updated)
    {
        if(cell_i.x() <= 1 || cell_i.x() >= gprms.GridXTotal-2 || cell_i.y() <= 1 || cell_i.y() >= gridY-2)
        {
            utility_data |= status_disabled;
            atomicAdd(pparams.disabled_points_count, 1);
        }
    }

    // ensure the coordinates are valid
    if(pos.x() > 0.5 || pos.x() < -0.5 || pos.y() > 0.5 || pos.y() < -0.5)
        gpu_error_indicator |= error_code_point_jump_cells;

    Fe = (Eigen::Matrix2d::Identity() + dt*p_Bp) * Fe;     // Bp plays the role of the gradient of the velocity vector
    if(isnan(Fe(0,0)) || isnan(Fe(1,0)) || isnan(Fe(0,1)) || isnan(Fe(1,1))) gpu_error_indicator |= error_code_point_Fe_nan;

    double Je_tr, p_tr, q_tr;
    ComputePQ(Je_tr, p_tr, q_tr, kappa, mu, Fe);    // computes P, Q, J

    Eigen::Matrix2d U, V;
    Eigen::Vector2d vSigma, vSigmaSquared, v_s_hat_tr;
    ComputeSVD(Fe, U, vSigma, V, vSigmaSquared, v_s_hat_tr, kappa, mu, Je_tr);

    if(!(utility_data & status_crushed)) CheckIfPointIsInsideFailureSurface(utility_data, 0, p_tr, q_tr, initial_thickness);
    if(utility_data & status_crushed)
    {
        Wolper_Drucker_Prager(initial_thickness, p_tr, q_tr, Je_tr, U, V, vSigmaSquared, v_s_hat_tr, Fe, Jp_inv);
    }


    // distribute the values of p back into GPU memory: pos, velocity, BP, Fe, Jp_inv, PQ
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

    // save crushed/disabled status
    *reinterpret_cast<uint32_t*>(&bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data]) = utility_data;

    if(cell_updated)
    {
        cell = ((uint32_t)cell_i[1] << 16) | (uint32_t)cell_i[0];
        *reinterpret_cast<uint32_t*>(&bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]) = cell;
    }

    // upon request, PQ are recorded for visualization
    if(recordPQ)
    {
        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_P] = p_tr;
        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Q] = q_tr;
    }
}



// ======================================== KERNELS RELATED TO MULTI-GPU IMPLEMENTATION


__global__ void partition_kernel_receive_subgrid(const PartitionParams pparams,
                                                 const size_t transfer_buffer_idx,
                                                 const size_t receive_offset,
                                                 const size_t receive_width)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t receive_elem_count = gprms.GridYTotal * receive_width;
    if(idx >= receive_elem_count) return;

    for(int i=0; i<3; i++)
    {
        const size_t elem_idx = idx + i*pparams.pitch_grid + receive_offset*gprms.GridYTotal;
        const size_t buffer_idx = idx + i*pparams.transfer_buffer_width*gprms.GridYTotal;
        pparams.buffer_grid[elem_idx] += pparams.halo_transfer_buffer[transfer_buffer_idx][buffer_idx];
    }
}

__global__ void partition_kernel_receive_render_subgrid(const PartitionParams pparams,
                                                 const size_t transfer_buffer_idx,
                                                 const size_t receive_offset,
                                                 const size_t receive_width,
                                                        const int nArrays)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t receive_elem_count = gprms.GridYTotal * receive_width;
    if(idx >= receive_elem_count) return;

    for(int i=0; i<nArrays; i++)
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
    const uint32_t utility_data = *reinterpret_cast<uint32_t*>(&bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data]);
    if(utility_data & status_disabled) return; // point is disabled

    const uint32_t cell = *reinterpret_cast<const uint32_t*>(&bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);

    const int cx = (int)(cell & 0xffff) - (int)gridX_offset; // x-index of the cell of the point

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
    uint32_t utility_data = *reinterpret_cast<uint32_t*>(&bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data]);
    if(utility_data & status_disabled) return; // point is disabled

    const uint32_t cell = *reinterpret_cast<const uint32_t*>(&bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);

    const int cx = (int)(cell & 0xffff) - (int)gridX_offset; // x-index of the cell in the current partition
    int transfer_threshold = 0;

    auto transferPoint = [&](int bufferIndex, unsigned int* counter) {
        unsigned flyIdx = atomicAdd(counter, 1);
        if (flyIdx < pparams.point_transfer_buffer_capacity)
        {
            double *ptb = pparams.point_transfer_buffer[bufferIndex];
            for(int i=0;i<SimParams::PtArrIdx::nPtsArrays;i++)
                ptb[i + flyIdx*SimParams::PtArrIdx::nPtsArrays] = bpts[pt_idx + pitch_pts*i];

            atomicAdd(pparams.disabled_points_count, 1);
            utility_data |= status_disabled;
            *reinterpret_cast<uint32_t*>(&bpts[pt_idx + pitch_pts * SimParams::PtArrIdx::idx_utility_data]) = utility_data;
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



__forceinline__ __device__ void CalculateWeightCoeffs(const Eigen::Vector2d &pos, Eigen::Array2d ww[3])
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
    const double &kappa, const double &mu, const Eigen::Matrix2d &F)
{
    Je_tr = F.determinant();
    p_tr = -(kappa/2.) * (Je_tr*Je_tr - 1.);
    q_tr = coeff1*mu*(1./Je_tr)*dev(F*F.transpose()).norm();
}




__device__ void CheckIfPointIsInsideFailureSurface(uint32_t &utility_data, const uint16_t &grain,
                            const double &p, const double &q, const double &strength)
{
    const double pmax = gprms.IceCompressiveStrength;
    const double pmin = -gprms.IceTensileStrength;

    const double qmax = gprms.IceShearStrength;
    const double pmin2 = -gprms.IceTensileStrength2;

    const double beta = gprms.IceTensileStrength/gprms.IceCompressiveStrength;
    const double M_sq = (4.*qmax*qmax*(1.+2.*beta))/((pmax-pmin)*(pmax-pmin));

    if(p<0)
    {
        if(p<pmin2) { utility_data |= status_crushed; return; }
        double q0 = 2*sqrt(-pmax*pmin)*qmax/(pmax-pmin);
        double k = -q0/pmin2;
        double q_limit = k*(p-pmin2);
        if(q > q_limit) { utility_data |= status_crushed; return; }
    }
    else
    {
        double y = (1.+2.*beta)*q*q + M_sq*(p+beta*pmax) * (p-pmax);
        if(y > 0) utility_data |= status_crushed;
    }
}




__device__ void Wolper_Drucker_Prager(const double &initial_thickness,
                                      const double &p_tr, const double &q_tr, const double &Je_tr,
const Eigen::Matrix2d &U, const Eigen::Matrix2d &V, const Eigen::Vector2d &vSigmaSquared, const Eigen::Vector2d &v_s_hat_tr,
                                      Eigen::Matrix2d &Fe, double &Jp_inv)
{
    const double &mu = gprms.mu;
    const double &kappa = gprms.kappa;
    double DP_threshold_p = gprms.DP_threshold_p;
    const double pmin = -gprms.IceTensileStrength;

    const double &pmax = gprms.IceCompressiveStrength;
    const double &qmax = gprms.IceShearStrength;

    const double tan_phi = tan(gprms.DP_phi*SimParams::pi/180);

    double q_yield = 0;
    double q_n_1 = 0, p_n_1 = 0;

    if(p_tr < DP_threshold_p)
    {
        // tension
        if(Jp_inv < 0.1)
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
            const double p_ridge_max = gprms.RidgeFormationCoeff * SimParams::g * gprms.IceDensity * initial_thickness * (Jp_inv);
            p_n_1 = min(p_tr, p_ridge_max);

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

    // check if something went wrong
    if(isnan(Fe(0,0)) || isnan(Fe(1,0)) || isnan(Fe(0,1)) || isnan(Fe(1,1)))
    {
        gpu_error_indicator |= error_code_point_Fe_nan;
    }
}


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
    const double &kappa = gprms.kappa;
    const double &mu = gprms.mu;

    // Kirchhoff stress as per Wolper (2019)
    double Je = F.determinant();
    Eigen::Matrix2d b = F*F.transpose();
    Eigen::Matrix2d PFt = mu*(1./Je)*dev(b) + kappa*0.5*(Je*Je-1.)*Eigen::Matrix2d::Identity();
    return PFt;
}






/*
__device__ Eigen::Vector2d get_wind_vector(float lat, float lon, float tb)
{
    const double &gridLatMin = gprms.gridLatMin;
    const double &gridLonMin = gprms.gridLonMin;

    // space
    int lat_cell = (int)((lat-gridLatMin)/WindInterpolator::gridCellSize);
    int lon_cell = (int)((lon-gridLonMin)/WindInterpolator::gridCellSize);

    // Compute local coordinates within the cell
    float localLon = lon - (gridLonMin + lon_cell * WindInterpolator::gridCellSize);
    float localLat = lat - (gridLatMin + lat_cell * WindInterpolator::gridCellSize);

    // Compute barycentric coordinates
    float ub = localLon / WindInterpolator::gridCellSize;
    float vb = localLat / WindInterpolator::gridCellSize;

    Eigen::Vector2d cell_values0[2][2], cell_values1[2][2];
    for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
        {
            cell_values0[i][j] = Eigen::Vector2d(wgrid[lat_cell+i][lon_cell+j][0], wgrid[lat_cell+i][lon_cell+j][1]);
            cell_values1[i][j] = Eigen::Vector2d(wgrid[lat_cell+i][lon_cell+j][2], wgrid[lat_cell+i][lon_cell+j][3]);
        }
    Eigen::Vector2d ipVal[2];

    ipVal[0] =
        (1 - ub) * (1 - vb) * cell_values0[0][0] +
        ub * (1 - vb) * cell_values0[0][1] +
        (1 - ub) * vb * cell_values0[1][0] +
        ub * vb * cell_values0[1][1];

    ipVal[1] =
        (1 - ub) * (1 - vb) * cell_values1[0][0] +
        ub * (1 - vb) * cell_values1[0][1] +
        (1 - ub) * vb * cell_values1[1][0] +
        ub * vb * cell_values1[1][1];

    Eigen::Vector2d final_result = (1-tb)*ipVal[0] + tb*ipVal[1];
    return final_result;
}
*/


/*
__device__ void Glen_Nye_flow_law(const double dt, const double &q_tr,
const Eigen::Vector2d &vSigmaSquared,
const Eigen::Matrix2d &U,
const Eigen::Matrix2d &V,
const Eigen::Vector2d &v_s_hat_tr,
                                  Eigen::Matrix2d &Fe, double &qp)

{
    const double &mu = gprms.mu;
    const double &A = gprms.GlenA;


    const double Je_tr = Fe.determinant();
    double epsilon_dot_dt = A * q_tr*q_tr*q_tr * dt;      // Glen's Law


    double q_n_1 = max(q_tr - mu*epsilon_dot_dt, 0.);


    double s_hat_n_1_norm = q_n_1*coeff1_inv;
    Eigen::Vector2d vB_hat_E_new = s_hat_n_1_norm*(Je_tr/mu)*v_s_hat_tr.normalized() +
                                 Eigen::Vector2d::Constant(1)*(vSigmaSquared.sum()/d);
    Eigen::Vector2d vSigma_new = vB_hat_E_new.array().sqrt().matrix();
    Fe = U*vSigma_new.asDiagonal()*V.transpose();
    qp *= (q_n_1/q_tr);
}
*/

