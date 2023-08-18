// cuda_params.h
#ifndef cuda_params_h
#define cuda_params_h

#include "cuda_kernels_def_decl/deviceAPI.h"
#include "cuda_kernels_def_decl/deviceLaunch.h"
#include "lb/lattices/D3Q19_gpu.h"

namespace hemelb {

/**
        Device function to investigate which Iolet Ind corresponds to a fluid with index fluid_Ind
                To be used for the inlet/outlet related collision-streaming kernels
                Checks through the local iolets (inlet/outlet) to determine the correct iolet ID
                        each iolet has fluid sites with indices in the range: [lower_limit,upper_limit]
        Function returns the iolet ID value: IdInlet.
*/
GPU_INLINE_FUNCTION  void
_determine_Iolet_ID(int num_local_Iolets, site_t *iolets_ID_range, site_t fluid_Ind, int *IdInlet) {
  // Loop over the number of local iolets (num_local_Iolets) and determine whether the fluid ID (fluid_Ind) falls whithin the range
  for (int i_local_iolet = 0; i_local_iolet < num_local_Iolets; i_local_iolet++) {
    // iolet range: [lower_limit,upper_limit)
    int64_t lower_limit = iolets_ID_range[3 * i_local_iolet + 1];   // Included in the fluids range
    int64_t upper_limit = iolets_ID_range[3 * i_local_iolet + 2];   // Value included in the fluids' range - CHANGED TO INCLUDE THE VALUE

    // if ((fluid_Ind - upper_limit +1) * (fluid_Ind - lower_limit) <= 0){	 	//When the upper_limit is NOT included
    if ((fluid_Ind - upper_limit) * (fluid_Ind - lower_limit) <= 0) {   // When the upper_limit is included
      *IdInlet = (int) iolets_ID_range[3 * i_local_iolet];
      return;
    }
  }   // closes the loop over the local iolets
}
//==============================================================================

#if 0
template<typename LatticeType>
GPU_INLINE_FUNCTION double *
_CalculatePiTensor(const distribn_t *const f, 
	 	   const std::array<int, c.NUMVECTORS>& c.CX, 
		   const std::array<int, c.NUMVECTORS>& c.CY,
		   const std::array<int, c.NUMVECTORS>& c.CZ) {
  // static 
  double ret_SecMomDistrFunc[6];

  // Explicitly calculate the elements (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
  // and saves these with this order in the array ret_SecMomDistrFunc

  // Element (0,0)
  ret_SecMomDistrFunc[0] = 0.0;
  for (unsigned int l = 0; l < c.NUMVECTORS; ++l) {
    ret_SecMomDistrFunc[0] += f[l] * (double)c.CX[l] * (double)c.CX[l];
  }

  // Element (1,0)
  ret_SecMomDistrFunc[1] = 0.0;
  for (unsigned int l = 0; l < c.NUMVECTORS; ++l) {
    ret_SecMomDistrFunc[1] += f[l] * (double)c.CY[l] * (double)c.CX[l];
  }

  // Element (1,1)
  ret_SecMomDistrFunc[2] = 0.0;
  for (unsigned int l = 0; l < c.NUMVECTORS; ++l) {
    ret_SecMomDistrFunc[2] += f[l] * (double)c.CY[l] * (double)c.CY[l];
  }

  // Element (2,0)
  ret_SecMomDistrFunc[3] = 0.0;
  for (unsigned int l = 0; l < c.NUMVECTORS; ++l) {
    ret_SecMomDistrFunc[3] += f[l] * (double)c.CZ[l] * (double)c.CX[l];
  }

  // Element (2,1)
  ret_SecMomDistrFunc[4] = 0.0;
  for (unsigned int l = 0; l < c.NUMVECTORS; ++l) {
    ret_SecMomDistrFunc[4] += f[l] * (double)c.CZ[l] * (double)c.CY[l];
  }

  // Element (2,2)
  ret_SecMomDistrFunc[5] = 0.0;
  for (unsigned int l = 0; l < c.NUMVECTORS; ++l) {
    ret_SecMomDistrFunc[5] += f[l] * (double)c.CZ[l] * (double)c.CZ[l];
  }

  return ret_SecMomDistrFunc;
}
#endif

// Functors
//
//============================================================================
// Apply the optimisations from the GPU hackathon 2022 on all the kernels in use
// Optimisations:
// 	I. 		Merge the loops (test first having 3 loops, before merging to have 2 loops in total)
//	II.		Loop unrolling: #pragma unroll 19
//	III.  Compile with the flag maxxregcount set to 100 (leads to no memory spills to GPU global memory)

// Currently using the following GPU kernels:
// 1. Stability check: GPU_Check_Stability
// 2. Collision-streaming kernels:
//		2.1. 	GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB						Done!!!
//		2.2.	GPU_CollideStream_Iolets_NashZerothOrderPressure_v2								Done!!!
//		2.3. 	GPU_CollideStream_Iolets_NashZerothOrderPressure									Done!!!
//		2.4. 	GPU_CollideStream_wall_sBB_iolet_Nash
// Done!!! 		2.5. 	GPU_CollideStream_wall_sBB_iolet_Nash_v2 Done!!! 		2.6. 	GPU_CollideStream_Iolets_Ladd_VelBCs
// Done!!! 		2.7. 	GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs
// 3. Case of Velocity BCs and subtype: File:
//		3.1.	GPU_WallMom_correction_File_Weights_NoSearch											Done!!!
//============================================================================

//------------------------------------------------------------------------------
//**************************************************************
/** GPU kernel for evaluating the wall momentum correction terms on the GPU
                For the Velocity BCs - LADDIOLET BCs

                Implement what is in the function
                                LatticeVelocity InOutLetFileVelocity::GetVelocity
                Note that the arguments are: (halfWay, bValues->GetTimeStep())
                                LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));

                        I. Case 1: No weights needed (useWeightsFromFile=false)
                                        TODO in a seperate GPU kernel

                        II. Case 2: weights needed (useWeightsFromFile=true)
                                        Need to have the following information on the GPU:
                                        1. normal vector to the iolet
                                        2. LatticePosition& x : coordinates of the point
                                                        LatticePosition: Vector3D<double>
                                                        see function memCpy_HtD_GPUmem_Coords_Iolets which fills the following:
                                                                type int64_t [x_coord, y_coord, z_coord, ...]
                                                                void *GPUDataAddr_Coords_Inlet_Edge;
                                                                void *GPUDataAddr_Coords_InletWall_Edge;
                                                                void *GPUDataAddr_Coords_Inlet_Inner;
                                                                void *GPUDataAddr_Coords_InletWall_Inner;
                                                                void *GPUDataAddr_Coords_Outlet_Edge;
                                                                void *GPUDataAddr_Coords_OutletWall_Edge;
                                                                void *GPUDataAddr_Coords_Outlet_Inner;
                                                                void *GPUDataAddr_Coords_OutletWall_Inner;

                                        3. LatticeTimeStep t : The time-step (type unsigned long)
                                                                units.h:  typedef unsigned long LatticeTimeStep

                                        4. weights table
                                                                std::map<std::vector<int>, double> weights_table;

                                                        The CPU version gets the weight (mapped value) using the key value.
                                                        Here, we have 2 options:
                                                        4.1. Save the index of the key value (perform the search at Initialise_GPU)
                                                                                        and then save these indices so that I can easily obtain the
   corresponding weight (mapped value) without performing the search at each time-step switch to version 1 for this OR 4.2. Perform the search on the GPU (at
   each time) of the key value (xyz coords) in order to obtain the corresponding weight (mapped value) switch to version 0

                                        5. velocityTable[t]

                                        6. arr_elementsInEachInlet[index_inlet] - n_arr_elementsInCurrentInlet_weightsTable
                                                                number of elements (lines) in the weights table
                                                                Note there is 1 such table for each iolet

                                        7. Consider moving this info in shared memory
                                                                        GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+2];
                                                                        GMem_pp_dbl_weightsTable_wei[inlet_ID][ii];

                        //---------------------
                        Evaluate a single value correction term for each Fluid site and
                                each of the directions (LB directions from 1 to 18, excluding the 0th LB dir)
                        i.e. what is below the value of correction:

                                LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));

                                distribn_t correction = 2. * c.EQMWEIGHTS[ii]
                                                * (wallMom.x * c.CX[ii]
                                                                + wallMom.y * c.CY[ii]
                                                                + wallMom.z * c.CZ[ii]) / Cs2;
                        //---------------------

                        Consider that the loop is going through the lattice points
                        hence, we need to be able to determine the iolet ID from the fluid index

                        TODO: pass the arr_elementsInEachInlet[index_inlet] to the GPU global memory
*/

//**************************************************************
/** GPU kernel for evaluating the wall momentum correction terms on the GPU
                1. 	For the Velocity BCs - LADDIOLET BCs
                1.1	Case 2: weights needed (useWeightsFromFile=true)
                                        Implement what is in the function
                                                LatticeVelocity InOutLetFileVelocity::GetVelocity
                                        Note that the arguments are: (halfWay, bValues->GetTimeStep())
                                                LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
                1.2. NO NEED TO SEARCH for the appropriate weight based on the coords (halfway) -

                                        Approach 1: Perform the search
                                        Approach 2: Read this information (which element to access)

                                Remove all the unnecessary elements (from the kernel GPU_WallMom_correction_File_Weights above copied below now)...

                                        Need to have the following information on the GPU:
                                        1. normal vector to the iolet
                                        2. LatticePosition& x : coordinates of the point
                                                        LatticePosition: Vector3D<double>
                                                        see function memCpy_HtD_GPUmem_Coords_Iolets which fills the following:
                                                                type int64_t [x_coord, y_coord, z_coord, ...]
                                                                void *GPUDataAddr_Coords_Inlet_Edge;
                                                                void *GPUDataAddr_Coords_InletWall_Edge;
                                                                void *GPUDataAddr_Coords_Inlet_Inner;
                                                                void *GPUDataAddr_Coords_InletWall_Inner;
                                                                void *GPUDataAddr_Coords_Outlet_Edge;
                                                                void *GPUDataAddr_Coords_OutletWall_Edge;
                                                                void *GPUDataAddr_Coords_Outlet_Inner;
                                                                void *GPUDataAddr_Coords_OutletWall_Inner;

                                        3. LatticeTimeStep t : The time-step (type unsigned long)
                                                                units.h:  typedef unsigned long LatticeTimeStep

                                        4. weights table
                                                                std::map<std::vector<int>, double> weights_table;

                                                        The CPU version gets the weight (mapped value) using the key value.
                                                        Here, we have 2 options:
                                                        4.1. Save the index of the key value (perform the search at Initialise_GPU)
                                                                                        and then save these indices so that I can easily obtain the
   corresponding weight (mapped value) without performing the search at each time-step switch to version 1 for this OR 4.2. Perform the search on the GPU (at
   each time) of the key value (xyz coords) in order to obtain the corresponding weight (mapped value) switch to version 0

                                        5. velocityTable[t]

                                        6. arr_elementsInEachInlet[index_inlet] - n_arr_elementsInCurrentInlet_weightsTable
                                                                number of elements (lines) in the weights table
                                                                Note there is 1 such table for each iolet

                                        7. Consider moving this info in shared memory
                                                                        GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+2];
                                                                        GMem_pp_dbl_weightsTable_wei[inlet_ID][ii];

                        //---------------------
                        Evaluate a single value correction term for each Fluid site and
                                each of the directions (LB directions from 1 to 18, excluding the 0th LB dir)
                        i.e. what is below the value of correction:

                                LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));

                                distribn_t correction = 2. * LatticeType::EQMWEIGHTS[ii]
                                                * (wallMom.x * LatticeType::c.CX[ii]
                                                                + wallMom.y * LatticeType::c.CY[ii]
                                                                + wallMom.z * LatticeType::c.CZ[ii]) / Cs2;
                        //---------------------

                        Consider that the loop is going through the lattice points
                        hence, we need to be able to determine the iolet ID from the fluid index

                        TODO: pass the arr_elementsInEachInlet[index_inlet] to the GPU global memory
*/
//**************************************************************

template <typename LatticeType> struct GPU_WallMom_correction_File_Weights_NoSearch_Functor {
  int64_t *GMem_Coords_iolets;
  int64_t **GMem_pp_int_weightsTable_coord;
  distribn_t **GMem_pp_dbl_weightsTable_wei;
  int64_t *GMem_index_key_weightTable;
  distribn_t *GMem_weightTable;
  distribn_t *GMem_dbl_WallMom;
  float *GMem_ioletNormal;
  uint32_t *GMem_uint32_Iolet_Link;
  int inlet_ID;
  distribn_t *GMem_Inlet_velocityTable;
  int n_arr_elementsInCurrentInlet_weightsTable;
  site_t start_Fluid_ID_givenColStreamType;
  site_t site_Count_givenColStreamType;
  site_t lower_limit;
  site_t upper_limit;
  unsigned long time_Step;
  unsigned long total_TimeSteps;
  const double Cs2;

  // Constructor
  GPU_WallMom_correction_File_Weights_NoSearch_Functor(int64_t *GMem_Coords_iolets_, int64_t **GMem_pp_int_weightsTable_coord_,
                                                       distribn_t **GMem_pp_dbl_weightsTable_wei_, int64_t *GMem_index_key_weightTable_,
                                                       distribn_t *GMem_weightTable_, distribn_t *GMem_dbl_WallMom_, float *GMem_ioletNormal_,
                                                       uint32_t *GMem_uint32_Iolet_Link_, int inlet_ID_, distribn_t *GMem_Inlet_velocityTable_,
                                                       int n_arr_elementsInCurrentInlet_weightsTable_, site_t start_Fluid_ID_givenColStreamType_,
                                                       site_t site_Count_givenColStreamType_, site_t lower_limit_, site_t upper_limit_,
                                                       unsigned long time_Step_, unsigned long total_TimeSteps_, double Cs2_)
      : GMem_Coords_iolets(GMem_Coords_iolets_), GMem_pp_int_weightsTable_coord(GMem_pp_int_weightsTable_coord_),
        GMem_pp_dbl_weightsTable_wei(GMem_pp_dbl_weightsTable_wei_), GMem_index_key_weightTable(GMem_index_key_weightTable_),
        GMem_weightTable(GMem_weightTable_), GMem_dbl_WallMom(GMem_dbl_WallMom_), GMem_ioletNormal(GMem_ioletNormal_),
        GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), inlet_ID(inlet_ID_), GMem_Inlet_velocityTable(GMem_Inlet_velocityTable_),
        n_arr_elementsInCurrentInlet_weightsTable(n_arr_elementsInCurrentInlet_weightsTable_),
        start_Fluid_ID_givenColStreamType(start_Fluid_ID_givenColStreamType_), site_Count_givenColStreamType(site_Count_givenColStreamType_),
        lower_limit(lower_limit_), upper_limit(upper_limit_), time_Step(time_Step_), total_TimeSteps(total_TimeSteps_), Cs2(Cs2_)  {}

  GPU_KERNEL void operator()(unsigned long long Ind) {
    Ind = Ind + lower_limit;
	const lb::lattices::D3Q19GPUConstants c;

    if (Ind >= (upper_limit + 1))
      return;

    // Done:
    /* Load the following info:
    1. Coordinates of the local point
                    Done!!! - Not needed with Approach 2
    2. normal vector to the iolet
                    in the arguments above add the following:
                    float* GMem_ioletNormal,
    3. Check why in the CPU code the normal is declared as double - not float
    4. Load the iolet link info (which direction has a link to an iolet)
    */

    // 1. Load the coordinates of the point for which we would like to evaluate the correction terms
    // Have in mind that (save registers per thread):
    // int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
    /* NOT NEEDED:
    int64_t x_coord = GMem_Coords_iolets[(Ind - start_Fluid_ID_givenColStreamType)*3];
    int64_t y_coord = GMem_Coords_iolets[(Ind - start_Fluid_ID_givenColStreamType)*3 + 1];
    int64_t z_coord = GMem_Coords_iolets[(Ind - start_Fluid_ID_givenColStreamType)*3 + 2];
    //printf("Fluid Index = %lld,  Shifted Index = %lld, Coordinates: (x, y, z) = (%lld, %lld, %lld) \n", Ind, shifted_Fluid_Ind, x_coord, y_coord, z_coord);
    //printf("Fluid Index = %lld, Coordinates: (x, y, z) = (%lld, %lld, %lld) \n", Ind, x_coord, y_coord, z_coord);
    */

    // Note that the normal vector components are of type float: float* GMem_ioletNormal
    double inletNormal_x = (double) GMem_ioletNormal[3 * inlet_ID];
    double inletNormal_y = (double) GMem_ioletNormal[3 * inlet_ID + 1];
    double inletNormal_z = (double) GMem_ioletNormal[3 * inlet_ID + 2];
    // printf("Inlet ID: %d, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", inlet_ID, inletNormal_x, inletNormal_y, inletNormal_z);

    //==========================================================================
    // Load the Iolet-Fluid links info
    uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

    // Here is the loop over the LB lattice directions
#pragma unroll 19
    for (int LB_Dir = 1; LB_Dir < c.NUMVECTORS; LB_Dir++)   // keep the loop from LB_Dir=1
    {
      unsigned mask = 1U << (LB_Dir - 1);   // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do:
                                            // compare against test_bool_Wall_Intersect as well)
      bool is_Iolet_link = (Iolet_Intersect & mask);

      distribn_t correction = 0.0;
      distribn_t vel_weight = 0.0;
      distribn_t vel_weight_1 = 0.0;   // Reading the index from the host (precalculated at Initialise_GPU)
      distribn_t vel_weight_2 = 0.0;   // Reading the actual vel weight (precalculated at Initialise_GPU)

      distribn_t max_vel = 0.0;
      distribn_t wallMom_x = 0.0;
      distribn_t wallMom_y = 0.0;
      distribn_t wallMom_z = 0.0;

      int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
      int index = shifted_Fluid_Ind * (c.NUMVECTORS - 1) + LB_Dir - 1;

      // Check which links are fluid-iolet links
      if (is_Iolet_link) {   // ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
                             // printf("Enters the loop with an iolet link - Fluid ID: %lld, LB_Dir = %d \n", Ind, LB_Dir);

                             // Step: LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
        /** Continue here what is in GetVelocity
                                Remember that the arguments are: GetVelocity(halfWay, bValues->GetTimeStep())
                                        LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
                                No need to Search for the halfWay xyz coordinates in the velocity weights table - Just read the index!
        */

        /*
        // Approach 2:
        // 	Obtain the index of the corresponding coordinates (xyz) in the weights_table
        // 	i.e. evaluate the index at initialisation, since it only depends on the
        //	 geometry and then read this at every time step to get the correction term
        int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
        int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

        int64_t index_key_weights_table = GMem_index_key_weightTable[index];
        //if(Ind==10025) printf("Ind: %lld, index_in_key_weightTable: %d, index_key_weights_table: %lld, INT_MAX: %lld \n", Ind, index, index_key_weights_table,
        INT64_MAX);

        // Corresponds to the cases that do not find the xyz coord in the loop: while (iterations < 3) in LatticeVelocity InOutLetFileVelocity::GetVelocity
        // In which case the correction term is set to 0
        if (index_key_weights_table==INT_MAX){
                //printf("No xyz coord match in function LatticeVelocity InOutLetFileVelocity::GetVelocity ... \n");
                correction = 0.0;
                //if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z:
        %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);
                //if(time_Step==1 && correction!=0) printf("GPU - Fluid shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long
        long)(LB_Dir
        - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind));

                int index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
                GMem_dbl_WallMom[index_wallMom_correction] = correction; // Maybe NOT needed - Check - TODO
                return;
        }

        if (index_key_weights_table==INT_MAX-1){
                printf("Shouldn't be in this loop ... Error with the iolet links and while getting the key index from GPU Global memory ... \n");
        }

        //------------------------------------------------------------------
        // Wall momentum returned:
        // 		v_tot = normal * weights_table.at(xyz) * velocityTable[t];

        // April 2023
        // Initial Approach that resulted in error for multi-inlet case
        vel_weight_1 = GMem_pp_dbl_weightsTable_wei[inlet_ID][index_key_weights_table];
        */
        // New Approach - Get the Vel - weight directly
        vel_weight_2 = GMem_weightTable[index];

        // if (time_Step==1) // && vel_weight_1!= vel_weight_2)
        //	printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight(1): %5.3e, vel_weight(2): %5.3e \n", Ind, LB_Dir, vel_weight_1, vel_weight_2);

        vel_weight = vel_weight_2;
        //------------------------------------------------------------------

        max_vel = GMem_Inlet_velocityTable[inlet_ID * (total_TimeSteps + 1) + time_Step];   // index_inlet*(total_TimeSteps+1)+timeStep

        wallMom_x = inletNormal_x * vel_weight * max_vel;
        wallMom_y = inletNormal_y * vel_weight * max_vel;
        wallMom_z = inletNormal_z * vel_weight * max_vel;

        //------------------------------------------------------------------
        // C. Step: Evaluate the single correction term as
        correction = 2. * c.EQMWEIGHTS[LB_Dir] *
                     (wallMom_x * (double)c.CX[LB_Dir] + wallMom_y * (double)c.CY[LB_Dir] + wallMom_z * (double)c.CZ[LB_Dir]) / Cs2;

        // if (time_Step==1000 && correction!=0 && Ind==10065) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e,
        // wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);

        // TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
        // Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
        // 	*** Do that inside the collision-streaming kernels Instead ***
        //		correction *= nn;
        //==================================================================

        //
        // Save the correction term in GPU global memory
        // TODO: Need to pass the:
        //		siteCount and shifted_Fluid_Ind
        // siteCount is site_Count_Inlet_Inner, site_Count_InletWall_Inner etc
        // shifted_Fluid_Ind is
        // int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

        // if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e,
        // correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction); if(time_Step==1 && correction!=0) printf("GPU - Fluid
        // shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType +
        // shifted_Fluid_Ind));

        int index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
        GMem_dbl_WallMom[index_wallMom_correction] = correction;
        //
      }   // Closes the loop if(is_Iolet_link)

          // if (time_Step==1 && correction!=0)
      // printf("GPU - Fluid ID: %lld, LB-Dir: %d, correction: %5.3e \n", Ind, LB_Dir, correction);

    }   // ends the loop over the LB_Dir directions
        //==========================================================================

  }     // Ends the GPU kernel
};      // End of Functor

template <typename LatticeType> struct GPU_WallMom_correction_File_prefactor_Functor {

  distribn_t *GMem_dbl_wallMom_prefactor_correction;
  distribn_t *GMem_dbl_WallMom;
  uint32_t *GMem_uint32_Iolet_Link;
  int num_local_Iolets;
  site_t *GMem_Iolets_info;
  distribn_t *GMem_Inlet_velocityTable;
  site_t start_Fluid_ID_givenColStreamType;
  site_t site_Count_givenColStreamType;
  site_t lower_limit;
  site_t upper_limit;
  unsigned long time_Step;
  unsigned long total_TimeSteps;

  GPU_WallMom_correction_File_prefactor_Functor(distribn_t *GMem_dbl_wallMom_prefactor_correction_, distribn_t *GMem_dbl_WallMom_,
                                                uint32_t *GMem_uint32_Iolet_Link_, int num_local_Iolets_, site_t *GMem_Iolets_info_,
                                                distribn_t *GMem_Inlet_velocityTable_, site_t start_Fluid_ID_givenColStreamType_,
                                                site_t site_Count_givenColStreamType_, site_t lower_limit_, site_t upper_limit_, unsigned long time_Step_,
                                                unsigned long total_TimeSteps_)
      : GMem_dbl_wallMom_prefactor_correction(GMem_dbl_wallMom_prefactor_correction_), GMem_dbl_WallMom(GMem_dbl_WallMom_),
        GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), num_local_Iolets(num_local_Iolets_), GMem_Iolets_info(GMem_Iolets_info_),
        GMem_Inlet_velocityTable(GMem_Inlet_velocityTable_), start_Fluid_ID_givenColStreamType(start_Fluid_ID_givenColStreamType_),
        site_Count_givenColStreamType(site_Count_givenColStreamType_), lower_limit(lower_limit_), upper_limit(upper_limit_), time_Step(time_Step_),
        total_TimeSteps(total_TimeSteps_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) {
    Ind = Ind + lower_limit;
	const lb::lattices::D3Q19GPUConstants c;
    if (Ind >= upper_limit)
      return;

    /** Load the following info:
                            I. 		Velocity table Information - Requires iolet ID
                            II. 	Identify the local iolet (to be used in the velocity table)
                            III. 	Iolet link info (which direction has a link to an iolet)
                            IV. 	WallMom prefactor correction term
    */

    //--------------------------------------------------------------------------
    // II. 	Identify the local iolet
    //  		There are 2 possible ways:
    // 			1. Using the information from GPU global mem (GMem_Iolets_info)
    //			2. Using struct Iolets containing the info (when number of iolets less than 30)

    int IdInlet = INT32_MAX;   // Iolet (Inlet/Outlet) ID
    if (num_local_Iolets == 1) {
      // Approach 1 - from GPU global mem (GMem_Iolets_info)
      IdInlet = GMem_Iolets_info[0];
      // Approach 2 - from struct array
      // IdInlet = Iolets_info.Iolets_ID_range[0];
    } else {
      // Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
      // iolets_ID_range Array:
      //	a. Size: num_local_Iolets * 3
      //	Contains the following (in the order below)
      // 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit)

      // Approach 1 - from GPU global mem (GMem_Iolets_info)
      _determine_Iolet_ID(num_local_Iolets, GMem_Iolets_info, Ind, &IdInlet);

      // Approach 2 - from struct array
      // _determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet);
    }

    // Debugging:
    if (IdInlet == INT32_MAX) {
      printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! FAILURE!!! Needs to abort... \n\n", Ind, IdInlet);
    }
    /*else{
            printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
    }*/
    //--------------------------------------------------------------------------

    //==========================================================================
    // III. Load the Iolet-Fluid links info
    uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

    // Here is the loop over the LB lattice directions
#pragma unroll 19
    for (int LB_Dir = 1; LB_Dir < c.NUMVECTORS; LB_Dir++)   // keep the loop from LB_Dir=1
    {
      unsigned mask = 1U << (LB_Dir - 1);   // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do:
                                            // compare against test_bool_Wall_Intersect as well)
      bool is_Iolet_link = (Iolet_Intersect & mask);

      distribn_t correction = 0.0;
      distribn_t max_vel = 0.0;   // Max Velocity to be read from the velocityTable(IdInlet,t)

      int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
      // int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

      // Check which links are fluid-iolet links
      if (is_Iolet_link) {
        // printf("Enters the loop with an iolet link - Fluid ID: %lld, LB_Dir = %d \n", Ind, LB_Dir);

        // A. Step: Load the prefactor correction term
        //	1. TODO
        site_t index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
        distribn_t prefactor_correction = GMem_dbl_wallMom_prefactor_correction[index_wallMom_correction];

        // Note that the prefactor contains the info (see Eval_wallMom_prefactor_correction in LaddIoletDelegate)
        /**
                        LatticeVelocity wallMom_prefactor(iolet->GetVelocity_prefactor(halfWay, bValues->GetTimeStep()));

                        distribn_t prefactor_correction = 2. * LatticeType::EQMWEIGHTS[ii]
        * (wallMom_prefactor.x * LatticeType::c.CX[ii] + wallMom_prefactor.y * LatticeType::c.CY[ii]
  + wallMom_prefactor.z * LatticeType::c.CZ[ii]) / Cs2;
        */

        // Just multiply with max Velocity(IdInlet,t) from velocityTable
        // 	Load max Vel
        max_vel = GMem_Inlet_velocityTable[IdInlet * (total_TimeSteps + 1) + time_Step];   // index_inlet*(total_TimeSteps+1)+timeStep

        // B. Step: Evaluate the single correction term as
        correction = prefactor_correction * max_vel;

        /*wallMom_x = inletNormal_x * vel_weight * max_vel;
        wallMom_y = inletNormal_y * vel_weight * max_vel;
        wallMom_z = inletNormal_z * vel_weight * max_vel;

        //------------------------------------------------------------------
        // C. Step: Evaluate the single correction term as
        correction = 2. * _EQMWEIGHTS_19[LB_Dir]
        * (	wallMom_x * (double)_c.CX_19[LB_Dir] +
                                                                        wallMom_y * (double)_c.CY_19[LB_Dir] +
            wallMom_z * (double)_c.CZ_19[LB_Dir]) / _Cs2;
        */

        // if (time_Step==1000 && correction!=0 && Ind==10065) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e,
        // wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);

        // TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
        // Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
        // 	*** Do that inside the collision-streaming kernels Instead ***
        //		correction *= nn;
        //==================================================================

        //
        // Save the correction term in GPU global memory
        // TODO: Need to pass the:
        //		siteCount and shifted_Fluid_Ind
        // siteCount is site_Count_Inlet_Inner, site_Count_InletWall_Inner etc
        // shifted_Fluid_Ind is
        // int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

        // if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e,
        // correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction); if(time_Step==1 && correction!=0) printf("GPU - Fluid
        // shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType +
        // shifted_Fluid_Ind));

        GMem_dbl_WallMom[index_wallMom_correction] = correction;
        //
      }   // Closes the loop if(is_Iolet_link)

          // if (time_Step==1 && correction!=0)
      // printf("GPU - Fluid ID: %lld, LB-Dir: %d, correction: %5.3e \n", Ind, LB_Dir, correction);

    }   // ends the loop over the LB_Dir directions
        //==========================================================================

  }     // Ends the GPU kernel
};      // End of functor

template <typename LatticeType> struct GPU_WallMom_correction_File_prefactor_v2_Functor {
  distribn_t *GMem_dbl_wallMom_prefactor_correction;
  distribn_t *GMem_dbl_WallMom;
  uint32_t *GMem_uint32_Iolet_Link;
  int num_local_Iolets;
  Iolets Iolets_info;
  distribn_t *GMem_Inlet_velocityTable;
  site_t start_Fluid_ID_givenColStreamType;
  site_t site_Count_givenColStreamType;
  site_t lower_limit;
  site_t upper_limit;
  unsigned long time_Step;
  unsigned long total_TimeSteps;

  GPU_WallMom_correction_File_prefactor_v2_Functor(distribn_t *GMem_dbl_wallMom_prefactor_correction_, distribn_t *GMem_dbl_WallMom_,
                                                   uint32_t *GMem_uint32_Iolet_Link_, int num_local_Iolets_, Iolets Iolets_info_,
                                                   distribn_t *GMem_Inlet_velocityTable_, site_t start_Fluid_ID_givenColStreamType_,
                                                   site_t site_Count_givenColStreamType_, site_t lower_limit_, site_t upper_limit_, unsigned long time_Step_,
                                                   unsigned long total_TimeSteps_)
      : GMem_dbl_wallMom_prefactor_correction(GMem_dbl_wallMom_prefactor_correction_), GMem_dbl_WallMom(GMem_dbl_WallMom_),
        GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), num_local_Iolets(num_local_Iolets_), Iolets_info(Iolets_info_),
        GMem_Inlet_velocityTable(GMem_Inlet_velocityTable_), start_Fluid_ID_givenColStreamType(start_Fluid_ID_givenColStreamType_),
        site_Count_givenColStreamType(site_Count_givenColStreamType_), lower_limit(lower_limit_), upper_limit(upper_limit_), time_Step(time_Step_),
        total_TimeSteps(total_TimeSteps_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) {

    Ind = Ind + lower_limit;
	const lb::lattices::D3Q19GPUConstants c;

    if (Ind >= upper_limit)
      return;

    /** Load the following info:
                            I. 		Velocity table Information - Requires iolet ID
                            II. 	Identify the local iolet (to be used in the velocity table)
                            III. 	Iolet link info (which direction has a link to an iolet)
                            IV. 	WallMom prefactor correction term
    */

    //--------------------------------------------------------------------------
    // II. 	Identify the local iolet
    //  		There are 2 possible ways:
    // 			1. Using the information from GPU global mem.
    //			2. Using struct Iolets containing the info (when number of iolets less than 30)

    /*
    // Approach 1
    // Read the Iolet info (iolet ids and fluid sites range) from GMem_Iolets_info
    site_t *Iolet_info = new site_t[3*num_local_Iolets];
    for (int index = 0; index< (3*num_local_Iolets); index++)
    {
            Iolet_info[index] = GMem_Iolets_info[index];
    }

    // Determine the IdInlet - Done!!!
    int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
    if(num_local_Iolets==1){
            IdInlet = Iolet_info[0]; //Iolets_info.Iolets_ID_range[0];// IdInlet = iolets_ID_range[0];
    }
    else{
            // Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
            // iolets_ID_range Array:
            //	a. Size: num_local_Iolets * 3
            // 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit)
            // TODO: Replace this: _determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet); // _determine_Iolet_ID(num_local_Iolets,
    iolets_ID_range, Ind, &IdInlet); _determine_Iolet_ID(num_local_Iolets, Iolet_info, Ind, &IdInlet);
    }
    */
    //
    // Approach 2 (Should be Faster - consider testing this):
    // Access the info from the GPU's constant memory: _Iolets_Inlet_Inner[local_iolets_MaxSIZE], local_iolets_MaxSIZE = 90 cuda_params.h (Assume 30 max iolets
    // per RANK) Determine the IdInlet - Done!!!
    int IdInlet = INT32_MAX;                      // Iolet (Inlet/Outlet) ID
    if (num_local_Iolets == 1) {
      IdInlet = Iolets_info.Iolets_ID_range[0];   // IdInlet = iolets_ID_range[0];
    } else {
      // Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
      // iolets_ID_range Array:
      //	a. Size: num_local_Iolets * 3
      // 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit)
      _determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind,
                          &IdInlet);   // _determine_Iolet_ID(num_local_Iolets, iolets_ID_range, Ind, &IdInlet);
    }
    //

    // Testing:
    if (IdInlet == INT32_MAX) {
      printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
    }
    /*else{
            printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
    }*/
    //--------------------------------------------------------------------------

    //==========================================================================
    // III. Load the Iolet-Fluid links info
    uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

    // Here is the loop over the LB lattice directions
#pragma unroll 19
    for (int LB_Dir = 1; LB_Dir < c.NUMVECTORS; LB_Dir++)   // keep the loop from LB_Dir=1
    {
      unsigned mask = 1U << (LB_Dir - 1);   // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do:
                                            // compare against test_bool_Wall_Intersect as well)
      bool is_Iolet_link = (Iolet_Intersect & mask);

      distribn_t correction = 0.0;
      distribn_t max_vel = 0.0;   // Max Velocity to be read from the velocityTable(IdInlet,t)

      int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
      // int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

      // Check which links are fluid-iolet links
      if (is_Iolet_link) {
        // printf("Enters the loop with an iolet link - Fluid ID: %lld, LB_Dir = %d \n", Ind, LB_Dir);

        // A. Step: Load the prefactor correction term
        //	1. TODO
        site_t index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
        distribn_t prefactor_correction = GMem_dbl_wallMom_prefactor_correction[index_wallMom_correction];

        // Note that the prefactor contains the info (see Eval_wallMom_prefactor_correction in LaddIoletDelegate)
        /**
                        LatticeVelocity wallMom_prefactor(iolet->GetVelocity_prefactor(halfWay, bValues->GetTimeStep()));

                        distribn_t prefactor_correction = 2. * LatticeType::EQMWEIGHTS[ii]
        * (wallMom_prefactor.x * LatticeType::c.CX[ii] + wallMom_prefactor.y * LatticeType::c.CY[ii]
  + wallMom_prefactor.z * LatticeType::c.CZ[ii]) / Cs2;
        */

        // Just multiply with max Velocity(IdInlet,t) from velocityTable
        // 	Load max Vel
        max_vel = GMem_Inlet_velocityTable[IdInlet * (total_TimeSteps + 1) + time_Step];   // index_inlet*(total_TimeSteps+1)+timeStep

        // B. Step: Evaluate the single correction term as
        correction = prefactor_correction * max_vel;

        /*wallMom_x = inletNormal_x * vel_weight * max_vel;
        wallMom_y = inletNormal_y * vel_weight * max_vel;
        wallMom_z = inletNormal_z * vel_weight * max_vel;

        //------------------------------------------------------------------
        // C. Step: Evaluate the single correction term as
        correction = 2. * _EQMWEIGHTS_19[LB_Dir]
        * (	wallMom_x * (double)_c.CX_19[LB_Dir] +
                                                                        wallMom_y * (double)_c.CY_19[LB_Dir] +
            wallMom_z * (double)_c.CZ_19[LB_Dir]) / _Cs2;
        */

        // if (time_Step==1000 && correction!=0 && Ind==10065) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e,
        // wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);

        // TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
        // Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
        // 	*** Do that inside the collision-streaming kernels Instead ***
        //		correction *= nn;
        //==================================================================

        //
        // Save the correction term in GPU global memory
        // TODO: Need to pass the:
        //		siteCount and shifted_Fluid_Ind
        // siteCount is site_Count_Inlet_Inner, site_Count_InletWall_Inner etc
        // shifted_Fluid_Ind is
        // int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

        // if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e,
        // correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction); if(time_Step==1 && correction!=0) printf("GPU - Fluid
        // shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType +
        // shifted_Fluid_Ind));

        GMem_dbl_WallMom[index_wallMom_correction] = correction;
        //
      }   // Closes the loop if(is_Iolet_link)

          // if (time_Step==1 && correction!=0)
      // printf("GPU - Fluid ID: %lld, LB-Dir: %d, correction: %5.3e \n", Ind, LB_Dir, correction);

    }   // ends the loop over the LB_Dir directions
        //==========================================================================

  }     // Ends the GPU kernel
};      // End of functor

//**************************************************************
/* April 2023
                New kernel for evaluating the wall momentum corection terms
                        on the GPU.

                Approach:
                        A. Use the geometric prefactor associated with the wall momentum
                                        correction terms.
                                        See ...
                        B. Just multiply with the maximum velocity(t) for the Corresponding
                                                iolet from the velocity table

                Identify the iolet index from the fluid index and info passed as an argument to the GPU kernel
                        Iolets Iolets_info containing:
                iolet ID and fluid sites range

                Consider moving the above information in GPU constant memory
*/
//**************************************************************
//**************************************************************
template <typename LatticeType> struct GPU_WallMom_correction_File_prefactor_NoIoletIDSearch_Functor {
  distribn_t *GMem_dbl_wallMom_prefactor_correction;
  distribn_t *GMem_dbl_WallMom;
  uint32_t *GMem_uint32_Iolet_Link;
  int IdInlet;
  distribn_t *GMem_Inlet_velocityTable;
  site_t start_Fluid_ID_givenColStreamType;
  site_t site_Count_givenColStreamType;
  site_t lower_limit;
  site_t upper_limit;
  unsigned long time_Step;
  unsigned long total_TimeSteps;

  GPU_WallMom_correction_File_prefactor_NoIoletIDSearch_Functor(distribn_t *GMem_dbl_wallMom_prefactor_correction_, distribn_t *GMem_dbl_WallMom_,
                                                                uint32_t *GMem_uint32_Iolet_Link_, int IdInlet_, distribn_t *GMem_Inlet_velocityTable_,
                                                                site_t start_Fluid_ID_givenColStreamType_, site_t site_Count_givenColStreamType_,
                                                                site_t lower_limit_, site_t upper_limit_, unsigned long time_Step_,
                                                                unsigned long total_TimeSteps_)
      : GMem_dbl_wallMom_prefactor_correction(GMem_dbl_wallMom_prefactor_correction_), GMem_dbl_WallMom(GMem_dbl_WallMom_),
        GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), IdInlet(IdInlet_), GMem_Inlet_velocityTable(GMem_Inlet_velocityTable_),
        start_Fluid_ID_givenColStreamType(start_Fluid_ID_givenColStreamType_), site_Count_givenColStreamType(site_Count_givenColStreamType_),
        lower_limit(lower_limit_), upper_limit(upper_limit_), time_Step(time_Step_), total_TimeSteps(total_TimeSteps_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) {
	const lb::lattices::D3Q19GPUConstants c;
    Ind = Ind + lower_limit;

    if (Ind >= upper_limit)
      return;

    /** Load the following info:
                            I. 		Velocity table Information - Requires iolet ID
                            II. 	local iolet ID provided in the GPU kernel's arguments now (to be used in the velocity table)
                            III. 	Iolet link info (which direction has a link to an iolet)
                            IV. 	WallMom prefactor correction term
    */

    //==========================================================================
    // III. Load the Iolet-Fluid links info
    uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

    // Here is the loop over the LB lattice directions
#pragma unroll 19
    for (int LB_Dir = 1; LB_Dir < c.NUMVECTORS; LB_Dir++)   // keep the loop from LB_Dir=1
    {
      unsigned mask = 1U << (LB_Dir - 1);   // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To
                                            // do: compare against test_bool_Wall_Intersect as well)
      bool is_Iolet_link = (Iolet_Intersect & mask);

      // Check which links are fluid-iolet links
      if (is_Iolet_link) {
        // printf("Enters the loop with an iolet link - Fluid ID: %lld, LB_Dir = %d \n", Ind, LB_Dir);

        // distribn_t correction = 0.0;
        // distribn_t max_vel=0.0; // Max Velocity to be read from the velocityTable(IdInlet,t)

        int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
        // int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

        // A. Step: Load the prefactor correction term
        site_t index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
        distribn_t prefactor_correction = GMem_dbl_wallMom_prefactor_correction[index_wallMom_correction];

        // Note that the prefactor contains the info (see Eval_wallMom_prefactor_correction in LaddIoletDelegate)
        /**
                        LatticeVelocity wallMom_prefactor(iolet->GetVelocity_prefactor(halfWay, bValues->GetTimeStep()));

                        distribn_t prefactor_correction = 2. * LatticeType::EQMWEIGHTS[ii]
        * (wallMom_prefactor.x * LatticeType::c.CX[ii] + wallMom_prefactor.y * LatticeType::c.CY[ii]
  + wallMom_prefactor.z * LatticeType::c.CZ[ii]) / Cs2;
        */

        // Just multiply with max Velocity(IdInlet,t) from velocityTable
        // 	Load max Vel
        distribn_t max_vel = GMem_Inlet_velocityTable[IdInlet * (total_TimeSteps + 1) + time_Step];   // index_inlet*(total_TimeSteps+1)+timeStep

        // B. Step: Evaluate the single correction term as
        distribn_t correction = prefactor_correction * max_vel;

        // TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
        // Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
        // 	*** Do that inside the collision-streaming kernels Instead ***
        //		correction *= nn;
        //==================================================================

        //
        // Save the correction term in GPU global memory
        // TODO: Need to pass the:
        //		siteCount and shifted_Fluid_Ind
        // siteCount is site_Count_Inlet_Inner, site_Count_InletWall_Inner etc
        // shifted_Fluid_Ind is
        // int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

        // if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e,
        // correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction); if(time_Step==1 && correction!=0) printf("GPU - Fluid
        // shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType +
        // shifted_Fluid_Ind));

        GMem_dbl_WallMom[index_wallMom_correction] = correction;
        //
      }   // Closes the loop if(is_Iolet_link)

          // if (time_Step==1 && correction!=0)
      // printf("GPU - Fluid ID: %lld, LB-Dir: %d, correction: %5.3e \n", Ind, LB_Dir, correction);

    }   // ends the loop over the LB_Dir directions
        //==========================================================================

  }     // Ends the GPU kernel
};      // End of the Functor

// Not sure if this kernel is currently functional. It is commented out in lb.hpp a lot
struct GPU_Check_Coordinates_Functor {

  int64_t *GMem_Coords_iolets;
  site_t start_Fluid_ID_givenColStreamType;
  site_t lower_limit;
  site_t upper_limit;

  GPU_Check_Coordinates_Functor(int64_t *GMem_Coords_iolets_, site_t start_Fluid_ID_givenColStreamType_, site_t lower_limit_, site_t upper_limit_)
      : GMem_Coords_iolets(GMem_Coords_iolets_), start_Fluid_ID_givenColStreamType(start_Fluid_ID_givenColStreamType_), lower_limit(lower_limit_),
        upper_limit(upper_limit_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) {

    // unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
    // Ind = Ind + lower_limit;

    // if(Ind >= upper_limit)
    // return;
    printf("Enter GPU_Check_Coordinates kernel... \n");
    for (int64_t Ind = lower_limit; Ind <= upper_limit; Ind++) {   // TODO: Check the limits

      // 1. Load the coordinates of the point for which we would like tp evaluate the correction terms
      // Have in mind that (save registers per thread):
      int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

      // Address is Misaligned (threads #0 to #31)
      int64_t x_coord = GMem_Coords_iolets[shifted_Fluid_Ind * 3];
      int64_t y_coord = GMem_Coords_iolets[shifted_Fluid_Ind * 3 + 1];
      int64_t z_coord = GMem_Coords_iolets[shifted_Fluid_Ind * 3 + 2];

      // printf("Inside GPU kernel - Fluid Index = %lld, start_Fluid_ID_givenColStreamType = %lld, Shifted Index = %lld \n", Ind,
      // start_Fluid_ID_givenColStreamType, shifted_Fluid_Ind);
      printf("Test coords kernel - Fluid Index = %lld, Shifted Index = %lld, Coordinates: (x, y, z) = (%lld, %lld, %lld) \n", Ind, shifted_Fluid_Ind, x_coord,
             y_coord, z_coord);
    }
  }
};   // End of Functor

//**************************************************************
/** Kernel for assessing the stability of the code
                Remember that the enum Stability is defined in SimulationState.h:
                                                enum Stability
                                                {
                                                        UndefinedStability = -1,
                                                        Unstable = 0,
                                                        Stable = 1,
                                                        StableAndConverged = 2
                                                };
                Initial value set to UndefinedStability(i.e. -1).

                *** CRITERION ***
                The kernel assesses the stability by:
                1. Examining whether f_new > 0.0
                                        SAME approach as the CPU version of hemeLB
                2. Consider in the future checking for NaNs values (maybe just the density will suffice)

                If unstable (see criterion above):
                        flag d_Stability_flag set to 0 (global memory int*).
*/
//**************************************************************

template <typename LatticeType> struct GPU_Check_Stability_Functor {

  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  int *d_Stability_flag;
  site_t nArr_dbl;
  site_t lower_limit;
  site_t upper_limit;
  int time_Step;

  GPU_Check_Stability_Functor(distribn_t *GMem_dbl_fOld_b_, distribn_t *GMem_dbl_fNew_b_, int *d_Stability_flag_, site_t nArr_dbl_, site_t lower_limit_,
                              site_t upper_limit_, int time_Step_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), d_Stability_flag(d_Stability_flag_), nArr_dbl(nArr_dbl_),
        lower_limit(lower_limit_), upper_limit(upper_limit_), time_Step(time_Step_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) {
	const lb::lattices::D3Q19GPUConstants c;
    Ind = Ind + lower_limit;

    if (Ind >= upper_limit)
      return;

    int Stability_GPU = *d_Stability_flag;
    // printf("Site ID = %lld - Stability flag: %d \n\n", Ind, Stability_GPU);

    /** At first, follow the same approach as in the CPU version of hemeLB,
                    i.e. examine whether the distribution functions are positive, see lb/StabilityTester.h
                    //--------------------------------------------------------------------
                    Also, see SimulationState.h for the enum Stability:
                    namespace lb
            {
            enum Stability
            {
                  UndefinedStability = -1,
                  Unstable = 0,
                  Stable = 1,
                  StableAndConverged = 2
                };
              }
                    //--------------------------------------------------------------------
    // Note that by testing for value > 0.0, we also catch stray NaNs.
    if (! (value > 0.0))
    {
            mUpwardsStability = Unstable;
            break;
    }
    */

    // Load the distribution functions fNew_GPU_b[19]
    // distribn_t dev_ff_new[19];

    for (int direction = 0; direction < c.NUMVECTORS; direction++) {
      distribn_t ff = GMem_dbl_fNew_b[(unsigned long long) direction * nArr_dbl + Ind];
      // dev_ff_new[direction] = ff;
      if (!(ff > 0.0))   // Unstable simulation
      {
        Stability_GPU = 0;
        *d_Stability_flag = 0;
        return;
      }
      if (Stability_GPU == 0)
        return;

    }   // Ends the loop over the LB-directions

    // Debugging test
    // if(time_Step%200 ==0) *d_Stability_flag = 0;

  }   // Ends the kernel GPU_Check_Stability

};    // End of functor

template <typename LatticeType> struct GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_Functor {

  distribn_t *GMem_dbl_fOld_b;       // read
  distribn_t *GMem_dbl_fNew_b;       // write
  distribn_t *GMem_dbl_MacroVars;    // write
  site_t *GMem_int64_Neigh;          // read
  uint32_t *GMem_uint32_Wall_Link;   // unused
  site_t nArr_dbl;
  site_t lower_limit_MidFluid;
  site_t upper_limit_MidFluid;
  site_t lower_limit_Wall;
  site_t upper_limit_Wall;
  site_t totalSharedFs;
  bool write_GlobalMem;
  double minusInvTau;
  int myPiD;

  GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_Functor(distribn_t *GMem_dbl_fOld_b_,       // read
                                                                  distribn_t *GMem_dbl_fNew_b_,       // write
                                                                  distribn_t *GMem_dbl_MacroVars_,    // write
                                                                  site_t *GMem_int64_Neigh_,          // read
                                                                  uint32_t *GMem_uint32_Wall_Link_,   // unused
                                                                  site_t nArr_dbl_, site_t lower_limit_MidFluid_, site_t upper_limit_MidFluid_,
                                                                  site_t lower_limit_Wall_, site_t upper_limit_Wall_, site_t totalSharedFs_,
                                                                  bool write_GlobalMem_, double minusInvTau_, int myPiD_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_),               // read
        GMem_dbl_fNew_b(GMem_dbl_fNew_b_),               // write
        GMem_dbl_MacroVars(GMem_dbl_MacroVars_),         // write
        GMem_int64_Neigh(GMem_int64_Neigh_),             // read
        GMem_uint32_Wall_Link(GMem_uint32_Wall_Link_),   // unused
        nArr_dbl(nArr_dbl_), lower_limit_MidFluid(lower_limit_MidFluid_), upper_limit_MidFluid(upper_limit_MidFluid_), lower_limit_Wall(lower_limit_Wall_),
        upper_limit_Wall(upper_limit_Wall_), totalSharedFs(totalSharedFs_), write_GlobalMem(write_GlobalMem_), minusInvTau(minusInvTau_),  myPiD(myPiD_){}

  GPU_KERNEL void operator()(unsigned long long Ind) {
	const lb::lattices::D3Q19GPUConstants c;
  Ind = Ind + lower_limit_MidFluid;

  if (Ind >= upper_limit_Wall)
    return;

  double dev_ff[19];   //, dev_fEq[19];
  double nn = 0.0;     // density
  double momentum_x, momentum_y, momentum_z;
  momentum_x = momentum_y = momentum_z = 0.0;

  double velx, vely, velz;   // Fluid Velocity

  //-----------------------------------------------------------------------------------------------------------
  // 1. Read the fOld_GPU_b distr. functions
  // 2. Calculate the nessessary elements for calculating the equilibrium distribution functions
  // 		a. Calculate density
  // 		b. Calculate momentum - Note: No body forces

#pragma unroll 19
  for (int direction = 0; direction < c.NUMVECTORS; direction++) {
    double ff = GMem_dbl_fOld_b[(unsigned long long) direction * nArr_dbl + Ind];
    dev_ff[direction] = ff;
    nn += ff;

    // Shows a lower number of registers per thread (51) compared to the the explicit method below!!!
    momentum_x += (double) c.CX[direction] * ff;
    momentum_y += (double) c.CY[direction] * ff;
    momentum_z += (double) c.CZ[direction] * ff;
  }

  double density_1 = 1.0 / nn;

  //-----------------------------------------------------------------------------------------------------------
  // c. Calculate equilibrium distr. functions

  // double momentumMagnitudeSquared = momentum_x * momentum_x
  //											+ momentum_y * momentum_y + momentum_z * momentum_z;

  double f_neq[19];
#pragma unroll 19
  for (int i = 0; i < c.NUMVECTORS; ++i) {
    double mom_dot_ei = (double) c.CX[i] * momentum_x + (double) c.CY[i] * momentum_y + (double) c.CZ[i] * momentum_z;

    double dev_fEq = c.EQMWEIGHTS[i] * (nn - (3.0 / 2.0) * (momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z) * density_1 +
                                          (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

    f_neq[i] = dev_ff[i] - dev_fEq;
    dev_ff[i] += (dev_ff[i] - dev_fEq) * minusInvTau;
  }


  // --------------------------------------------------------------------------------
  // Streaming Step:
  // a. Load the streaming indices

	  		// modifies: mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
	  		//  		 GPUDataAddr_dbl_MacroVars
  // b. If within the limits for the mWallCollision
  //		LOAD the Wall-Fluid links info - Remember that this is done for all the fluid nNodes
  //		Memory allocation in the future must be restricted to just the fluid nodes next to walls (i.e. the siteCount involved)

  site_t index_wall = nArr_dbl * c.NUMVECTORS;   // typedef int64_t site_t;

  GMem_dbl_fNew_b[Ind] = dev_ff[0];

#pragma unroll 18
  for (int LB_Dir = 1; LB_Dir < c.NUMVECTORS; LB_Dir++) {
    int64_t dev_NeighInd =
        GMem_int64_Neigh[(unsigned long long) LB_Dir * nArr_dbl + Ind];   // Neighbouring index refers to the index to be streamed to in the global memory. Here
                                                                          // it Refers to Data Address NOT THE STREAMING FLUID ID!!!

    // Is there a performance gain in choosing Option 1 over Option 2 or Option 3 below???
    // Option 1:
    if (dev_NeighInd == index_wall)   // Wall Link
    {
      // Simple Bounce Back case:
      GMem_dbl_fNew_b[(unsigned long long) c.INVERSEDIRECTIONS[LB_Dir] * nArr_dbl + Ind] = dev_ff[LB_Dir];   // Bounce Back - Same fluid ID - Reverse LB_Dir
    } else {
      GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir];                                                      // If neigh_d is selected
    }

  }

  //=============================================================================================
  // Write old density and velocity to memory -
  // if (time_Step%_Send_MacroVars_DtH ==0){
  if (write_GlobalMem) {
    GMem_dbl_MacroVars[Ind] = nn;

    velx = momentum_x * density_1;
    vely = momentum_y * density_1;
    velz = momentum_z * density_1;

    GMem_dbl_MacroVars[1ULL * nArr_dbl + Ind] = velx;
    GMem_dbl_MacroVars[2ULL * nArr_dbl + Ind] = vely;
    GMem_dbl_MacroVars[3ULL * nArr_dbl + Ind] = velz;
  }

    //==========================================================================================
  }    // Ends the merged kernels GPU_Collide Types 1 & 2: mMidFluidCollision & mWallCollision
};    // End of functor

//==========================================================================================
// Save the fNew post-collision distribution functions in the fOld array
// Each thread is responsible for reading the fNew_GPU_b distr. functions for a lattice fluid node
// i.e. the range for this kernel should be [0, nFluid_nodes) -
// ***	 Does not swap the totalSharedFs distr. *** //
// and then saves these values in fOld_GPU_b.
// Check the discussion here:
// https://stackoverflow.com/questions/22345391/cuda-device-memory-copies-cudamemcpydevicetodevice-vs-copy-kernel
//==========================================================================================

// Not sure if this kernel is in use right now...
template <typename LatticeType> struct GPU_SwapOldAndNew_Functor {

  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  site_t nArr_dbl;
  site_t lower_limit;
  site_t upper_limit;

  GPU_SwapOldAndNew_Functor(distribn_t *__restrict__ GMem_dbl_fOld_b_, distribn_t *__restrict__ GMem_dbl_fNew_b_, site_t nArr_dbl_, site_t lower_limit_,
                            site_t upper_limit_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), nArr_dbl(nArr_dbl_), lower_limit(lower_limit_), upper_limit(upper_limit_) {}

  GPU_KERNEL void operator()(unsigned long long Ind, unsigned long long Stride) {
   const lb::lattices::D3Q19GPUConstants c;
    Ind = Ind + lower_limit;

    if (Ind >= upper_limit)
      return;

    for (int unsigned long long Index = Ind; Index < upper_limit; Index += Stride) {
      // Just copy the populations  - fNew in fOld
      // Read in the fNew[19][Ind] and copy to fOld[19][Ind]
      for (int i = 0; i < c.NUMVECTORS; i++) {
        GMem_dbl_fOld_b[(unsigned long long) i * nArr_dbl + Index] = GMem_dbl_fNew_b[(unsigned long long) i * nArr_dbl + Index];
      }
    }

  }   // Ends the GPU_SwapOldAndNew kernel
  //==========================================================================================
};   // End functor

//==========================================================================================
// GPU kernel to do the appropriate re-allocation of the received distr. functions
// placed in totalSharedFs in fOld in the RECEIVING rank (host-to-device memcpy preceded this kernel)
// into the destination buffer "f_new"
// using the streamingIndicesForReceivedDistributions (GPUDataAddr_int64_streamInd)
// 		see: *GetFNew(streamingIndicesForReceivedDistributions[i]) = *GetFOld(neighbouringProcs[0].FirstSharedDistribution + i);
// 		from LatticeData::CopyReceived()
//==========================================================================================
template <typename LatticeType> struct GPU_StreamReceivedDistr_Functor {

  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  site_t *GMem_int64_streamInd;
  site_t nArr_dbl;
  site_t upper_limit;

  GPU_StreamReceivedDistr_Functor(distribn_t *GMem_dbl_fOld_b_, distribn_t *GMem_dbl_fNew_b_, site_t *GMem_int64_streamInd_, site_t nArr_dbl_,
                                  site_t upper_limit_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), GMem_int64_streamInd(GMem_int64_streamInd_), nArr_dbl(nArr_dbl_),
        upper_limit(upper_limit_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) {
	const lb::lattices::D3Q19GPUConstants c;
    // Ind =Ind + lower_limit; // limits are: for (site_t i = 0; i < totalSharedFs; i++)
    if (Ind >= upper_limit)
      return;


    // Read in the fOld[neighbouringProcs[0].FirstSharedDistribution + Ind] and then place this in the appropriate index in fNew
    distribn_t dev_fOld;
    dev_fOld = GMem_dbl_fOld_b[(unsigned long long) c.NUMVECTORS * nArr_dbl + 1 + Ind];

    // Read the corresponding Index from the streaming Indices For Received Distributions
    // Remeber that this index refers to data layout method (a),
    //	i.e. Arrange by fluid index (as is hemeLB CPU version), i.e f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind],
    //..., fq[Ind]
    // Need to convert to data layout method (b),
    // 	i.e. Arrange by index_LB, i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., fq[0 to (nFluid_nodes-1)]
    site_t streamIndex_method_a = GMem_int64_streamInd[Ind];

    // Convert to data layout method (b)
    // 		The streamed array index (method_a) is within the domain, i.e. [0,nFluid_nodes*_NUMVECTORS)
    // 		a. The LB_dir, [0,_NUMVECTORS), will then be the value returned by modulo(_NUMVECTORS):
    int LB_Dir = streamIndex_method_a % c.NUMVECTORS;
    // 		b. Fluid ID
    site_t fluid_ID = (streamIndex_method_a - LB_Dir) / c.NUMVECTORS;   // Evaluate the ACTUAL fluid ID index

    site_t streamIndex_method_b = LB_Dir * nArr_dbl + fluid_ID;

    GMem_dbl_fNew_b[streamIndex_method_b] = dev_fOld;

  }   // Ends the GPU_StreamReceivedDistr kernel
  //==========================================================================================
};

}   // namespace hemelb

#endif
