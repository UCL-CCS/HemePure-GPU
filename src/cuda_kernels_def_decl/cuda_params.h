// cuda_params.h
#ifndef cuda_params_h
#define cuda_params_h

#include <stdint.h> // to use uint64_t below
#include "units.h"
#include "cuda_kernels_def_decl/deviceAPI.h"
#include "cuda_kernels_def_decl/deviceLaunch.h"

#include "lb/lattices/D3Q19_gpu.h"

#define local_iolets_MaxSIZE 90 // This is the max array size with the iolet info (Iolet ID and fluid sites range, min and max, i.e. size = 3*local number of iolets). Assume that maximum number of iolets per RANK = local_iolets_MaxSIZE/3, i.e 30 here
#define frequency_WriteGlobalMem 1000 // Frequency to write macroVariables to GPU global memory

namespace hemelb
{
	// Struct to hold the info for the Iolets: Iolet ID and fluid sites ranges
	// Definition of the struct needs to be visible to all files
	struct Iolets{
		int n_local_iolets;						// 	Number of local Rank Iolets - NOTE: Some Iolet IDs may repeat, depending on the fluid ID numbering - see the value of unique iolets, (for example n_unique_LocalInlets_mInlet_Edge)
		site_t Iolets_ID_range[local_iolets_MaxSIZE]; 	//	Iolet ID and fluid sites range: [min_Fluid_Index, max_Fluid_Index], i.e 3 site_t values per iolet
	};
	extern struct Iolets Inlet_Edge, Inlet_Inner, InletWall_Edge, InletWall_Inner;
	extern struct Iolets Outlet_Edge, Outlet_Inner, OutletWall_Edge, OutletWall_Inner;

	inline void check_cuda_errors(const char *filename, const int line_number, int myProc);

	//==============================================================================
	/**
	  Device function to investigate which Iolet Ind corresponds to a fluid with index fluid_Ind
	  To be used for the inlet/outlet related collision-streaming kernels
	  Checks through the local iolets (inlet/outlet) to determine the correct iolet ID
	  each iolet has fluid sites with indices in the range: [lower_limit,upper_limit]
	  Function returns the iolet ID value: IdInlet.
	 */
	GPU_INLINE_DEVICE_FUNCTION void _determine_Iolet_ID(int num_local_Iolets, const site_t* iolets_ID_range, site_t fluid_Ind, int* IdInlet)
	{
		//printf("(cuda_params.h) - Enters _determine_Iolet_ID !!! \n");
		// Loop over the number of local iolets (num_local_Iolets) and determine whether the fluid ID (fluid_Ind) falls whithin the range
		for (int i_local_iolet = 0; i_local_iolet<num_local_Iolets; i_local_iolet++)
		{
			// iolet range: [lower_limit,upper_limit)
			int64_t lower_limit = iolets_ID_range[3*i_local_iolet+1];	// Included in the fluids range
			int64_t upper_limit = iolets_ID_range[3*i_local_iolet+2];	// Value included in the fluids' range - CHANGED TO INCLUDE THE VALUE

			//if ((fluid_Ind - upper_limit +1) * (fluid_Ind - lower_limit) <= 0){	 	//When the upper_limit is NOT included
			if ((fluid_Ind - upper_limit) * (fluid_Ind - lower_limit) <= 0){ 				// When the upper_limit is included
				*IdInlet =(int)(iolets_ID_range[3*i_local_iolet]);
				return;
			}
		}// closes the loop over the local iolets
	}


//==============================================================================
// IZ 9 July 2024
// Evaluate wall shear stress magnitude on the device

//==============================================================================
// May 2023
// Struct to contain the array for the second moment of distr. functions
// 6 elements are sufficient
struct structSecMomDistrFun
{
	//array declared inside structure
	double arr[6];
};

//==============================================================================
	/* Device function to evaluate second moment of a distr. function
	/* Despite its name, this method does not compute the whole pi tensor (i.e. momentum flux tensor). What it does is
	* computing the second moment of a distribution function. If this distribution happens to be f_eq, the resulting
	* tensor will be the equilibrium part of pi. However, if the distribution function is f_neq, the result WON'T be
	* the non equilibrium part of pi. In order to get it, you will have to multiply by (1 - timestep/2*tau)
	*
	* @param f distribution function
	* @return second moment of the distribution function f
	* using the array declared in structure
	*/
GPU_INLINE_DEVICE_FUNCTION struct structSecMomDistrFun _structCalculatePiTensor(const distribn_t* const f) //return type is struct structSecMomDistrFun
{
	struct structSecMomDistrFun ret_SecMomDistrFunc; //demo structure member declared

	const lb::lattices::D3Q19GPUConstants c;

	// Fill the elements SecMomDistrFun.arr[i]; i=0 to 5
	// Explicitly calculate the elements (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
	// and saves these with this order in the struct array ret_SecMomDistrFunc.arr

	// Element (0,0)
	ret_SecMomDistrFunc.arr[0] = 0.0;
	for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[0] += f[l] * c.CX[l]* c.CX[l];
	}

	// Element (1,0)
	ret_SecMomDistrFunc.arr[1] = 0.0;
	for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[1] += f[l] * c.CY[l]* c.CX[l];
	}

	// Element (1,1)
	ret_SecMomDistrFunc.arr[2] = 0.0;
	for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[2] += f[l] * c.CY[l]* c.CY[l];
	}

	// Element (2,0)
	ret_SecMomDistrFunc.arr[3] = 0.0;
	for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[3] += f[l] * c.CZ[l]* c.CX[l];
	}

	// Element (2,1)
	ret_SecMomDistrFunc.arr[4] = 0.0;
	for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[4] += f[l] * c.CZ[l]* c.CY[l];
	}

	// Element (2,2)
	ret_SecMomDistrFunc.arr[5] = 0.0;
	for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[5] += f[l] * c.CZ[l]* c.CZ[l];
	}

	return ret_SecMomDistrFunc; //address of structure member returned
}


//==============================================================================
	/* Device function to evaluate second moment of a distr. function
	/* Despite its name, this method does not compute the whole pi tensor (i.e. momentum flux tensor). What it does is
	* computing the second moment of a distribution function. If this distribution happens to be f_eq, the resulting
	* tensor will be the equilibrium part of pi. However, if the distribution function is f_neq, the result WON'T be
	* the non equilibrium part of pi. In order to get it, you will have to multiply by (1 - timestep/2*tau)
	*
	* @param f distribution function
	* @return second moment of the distribution function f
	* 	using a pointer to the array.
	* 		Note that the array needs to be declared as static, otherwise compiling issues might occur
	*/
	GPU_INLINE_DEVICE_FUNCTION double *_CalculatePiTensor(const distribn_t* const f)
	{
			static double ret_SecMomDistrFunc[6]; // Needs to be static

			const lb::lattices::D3Q19GPUConstants c;

			/*
			// Fill in (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
			for (int ii = 0; ii < 3; ++ii)
			{
				for (int jj = 0; jj <= ii; ++jj)
				{
					ret[ii][jj] = 0.0;
						for (unsigned int l = 0; l < DmQn::NUMVECTORS; ++l)
						{
							ret[ii][jj] += f[l] * DmQn::discreteVelocityVectors[ii][l]
													* DmQn::discreteVelocityVectors[jj][l];
						}
				}
			}
			*/

			// Explicitly calculate the elements (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
			// and saves these with this order in the array ret_SecMomDistrFunc

			// Element (0,0)
			ret_SecMomDistrFunc[0] = 0.0;
			for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[0] += f[l] * c.CX[l]* c.CX[l];
			}

			// Element (1,0)
			ret_SecMomDistrFunc[1] = 0.0;
			for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[1] += f[l] * c.CY[l]* c.CX[l];
			}

			// Element (1,1)
			ret_SecMomDistrFunc[2] = 0.0;
			for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[2] += f[l] * c.CY[l]* c.CY[l];
			}

			// Element (2,0)
			ret_SecMomDistrFunc[3] = 0.0;
			for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[3] += f[l] * c.CZ[l]* c.CX[l];
			}

			// Element (2,1)
			ret_SecMomDistrFunc[4] = 0.0;
			for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[4] += f[l] * c.CZ[l]* c.CY[l];
			}

			// Element (2,2)
			ret_SecMomDistrFunc[5] = 0.0;
			for (unsigned int l = 0; l < c.NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[5] += f[l] * c.CZ[l]* c.CZ[l];
			}

			return ret_SecMomDistrFunc;
	}
//==============================================================================

	GPU_INLINE_DEVICE_FUNCTION double _CalculateWallShearStressMagnitude(const distribn_t density,
			const distribn_t* const f_neq,
			const double normal_x, const double normal_y, const double normal_z,
			const double &iStressParameter)
	{
		distribn_t wall_shear_stress_magn;

		//printf("Wall normal components: (%5.5e, %5.5e, %5.5e)\n", normal_x, normal_y, normal_z);

		// sigma_ij is the force
		// per unit area in
		// direction i on the
		// plane with the normal
		// in direction j
		distribn_t stress_vector[] = { 0.0, 0.0, 0.0 }; // Force per unit area in
		// direction i on the
		// plane perpendicular to
		// the surface normal
		distribn_t square_stress_vector = 0.0;
		distribn_t normal_stress = 0.0; // Magnitude of force per
		// unit area normal to the
		// surface

		// Multiplying the second moment of the non equilibrium function by temp gives the non equilibrium part
		// of the moment flux tensor pi.
		distribn_t temp = iStressParameter * (-sqrt(2.0));

		// Computes the second moment of the argument passed ( non equilibrium part of f).
		// This will initially evaluate the second moments of the distr. functions
		// Explicitly calculate the elements (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
		// and saves these with this order in the array ret_SecMomDistrFunc
		// 	Need then to exploit symmetry to fill the elements (0,1) (0,2) (1,2)

		/*
		// Old approach
		double *SecMomDistrFunc;
		SecMomDistrFunc = _CalculatePiTensor(f_neq);
		*/

		double *SecMomDistrFunc;
		//--------------------------------------------------------------------------
		/*
		// Approach 1: Using a pointer to the array
		double *SecMomDistrFunc_returned;
		SecMomDistrFunc_returned = _CalculatePiTensor(f_neq);
		SecMomDistrFunc = SecMomDistrFunc_returned;
		*/

		// Approach 2: Using a struct and array declared in that struct
		struct structSecMomDistrFun SecMomDistrFunc_returned;
		SecMomDistrFunc_returned = _structCalculatePiTensor(f_neq);
		SecMomDistrFunc = SecMomDistrFunc_returned.arr;

		//--------------------------------------------------------------------------

		// Does not need the following - Use symmetry
		//SecMomDistrFunc[6] = SecMomDistrFunc[1];
		//SecMomDistrFunc[7] = SecMomDistrFunc[3];
		//SecMomDistrFunc[8] = SecMomDistrFunc[4];
		/*// Debugging
		for (int i = 0; i < 6; ++i) {
			printf("Second Mom. Distr. funct. %5.5e\n", SecMomDistrFunc[i]);
		}*/

		// Original loop:
		/*for (unsigned i = 0; i < 3; i++)
		{
			for (unsigned j = 0; j < 3; j++){
				stress_vector[i] += pi[i][j] * nor[j] * temp;
			}
			square_stress_vector += stress_vector[i] * stress_vector[i];

			//normal_stress += stress_vector[i] * nor[i];
		}
		*/

		// Unrolled loops:
		stress_vector[0] = ( SecMomDistrFunc[0] * normal_x +
												SecMomDistrFunc[1] * normal_y +
												SecMomDistrFunc[3] * normal_z) * temp;
		stress_vector[1] = ( SecMomDistrFunc[1] * normal_x +
												SecMomDistrFunc[2] * normal_y +
												SecMomDistrFunc[4] * normal_z) * temp;
		stress_vector[2] = ( SecMomDistrFunc[3] * normal_x +
												SecMomDistrFunc[4] * normal_y +
												SecMomDistrFunc[5] * normal_z) * temp;

		square_stress_vector = 	stress_vector[0] * stress_vector[0] +
														stress_vector[1] * stress_vector[1] +
														stress_vector[2] * stress_vector[2];

		normal_stress = 	stress_vector[0] * normal_x
										+ stress_vector[1] * normal_y
										+ stress_vector[2] * normal_z;

		// shear_stress^2 + normal_stress^2 = stress_vector^2
		//stress = sqrt(square_stress_vector - normal_stress * normal_stress);
		wall_shear_stress_magn = sqrt(square_stress_vector - normal_stress * normal_stress);;

		//printf("Wall Shear Stress = %5.5e, Sq.StressVect = %5.5e, NormStressSq = %5.5e\n", wall_shear_stress_magn, square_stress_vector, normal_stress * normal_stress);

		return wall_shear_stress_magn;
	}
	//==============================================================================

} // namespace HemeLB

#include "cuda_kernels_def_decl/GPU_BaseKernels.hpp"
#include "cuda_kernels_def_decl/GPU_Collide_Stream_Iolets.hpp"
#include "cuda_kernels_def_decl/GPU_Collide_Stream_wall_sBB_Iolets.hpp"

#endif
