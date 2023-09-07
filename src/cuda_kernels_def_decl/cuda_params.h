// cuda_params.h
#ifndef cuda_params_h
#define cuda_params_h

#include <stdint.h> // to use uint64_t below
#include "units.h"
#include "cuda_kernels_def_decl/deviceAPI.h"
#include "cuda_kernels_def_decl/deviceLaunch.h"

#define local_iolets_MaxSIZE 90 // This is the max array size with the iolet info (Iolet ID and fluid sites range, min and max, i.e. size = 3*local number of iolets). Assume that maximum number of iolets per RANK = local_iolets_MaxSIZE/3, i.e 30 here
#define frequency_WriteGlobalMem 100 // Frequency to write macroVariables to GPU global memory

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

	/* Device function to evaluate second moment of a distr. function
	 * Despite its name, this method does not compute the whole pi tensor (i.e. momentum flux tensor). What it does is
	 * computing the second moment of a distribution function. If this distribution happens to be f_eq, the resulting
	 * tensor will be the equilibrium part of pi. However, if the distribution function is f_neq, the result WON'T be
	 * the non equilibrium part of pi. In order to get it, you will have to multiply by (1 - timestep/2*tau)
	 *
	 * @param f distribution function
	 * @return second moment of the distribution function f
	 */
	GPU_INLINE_DEVICE_FUNCTION void _CalculatePiTensor(const distribn_t* const f, double* SecMomDistrFunc,
													  Direction NUMVECTORS, 
													  const double* CX,
													  const double* CY,
													  const double* CZ)
	{
		// Explicitly calculate the elements (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
		// and saves these with this order in the array SecMomDistrFunc

		// Element (0,0)
		SecMomDistrFunc[0] = 0.0;
		for (unsigned int l = 0; l < NUMVECTORS; ++l)
		{
			SecMomDistrFunc[0] += f[l] * CX[l]* CX[l];
		}

		// Element (1,0)
		SecMomDistrFunc[1] = 0.0;
		for (unsigned int l = 0; l < NUMVECTORS; ++l)
		{
			SecMomDistrFunc[1] += f[l] * CY[l]* CX[l];
		}

		// Element (1,1)
		SecMomDistrFunc[2] = 0.0;
		for (unsigned int l = 0; l < NUMVECTORS; ++l)
		{
			SecMomDistrFunc[2] += f[l] * CY[l]* CY[l];
		}

		// Element (2,0)
		SecMomDistrFunc[3] = 0.0;
		for (unsigned int l = 0; l < NUMVECTORS; ++l)
		{
			SecMomDistrFunc[3] += f[l] * CZ[l]* CX[l];
		}

		// Element (2,1)
		SecMomDistrFunc[4] = 0.0;
		for (unsigned int l = 0; l < NUMVECTORS; ++l)
		{
			SecMomDistrFunc[4] += f[l] * CZ[l]* CY[l];
		}

		// Element (2,2)
		SecMomDistrFunc[5] = 0.0;
		for (unsigned int l = 0; l < NUMVECTORS; ++l)
		{
			SecMomDistrFunc[5] += f[l] * CZ[l]* CZ[l];
		}

	}


} // namespace HemeLB

#include "cuda_kernels_def_decl/GPU_BaseKernels.hpp"
#include "cuda_kernels_def_decl/GPU_Collide_Stream_Iolets.hpp"
#include "cuda_kernels_def_decl/GPU_Collide_Stream_wall_sBB_Iolets.hpp"

#endif
