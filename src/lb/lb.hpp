// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_LB_HPP
#define HEMELB_LB_LB_HPP
#include <omp.h>
#include "io/writers/xdr/XdrMemWriter.h"
#include "lb/lb.h"
#include "util/unique.h"
#include "lb/InitialCondition.h"
#include "lb/InitialCondition.hpp"

#ifdef HEMELB_USE_GPU
#include "lb/kernels/LBGKSpongeLayer.h"
#ifndef HEMELB_USE_HIP
#include <cuda_profiler_api.h>
#endif
#endif



namespace hemelb
{
	namespace lb
	{

#ifdef HEMELB_USE_GPU

// Define static variables associated with pinned memory
template<class LatticeType>
	distribn_t* LBM<LatticeType>::h_ghostDensity_inlet = nullptr;

template<class LatticeType>
	distribn_t* LBM<LatticeType>::h_ghostDensity_outlet = nullptr;

// Define static variables associated with pinned memory in Read_Macrovariables_GPU_to_CPU
template<class LatticeType>
distribn_t* LBM<LatticeType>::dens_GPU = nullptr;

template<class LatticeType>
distribn_t* LBM<LatticeType>::vx_GPU = nullptr;

template<class LatticeType>
distribn_t* LBM<LatticeType>::vy_GPU = nullptr;

template<class LatticeType>
distribn_t* LBM<LatticeType>::vz_GPU = nullptr;

template<class LatticeType>
distribn_t* LBM<LatticeType>::WallShearStressMagn_Edge_Type2_GPU = nullptr;

template<class LatticeType>
distribn_t* LBM<LatticeType>::WallShearStressMagn_Edge_Type5_GPU = nullptr;

template<class LatticeType>
distribn_t* LBM<LatticeType>::WallShearStressMagn_Edge_Type6_GPU = nullptr;

template<class LatticeType>
distribn_t* LBM<LatticeType>::WallShearStressMagn_Inner_Type2_GPU = nullptr;

template<class LatticeType>
distribn_t* LBM<LatticeType>::WallShearStressMagn_Inner_Type5_GPU = nullptr;

template<class LatticeType>
distribn_t* LBM<LatticeType>::WallShearStressMagn_Inner_Type6_GPU = nullptr;
#endif


		template<class LatticeType>
			hemelb::lb::LbmParameters* LBM<LatticeType>::GetLbmParams()
			{
				return &mParams;
			}

		template<class LatticeType>
			lb::MacroscopicPropertyCache& LBM<LatticeType>::GetPropertyCache()
			{
				return propertyCache;
			}

		template<class LatticeType>
			LBM<LatticeType>::LBM(configuration::SimConfig *iSimulationConfig,
					net::Net* net,
					geometry::LatticeData* latDat,
					SimulationState* simState,
					reporting::Timers &atimings,
					geometry::neighbouring::NeighbouringDataManager *neighbouringDataManager) :
				mSimConfig(iSimulationConfig), mNet(net), mLatDat(latDat), mState(simState),
				mParams(iSimulationConfig->GetTimeStepLength(), iSimulationConfig->GetVoxelSize()), timings(atimings),
				propertyCache(*simState, *latDat), neighbouringDataManager(neighbouringDataManager)
		{
			ReadParameters();
		}

		template<class LatticeType>
			void LBM<LatticeType>::InitInitParamsSiteRanges(kernels::InitParams& initParams, unsigned& state)
			{
				initParams.siteRanges.resize(2);

				initParams.siteRanges[0].first = 0;
				initParams.siteRanges[1].first = mLatDat->GetMidDomainSiteCount();
				state = 0;
				initParams.siteRanges[0].second = initParams.siteRanges[0].first + mLatDat->GetMidDomainCollisionCount(state);
				initParams.siteRanges[1].second = initParams.siteRanges[1].first + mLatDat->GetDomainEdgeCollisionCount(state);

				initParams.siteCount = mLatDat->GetMidDomainCollisionCount(state) + mLatDat->GetDomainEdgeCollisionCount(state);
			}

		template<class LatticeType>
			void LBM<LatticeType>:: AdvanceInitParamsSiteRanges(kernels::InitParams& initParams, unsigned& state)
			{
				initParams.siteRanges[0].first += mLatDat->GetMidDomainCollisionCount(state);
				initParams.siteRanges[1].first += mLatDat->GetDomainEdgeCollisionCount(state);
				++state;
				initParams.siteRanges[0].second = initParams.siteRanges[0].first + mLatDat->GetMidDomainCollisionCount(state);
				initParams.siteRanges[1].second = initParams.siteRanges[1].first + mLatDat->GetDomainEdgeCollisionCount(state);

				initParams.siteCount = mLatDat->GetMidDomainCollisionCount(state) + mLatDat->GetDomainEdgeCollisionCount(state);
			}

		template<class LatticeType>
			void LBM<LatticeType>::InitCollisions()
			{
				/**
				 * Ensure the boundary objects have all info necessary.
				 */
				PrepareBoundaryObjects();

				// TODO Note that the convergence checking is not yet implemented in the
				// new boundary condition hierarchy system.
				// It'd be nice to do this with something like
				// MidFluidCollision = new ConvergenceCheckingWrapper(new WhateverMidFluidCollision());


				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();
				// std::printf("Local Rank = %i \n\n", myPiD);

				kernels::InitParams initParams = kernels::InitParams();
				initParams.latDat = mLatDat;
				initParams.lbmParams = &mParams;
				initParams.neighbouringDataManager = neighbouringDataManager;

				//---------------------------
				// IZ-Added July 2024 for the LES/SL (sponge layer) implementation
				// Here outletPositions/inletPositions contain the full list of iolets not just the local ones...
				// as the loop is going up to GetTotalIoletCount() and we get each iolet info through GetIolets()[indexIolet]
				// TODO: I think I should Consider only the local iolets
				for (unsigned outlet = 0; outlet < mOutletValues->GetTotalIoletCount(); ++outlet)
				{
					initParams.outletPositions.push_back(mOutletValues->GetIolets()[outlet]->GetPosition());
				}
				for (unsigned intlet = 0; intlet < mInletValues->GetTotalIoletCount(); ++intlet)
				{
					initParams.inletPositions.push_back(mInletValues->GetIolets()[intlet]->GetPosition());
				}
				//---------------------------

				unsigned collId;
				InitInitParamsSiteRanges(initParams, collId);
				mMidFluidCollision = new tMidFluidCollision(initParams);

				AdvanceInitParamsSiteRanges(initParams, collId);
				mWallCollision = new tWallCollision(initParams);

				AdvanceInitParamsSiteRanges(initParams, collId);
				initParams.boundaryObject = mInletValues;
				mInletCollision = new tInletCollision(initParams);

				AdvanceInitParamsSiteRanges(initParams, collId);
				initParams.boundaryObject = mOutletValues;
				mOutletCollision = new tOutletCollision(initParams);

				AdvanceInitParamsSiteRanges(initParams, collId);
				initParams.boundaryObject = mInletValues;
				mInletWallCollision = new tInletWallCollision(initParams);

				AdvanceInitParamsSiteRanges(initParams, collId);
				initParams.boundaryObject = mOutletValues;
				mOutletWallCollision = new tOutletWallCollision(initParams);

			}

		template<class LatticeType>
			void LBM<LatticeType>::Initialise(iolets::BoundaryValues* iInletValues,
					iolets::BoundaryValues* iOutletValues,
					const util::UnitConverter* iUnits)
			{
				mInletValues = iInletValues;
				mOutletValues = iOutletValues;
				mUnits = iUnits;

				InitCollisions();

				// IZ Nov 2023 - Commented out after bringing in Checkpointing functionality (following Jon's approach)
				// SetInitialConditions();
			}

		template<class LatticeType>
			void LBM<LatticeType>::PrepareBoundaryObjects()
			{
				// First, iterate through all of the inlet and outlet objects, finding out the minimum density seen in the simulation.
				distribn_t minDensity = std::numeric_limits<distribn_t>::max();

				for (unsigned inlet = 0; inlet < mInletValues->GetLocalIoletCount(); ++inlet)
				{
					minDensity = std::min(minDensity, mInletValues->GetLocalIolet(inlet)->GetDensityMin());
				}

				for (unsigned outlet = 0; outlet < mOutletValues->GetLocalIoletCount(); ++outlet)
				{
					minDensity = std::min(minDensity, mOutletValues->GetLocalIolet(outlet)->GetDensityMin());
				}

				// Now go through them again, informing them of the minimum density.
				for (unsigned inlet = 0; inlet < mInletValues->GetLocalIoletCount(); ++inlet)
				{
					mInletValues->GetLocalIolet(inlet)->SetMinimumSimulationDensity(minDensity);
				}

				for (unsigned outlet = 0; outlet < mOutletValues->GetLocalIoletCount(); ++outlet)
				{
					mOutletValues->GetLocalIolet(outlet)->SetMinimumSimulationDensity(minDensity);
				}
			}


#ifdef HEMELB_USE_GPU
			/**
				Function to:
				a. Read the received distr. functions at the RECEIVING rank (host)
						after completing the colision-streaming at the domain edges
						and
				b. Send these populations to the GPU: host-to-device memcopy.
				Comments:
				1.	SENDING rank sends the fNew distr. functions in totalSharedFs
						RECEIVING rank places these values in fOld (in totalSharedFs): Note that shift +1 at the end of the array
				2.	This should be called as soon as the MPI exchange at domain edges has been successfully completed!!!
			*/
		template<class LatticeType>
			bool LBM<LatticeType>::Read_DistrFunctions_CPU_to_GPU_totalSharedFs()
			{

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				// Fluid sites details
				int64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount();	// Total number of fluid sites: GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)
				int64_t totSharedFs = mLatDat->totalSharedFs;	// totalSharedFs: Total number of data elements (distr. functions) to be transferred to the GPU

				//std::printf("Proc# %i : #data elements (distr. functions) to be transferred = %i \n\n", myPiD, totSharedFs);	// Test that I can access these values
/*
				distribn_t* Data_dbl_fOld_Tr = new distribn_t[totSharedFs];	// distribn_t (type double)

				if(!Data_dbl_fOld_Tr){ std::cout << "Memory allocation error at Read_DistrFunctions_CPU_to_GPU" << std::endl;  return false;}

				// Copy the data from *GetFOld(neighbouringProcs[0].FirstSharedDistribution + i);
				Data_dbl_fOld_Tr = mLatDat->GetFOld(mLatDat->neighbouringProcs[0].FirstSharedDistribution); // Carefull: Starting Address mLatDat->neighbouringProcs[0].FirstSharedDistribution = (nFluid_nodes * LatticeType::NUMVECTORS +1)
*/
				/*
				// Check the addreses: It is expected that totalSharedFs will be placed at the end of the distr.functions for the domain, i.e. after nFluid_nodes*19 +1
				Data_dbl_fOld_Tr_test = mLatDat->GetFOld(nFluid_nodes * LatticeType::NUMVECTORS +1); // Carefull: +1 - Starts at the end of the distr. functions for the domain (nFluid_nodes*num_Vectors +1)
				printf("Rank: %d, End of Array (nFluid_nodes*19+1) = %lld - Value 1 (neighbouringProcs[0].FirstSharedDistribution) = %lld \n\n", myPiD, nFluid_nodes * LatticeType::NUMVECTORS +1,mLatDat->neighbouringProcs[0].FirstSharedDistribution);

				//----------------------------------------------------------------------------------------
				// Debugging - testing. Remove later...
				for (site_t i = 0; i < totSharedFs; i++)
				{
					double ff =  Data_dbl_fOld_Tr[i];
					double ff_test =  Data_dbl_fOld_Tr_test[i];

					Data_dbl_fOld_Tr_test2[i] = *(mLatDat->GetFOld(mLatDat->neighbouringProcs[0].FirstSharedDistribution+i));

					if (ff != ff_test)
						printf("Value of  distr. ff = %.5f Vs ff_test = %.5f  \n\n", ff, ff_test);

					if (ff != Data_dbl_fOld_Tr_test2[i])
						printf("Value of  distr. ff = %.5f Vs Data_dbl_fOld_Tr_test2[%lld] = %.5f  \n\n", ff, Data_dbl_fOld_Tr_test2[i], i);

				} // Remove later...
				//----------------------------------------------------------------------------------------
				*/
				// Send the data from host (Data_dbl_fOld_Tr) to the Device GPU global memory
				// Memory copy from host (Data_dbl_fOld) to Device (GPUDataAddr_dbl_fOld)

				unsigned long long MemSz = totSharedFs  * sizeof(distribn_t); // Total memory size


				// Method 1: Using pageable memory (on the host)
				//status = deviceMemcpyAsync(&(((distribn_t*)GPUDataAddr_dbl_fOld_b)[nFluid_nodes * LatticeType::NUMVECTORS +1]), &(Data_dbl_fOld_Tr[0]), MemSz, memcpyHostToDevice, stream_memCpy_CPU_GPU_domainEdge);
				// This works as well:
				// Sept 2020 - Switch to Unified Memory (from GPUDataAddr_dbl_fOld_b to mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat)
				bool status = deviceMemcpyAsync(&(((distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat)[nFluid_nodes * LatticeType::NUMVECTORS +1]), mLatDat->GetFOld(mLatDat->neighbouringProcs[0].FirstSharedDistribution), MemSz, memcpyHostToDevice, stream_ReceivedDistr); // stream_memCpy_CPU_GPU_domainEdge);

				//cudaStatus = deviceMemcpy(&(((distribn_t*)GPUDataAddr_dbl_fOld_b)[nFluid_nodes * LatticeType::NUMVECTORS +1]), &(Data_dbl_fOld_Tr[0]), MemSz, memcpyHostToDevice);
				if (!status) fprintf(stderr, "GPU memory copy host-to-device failed ... Rank = %d, Time = %d \n", myPiD, mState->GetTimeStep());


				/*
				// Method 2: Using pinned
				// Data_H2D_memcpy_totalSharedFs = mLatDat->GetFOld(mLatDat->neighbouringProcs[0].FirstSharedDistribution);

				distribn_t *f_old = mLatDat->GetFOld(mLatDat->neighbouringProcs[0].FirstSharedDistribution);
			 	for (site_t i = 0; i < totSharedFs; ++i){
				 	Data_H2D_memcpy_totalSharedFs[i] = f_old[i];
			 	}

				cudaStatus = deviceMemcpyAsync(&(((distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat)[nFluid_nodes * LatticeType::NUMVECTORS +1]),
																			&Data_H2D_memcpy_totalSharedFs[0], MemSz, memcpyHostToDevice, stream_ReceivedDistr); // stream_memCpy_CPU_GPU_domainEdge);
				if (!status) fprintf(stderr, "GPU memory copy (using Pinned Memory) host-to-device failed... \
																								Trasnfering %.2fGB on Rank %d, Time = %d \n", (double)MemSz/1073741824.0, myPiD, mState->GetTimeStep());
				*/

				// Delete when the mem.copy is complete
				//delete[] Data_dbl_fOld_Tr; 				// Cannot delete as it is pointing to the following: mLatDat->GetFOld(mLatDat->neighbouringProcs[0].FirstSharedDistribution);
				//delete[] Data_dbl_fOld_Tr_test;		// Cannot delete for the same reason!!! points to : mLatDat->GetFOld(nFluid_nodes * LatticeType::NUMVECTORS +1)
				// delete[] Data_dbl_fOld_Tr_test2; 	// This one is possible



				return true;
			}



		template<class LatticeType>
			bool LBM<LatticeType>::Read_DistrFunctions_CPU_to_GPU(int64_t firstIndex, int64_t siteCount)
			{

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				// Fluid sites details
				int64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount();	// Total number of fluid sites: GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)
				int64_t nDistr_Tr = siteCount * LatticeType::NUMVECTORS;	// Total number of data elements (distr. functions) to be transferred to the GPU

				// std::printf("Proc# %i : #data elements (distr. functions) to be transferred = %i \n\n", myPiD, nDistr_Tr);	// Test that I can access these values

				distribn_t* Data_dbl_fOld_Tr = new distribn_t[nDistr_Tr];	// distribn_t (type double)

				if(!Data_dbl_fOld_Tr){ std::cout << "Memory allocation error at Read_DistrFunctions_CPU_to_GPU" << std::endl;  return false;}

				// 	f Distr. - To do!!!
				// Convert from method_a (CPU) to method_b to be send to the GPU
#pragma omp parallel for collapse(2)
				for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++)
				{
					for (site_t i = firstIndex; i < (firstIndex + siteCount); i++)
					{
						// make a shift of the data index in Data_dbl_fOld_Tr so that it starts from 0
						*(&Data_dbl_fOld_Tr[l * siteCount + (i - firstIndex)]) = *(mLatDat->GetFOld(i * LatticeType::NUMVECTORS + l)); // distribn_t (type double) - Data_dbl_fOld contains the oldDistributions re-arranged
					}
				}

				// Send the data from host (Data_dbl_fOld_Tr) to the Device GPU global memory
				// Memory copy from host (Data_dbl_fOld) to Device (GPUDataAddr_dbl_fOld)
				// cudaStatus = deviceMemcpy(GPUDataAddr_dbl_fOld, Data_dbl_fOld, nArray_oldDistr * sizeof(distribn_t), memcpyHostToDevice);
				// if(!status){ fprintf(stderr, "GPU memory transfer Host To Device failed - \n"); return false; }

				// Send iteratively the f_0, f_1, f_2, ..., f_(q-1) to the corresponding GPU mem. address
				long long MemSz = siteCount * sizeof(distribn_t);	// Memory size for each of the fi's send - Carefull: This is not the total Memory Size!!!

				for (int LB_ind=0; LB_ind < LatticeType::NUMVECTORS; LB_ind++)
				{
					bool status = deviceMemcpy(&(((distribn_t*)GPUDataAddr_dbl_fOld_b)[(LB_ind*nFluid_nodes)+firstIndex]),
												&(Data_dbl_fOld_Tr[LB_ind * siteCount]), MemSz, memcpyHostToDevice);
					if (!status) fprintf(stderr, "GPU memory copy failed (%d)\n", LB_ind);
				}


				delete[] Data_dbl_fOld_Tr;

				return true;
			}


			// Function for reading:
			//	a. the Distribution Functions post-collision, fNew, in totalSharedFs
			// 				that will be send to the neighbouring ranks
			// 		from the GPU and copying to the CPU (device-to-host mem. copy - Asynchronous)

			//
			// If we use deviceMemcpy: Remember that from the host perspective the mem copy is synchronous, i.e. blocking
			// so the host will wait the data transfer to complete and then proceed to the next function call

			// Switched to deviceMemcpyAsync(): non-blocking on the host,
			//		so control returns to the host thread immediately after the transfer is issued.
			// 	cuda stream: mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2()
			//=================================================================================================
		template<class LatticeType>
			bool LBM<LatticeType>::Read_DistrFunctions_GPU_to_CPU_totalSharedFs()
			{


				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				// Total number of fluid sites
				uint64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount(); // Actually GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)
				uint64_t totSharedFs = mLatDat->totalSharedFs;

				//======================================================================
				/* Approach 1
				//--------------------------------------------------------------------------
				// a. Distribution functions fNew, i.e. post collision populations:
				// unsigned long long MemSz = (1 + totSharedFs)  * sizeof(distribn_t); // Total memory size
				unsigned long long MemSz = (1+totSharedFs)  * sizeof(distribn_t); // Total memory size

				distribn_t* fNew_GPU_totalSharedFs = new distribn_t[1+totSharedFs];	// distribn_t (type double)

				if(!fNew_GPU_totalSharedFs){ std::cout << "Memory allocation error - ReadGPU_distr totalSharedFs" << std::endl; return false;}

				// THink about the following: Starting addres of totalSharedFs: a) nFluid_nodes*LatticeType::NUMVECTORS 0R b) nFluid_nodes*LatticeType::NUMVECTORS +1 ??? To do!!!

				// Get the cuda stream created in BaseNet using the class member function Get_stream_memCpy_GPU_CPU_domainEdge_new2():
				hemelb::net::Net& mNet_cuda_stream = *mNet;
				cudaStatus = deviceMemcpyAsync(&(fNew_GPU_totalSharedFs[0]), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[nFluid_nodes * LatticeType::NUMVECTORS]), MemSz, memcpyDeviceToHost, mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2() );
				//cudaStatus = deviceMemcpyAsync(&(fNew_GPU_totalSharedFs[0]), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[nFluid_nodes * LatticeType::NUMVECTORS]), MemSz, memcpyDeviceToHost, stream_memCpy_GPU_CPU_domainEdge);

				//cudaStatus = deviceMemcpy(&(fNew_GPU_totalSharedFs[0]), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[nFluid_nodes * LatticeType::NUMVECTORS]), MemSz, memcpyDeviceToHost);

				if(!status){
					const char * eStr = deviceGetErrorString();
					printf("GPU memory transfer for ReadGPU_distr totalSharedFs failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					delete[] fNew_GPU_totalSharedFs;
					return false;
				}

				// Read the fNew distributions from the array
				//mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS) = fNew_GPU_totalSharedFs;  // distribn_t (type double) - Data_dbl points to &newDistributions[0]

				for (site_t i = 0; i < totSharedFs+1; i++)
				{

					//distribn_t ff = fNew_GPU_totalSharedFs[i];
					//distribn_t GetFNew_value = *(mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS + i));
					//if(GetFNew_value !=ff )
					//	printf("Error!!! Value pointed by GetFNew = %.5f Vs value of ff = %.5f \n\n", GetFNew_value, ff);

					*(mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS + i)) = fNew_GPU_totalSharedFs[i];
					//printf("Value of  distr. f = %.5f \n\n", *(mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS + i)));

				}

				// Delete the variables when copy is completed
				delete[] fNew_GPU_totalSharedFs;
				*/
				//======================================================================

				//======================================================================
				// Alternative Approach 2
				//	Put the data from the GPU directly in *(mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS)
				// NO need to define and use the fNew_GPU_totalSharedFs

				unsigned long long MemSz = (1+totSharedFs)  * sizeof(distribn_t); // Total memory size

				// Get the cuda stream created in BaseNet using the class member function Get_stream_memCpy_GPU_CPU_domainEdge_new2():
				hemelb::net::Net& mNet_cuda_stream = *mNet;


				// Method 1: Using pageable memory (on the host)
				//cudaStatus = deviceMemcpyAsync( mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[nFluid_nodes * LatticeType::NUMVECTORS]), MemSz, memcpyDeviceToHost, mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2() );
				// Sept 2020 - Switching to cuda-aware mpi makes the D2H mem.copy not necessary. Also switching to Using Unified Memory
				// Does the following make sense though: case of NO cuda-aware mpi (in which case have to call D2H memcpy) and Unified Memory???
				bool status = deviceMemcpyAsync( mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS),
																			&(((distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat)[nFluid_nodes * LatticeType::NUMVECTORS]),
																			MemSz, memcpyDeviceToHost, mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2() );

				if(!status){
					const char * eStr = deviceGetErrorString();
					printf("GPU memory transfer for ReadGPU_distr totalSharedFs failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					return false;
				}

				/*
				// Method 2: Using pinned memory
				//mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS) = Data_D2H_memcpy_totalSharedFs;
				// Data_D2H_memcpy_totalSharedFs = mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS);

				cudaStatus = deviceMemcpyAsync( &Data_D2H_memcpy_totalSharedFs[0],
																			&(((distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat)[nFluid_nodes * LatticeType::NUMVECTORS]),
																			MemSz, memcpyDeviceToHost, mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2() );

				if(!status){
					const char * eStr = deviceGetErrorString() ;
					printf("GPU memory transfer for ReadGPU_distr totalSharedFs failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					return false;
				}

				// Wait the mem copy to complete
				deviceStreamSynchronize(mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2());

				distribn_t *f_new = mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS);
				for (site_t i = 0; i < 1 + totSharedFs; ++i){
					f_new[i] = Data_D2H_memcpy_totalSharedFs[i];
				}
				*/
				//======================================================================

				return true;
			} // Ends the Read_DistrFunctions_GPU_to_CPU_totalSharedFs






			//=================================================================================================
			/** Function to:
			 		Read the coordinates of the iolet points
			 		type site_t (int64_t)
			 		Perform a memory copy from Host to Device (to GPU global memory) for the case of Velocity Inlet/Outlet BCs
						TODO: change to Asynchronous memcpy (if applicable - CHECK)
						Any gain (from the asynchronous memcpy option) will be just on the initialisation time.
						Not a priority now...

					From: geometry/Site.h  - coordinates are of type site_t (int64_t)
					inline const util::Vector3D<site_t>& GetGlobalSiteCoords() const
	        {
	          return latticeData.GetGlobalSiteCoords(index);
	        }

					which however then are converted to double:
						see: LatticePosition sitePos(site.GetGlobalSiteCoords());
						as LatticePosition: Vector3D<double>  units.h:  typedef util::Vector3D<LatticeDistance> LatticePosition; // origin of lattice is at {0.0,0.0,0.0}
			*/
			//=================================================================================================
			template<class LatticeType>
			bool LBM<LatticeType>::memCpy_HtD_GPUmem_Coords_Iolets(site_t firstIndex, site_t siteCount,
			                      void *GPUDataAddr_Coords_iolets)
			{

				bool memCpy_function_output = true;

			  // Local rank
			  const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
			  int myPiD = rank_Com.Rank();

			  //======================================================================
			  site_t nArr_Coords_iolets = 3 * siteCount; // Number of elements (the fluid index can be used to get the x,y,z Coords) of type int64_t (units.h:  typedef int64_t site_t;)
			  site_t* Data_int64_Coords_iolets = new site_t[nArr_Coords_iolets];	// site_t (type int64_t)

				// Allocate memory on the GPU (global memory)
				site_t MemSz = nArr_Coords_iolets *  sizeof(int64_t); 	// site_t (int64_t) Check that will remain like this in the future
				bool status = deviceMalloc((void**)&GPUDataAddr_Coords_iolets, MemSz);

				if(!status){
					fprintf(stderr, "GPU memory allocation - Coords for Iolets - failed...\n");
					memCpy_function_output = false;
				}
				else{
					printf("GPU memory allocation - Coords for Iolets - Bytes: %lld - SUCCESS from Rank: %d \n", MemSz, myPiD);
				}

			  for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
			  {
					// Save the coords to Data_int64_Coords_iolets
					int64_t shifted_Fluid_Ind = siteIndex - firstIndex;

			    geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);
			    //LatticePosition sitePos(site.GetGlobalSiteCoords());	// LatticePosition: Vector3D<double>  units.h:  typedef util::Vector3D<LatticeDistance> LatticePosition; // origin of lattice is at {0.0,0.0,0.0}
					//printf("SiteIndex: %lld with Coordinates: x=%4.2f, y=%4.2f, z=%4.2f \n", siteIndex, sitePos.x, sitePos.y, sitePos.z); // Note that sitePos is of type Vector3D<double>, hence the printf ... %4.2f

					/*Data_int64_Coords_iolets[shifted_Fluid_Ind*3] = sitePos.x;
					Data_int64_Coords_iolets[shifted_Fluid_Ind*3+1] = sitePos.y;
					Data_int64_Coords_iolets[shifted_Fluid_Ind*3+2] = sitePos.z;
					*/

					int64_t x_coord = site.GetGlobalSiteCoords().x;
					int64_t y_coord = site.GetGlobalSiteCoords().y;
					int64_t z_coord = site.GetGlobalSiteCoords().z;

					Data_int64_Coords_iolets[shifted_Fluid_Ind*3] = x_coord;
					Data_int64_Coords_iolets[shifted_Fluid_Ind*3+1] = y_coord;
					Data_int64_Coords_iolets[shifted_Fluid_Ind*3+2] = z_coord;

					//if(myPiD==3)
						//printf("SiteIndex: %lld with Coordinates (Data): x=%lld, y=%lld, z=%lld \n", siteIndex, Data_int64_Coords_iolets[shifted_Fluid_Ind*3], Data_int64_Coords_iolets[shifted_Fluid_Ind*3+1], Data_int64_Coords_iolets[shifted_Fluid_Ind*3+2]);
						//printf("SiteIndex: %lld with Coordinates: x=%lld, y=%lld, z=%lld \n", siteIndex, x_coord, y_coord, z_coord);
			 }

			 // Perform a HtD memcpy (from Data_int64_Coords_iolets to GPUDataAddr_Coords_iolets)
			 status = deviceMemcpy(GPUDataAddr_Coords_iolets,
				 												Data_int64_Coords_iolets, MemSz, memcpyHostToDevice);
			 if(! status ){
		     const char * eStr = deviceGetErrorString();
		     printf("GPU memory copy for IOLETS coordinates failed with error: \"%s\" at proc# %i - SiteCount: %lld \n", eStr, myPiD, siteCount);
				 memCpy_function_output = false;
		   }
			 //cudaDeviceSynchronize();
			 delete[] Data_int64_Coords_iolets;
		   //======================================================================
		   /*if(!status){
		     const char * eStr = deviceGetErrorString();
		     printf("GPU memory copy for IOLETS coordinates failed with error: \"%s\" at proc# %i - SiteCount: %lld \n", eStr, myPiD, siteCount);
		     return false;
		   }
		   else{
		     return true;
		   }*/
		   //======================================================================
			 return memCpy_function_output;
			}




			//=================================================================================================
			// Function to:
			// 	Perform a memory copy from Host to Device (to GPU global memory) for the wall momentum and the case of Velocity Inlet/Outlet BCs
			// 		Synchronous memcpy at the moment. TODO: change to Asynchronous memcpy.
			//=================================================================================================
		template<class LatticeType>
			bool LBM<LatticeType>::memCpy_HtD_GPUmem_WallMom(site_t firstIndex, site_t siteCount, std::vector<util::Vector3D<double> >& wallMom_Iolet, void *GPUDataAddr_wallMom)
			{

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				//======================================================================
				site_t nArr_wallMom = siteCount * (LatticeType::NUMVECTORS -1); // Number of elements of type distribn_t(double)
				distribn_t* Data_dbl_WallMom = new distribn_t[3*nArr_wallMom];	// distribn_t (type double)

				distribn_t* WallMom_x = &Data_dbl_WallMom[0];
				distribn_t* WallMom_y = &Data_dbl_WallMom[1*nArr_wallMom];
				distribn_t* WallMom_z = &Data_dbl_WallMom[2*nArr_wallMom];

				// Arrange the WallMom data as in method B for the distr. functions - TODO...
				//	Method b: Arrange by index_LB, i.e. wallMom_Dir_1[0 to (nFluid_nodes_Iolet-1)], wallMom_Dir_2[0 to (nFluid_nodes_Iolet-1)], ..., wallMom_Dir_q[0 to (nFluid_nodes_Iolet-1)]
				for (unsigned int direction = 1; direction < LatticeType::NUMVECTORS; direction++) // Ignore LB_Dir=0 (resting)
				{
					for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
					{
						site_t shifted_Fluid_Ind = siteIndex-firstIndex;
						WallMom_x[(direction-1)*siteCount + shifted_Fluid_Ind] = wallMom_Iolet[shifted_Fluid_Ind * LatticeType::NUMVECTORS + direction].x;
						WallMom_y[(direction-1)*siteCount + shifted_Fluid_Ind] = wallMom_Iolet[shifted_Fluid_Ind * LatticeType::NUMVECTORS + direction].y;
						WallMom_z[(direction-1)*siteCount + shifted_Fluid_Ind] = wallMom_Iolet[shifted_Fluid_Ind * LatticeType::NUMVECTORS + direction].z;

						/*if(WallMom_x[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0 &&
								WallMom_y[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0 &&
								WallMom_z[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0)
								printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction,
																		WallMom_x[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction],
																		WallMom_y[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction],
																		WallMom_z[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]);
						*/
					}
				}
				// Memory copy from host (Data_dbl_WallMom) to Device (e.g. GPUDataAddr_wallMom_Inlet_Edge)
				bool status = deviceMemcpy(GPUDataAddr_wallMom, Data_dbl_WallMom, 3*nArr_wallMom * sizeof(distribn_t), memcpyHostToDevice);

				//======================================================================
				if(!status){
					const char * eStr = deviceGetErrorString();
					printf("GPU memory allocation for wallMom failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					return false;
				}
				/*else{
					printf("GPU memory allocation for wallMom successful!!! at proc# %i\n", myPiD);
				}*/
				//======================================================================

				delete[] Data_dbl_WallMom;

				return true;
			}


			//=================================================================================================
			// Function to:
			// 	Perform a memory copy from Host to Device (to GPU global memory) for the wall momentum correction and the case of Velocity Inlet/Outlet BCs
			// 		Synchronous memcpy at the moment. TODO: change to Asynchronous memcpy.
			//=================================================================================================
		template<class LatticeType>
			bool LBM<LatticeType>::memCpy_HtD_GPUmem_WallMom_correction(site_t firstIndex, site_t siteCount, std::vector<double>& wallMom_correction_Iolet, void *GPUDataAddr_wallMom)
			{
				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				//======================================================================
				site_t nArr_wallMom = siteCount * (LatticeType::NUMVECTORS -1); // Number of elements of type distribn_t(double)
				distribn_t* Data_dbl_WallMom = new distribn_t[nArr_wallMom];	// distribn_t (type double)

				distribn_t* WallMom_x = &Data_dbl_WallMom[0];

				// Arrange the WallMom data as in method B for the distr. functions - TODO...
				//	Method b: Arrange by index_LB, i.e. wallMom_Dir_1[0 to (nFluid_nodes_Iolet-1)], wallMom_Dir_2[0 to (nFluid_nodes_Iolet-1)], ..., wallMom_Dir_q[0 to (nFluid_nodes_Iolet-1)]
				for (unsigned int direction = 1; direction < LatticeType::NUMVECTORS; direction++) // Ignore LB_Dir=0 (resting)
				{
					for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
					{
						site_t shifted_Fluid_Ind = siteIndex-firstIndex;
						WallMom_x[(direction-1)*siteCount + shifted_Fluid_Ind] = wallMom_correction_Iolet[shifted_Fluid_Ind * LatticeType::NUMVECTORS + direction];

						/*if(WallMom_x[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0 &&
								WallMom_y[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0 &&
								WallMom_z[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0)
								printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction,
																		WallMom_x[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction],
																		WallMom_y[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction],
																		WallMom_z[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]);
						*/
					}
				}
				// Memory copy from host (Data_dbl_WallMom) to Device (e.g. GPUDataAddr_wallMom_Inlet_Edge)
				bool status = deviceMemcpy(GPUDataAddr_wallMom, Data_dbl_WallMom, nArr_wallMom * sizeof(distribn_t), memcpyHostToDevice);

				//======================================================================
				if(!status){
					const char * eStr = deviceGetErrorString();
					printf("GPU memory allocation for wallMom correction term failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					return false;
				}
				/*else{
					printf("GPU memory allocation for wallMom successful!!! at proc# %i\n", myPiD);
				}*/
				//======================================================================

				delete[] Data_dbl_WallMom;

				return true;
			}


			//=================================================================================================
			// Function to:
			// 	Perform a memory copy from Host to Device (to GPU global memory) for the wall momentum correction and the case of Velocity Inlet/Outlet BCs
			// 		TODO: change to Asynchronous memcpy.
			//=================================================================================================
			template<class LatticeType>
			bool LBM<LatticeType>::compare_CPU_GPU_WallMom_correction(site_t firstIndex, site_t siteCount, std::vector<double>& wallMom_correction_Iolet, void *GPUDataAddr_wallMom)
			{


				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				site_t nArr_wallMom = siteCount * (LatticeType::NUMVECTORS -1); // Number of elements of type distribn_t(double)
				distribn_t* Data_dbl_WallMom_CPU = new distribn_t[nArr_wallMom];	// distribn_t (type double)
				distribn_t* Data_dbl_WallMom_GPU = new distribn_t[nArr_wallMom];	// distribn_t (type double)

				//======================================================================
				// CPU results - Transform to the same data layout as the GPU results

				distribn_t* WallMom_x = &Data_dbl_WallMom_CPU[0];

				// Arrange the WallMom data as in method B for the distr. functions - TODO...
				//	Method b: Arrange by index_LB, i.e. wallMom_Dir_1[0 to (nFluid_nodes_Iolet-1)], wallMom_Dir_2[0 to (nFluid_nodes_Iolet-1)], ..., wallMom_Dir_q[0 to (nFluid_nodes_Iolet-1)]
				for (unsigned int direction = 1; direction < LatticeType::NUMVECTORS; direction++) // Ignore LB_Dir=0 (resting)
				{
					for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
					{
						site_t shifted_Fluid_Ind = siteIndex-firstIndex;
						WallMom_x[(direction-1)*siteCount + shifted_Fluid_Ind] = wallMom_correction_Iolet[shifted_Fluid_Ind * LatticeType::NUMVECTORS + direction];

						//if(WallMom_x[(direction-1)*siteCount + shifted_Fluid_Ind]!=0)
						//		printf("Time: %d, Site: %d, Dir: %d, Wall mom correction: %.5e \n", mState->GetTimeStep(), siteIndex, direction, WallMom_x[(direction-1)*siteCount + shifted_Fluid_Ind]);

					}
				}
				// Memory copy from host (Data_dbl_WallMom) to Device (e.g. GPUDataAddr_wallMom_Inlet_Edge)
				// cudaStatus = deviceMemcpy(GPUDataAddr_wallMom, Data_dbl_WallMom, nArr_wallMom * sizeof(distribn_t), memcpyHostToDevice);

				//======================================================================
				// GPU results
				bool status  = deviceMemcpy(Data_dbl_WallMom_GPU, GPUDataAddr_wallMom, nArr_wallMom * sizeof(distribn_t), memcpyDeviceToHost);

				if(!status){
					const char * eStr = deviceGetErrorString ();
					printf("GPU memory allocation for wallMom correction term failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					return false;
				}
				/*else{
					printf("GPU memory copy for wallMom GPU successful!!! at proc# %i\n", myPiD); //return true;
				}*/
				//======================================================================

				// Comparison of the CPU and GPU results
				for (unsigned int direction = 1; direction < LatticeType::NUMVECTORS; direction++) // Ignore LB_Dir=0 (resting)
				{
					for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
					{
						site_t shifted_Fluid_Ind = siteIndex-firstIndex;
						// CPU: Data_dbl_WallMom_CPU: WallMom_x[(direction-1)*siteCount + shifted_Fluid_Ind]
						// GPU:

						/*
						if(Data_dbl_WallMom_CPU[(direction-1)*siteCount + shifted_Fluid_Ind] !=0)
							printf("aTime: %d, Site: %d, Dir: %d, CPU correction: %.5e, GPU correction: %.5e \n", mState->GetTimeStep(), siteIndex, direction, Data_dbl_WallMom_CPU[(direction-1)*siteCount + shifted_Fluid_Ind], Data_dbl_WallMom_GPU[(direction-1)*siteCount + shifted_Fluid_Ind]);
							*/
					}
				}


			}


			//=================================================================================================
			// Function to:
			// 	Perform a memory copy from Host to Device (to GPU global memory) for the wall momentum correction and the case of Velocity Inlet/Outlet BCs
			// 		TODO: change to Asynchronous memcpy.
			//=================================================================================================
			template<class LatticeType>
			bool LBM<LatticeType>::memCpy_HtD_GPUmem_WallMom_correction_cudaStream(site_t firstIndex, site_t siteCount, std::vector<double>& wallMom_correction_Iolet, void *GPUDataAddr_wallMom, Stream_t ptrStream)
			{


				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				//======================================================================
				site_t nArr_wallMom = siteCount * (LatticeType::NUMVECTORS -1); // Number of elements of type distribn_t(double)
				distribn_t* Data_dbl_WallMom = new distribn_t[nArr_wallMom];	// distribn_t (type double)

				distribn_t* WallMom_x = &Data_dbl_WallMom[0];

				// Arrange the WallMom data as in method B for the distr. functions - TODO...
				//	Method b: Arrange by index_LB, i.e. wallMom_Dir_1[0 to (nFluid_nodes_Iolet-1)], wallMom_Dir_2[0 to (nFluid_nodes_Iolet-1)], ..., wallMom_Dir_q[0 to (nFluid_nodes_Iolet-1)]
				for (unsigned int direction = 1; direction < LatticeType::NUMVECTORS; direction++) // Ignore LB_Dir=0 (resting)
				{
					for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
					{
						site_t shifted_Fluid_Ind = siteIndex-firstIndex;
						WallMom_x[(direction-1)*siteCount + shifted_Fluid_Ind] = wallMom_correction_Iolet[shifted_Fluid_Ind * LatticeType::NUMVECTORS + direction];

						/*if(WallMom_x[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0 &&
								WallMom_y[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0 &&
								WallMom_z[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0)
								printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction,
																		WallMom_x[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction],
																		WallMom_y[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction],
																		WallMom_z[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]);
						*/

					}
				}
				// Memory copy from host (Data_dbl_WallMom) to Device (e.g. GPUDataAddr_wallMom_Inlet_Edge)
				// cudaStatus = deviceMemcpy(GPUDataAddr_wallMom, Data_dbl_WallMom, nArr_wallMom * sizeof(distribn_t), memcpyHostToDevice);
				bool status = deviceMemcpyAsync(GPUDataAddr_wallMom, Data_dbl_WallMom, nArr_wallMom * sizeof(distribn_t), memcpyHostToDevice, ptrStream);


				delete[] Data_dbl_WallMom;
				//======================================================================
				if(!status){
					const char * eStr = deviceGetErrorString ();
					printf("GPU memory allocation for wallMom correction term failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					return false;
				}
				else{
					return true;
					//printf("GPU memory allocation for wallMom successful!!! at proc# %i\n", myPiD);
				}
				//======================================================================
			}


			//=================================================================================================
			// 	April 2023 - IZ
			//	Function to:
			// 	Perform a memory copy from Host to Device (to GPU global memory) for the wall momentum Prefactor correction and the case of Velocity Inlet/Outlet BCs
			// 	synchronous memcpy.
			//=================================================================================================
			template<class LatticeType>
			bool LBM<LatticeType>::memCpy_HtD_GPUmem_WallMom_prefactor_correction(site_t firstIndex, site_t siteCount, std::vector<double>& wallMom_prefactor_correction_Iolet, void *GPUDataAddr_wallMom_prefactor)
			{


				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				//======================================================================
				site_t nArr_wallMom_prefactor = siteCount * (LatticeType::NUMVECTORS -1); // Number of elements of type distribn_t(double)
				distribn_t* Data_dbl_WallMom_prefactor = new distribn_t[nArr_wallMom_prefactor];	// distribn_t (type double)

				distribn_t* WallMom_x_prefactor = &Data_dbl_WallMom_prefactor[0];

				// Arrange the WallMom_prefactor data as in method B for the distr. functions - TODO...
				//	Method b: Arrange by index_LB, i.e. wallMom_prefactor_Dir_1[0 to (nFluid_nodes_Iolet-1)], wallMom_Dir_2[0 to (nFluid_nodes_Iolet-1)], ..., wallMom_Dir_q[0 to (nFluid_nodes_Iolet-1)]
				for (unsigned int direction = 1; direction < LatticeType::NUMVECTORS; direction++) // Ignore LB_Dir=0 (resting)
				{
					for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
					{
						site_t shifted_Fluid_Ind = siteIndex-firstIndex;
						WallMom_x_prefactor[(direction-1)*siteCount + shifted_Fluid_Ind] = wallMom_prefactor_correction_Iolet[shifted_Fluid_Ind * LatticeType::NUMVECTORS + direction];

						/*if(WallMom_x[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0 &&
								WallMom_y[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0 &&
								WallMom_z[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]!=0)
								printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction,
																		WallMom_x[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction],
																		WallMom_y[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction],
																		WallMom_z[(siteIndex-firstIndex) * LatticeType::NUMVECTORS + direction]);
						*/

					}
				}
				// Memory copy from host (Data_dbl_WallMom_prefactor) to Device (e.g. GPUDataAddr_wallMom_prefactor_Inlet_Edge)
				bool status =  deviceMemcpy(GPUDataAddr_wallMom_prefactor, Data_dbl_WallMom_prefactor, nArr_wallMom_prefactor * sizeof(distribn_t), memcpyHostToDevice);

				delete[] Data_dbl_WallMom_prefactor;
				//======================================================================
				if(!status){
					const char * eStr = deviceGetErrorString ();
					printf("GPU memory allocation for wallMom prefactor correction term failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					return false;
				}
				else{
					return true;
					//printf("GPU memory allocation for wallMom successful!!! at proc# %i\n", myPiD);
				}
				//======================================================================
			}


			//=================================================================================================
			// Function to:
			// 1. read the wall momentum for the case of Velocity Inlet/Outlet BCs
			// 2. fill the appropriate vector that will be used to send the data to the GPU global memory
			//=================================================================================================
		template<class LatticeType>
			void LBM<LatticeType>::read_WallMom_from_propertyCache(site_t firstIndex, site_t siteCount, const lb::MacroscopicPropertyCache& propertyCache,
																															std::vector<util::Vector3D<double> >& wallMom_Iolet)
			{
				std::vector<util::Vector3D<double> > wallMom_forReadFromCache;

				for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
				{
					for (unsigned int direction = 0; direction < LatticeType::NUMVECTORS; direction++)
					{
						// If I already wrote to propertyCache starting from location 0
						//LatticeVelocity site_WallMom = propertyCache.wallMom_Cache.Get((siteIndex-firstIndex)*LatticeType::NUMVECTORS+direction);

						// If I already wrote to propertyCache starting from the location based on fluid ID index
						LatticeVelocity site_WallMom = propertyCache.wallMom_Cache.Get(siteIndex*LatticeType::NUMVECTORS+direction);
						/*
						if (siteIndex==9919 && direction==18){
						if(site_WallMom.x !=0 || site_WallMom.y !=0 || site_WallMom.z !=0)
							printf("Received Wall Mom in LBM - Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction,
																			site_WallMom.x,
																			site_WallMom.y,
																			site_WallMom.z);
						}
						*/
						//wallMom_Iolet.push_back(site_WallMom);
							wallMom_forReadFromCache.push_back(site_WallMom);
					}
				}

				wallMom_Iolet = wallMom_forReadFromCache;
				//wallMom_Iolet.insert(wallMom_Iolet.begin(),wallMom_forReadFromCache);

			}


			//=================================================================================================
			// Function to:
			// 1. read the wall momentum correction for the case of Velocity Inlet/Outlet BCs
			// 2. fill the appropriate vector that will be used to send the data to the GPU global memory
			//=================================================================================================
		template<class LatticeType>
			void LBM<LatticeType>::read_WallMom_correction_from_propertyCache(site_t firstIndex, site_t siteCount, const lb::MacroscopicPropertyCache& propertyCache, std::vector<double>& wallMom_correction_Iolet)
			{
				std::vector<double> wallMom_correction_forReadFromCache;
				wallMom_correction_forReadFromCache.reserve(siteCount*LatticeType::NUMVECTORS);

				for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
				{
					for (unsigned int direction = 0; direction < LatticeType::NUMVECTORS; direction++)
					{
						// If I already wrote to propertyCache starting from location 0
						//double site_WallMom_correction = propertyCache.wallMom_correction_Cache.Get((siteIndex-firstIndex)*LatticeType::NUMVECTORS+direction);

						// If I already wrote to propertyCache starting from the location based on fluid ID index
						double site_WallMom_correction = propertyCache.wallMom_correction_Cache.Get(siteIndex*LatticeType::NUMVECTORS+direction);
						/*
						if (siteIndex==9919 && direction==18){
						if(site_WallMom.x !=0 || site_WallMom.y !=0 || site_WallMom.z !=0)
							printf("Received Wall Mom in LBM - Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction,
																			site_WallMom.x,
																			site_WallMom.y,
																			site_WallMom.z);
						}
						*/
						//wallMom_Iolet.push_back(site_WallMom);
							wallMom_correction_forReadFromCache.push_back(site_WallMom_correction);
					}
				}

				wallMom_correction_Iolet = wallMom_correction_forReadFromCache;
				//wallMom_correction_Iolet.insert(wallMom_correction_Iolet.begin(),wallMom_correction_forReadFromCache);
			}


			template<class LatticeType>
				void LBM<LatticeType>::swap_Pointers_GPU_glb_mem(void **pointer_GPU_glb_left, void **pointer_GPU_gbl_right)
				{
					void *pSwap = *pointer_GPU_glb_left;
			    *pointer_GPU_glb_left = *pointer_GPU_gbl_right;
			    *pointer_GPU_gbl_right = pSwap;
				}




			//=================================================================================================
			// Function for reading the macroVariables:
			//	a. Density [nFluid nodes]
			//	b. Velocity[nFluid nodes*3]
			// from the GPU and copying to the CPU (device-to-host mem. copy - Asynchronous: stream stream_Read_Data_GPU_Dens)
			//
			// When data needs to be saved to the disk on the CPU
			//
			// Remember that from the host perspective the mem copy is synchronous, i.e. blocking
			// so the host will wait the data transfer to complete and then proceed to the next function call
			// To do:
			// Address the issue pointed below with the Collision Implementation type
			//
			// April 2023
			// 	c. Transfer the wall Shear Stress magnitude from the GPU

			// Modified the following so that it is invoked only when necessary
			//				i.e. if (propertyCache.densityCache.RequiresRefresh())
			//					or if (propertyCache.velocityCache.RequiresRefresh())
			//					or if (propertyCache.wallShearStressMagnitudeCache.RequiresRefresh())

			// Note that dens_GPU, velx, wall shear stresses etc are static (declared in lb.h (public members of class LBM))
			// and defined in a global scope in the source file
			//=================================================================================================
		template<class LatticeType>
			//bool LBM<LatticeType>::Read_Macrovariables_GPU_to_CPU(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache, kernels::HydroVars<LB_KERNEL>& hydroVars(geometry::Site<geometry::LatticeData>&_site)) // Is it necessary to use lb::MacroscopicPropertyCache& propertyCache or just propertyCache, as it is being initialised with the LBM constructor???
			bool LBM<LatticeType>::Read_Macrovariables_GPU_to_CPU(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache)
			{
				/**
				Remember to address the following point in the future - Only valid for the LBGK collision kernel:
				Need to make it more general - Pass the Collision Kernel Impl. typename - To do!!!
				kernels::HydroVars<lb::kernels::LBGK<lb::lattices::D3Q19> > hydroVars(site);
				*/

				bool res_Read_MacroVars = true;

				// Total number of fluid sites
				uint64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount(); // Actually GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)

				//static size_t allocatedSize = 0;
				unsigned long long MemSz = siteCount * sizeof(distribn_t);
				//======================================================================

				// Allocate memory first (if not already allocated)
				// a. Density
				if (dens_GPU == nullptr){
					bool status = deviceHostAlloc((void**)&dens_GPU, MemSz);
					memset(dens_GPU, 0, MemSz);
					if(!status){
								printf("Density Memory allocation failure...\n");
								fflush(stdout);
								res_Read_MacroVars = false;
								return res_Read_MacroVars;
						}
				}

				// b. Velocity
				if (vx_GPU == nullptr){
					bool status = deviceHostAlloc((void**)&vx_GPU, MemSz);
					memset(vx_GPU, 0, MemSz);
					if(!status){
								printf("Vx Memory allocation failure...\n");
								fflush(stdout);
								res_Read_MacroVars = false;
								return res_Read_MacroVars;
						}
				}
				if (vy_GPU == nullptr){
					bool status = deviceHostAlloc((void**)&vy_GPU, MemSz);
					memset(vy_GPU, 0, MemSz);
					if(!status){
								printf("Vy Memory allocation failure...\n");
								fflush(stdout);
								res_Read_MacroVars = false;
								return res_Read_MacroVars;
						}
				}
				if (vz_GPU == nullptr){
					bool status = deviceHostAlloc((void**)&vz_GPU, MemSz);
					memset(vz_GPU, 0, MemSz);
					if(!status){
								printf("Vz Memory allocation failure...\n");
								fflush(stdout);
								res_Read_MacroVars = false;
								return res_Read_MacroVars;
						}
				}

				//--------------------------------------------------------------------------
				// Proceed with the memory transfer operations
				// a. Density
				if (propertyCache.densityCache.RequiresRefresh()) {

					bool status = deviceMemcpyAsync(dens_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[firstIndex]),
			  					MemSz, memcpyDeviceToHost, stream_Read_Data_GPU_Dens);

					if(!status){
						printf("GPU memory transfer for density failed\n");
						fflush(stdout);
						deviceFreeHost(dens_GPU);
						dens_GPU = nullptr;
						res_Read_MacroVars = false;
					}
				}
				//--------------------------------------------------------------------------
				// b. Velocity
				if (propertyCache.velocityCache.RequiresRefresh()) {

						if (vx_GPU && vy_GPU && vz_GPU) {
							bool status = deviceMemcpyAsync(vx_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[1ULL*nFluid_nodes + firstIndex]), MemSz,
				            memcpyDeviceToHost, stream_Read_Data_GPU_Dens);

							if(!status){
								printf("GPU memory transfer Vel(1) failed\n");
								fflush(stdout);
								deviceFreeHost(vx_GPU);
								vx_GPU = nullptr;
								res_Read_MacroVars = false;
							}

							status = deviceMemcpyAsync(vy_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[2ULL*nFluid_nodes + firstIndex]), MemSz,
				            memcpyDeviceToHost, stream_Read_Data_GPU_Dens);

							if(!status){
								printf("GPU memory transfer Vel(2) failed\n");
								fflush(stdout);
								deviceFreeHost(vy_GPU);
								vy_GPU = nullptr;
								res_Read_MacroVars = false;
							}

							status = deviceMemcpyAsync(vz_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[3ULL*nFluid_nodes + firstIndex]), MemSz,
				            memcpyDeviceToHost, stream_Read_Data_GPU_Dens);

							if(!status){
								printf("GPU memory transfer Vel(3) failed\n");
								fflush(stdout);
								deviceFreeHost(vz_GPU);
								vz_GPU = nullptr;
								res_Read_MacroVars = false;
							}

						}
				}

				// Synchronize the stream to ensure all memcpy operations (density and velocity) are completed
				deviceStreamSynchronize(stream_Read_Data_GPU_Dens);

				// Copy the density - velocity values in hydroVars and propertyCache
				if (propertyCache.densityCache.RequiresRefresh() || propertyCache.velocityCache.RequiresRefresh()) {
						for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++) {
								geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

								// Need to make it more general - Pass the Collision Kernel Impl. typename - To do!!!
								kernels::HydroVars<LB_KERNEL> hydroVars(site);

								// Pass the density and velocity to the hydroVars and the densityCache, velocityCache
								if (propertyCache.densityCache.RequiresRefresh()) {
										hydroVars.density = dens_GPU[siteIndex - firstIndex];
										propertyCache.densityCache.Put(siteIndex, hydroVars.density);
								}

								if (propertyCache.velocityCache.RequiresRefresh()) {
										hydroVars.velocity.x = vx_GPU[siteIndex - firstIndex];
										hydroVars.velocity.y = vy_GPU[siteIndex - firstIndex];
										hydroVars.velocity.z = vz_GPU[siteIndex - firstIndex];
										propertyCache.velocityCache.Put(siteIndex, hydroVars.velocity);
								}
						}
				}

				// Wall Shear Stress Magnitude Case - Fill First the non-adjacent to walls sites
				if (propertyCache.wallShearStressMagnitudeCache.RequiresRefresh()) {
					for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++) {
						geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);
						//
						if (!site.IsWall()) {
								distribn_t stress = NO_VALUE; // constants.h: const distribn_t NO_VALUE = std::numeric_limits<distribn_t>::max();
								propertyCache.wallShearStressMagnitudeCache.Put(site.GetIndex(), stress);
						}
						//
					}
				}

				// Wall Shear Stress Magnitude Case - Fill the adjacent to walls sites
				if (propertyCache.wallShearStressMagnitudeCache.RequiresRefresh()) {
						distribn_t stress;

						auto copyWallShearStress_to_propertyCache = [&](distribn_t* stressArray, site_t start_Index, site_t total_numElements) {
								for (site_t siteIndex = start_Index; siteIndex < (start_Index + total_numElements); siteIndex++) {
										geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);
										stress = stressArray[siteIndex - start_Index];
										propertyCache.wallShearStressMagnitudeCache.Put(site.GetIndex(), stress);
								}
						};

						//--------------------
						// Site limits
						// A. Domain Edge
						site_t start_Index_Edge_Type2 = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0);
						site_t total_numElements_Edge_Type2 = mLatDat->GetDomainEdgeCollisionCount(1);
						site_t start_Index_Edge_Type5 = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
																						+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);
						site_t total_numElements_Edge_Type5 = mLatDat->GetDomainEdgeCollisionCount(4);
						site_t start_Index_Edge_Type6 = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
																						+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3) + mLatDat->GetDomainEdgeCollisionCount(4);
						site_t total_numElements_Edge_Type6 = mLatDat->GetDomainEdgeCollisionCount(5);

						// B. Inner Domain
						site_t start_Index_Inner_Type2 = mLatDat->GetMidDomainCollisionCount(0);
						site_t total_numElements_Inner_Type2 = mLatDat->GetMidDomainCollisionCount(1);
						site_t start_Index_Inner_Type5 = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
						site_t total_numElements_Inner_Type5 = mLatDat->GetMidDomainCollisionCount(4);
						site_t start_Index_Inner_Type6 = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2)
																						 + mLatDat->GetMidDomainCollisionCount(3) + mLatDat->GetMidDomainCollisionCount(4);
						site_t total_numElements_Inner_Type6 = mLatDat->GetMidDomainCollisionCount(5);
						//--------------------

						auto allocateAndCopyD2H = [&](distribn_t*& stressArray, void* GPUDataAddr, site_t total_numElements) {
								if (total_numElements != 0) {

									// Allocate only the first time
									if (stressArray == nullptr) {
										bool status = deviceHostAlloc((void**)&stressArray, total_numElements * sizeof(distribn_t));
										memset(stressArray, 0, total_numElements * sizeof(distribn_t));
										if (stressArray == nullptr) {
												printf("Wall Shear Stress magnitude allocation failure");
												fflush(stdout);
												res_Read_MacroVars = false;
										}
									}

									// D2H mem. copy  wall shear stress magnitudes
									MemSz = total_numElements * sizeof(distribn_t);
									bool status = deviceMemcpyAsync(stressArray, GPUDataAddr, MemSz, memcpyDeviceToHost, stream_Read_Data_GPU_Dens);
									if(!status){
										printf("GPU memory transfer for Wall Shear Stress magnitude failed...\n");
										fflush(stdout);
										deviceFreeHost(stressArray);
										stressArray = nullptr;
										res_Read_MacroVars = false;
									}
								}
							};

						// Domain Edge
						allocateAndCopyD2H(WallShearStressMagn_Edge_Type2_GPU, GPUDataAddr_WallShearStressMagn_Edge_Type2, total_numElements_Edge_Type2);
						allocateAndCopyD2H(WallShearStressMagn_Edge_Type5_GPU, GPUDataAddr_WallShearStressMagn_Edge_Type5, total_numElements_Edge_Type5);
						allocateAndCopyD2H(WallShearStressMagn_Edge_Type6_GPU, GPUDataAddr_WallShearStressMagn_Edge_Type6, total_numElements_Edge_Type6);

						// Inner Domain
						allocateAndCopyD2H(WallShearStressMagn_Inner_Type2_GPU, GPUDataAddr_WallShearStressMagn_Inner_Type2, total_numElements_Inner_Type2);
						allocateAndCopyD2H(WallShearStressMagn_Inner_Type5_GPU, GPUDataAddr_WallShearStressMagn_Inner_Type5, total_numElements_Inner_Type5);
						allocateAndCopyD2H(WallShearStressMagn_Inner_Type6_GPU, GPUDataAddr_WallShearStressMagn_Inner_Type6, total_numElements_Inner_Type6);

						// Ensure that the memory allocations and copies above have completed
						deviceStreamSynchronize(stream_Read_Data_GPU_Dens);


						if (WallShearStressMagn_Edge_Type2_GPU) copyWallShearStress_to_propertyCache(WallShearStressMagn_Edge_Type2_GPU, start_Index_Edge_Type2, total_numElements_Edge_Type2);
						if (WallShearStressMagn_Edge_Type5_GPU) copyWallShearStress_to_propertyCache(WallShearStressMagn_Edge_Type5_GPU, start_Index_Edge_Type5, total_numElements_Edge_Type5);
						if (WallShearStressMagn_Edge_Type6_GPU) copyWallShearStress_to_propertyCache(WallShearStressMagn_Edge_Type6_GPU, start_Index_Edge_Type6, total_numElements_Edge_Type6);

						if (WallShearStressMagn_Inner_Type2_GPU) copyWallShearStress_to_propertyCache(WallShearStressMagn_Inner_Type2_GPU, start_Index_Inner_Type2, total_numElements_Inner_Type2);
						if (WallShearStressMagn_Inner_Type5_GPU) copyWallShearStress_to_propertyCache(WallShearStressMagn_Inner_Type5_GPU, start_Index_Inner_Type5, total_numElements_Inner_Type5);
						if (WallShearStressMagn_Inner_Type6_GPU) copyWallShearStress_to_propertyCache(WallShearStressMagn_Inner_Type6_GPU, start_Index_Inner_Type6, total_numElements_Inner_Type6);
				}

				return res_Read_MacroVars;
			}


			//=================================================================================================
			// Function for cleaning up the static pointers
			// 	associated with the function Read_Macrovariables_GPU_to_CPU
			// 	A. density, velocity
			// 	B. wall shear stress magnitude
			//=================================================================================================
		template<class LatticeType>
			void LBM<LatticeType>::Cleanup_GPU_pinned_Memory() {
			    if (dens_GPU) {
			        deviceFreeHost(dens_GPU);
			        dens_GPU = nullptr;
			    }
			    if (vx_GPU) {
			        deviceFreeHost(vx_GPU);
			        vx_GPU = nullptr;
			    }
			    if (vy_GPU) {
			        deviceFreeHost(vy_GPU);
			        vy_GPU = nullptr;
			    }
			    if (vz_GPU) {
			        deviceFreeHost(vz_GPU);
			        vz_GPU = nullptr;
			    }
			    if (WallShearStressMagn_Edge_Type2_GPU) {
			        deviceFreeHost(WallShearStressMagn_Edge_Type2_GPU);
			        WallShearStressMagn_Edge_Type2_GPU = nullptr;
			    }
			    if (WallShearStressMagn_Edge_Type5_GPU) {
			        deviceFreeHost(WallShearStressMagn_Edge_Type5_GPU);
			        WallShearStressMagn_Edge_Type5_GPU = nullptr;
			    }
			    if (WallShearStressMagn_Edge_Type6_GPU) {
			        deviceFreeHost(WallShearStressMagn_Edge_Type6_GPU);
			        WallShearStressMagn_Edge_Type6_GPU = nullptr;
			    }
			    if (WallShearStressMagn_Inner_Type2_GPU) {
			        deviceFreeHost(WallShearStressMagn_Inner_Type2_GPU);
			        WallShearStressMagn_Inner_Type2_GPU = nullptr;
			    }
			    if (WallShearStressMagn_Inner_Type5_GPU) {
			        deviceFreeHost(WallShearStressMagn_Inner_Type5_GPU);
			        WallShearStressMagn_Inner_Type5_GPU = nullptr;
			    }
			    if (WallShearStressMagn_Inner_Type6_GPU) {
			        deviceFreeHost(WallShearStressMagn_Inner_Type6_GPU);
			        WallShearStressMagn_Inner_Type6_GPU = nullptr;
			    }
			}


			//=================================================================================================
			/** Check the following!!! TODO!!!
				Function for reading:
							a. the Distribution Functions post-collision, fNew,
							b. Density [nFluid nodes]
							c. Velocity[nFluid nodes*3]
				 		ONLY FOR THE FLUID NODES at the current rank (WITHOUT the f's in totalSharedFs)
						from the GPU and copying to the CPU (device-to-host mem. copy - Synchronous)


				//	b.	When data needs to be saved to the disk on the CPU
				//
				// Remember that from the host perspective the mem copy is synchronous, i.e. blocking
				// so the host will wait the data transfer to complete and then proceed to the next function call
				//=================================================================================================
			*/
		template<class LatticeType>
			bool LBM<LatticeType>::Read_DistrFunctions_GPU_to_CPU_FluidSites()
			{


				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				// Total number of fluid sites
				uint64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount(); // Actually GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)
				uint64_t totSharedFs = mLatDat->totalSharedFs;

				//--------------------------------------------------------------------------
				// a. Distribution functions fNew, i.e. post collision populations:
				// unsigned long long TotalMem_dbl_fOld_b = ( nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs)  * sizeof(distribn_t); // Total memory size
				unsigned long long TotalMem_dbl_fNew_b = ( nFluid_nodes * LatticeType::NUMVECTORS)  * sizeof(distribn_t); // Total memory size

				//distribn_t* fNew_GPU_b = new distribn_t[TotalMem_dbl_fNew_b/sizeof(distribn_t)];	// distribn_t (type double)
				distribn_t* fNew_GPU_b = new distribn_t[nFluid_nodes * LatticeType::NUMVECTORS ];	// distribn_t (type double)

				//if(!fOld_GPU_b || !fNew_GPU_b){ std::cout << "Memory allocation error - ReadGPU_distr" << std::endl; return false;}
				/* else{ std::printf("Memory allocation for ReadGPU_distr successful from Proc# %i \n\n", myPiD); } */
				if(!fNew_GPU_b){ std::cout << "Memory allocation error - ReadGPU_distr" << std::endl; return false;}

				//cudaStatus = deviceMemcpyAsync(fNew_GPU_b, &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[0]), TotalMem_dbl_fNew_b, memcpyDeviceToHost, stream_Read_distr_Data_GPU);
				bool status = deviceMemcpy(&(fNew_GPU_b[0]), &(((distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat)[0]), TotalMem_dbl_fNew_b, memcpyDeviceToHost);
				if(!status){
					const char * eStr = deviceGetErrorString();
					printf("GPU memory transfer for ReadGPU_distr failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					delete[] fNew_GPU_b;
					return false;
				}

				/**
				 	Place the received distributions (fNew_GPU_b) in the PROPER location
					TAKING INTO ACCOUNT the different data layout in the CPU version of hemeLB
				*/
				// Read fNew[] from all the fluid nodes - Ignore what is in totalSharedFs
				site_t firstIndex = 0;
				site_t siteCount = nFluid_nodes;

				for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
				{
					geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

					for (int ii = 0; ii < LatticeType::NUMVECTORS; ii++)
					{
						*(mLatDat->GetFNew(siteIndex * LatticeType::NUMVECTORS + ii)) = fNew_GPU_b[ii* nFluid_nodes + siteIndex];
						*(mLatDat->GetFOld(siteIndex * LatticeType::NUMVECTORS + ii)) = fNew_GPU_b[ii* nFluid_nodes + siteIndex];

						//******************************************************************************
						// FNew index in hemeLB HOST array (after streaming):
						//			site.GetStreamedIndex<LatticeType> (ii) = the element in the array neighbourIndices[iSiteIndex * LatticeType::NUMVECTORS + iDirectionIndex];
						//
						// 			int64_t streamedIndex = site.GetStreamedIndex<LatticeType> (ii); // ii: direction

						// given the streamed index value find the fluid ID index: iFluidIndex = (Array_Index - iDirectionIndex)/NumVectors,
						//	i.e. iFluidIndex = (site.GetStreamedIndex<LatticeType> (ii) - ii)/NumVectors;
						// Applies if streaming ends within the domain in the same rank.
						// If not then the postcollision fNew will stream in the neighbouring rank.
						// It will be placed then in location for the totalSharedFs

						// Need to include the case of inlet BCs - Unstreamed Unknown populations - Done!!!
						//******************************************************************************

						/*
						if (site.HasIolet(ii)) //ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
						{
							int unstreamed_dir = LatticeType::INVERSEDIRECTIONS[ii];

							// unsigned long long heme_Index_Array = siteIndex * LatticeType::NUMVECTORS + unstreamed_dir;
							*(mLatDat->GetFNew(siteIndex * LatticeType::NUMVECTORS + unstreamed_dir)) = fNew_GPU_b[unstreamed_dir* nFluid_nodes + siteIndex] ; // ghostHydrovars.GetFEq()[unstreamed];
							*(mLatDat->GetFOld(siteIndex * LatticeType::NUMVECTORS + unstreamed_dir)) = fNew_GPU_b[unstreamed_dir* nFluid_nodes + siteIndex] ; // ghostHydrovars.GetFEq()[unstreamed];
						}
						else if (site.HasWall(ii)){
							// Propagate the post-collisional f into the opposite direction - Simple Bounce Back: same FluidIndex
							unsigned long long BB_Index_Array = siteIndex * LatticeType::NUMVECTORS + LatticeType::INVERSEDIRECTIONS[ii];
							*(mLatDat->GetFNew(BB_Index_Array)) = fNew_GPU_b[(LatticeType::INVERSEDIRECTIONS[ii])* nFluid_nodes + siteIndex];
							*(mLatDat->GetFOld(BB_Index_Array)) = fNew_GPU_b[(LatticeType::INVERSEDIRECTIONS[ii])* nFluid_nodes + siteIndex];

							// printf("Site ID = %lld - Wall in Dir: %d, Streamed Array Index = %lld /(%lld), Value fNew = %.5e \n\n", siteIndex, ii, BB_Index_Array, (nFluid_nodes * LatticeType::NUMVECTORS), fNew_GPU_b[(LatticeType::INVERSEDIRECTIONS[ii])* nFluid_nodes + siteIndex]);
						}
						else{ // If Bulk-link

							if((site.GetStreamedIndex<LatticeType> (ii)) < (nFluid_nodes * LatticeType::NUMVECTORS)){		// Within the domain
								// fNew_GPU_b index should be:
								// Dir(b) * nFluidnodes + iFluidIndex, i.e. fNew_GPU_b[ii * mLatDat->GetLocalFluidSiteCount() + iFluidIndex]
								uint64_t iFluidIndex = ((site.GetStreamedIndex<LatticeType> (ii)) - ii)/LatticeType::NUMVECTORS;

								*(mLatDat->GetFNew(site.GetStreamedIndex<LatticeType> (ii))) = fNew_GPU_b[ii * nFluid_nodes + iFluidIndex]; // When streaming on the GPU
								*(mLatDat->GetFOld(site.GetStreamedIndex<LatticeType> (ii))) = fNew_GPU_b[ii * nFluid_nodes + iFluidIndex]; // When streaming on the GPU
								// * (mLatDat->GetFNew(site.GetStreamedIndex<LatticeType> (ii))) = fNew_GPU_b[ii * (mLatDat->GetLocalFluidSiteCount()) + siteIndex]; // no streaming on the GPU

								//printf("Fluid ID: %lld (/%lld), Data ADddres To Stream: %lld, fNew_GPU[%d] = %.5f \n\n", iFluidIndex, nFluid_nodes, site.GetStreamedIndex<LatticeType> (ii), ii, fNew_GPU_b[ii * nFluid_nodes + iFluidIndex]);
							}
							//else	// Will Stream out of the domain to neighbour ranks (put in totalSharedFs)
							//{
							//	*(mLatDat->GetFNew(site.GetStreamedIndex<LatticeType> (ii))) = fNew_GPU_b[site.GetStreamedIndex<LatticeType> (ii)];
							//	*(mLatDat->GetFOld(site.GetStreamedIndex<LatticeType> (ii))) = fNew_GPU_b[site.GetStreamedIndex<LatticeType> (ii)];
							//	// printf("Data ADddres: %lld, fNew_GPU[%d] = %.5f \n\n", site.GetStreamedIndex<LatticeType> (ii), ii, fNew_GPU_b[site.GetStreamedIndex<LatticeType> (ii)]);
							//	if (site.GetStreamedIndex<LatticeType> (ii) >= (nFluid_nodes * LatticeType::NUMVECTORS+1+totSharedFs)) printf("Error!!! Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", ii, nFluid_nodes * LatticeType::NUMVECTORS+1+totSharedFs, site.GetStreamedIndex<LatticeType> (ii) );
							//}

						} // Ends the if Bulk link case
						*/

						/*
						//
						// Debugging
						uint64_t max_Perm_Ind = nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs;
						uint64_t ind_fNew_GPU_b = ii * nFluid_nodes + iFluidIndex;

						uint64_t max_Perm_Ind_CPU = nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs;
						uint64_t ind_GetFNew = site.GetStreamedIndex<LatticeType> (ii);

						// if(iFluidIndex > nFluid_nodes) printf("Attempting to access Fluid ID index = %lld - Max. Fluid nodes = %lld  \n\n", iFluidIndex, nFluid_nodes);
						if(ind_GetFNew > max_Perm_Ind_CPU) printf("Wow!!! Attempting to access CPU index = %lld - Max. Permited = %lld  \n\n", ind_GetFNew, max_Perm_Ind_CPU);
						if(ind_fNew_GPU_b > max_Perm_Ind) printf("Error!!! Attempting to access index = %lld - Max. Permited = %lld  \n\n", ind_fNew_GPU_b, max_Perm_Ind);

						// printf("Index in fNew: Method 1: SiteIndex = %lld, Index of fNew[%d] = %lld Vs Index_2 = %lld \n\n", siteIndex, ii, (ii * mLatDat->GetLocalFluidSiteCount() + iFluidIndex), (siteIndex*LatticeType::NUMVECTORS + ii));
						// printf("SiteIndex = %lld, Streamed Fluid SiteIndex = %lld, fNew[%d] = %.5f \n\n", siteIndex, iFluidIndex, ii, fNew_GPU_b[ii * mLatDat->GetLocalFluidSiteCount() + iFluidIndex]);
						//
						*/
					} // Ends the loop over the lattice directions (	for (int ii = 0; ii < LatticeType::NUMVECTORS; ii++))
			}// Ends the loop over the sites
				//

				// Delete the variables when copy is completed
				delete[] fNew_GPU_b;

				return true;
			}


			//=================================================================================================
			// Function for reading:
			//	a. the Distribution Functions post-collision, fNew,
			//	b. Density [nFluid nodes]
			//	c. Velocity[nFluid nodes*3]
			// from the GPU and copying to the CPU (device-to-host mem. copy - Synchronous)
			//
			//	Total (Read_DistrFunctions_GPU_to_CPU_tot) refers to the fact that these distr. functions include the totalSharedFs as well
			//
			// Development phase:
			//	Necessary at each time step as ALL data need to reside on the CPU
			//
			// Final phase: (All collision/streaming types implemented)
			//	a.	to be called at the domain bundaries
			//		for the exchange of the fNew to be exchanged
			//	b.	When data needs to be saved to the disk on the CPU
			//
			// Remember that from the host perspective the mem copy is synchronous, i.e. blocking
			// so the host will wait the data transfer to complete and then proceed to the next function call
			//=================================================================================================
		template<class LatticeType>
			bool LBM<LatticeType>::Read_DistrFunctions_GPU_to_CPU_tot(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache) // Is it necessary to use lb::MacroscopicPropertyCache& propertyCache or just propertyCache, as it is being initialised with the LBM constructor???
			{

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				// Total number of fluid sites
				uint64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount(); // Actually GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)
				uint64_t totSharedFs = mLatDat->totalSharedFs;

				//--------------------------------------------------------------------------
				// a. Distribution functions fNew, i.e. post collision populations:
				// unsigned long long TotalMem_dbl_fOld_b = ( nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs)  * sizeof(distribn_t); // Total memory size
				unsigned long long TotalMem_dbl_fNew_b = ( nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs)  * sizeof(distribn_t); // Total memory size

				//distribn_t* fNew_GPU_b = new distribn_t[TotalMem_dbl_fNew_b/sizeof(distribn_t)];	// distribn_t (type double)
				distribn_t* fNew_GPU_b = new distribn_t[nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs];	// distribn_t (type double)

				//if(!fOld_GPU_b || !fNew_GPU_b){ std::cout << "Memory allocation error - ReadGPU_distr" << std::endl; return false;}
				/* else{ std::printf("Memory allocation for ReadGPU_distr successful from Proc# %i \n\n", myPiD); } */
				if(!fNew_GPU_b){ std::cout << "Memory allocation error - ReadGPU_distr" << std::endl; return false;}

				//cudaStatus = deviceMemcpyAsync(fNew_GPU_b, &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[0]), TotalMem_dbl_fNew_b, memcpyDeviceToHost, stream_Read_distr_Data_GPU);
				bool status = deviceMemcpy(&(fNew_GPU_b[0]), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[0]), TotalMem_dbl_fNew_b, memcpyDeviceToHost);
				if(!status){
					const char * eStr = deviceGetErrorString ();
					printf("GPU memory transfer for ReadGPU_distr failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					delete[] fNew_GPU_b;
					return false;
				}

				//--------------------------------------------------------------------------
				//	b. Density

				distribn_t* dens_GPU = new distribn_t[nFluid_nodes];

				if(dens_GPU==0){printf("Density Memory allocation failure"); return false;}

				unsigned long long MemSz = nFluid_nodes*sizeof(distribn_t);

				status = deviceMemcpy(dens_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[0]), MemSz, memcpyDeviceToHost);
				//cudaStatus = deviceMemcpyAsync(dens_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[0]), MemSz, memcpyDeviceToHost, stream_Read_Data_GPU_Dens);

				if(!status){
					printf("GPU memory transfer for density failed\n");
					delete[] dens_GPU;
					return false;
				}

				// c. Velocity
				distribn_t* vx_GPU = new distribn_t[nFluid_nodes];
				distribn_t* vy_GPU = new distribn_t[nFluid_nodes];
				distribn_t* vz_GPU = new distribn_t[nFluid_nodes];

				if(vx_GPU==0 || vy_GPU==0 || vz_GPU==0){ printf("Memory allocation failure"); return false;}

				status = deviceMemcpy(vx_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[1ULL*nFluid_nodes]), MemSz, memcpyDeviceToHost);
				//cudaStatus = deviceMemcpyAsync(vx_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[1ULL*nFluid_nodes]), MemSz, memcpyDeviceToHost, stream_Read_Data_GPU_Vel);
				if(!status){
					printf("GPU memory transfer Vel(1) failed\n");
					delete[] vx_GPU;
					return false;
				}

				status = deviceMemcpy(vy_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[2ULL*nFluid_nodes]), MemSz, memcpyDeviceToHost);
				//cudaStatus = deviceMemcpyAsync(vy_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[2ULL*nFluid_nodes]), MemSz, memcpyDeviceToHost, stream_Read_Data_GPU_Vel);
				if(!status){
					printf("GPU memory transfer Vel(2) failed\n");
					delete[] vy_GPU;
					return false;
				}

				status = deviceMemcpy(vz_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[3ULL*nFluid_nodes]), MemSz, memcpyDeviceToHost);
				//cudaStatus = deviceMemcpyAsync(vz_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[3ULL*nFluid_nodes]), MemSz, memcpyDeviceToHost, stream_Read_Data_GPU_Vel);
				if(!status){
					printf("GPU memory transfer Vel(2) failed\n");
					delete[] vz_GPU;
					return false;
				}
				//--------------------------------------------------------------------------
				//hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // Check for last cuda error: Remember that it is in DEBUG flag


				//
				// Read only the density, velocity and fNew[] that needs to be passed to the CPU at the updated sites: The ones that had been updated in the GPU collision kernel
				for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
				{
					geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);
					// printf("site.GetIndex() = %lld Vs siteIndex = %lld \n\n", site.GetIndex(), siteIndex); // Works fine - Access to the correct site

					//
					// Need to make it more general - Pass the Collision Kernel Impl. typename - To do!!!
					kernels::HydroVars<lb::kernels::LBGK<lb::lattices::D3Q19> > hydroVars(site);

					// Pass the density and velocity to the hydroVars and the densityCache, velocityCache
					hydroVars.density = dens_GPU[siteIndex];
					hydroVars.velocity.x = vx_GPU[siteIndex];
					hydroVars.velocity.y = vy_GPU[siteIndex];
					hydroVars.velocity.z = vz_GPU[siteIndex];
					propertyCache.densityCache.Put(siteIndex, hydroVars.density);		//propertyCache.densityCache.Put(site.GetIndex(), hydroVars.density);
					propertyCache.velocityCache.Put(siteIndex, hydroVars.velocity);	//propertyCache.velocityCache.Put(site.GetIndex(), hydroVars.velocity);

					// printf("propertyCache.densityCache.RequiresRefresh() = %d and propertyCache.velocityCache.RequiresRefresh() = %d \n\n", propertyCache.densityCache.RequiresRefresh(), propertyCache.velocityCache.RequiresRefresh());
					// Checked - Values set to 1 (true) at each time-step -> No Need to include the if statement for these variables as below. Remove all commented out code
					/*
						// Either the following or the whole function UpdateMinsAndMaxes - Check that the above works first.
						if (propertyCache.densityCache.RequiresRefresh())
						{
							propertyCache.densityCache.Put(site.GetIndex(), hydroVars.density);
						}

						if (propertyCache.velocityCache.RequiresRefresh())
						{
							propertyCache.velocityCache.Put(site.GetIndex(), hydroVars.velocity);
						}
					*/
						/*
						streamers::BaseStreamer<streamers::SimpleCollideAndStream>::template UpdateMinsAndMaxes<tDoRayTracing>(site,
								hydroVars,
								lbmParams,
								propertyCache);
						*/

						// Need to add the function UpdateMinsAndMaxes OR maybe just get the density and velocity
						// Need to calculate these variables - Done!!!
						// To do:
						// 1. Allocate memory on the GPU global memory for density and velocity - Done!!!
						// 2. Calculate these MacroVariables on the GPU - either in the collision/streaming kernel or in a separate kernel -Think about this!!!
						// 3. Memory Copy of density and Velocity from the GPU to the CPU - Done!!!
						//		and then do:
						//	3.a. propertyCache.densityCache.Put(site.GetIndex(), hydroVars.density);	Done!!!
						//	3.b. propertyCache.velocityCache.Put(site.GetIndex(), hydroVars.velocity);	Done!!!


					//

					for (int ii = 0; ii < LatticeType::NUMVECTORS; ii++)
					{
						//******************************************************************************
						// FNew index in hemeLB HOST array (after streaming):
						//			site.GetStreamedIndex<LatticeType> (ii) = the element in the array neighbourIndices[iSiteIndex * LatticeType::NUMVECTORS + iDirectionIndex];
						//
						// 			int64_t streamedIndex = site.GetStreamedIndex<LatticeType> (ii); // ii: direction

						// given the streamed index value find the fluid ID index: iFluidIndex = (Array_Index - iDirectionIndex)/NumVectors,
						//	i.e. iFluidIndex = (site.GetStreamedIndex<LatticeType> (ii) - ii)/NumVectors;
						// Applies if streaming ends within the domain in the same rank.
						// If not then the postcollision fNew will stream in the neighbouring rank.
						// It will be placed then in location for the totalSharedFs

						// Need to include the case of inlet BCs - Unstreamed Unknown populations - Done!!!
						//******************************************************************************


						if (site.HasIolet(ii)) //ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
						{
							int unstreamed_dir = LatticeType::INVERSEDIRECTIONS[ii];

							// unsigned long long heme_Index_Array = siteIndex * LatticeType::NUMVECTORS + unstreamed_dir;
							*(mLatDat->GetFNew(siteIndex * LatticeType::NUMVECTORS + unstreamed_dir)) = fNew_GPU_b[unstreamed_dir* nFluid_nodes + siteIndex] ; // ghostHydrovars.GetFEq()[unstreamed];

						}
						else if (site.HasWall(ii)){
							// Propagate the post-collisional f into the opposite direction - Simple Bounce Back: same FluidIndex
							unsigned long long BB_Index_Array = siteIndex * LatticeType::NUMVECTORS + LatticeType::INVERSEDIRECTIONS[ii];
							*(mLatDat->GetFNew(BB_Index_Array)) = fNew_GPU_b[(LatticeType::INVERSEDIRECTIONS[ii])* nFluid_nodes + siteIndex];
							// printf("Site ID = %lld - Wall in Dir: %d, Streamed Array Index = %lld /(%lld), Value fNew = %.5e \n\n", siteIndex, ii, BB_Index_Array, (nFluid_nodes * LatticeType::NUMVECTORS), fNew_GPU_b[(LatticeType::INVERSEDIRECTIONS[ii])* nFluid_nodes + siteIndex]);
						}
						else{ // If Bulk-link

							if((site.GetStreamedIndex<LatticeType> (ii)) < (nFluid_nodes * LatticeType::NUMVECTORS)){		// Within the domain
								// fNew_GPU_b index should be:
								// Dir(b) * nFluidnodes + iFluidIndex, i.e. fNew_GPU_b[ii * mLatDat->GetLocalFluidSiteCount() + iFluidIndex]
								uint64_t iFluidIndex = ((site.GetStreamedIndex<LatticeType> (ii)) - ii)/LatticeType::NUMVECTORS;

								*(mLatDat->GetFNew(site.GetStreamedIndex<LatticeType> (ii))) = fNew_GPU_b[ii * nFluid_nodes + iFluidIndex]; // When streaming on the GPU
								// * (mLatDat->GetFNew(site.GetStreamedIndex<LatticeType> (ii))) = fNew_GPU_b[ii * (mLatDat->GetLocalFluidSiteCount()) + siteIndex]; // no streaming on the GPU

								//printf("Fluid ID: %lld (/%lld), Data ADddres To Stream: %lld, fNew_GPU[%d] = %.5f \n\n", iFluidIndex, nFluid_nodes, site.GetStreamedIndex<LatticeType> (ii), ii, fNew_GPU_b[ii * nFluid_nodes + iFluidIndex]);
							}
							else	// Will Stream out of the domain to neighbour ranks (put in totalSharedFs)
							{
								*(mLatDat->GetFNew(site.GetStreamedIndex<LatticeType> (ii))) = fNew_GPU_b[site.GetStreamedIndex<LatticeType> (ii)];
								// printf("Data ADddres: %lld, fNew_GPU[%d] = %.5f \n\n", site.GetStreamedIndex<LatticeType> (ii), ii, fNew_GPU_b[site.GetStreamedIndex<LatticeType> (ii)]);
								if (site.GetStreamedIndex<LatticeType> (ii) >= (nFluid_nodes * LatticeType::NUMVECTORS+1+totSharedFs)) printf("Error!!! Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", ii, nFluid_nodes * LatticeType::NUMVECTORS+1+totSharedFs, site.GetStreamedIndex<LatticeType> (ii) );
							}
						} // Ends the if Bulk link case


						/*
						//
						// Debugging
						uint64_t max_Perm_Ind = nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs;
						uint64_t ind_fNew_GPU_b = ii * nFluid_nodes + iFluidIndex;

						uint64_t max_Perm_Ind_CPU = nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs;
						uint64_t ind_GetFNew = site.GetStreamedIndex<LatticeType> (ii);

						// if(iFluidIndex > nFluid_nodes) printf("Attempting to access Fluid ID index = %lld - Max. Fluid nodes = %lld  \n\n", iFluidIndex, nFluid_nodes);
						if(ind_GetFNew > max_Perm_Ind_CPU) printf("Wow!!! Attempting to access CPU index = %lld - Max. Permited = %lld  \n\n", ind_GetFNew, max_Perm_Ind_CPU);
						if(ind_fNew_GPU_b > max_Perm_Ind) printf("Error!!! Attempting to access index = %lld - Max. Permited = %lld  \n\n", ind_fNew_GPU_b, max_Perm_Ind);

						// printf("Index in fNew: Method 1: SiteIndex = %lld, Index of fNew[%d] = %lld Vs Index_2 = %lld \n\n", siteIndex, ii, (ii * mLatDat->GetLocalFluidSiteCount() + iFluidIndex), (siteIndex*LatticeType::NUMVECTORS + ii));
						// printf("SiteIndex = %lld, Streamed Fluid SiteIndex = %lld, fNew[%d] = %.5f \n\n", siteIndex, iFluidIndex, ii, fNew_GPU_b[ii * mLatDat->GetLocalFluidSiteCount() + iFluidIndex]);
						//
						*/
					}

				}
				//


				// Delete the variables when copy is completed
				//delete[] fOld_GPU_b;
				delete[] fNew_GPU_b;
				delete[] dens_GPU;
				delete[] vx_GPU;
				delete[] vy_GPU;
				delete[] vz_GPU;

				return true;
			}



		template<class LatticeType>
			void LBM<LatticeType>::get_Iolet_BCs(std::string& hemeLB_IoletBC_Inlet, std::string& hemeLB_IoletBC_Outlet)
			{
				// Check If I can get the type of Iolet BCs from the CMake file
				#define QUOTE_RAW(x) #x
				#define QUOTE_CONTENTS(x) QUOTE_RAW(x)

				//std::string hemeIoletBC_Inlet, hemeIoletBC_Outlet;
				hemeLB_IoletBC_Inlet = QUOTE_CONTENTS(HEMELB_INLET_BOUNDARY);
				//printf("Function Call: Inlet Type of BCs: hemeIoletBC_Inlet: %s \n", hemeLB_IoletBC_Inlet.c_str()); //note the use of c_str

				hemeLB_IoletBC_Outlet = QUOTE_CONTENTS(HEMELB_OUTLET_BOUNDARY);
				//printf("Function Call: Outlet Type of BCs: hemeIoletBC_Outlet: %s \n\n", hemeLB_IoletBC_Outlet.c_str()); //note the use of c_str

			}

//=================================================================================================
/**		Function for applying Velocity Boundary conditions
			i.e. evaluating the wall momentum correction term on the GPU (single value - not 3 components)
			Case of LADDIOLET and
				subtype File

			1. Examine each case (loop through the fluid sites)
					whether domain edge or inner domain fluid sites

			NOTE that, for example, the
				GPUDataAddr_wallMom_correction_Inlet_Inner
			is used instead of
				GPUDataAddr_wallMom_correction_Inlet_Inner_Direct
			Same applies to the other GPU global memory pointers

			Store the correction term using these pointers:
				void *GPUDataAddr_wallMom_correction_Inlet_Edge;
				void *GPUDataAddr_wallMom_correction_InletWall_Edge;
				void *GPUDataAddr_wallMom_correction_Inlet_Inner;
				void *GPUDataAddr_wallMom_correction_InletWall_Inner;
				void *GPUDataAddr_wallMom_correction_Outlet_Edge;
				void *GPUDataAddr_wallMom_correction_OutletWall_Edge;
				void *GPUDataAddr_wallMom_correction_Outlet_Inner;
				void *GPUDataAddr_wallMom_correction_OutletWall_Inner;

			2. Use the struct hemelb::Iolets
						struct Iolets Inlet_Edge, Inlet_Inner, InletWall_Edge, InletWall_Inner;
						struct Iolets Outlet_Edge, Outlet_Inner, OutletWall_Edge, OutletWall_Inner;
					with the iolets' info
					[local Iolet ID #0, min_index #0, max_index #0, local Iolet ID #1, min_index #1, max_index #1, ..., local Iolet ID #(number_elements_1), min_index #(number_elements_1), max_index #(number_elements_1)]
						i. 	Local Iolet ID
						ii. Range of fluid sites associated with each one of these iolets.
								[min_index, max_index] : NOTE INCLUDING THE max_index !!!
					//------------------------
					**IMPORTANT**
						Seems to be a bug with the above struct (when it should be zero it returns non-zero values)
					Use the other params associated with this info,
						see function identify_Range_iolets_ID
					and for example the following:
							std::vector<site_t> Iolets_Inlet_Edge; 			// vector with Inlet IDs and range associated with PreSend collision-streaming Type 3 (mInletCollision)
							int n_LocalInlets_mInlet_Edge; 							// number of local Inlets involved during the PreSend mInletCollision collision - needed for the range of fluid sites involved
							int n_unique_LocalInlets_mInlet_Edge;				// number of unique local Inlets

					Status: Fixed the above bug.
					//------------------------

			3. New approach - April 2023

*/
//=================================================================================================
		template<class LatticeType>
			void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction()
			{
				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				//----------------------------------------------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_WallMom_correct = 256;				//Number of threads per block for the Collision step
				dim3 nThreads_WallMom(nThreadsPerBlock_WallMom_correct);
				//----------------------------------------------------------------------

				//----------------------------------------------------------------------
				// Need the following as arguments to the function that will apply the BCs (replace what is below)
				// struct hemelb::Iolets Iolets_For_Eval_WallMom; // revert to the std::vector<site_t> Iolets_Inlet_Edge; etc (see bug info above)
				int n_LocalInlets=0;
				std::vector<site_t> Iolet_ID_AssocRange_Iolets_vect;
				void *GPUDataAddr_Coords_iolets;
				site_t start_Fluid_ID_givenColStreamType;
				site_t site_Count_givenColStreamType;

				//----------------------------------------------------------------------
				// Start fluid ID index for each collision-streaming type
				site_t start_Index_Inlet_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
				site_t start_Index_InletWall_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
																						+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);

				site_t start_Index_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
				site_t start_Index_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
				//----------------------------------------------------------------------
				// SiteCount for each collision-streaming type
				site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
				site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
				site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
				site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
				//----------------------------------------------------------------------

				//======================================================================
				// PreSend Fluid sites
				//======================================================================

				// I. Domain edge sites
				// I.1. Inlet_Edge

				// 		if(Inlet_Edge.n_local_iolets != 0){ // Bug in these values because not properly initialised
				n_LocalInlets = n_LocalInlets_mInlet_Edge;
				Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
				Iolet_ID_AssocRange_Iolets_vect = Iolets_Inlet_Edge;

				GPUDataAddr_Coords_iolets = GPUDataAddr_Coords_Inlet_Edge;
				start_Fluid_ID_givenColStreamType = start_Index_Inlet_Edge;
				site_Count_givenColStreamType = site_Count_Inlet_Edge;
				//cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreSend_3;

				if(n_LocalInlets != 0){

					if (mState->GetTimeStep() ==1){
						printf("Enters Inlet Edge - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_Inlet_Edge: %lld, site_Count_Inlet_Edge: %lld \n\n", mState->GetTimeStep(), myPiD, Inlet_Edge.n_local_iolets, n_LocalInlets_mInlet_Edge, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);
					}

					for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

						//------------------------------------------------------------------
						// Details of the iolet obtained from the following
						/* // See the comment above regarding the bug in struct Iolets
						int iolet_ID = Inlet_Edge.Iolets_ID_range[3*index_iolet];
						site_t lower_Fluid_index = Inlet_Edge.Iolets_ID_range[3*index_iolet+1];
						site_t max_Fluid_index = Inlet_Edge.Iolets_ID_range[3*index_iolet+2]; // INCLUDED fluid index
						*/
						int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
						site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
						site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

						// Debugging - Remove later
						//if (mState->GetTimeStep() ==1) 	printf("Lower Fluid ID: %lld - Max ID: %lld \n\n", lower_Fluid_index, max_Fluid_index);
						//------------------------------------------------------------------

						// Number of fluid sites involved
						site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

						int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
						//------------------------------------------------------------------

						// 8 Jan 2023
						// TODO: Think about this, whether to keep the shared memory (search for the map key on the GPU)
						//				OR load the address from Global memory of the map key to get the appropriate weight
						unsigned int SMemPerBlock_wallMomCorr = 3 * arr_elementsInEachInlet[iolet_ID] * sizeof(int64_t); // nThreadsPerBlock_Sat*sizeof(long)*2;

						// Launch the GPU kernel here
						if (nBlocks_WallMom!=0)
								//hemelb::GPU_WallMom_correction_File_Weights_NoSearch <<<nBlocks_WallMom, nThreads_WallMom, SMemPerBlock_wallMomCorr>>>
								hemelb::GPU_WallMom_correction_File_Weights_NoSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_3>>>
																												( (int64_t*)GPUDataAddr_Coords_Inlet_Edge,
																													(int64_t**)GPUDataAddr_pp_Inlet_weightsTable_coord,
																													(distribn_t**)GPUDataAddr_pp_Inlet_weightsTable_wei,
																													(int64_t*)GPUDataAddr_index_weightTable_Inlet_Edge,
																													(distribn_t*)GPUDataAddr_weightTable_Inlet_Edge,
																													(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Edge,
																													(float*)d_inletNormal,
																													(uint32_t*)GPUDataAddr_uint32_Iolet,
																													iolet_ID,
																													(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																													arr_elementsInEachInlet[iolet_ID],
																													start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																													lower_Fluid_index, max_Fluid_index,
																													mState->GetTimeStep(), mState->GetTotalTimeSteps(),mState->GetInitTimeStep()
																													);
					}
				} // Closes the if(n_LocalInlets != 0)

				//----------------------------------------------------------------------
				// I.2. InletWall Edge

				n_LocalInlets = n_LocalInlets_mInletWall_Edge;
				Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
				Iolet_ID_AssocRange_Iolets_vect = Iolets_InletWall_Edge;

				GPUDataAddr_Coords_iolets = GPUDataAddr_Coords_InletWall_Edge;
				start_Fluid_ID_givenColStreamType = start_Index_InletWall_Edge;
				site_Count_givenColStreamType = site_Count_InletWall_Edge;
				// cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreSend_5;

				if(n_LocalInlets != 0){

					if (mState->GetTimeStep() ==1){
						printf("Enters InletWall Edge - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_InletWall_Edge: %lld, site_Count_InletWall_Edge: %lld \n\n", mState->GetTimeStep(), myPiD, InletWall_Edge.n_local_iolets, n_LocalInlets_mInletWall_Edge, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);
					}

					for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

						//------------------------------------------------------------------
						// Details of the iolet obtained from the following
						/* // See the comment above regarding the bug in struct Iolets
						int iolet_ID = Inlet_Edge.Iolets_ID_range[3*index_iolet];
						site_t lower_Fluid_index = Inlet_Edge.Iolets_ID_range[3*index_iolet+1];
						site_t max_Fluid_index = Inlet_Edge.Iolets_ID_range[3*index_iolet+2]; // INCLUDED fluid index
						*/
						int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
						site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
						site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

						// Debugging - Remove later
						//if (mState->GetTimeStep() ==1) 	printf("Lower Fluid ID: %lld - Max ID: %lld \n\n", lower_Fluid_index, max_Fluid_index);
						//------------------------------------------------------------------

						// Number of fluid sites involved
						site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

						int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
						//------------------------------------------------------------------

						// 8 Jan 2023
						// TODO: Think about this, whether to keep the shared memory (search for the map key on the GPU)
						//				OR load the address from Global memory of the map key to get the appropriate weight
						unsigned int SMemPerBlock_wallMomCorr = 3 * arr_elementsInEachInlet[iolet_ID] * sizeof(int64_t); // nThreadsPerBlock_Sat*sizeof(long)*2;

						// Launch the GPU kernel here
						if (nBlocks_WallMom!=0)
								//hemelb::GPU_WallMom_correction_File_Weights_NoSearch <<<nBlocks_WallMom, nThreads_WallMom, SMemPerBlock_wallMomCorr>>>
								hemelb::GPU_WallMom_correction_File_Weights_NoSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_5>>>
																												( (int64_t*)GPUDataAddr_Coords_InletWall_Edge,
																													(int64_t**)GPUDataAddr_pp_Inlet_weightsTable_coord,
																													(distribn_t**)GPUDataAddr_pp_Inlet_weightsTable_wei,
																													(int64_t*)GPUDataAddr_index_weightTable_InletWall_Edge,
																													(distribn_t*)GPUDataAddr_weightTable_InletWall_Edge,
																													(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Edge,
																													(float*)d_inletNormal,
																													(uint32_t*)GPUDataAddr_uint32_Iolet,
																													iolet_ID,
																													(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																													arr_elementsInEachInlet[iolet_ID],
																													start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																													lower_Fluid_index, max_Fluid_index,
																													mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																													);
					}
				} // Closes the if(n_LocalInlets != 0)

				//======================================================================

				//======================================================================
				//  PreReceive Fluid sites
				//====================================================================
				// II. Inner domain
				// II.1. Inlet Inner

				n_LocalInlets = n_LocalInlets_mInlet;
				Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
				Iolet_ID_AssocRange_Iolets_vect = Iolets_Inlet_Inner;

				GPUDataAddr_Coords_iolets = GPUDataAddr_Coords_Inlet_Inner;
				start_Fluid_ID_givenColStreamType = start_Index_Inlet_Inner;
				site_Count_givenColStreamType = site_Count_Inlet_Inner;
				//cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreRec_3;

				if(n_LocalInlets != 0){

					if (mState->GetTimeStep() ==1){
						printf("Enters Inlet Inner - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_Inlet_Inner: %lld, site_Count_Inlet_Inner: %lld \n\n", mState->GetTimeStep(), myPiD, Inlet_Inner.n_local_iolets, n_LocalInlets_mInlet, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);
					}

					for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

						//------------------------------------------------------------------
						// Details of the iolet obtained from the following
						/* // See the comment above regarding the bug in struct Iolets
						int iolet_ID = Inlet_Edge.Iolets_ID_range[3*index_iolet];
						site_t lower_Fluid_index = Inlet_Edge.Iolets_ID_range[3*index_iolet+1];
						site_t max_Fluid_index = Inlet_Edge.Iolets_ID_range[3*index_iolet+2]; // INCLUDED fluid index
						*/
						int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
						site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
						site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

						// Debugging - Remove later
						//if (mState->GetTimeStep() ==1) 	printf("Lower Fluid ID: %lld - Max ID: %lld \n\n", lower_Fluid_index, max_Fluid_index);
						//------------------------------------------------------------------

						// Number of fluid sites involved
						site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

						int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
						//------------------------------------------------------------------

						// 8 Jan 2023
						// TODO: Think about this, whether to keep the shared memory (search for the map key on the GPU)
						//				OR load the address from Global memory of the map key to get the appropriate weight
						unsigned int SMemPerBlock_wallMomCorr = 3 * arr_elementsInEachInlet[iolet_ID] * sizeof(int64_t); // nThreadsPerBlock_Sat*sizeof(long)*2;

						// Launch the GPU kernel here
						if (nBlocks_WallMom!=0)
								//hemelb::GPU_WallMom_correction_File_Weights_NoSearch <<<nBlocks_WallMom, nThreads_WallMom, SMemPerBlock_wallMomCorr>>>
								hemelb::GPU_WallMom_correction_File_Weights_NoSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_3>>>
																												( (int64_t*)GPUDataAddr_Coords_Inlet_Inner,
																													(int64_t**)GPUDataAddr_pp_Inlet_weightsTable_coord,
																													(distribn_t**)GPUDataAddr_pp_Inlet_weightsTable_wei,
																													(int64_t*)GPUDataAddr_index_weightTable_Inlet_Inner,
																													(distribn_t*)GPUDataAddr_weightTable_Inlet_Inner,
																													(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Inner,
																													(float*)d_inletNormal,
																													(uint32_t*)GPUDataAddr_uint32_Iolet,
																													iolet_ID,
																													(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																													arr_elementsInEachInlet[iolet_ID],
																													start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																													lower_Fluid_index, max_Fluid_index,
																													mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																													);
					}
				} // Closes the if(n_LocalInlets != 0){

				//----------------------------------------------------------------------
				// II.2. InletWall_Inner
				n_LocalInlets = n_LocalInlets_mInletWall;
				Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
				Iolet_ID_AssocRange_Iolets_vect = Iolets_InletWall_Inner;

				GPUDataAddr_Coords_iolets = GPUDataAddr_Coords_InletWall_Inner;
				start_Fluid_ID_givenColStreamType = start_Index_InletWall_Inner;
				site_Count_givenColStreamType = site_Count_InletWall_Inner;
				// cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreRec_5;

				if(n_LocalInlets != 0){

					if (mState->GetTimeStep() ==1){
						printf("Enters InletWall_Inner - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_InletWall_Inner: %lld, site_Count_InletWall_Inner: %lld \n\n", mState->GetTimeStep(), myPiD, Inlet_Inner.n_local_iolets, n_LocalInlets_mInlet, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);
					}

					for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

						//------------------------------------------------------------------
						// Details of the iolet obtained from the following
						int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
						site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
						site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

						//------------------------------------------------------------------

						// Number of fluid sites involved
						site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

						// Debugging - Remove later
						//if (mState->GetTimeStep() ==1) 	printf("Lower Fluid ID: %lld - Max ID: %lld \n\n", lower_Fluid_index, max_Fluid_index, site_Count_WallMom);

						int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
						//------------------------------------------------------------------

						// 8 Jan 2023
						// TODO: Think about this, whether to keep the shared memory (search for the map key on the GPU)
						//				OR load the address from Global memory of the map key to get the appropriate weight
						unsigned int SMemPerBlock_wallMomCorr = 3 * arr_elementsInEachInlet[iolet_ID] * sizeof(int64_t); // nThreadsPerBlock_Sat*sizeof(long)*2;

						// Launch the GPU kernel here
						if (nBlocks_WallMom!=0)
								//hemelb::GPU_WallMom_correction_File_Weights_NoSearch <<<nBlocks_WallMom, nThreads_WallMom, SMemPerBlock_wallMomCorr>>>
								hemelb::GPU_WallMom_correction_File_Weights_NoSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_5>>>
																												( (int64_t*)GPUDataAddr_Coords_InletWall_Inner,
																													(int64_t**)GPUDataAddr_pp_Inlet_weightsTable_coord,
																													(distribn_t**)GPUDataAddr_pp_Inlet_weightsTable_wei,
																													(int64_t*)GPUDataAddr_index_weightTable_InletWall_Inner,
																													(distribn_t*)GPUDataAddr_weightTable_InletWall_Inner,
																													(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Inner,
																													(float*)d_inletNormal,
																													(uint32_t*)GPUDataAddr_uint32_Iolet,
																													iolet_ID,
																													(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																													arr_elementsInEachInlet[iolet_ID],
																													start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																													lower_Fluid_index, max_Fluid_index,
																													mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																													);
					}
				} // Closes the if(n_LocalInlets != 0){
				//====================================================================

				//======================================================================
				// 25 April 2023 - Debugging - small difference between CPU and GPU versions
				// Compare the wall momentum correction terms:
				// 1. as obtained on the GPU and
				// 2. as evaluated on the host

				// 1. The GPU wall momentum correction terms are contained in the following
				//		void *GPUDataAddr_wallMom_correction_Inlet_Edge;
				//		void *GPUDataAddr_wallMom_correction_InletWall_Edge;
				//		void *GPUDataAddr_wallMom_correction_Inlet_Inner;
				//		void *GPUDataAddr_wallMom_correction_InletWall_Inner;
				//
				//		in the data layout




				// 2. The host data:
				//		Check
				//			GetWallMom_correction_Direct
				//				DoGetWallMom_correction_Direct
				//					Eval_wallMom_correction


				// Domain Edge
				// Collision Type 3 (mInletCollision):
				start_Fluid_ID_givenColStreamType = start_Index_Inlet_Edge;
				site_Count_givenColStreamType = site_Count_Inlet_Edge;
				if (site_Count_Inlet_Edge!=0){
					if(mState->GetTimeStep()==1)
						printf("Comparison - Enters Inlet Edge - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_Inlet_Edge: %lld, site_Count_Inlet_Edge: %lld \n\n", mState->GetTimeStep(), myPiD, Inlet_Edge.n_local_iolets, n_LocalInlets_mInlet_Edge, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);

					// Directly getting the correction term without passing through propertyCache
					// Note that:
					//		a) wallMom_correction_Inlet_Edge_Direct contains the correction for LB_dir=0 as well
					// 		b) Has different layout
					GetWallMom_correction_Direct(mInletCollision, start_Index_Inlet_Edge, site_Count_Inlet_Edge, propertyCache, wallMom_correction_Inlet_Edge_Direct);
					wallMom_correction_Inlet_Edge_Direct.resize(site_Count_Inlet_Edge*LatticeType::NUMVECTORS);

					//cudaDeviceSynchronize();
					// Compare the GPU and CPU results
					compare_CPU_GPU_WallMom_correction(start_Index_Inlet_Edge, site_Count_Inlet_Edge, wallMom_correction_Inlet_Edge_Direct, GPUDataAddr_wallMom_correction_Inlet_Edge);//
					//memCpy_HtD_GPUmem_WallMom_correction_cudaStream(start_Index_Inlet_Edge, site_Count_Inlet_Edge, wallMom_correction_Inlet_Edge_Direct,
					//																								GPUDataAddr_wallMom_correction_Inlet_Edge, Collide_Stream_PreSend_3);
				}
				//======================================================================
			} // Ends void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction()


//=================================================================================================
		template<class LatticeType>
			void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction_ApprPref()
						{
							// Local rank
							const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
							int myPiD = rank_Com.Rank();
							//if (mState->GetTimeStep()==1)
							//	printf("Rank %d - Enters apply_Vel_BCs_File_GetWallMom_correction_ApprPref \n\n", myPiD);

							bool use_approach_1;	// iolets' information on the GPU global memory (e.g. (site_t*)GPUDataAddr_Inlet_Edge) - Use 1st version
							bool use_approach_2;	// iolets' information passed using the struct Iolets (e.g. Iolets Inlet_Edge) to the GPU kernel below - Use 2nd version

							//----------------------------------------------------------------------
							// Cuda kernel set-up
							int nThreadsPerBlock_WallMom_correct = 256;				//Number of threads per block for the evaluation of wall momentum correction terms
							dim3 nThreads_WallMom(nThreadsPerBlock_WallMom_correct);
							//----------------------------------------------------------------------

							//----------------------------------------------------------------------
							// Need the following as arguments to the function that will apply the BCs (replace what is below inlined when initial testing completed)
							// April 2023 - updated
							void *GPUDataAddr_wallMom_prefactor_correction; // prefactor term associated with wall momentum correction
							void *GPUDataAddr_IoletsInfo; // GPUDataAddr_Inlet_Edge
							void *GPUDataAddr_wallMom_correction;
							site_t start_Fluid_ID_givenColStreamType;
							site_t site_Count_givenColStreamType;
							int n_LocalInlets=0;
							//-----------------------------------

							//----------------------------------------------------------------------
							// Start fluid ID index for each collision-streaming type
							site_t start_Index_Inlet_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
							site_t start_Index_InletWall_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
																									+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);

							site_t start_Index_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
							site_t start_Index_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
							//----------------------------------------------------------------------
							// SiteCount for each collision-streaming type
							site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
							site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
							site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
							site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
							//----------------------------------------------------------------------

							//======================================================================
							// PreSend Fluid sites
							//======================================================================

							// I. Domain edge sites
							// I.1. Inlet_Edge
							n_LocalInlets = n_LocalInlets_mInlet_Edge;
							start_Fluid_ID_givenColStreamType = start_Index_Inlet_Edge;
							site_Count_givenColStreamType = site_Count_Inlet_Edge;
							//cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreSend_3;

							//
							/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge;
							GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_Inlet_Edge;
							GPUDataAddr_IoletsInfo = GPUDataAddr_Inlet_Edge;
							*/

							int nBlocks_WallMom = (site_Count_givenColStreamType)/nThreadsPerBlock_WallMom_correct
																	+ ((site_Count_givenColStreamType % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);

							use_approach_1 = ((n_LocalInlets <=(local_iolets_MaxSIZE/3))? 0:1);
							use_approach_2 = (!use_approach_1);
							//if(mState->GetTimeStep()==1) printf("Use approach 1 (GPU global mem.): %d  - Use approach 2 (struct Iolets): %d \n", use_approach_1, use_approach_2);

							if(nBlocks_WallMom!=0){
								if(use_approach_1)
									hemelb::GPU_WallMom_correction_File_prefactor <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_3>>>
																											( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge,
																												(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Edge,
																												(uint32_t*)GPUDataAddr_uint32_Iolet,
																												n_LocalInlets, (site_t*)GPUDataAddr_Inlet_Edge,
																												(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																												start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																												start_Fluid_ID_givenColStreamType, (start_Fluid_ID_givenColStreamType + site_Count_givenColStreamType),
																												mState->GetTimeStep(), mState->GetTotalTimeSteps()
																												);
								if(use_approach_2)
									hemelb::GPU_WallMom_correction_File_prefactor_v2 <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_3>>>
																											( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge,
																												(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Edge,
																												(uint32_t*)GPUDataAddr_uint32_Iolet,
																												n_LocalInlets, Inlet_Edge,
																												(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																												start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																												start_Fluid_ID_givenColStreamType, (start_Fluid_ID_givenColStreamType + site_Count_givenColStreamType),
																												mState->GetTimeStep(), mState->GetTotalTimeSteps()
																												);

							}
							//----------------------------------------------------------------------
							// I.2. InletWall Edge

							n_LocalInlets = n_LocalInlets_mInletWall_Edge;
							start_Fluid_ID_givenColStreamType = start_Index_InletWall_Edge;
							site_Count_givenColStreamType = site_Count_InletWall_Edge;
							// cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreSend_5;

							//
							/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge;
							GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_InletWall_Edge;
							GPUDataAddr_IoletsInfo = GPUDataAddr_InletWall_Edge;
							*/

							nBlocks_WallMom = (site_Count_givenColStreamType)/nThreadsPerBlock_WallMom_correct
																	+ ((site_Count_givenColStreamType % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);

							use_approach_1 = ((n_LocalInlets <=(local_iolets_MaxSIZE/3))? 0:1);
							use_approach_2 = (!use_approach_1);
							if(nBlocks_WallMom!=0){
								if(use_approach_1)
									hemelb::GPU_WallMom_correction_File_prefactor <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_5>>>
																											( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge,
																												(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Edge,
																												(uint32_t*)GPUDataAddr_uint32_Iolet,
																												n_LocalInlets, (site_t*)GPUDataAddr_InletWall_Edge,
																												(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																												start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																												start_Fluid_ID_givenColStreamType, (start_Fluid_ID_givenColStreamType + site_Count_givenColStreamType),
																												mState->GetTimeStep(), mState->GetTotalTimeSteps()
																												);

								if(use_approach_2)
									hemelb::GPU_WallMom_correction_File_prefactor_v2 <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_5>>>
																										( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge,
																										(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Edge,
																										(uint32_t*)GPUDataAddr_uint32_Iolet,
																										n_LocalInlets, InletWall_Edge,
																										(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																										start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																										start_Fluid_ID_givenColStreamType, (start_Fluid_ID_givenColStreamType + site_Count_givenColStreamType),
																										mState->GetTimeStep(), mState->GetTotalTimeSteps()
																										);

							}
							//======================================================================

							//======================================================================
							//  PreReceive Fluid sites
							//====================================================================
							// II. Inner domain
							// II.1. Inlet Inner

							n_LocalInlets = n_LocalInlets_mInlet;
							start_Fluid_ID_givenColStreamType = start_Index_Inlet_Inner;
							site_Count_givenColStreamType = site_Count_Inlet_Inner;
							//cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreRec_3;

							//
							/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner;
							GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_Inlet_Inner;
							GPUDataAddr_IoletsInfo = GPUDataAddr_Inlet_Inner;
							*/

							nBlocks_WallMom = (site_Count_givenColStreamType)/nThreadsPerBlock_WallMom_correct
																	+ ((site_Count_givenColStreamType % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);

							use_approach_1 = ((n_LocalInlets <=(local_iolets_MaxSIZE/3))? 0:1);
							use_approach_2 = (!use_approach_1);
							if(nBlocks_WallMom!=0){
								if(use_approach_1)
									hemelb::GPU_WallMom_correction_File_prefactor <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_3>>>
																											( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner,
																												(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Inner,
																												(uint32_t*)GPUDataAddr_uint32_Iolet,
																												n_LocalInlets, (site_t*)GPUDataAddr_Inlet_Inner,
																												(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																												start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																												start_Fluid_ID_givenColStreamType, (start_Fluid_ID_givenColStreamType + site_Count_givenColStreamType),
																												mState->GetTimeStep(), mState->GetTotalTimeSteps()
																											);

								if(use_approach_2)
									hemelb::GPU_WallMom_correction_File_prefactor_v2 <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_3>>>
																										( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner,
																										(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Inner,
																										(uint32_t*)GPUDataAddr_uint32_Iolet,
																										n_LocalInlets, Inlet_Inner,
																										(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																										start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																										start_Fluid_ID_givenColStreamType, (start_Fluid_ID_givenColStreamType + site_Count_givenColStreamType),
																										mState->GetTimeStep(), mState->GetTotalTimeSteps()
																									);

							}
							//----------------------------------------------------------------------
							// II.2. InletWall_Inner
							n_LocalInlets = n_LocalInlets_mInletWall;
							start_Fluid_ID_givenColStreamType = start_Index_InletWall_Inner;
							site_Count_givenColStreamType = site_Count_InletWall_Inner;
							// cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreRec_5;

							//
							/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner;
							GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_InletWall_Inner;
							GPUDataAddr_IoletsInfo = GPUDataAddr_InletWall_Inner;
							*/

							nBlocks_WallMom = (site_Count_givenColStreamType)/nThreadsPerBlock_WallMom_correct
																	+ ((site_Count_givenColStreamType % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);

							use_approach_1 = ((n_LocalInlets <=(local_iolets_MaxSIZE/3))? 0:1);
							use_approach_2 = (!use_approach_1);
							if(nBlocks_WallMom!=0){
								if(use_approach_1)
									hemelb::GPU_WallMom_correction_File_prefactor <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_5>>>
																											( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner,
																												(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Inner,
																												(uint32_t*)GPUDataAddr_uint32_Iolet,
																												n_LocalInlets, (site_t*)GPUDataAddr_InletWall_Inner,
																												(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																												start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																												start_Fluid_ID_givenColStreamType, (start_Fluid_ID_givenColStreamType + site_Count_givenColStreamType),
																												mState->GetTimeStep(), mState->GetTotalTimeSteps()
																												);

								if(use_approach_2)
									hemelb::GPU_WallMom_correction_File_prefactor_v2 <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_5>>>
																						( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner,
																						(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Inner,
																						(uint32_t*)GPUDataAddr_uint32_Iolet,
																						n_LocalInlets, InletWall_Inner,
																						(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																						start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																						start_Fluid_ID_givenColStreamType, (start_Fluid_ID_givenColStreamType + site_Count_givenColStreamType),
																						mState->GetTimeStep(), mState->GetTotalTimeSteps()
																						);

							}
							//====================================================================
						} // Ends void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction_ApprPref()

						//=================================================================================================
							/**
								Evaluate the Vel BCs wall momentum correction terms for the PreSend related kernels (Domain Edges)
							*/
								template<class LatticeType>
									void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch_PreSend()
									{
										// Local rank
										const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
										int myPiD = rank_Com.Rank();

										//if (mState->GetTimeStep()==1)
										//	printf("Rank %d - Enters apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch \n\n", myPiD);

										//----------------------------------------------------------------------
										// Cuda kernel set-up
										int nThreadsPerBlock_WallMom_correct = 256;				//Number of threads per block for the evaluation of wall momentum correction terms
										dim3 nThreads_WallMom(nThreadsPerBlock_WallMom_correct);
										//----------------------------------------------------------------------

										//----------------------------------------------------------------------
										// Need the following as arguments to the function that will apply the BCs (replace what is below inlined when initial testing completed)
										// April 2023 - updated
										void *GPUDataAddr_wallMom_prefactor_correction; // prefactor term associated with wall momentum correction
										void *GPUDataAddr_IoletsInfo; // GPUDataAddr_Inlet_Edge
										void *GPUDataAddr_wallMom_correction;
										site_t start_Fluid_ID_givenColStreamType;
										site_t site_Count_givenColStreamType;
										int n_LocalInlets=0;

										std::vector<site_t> Iolet_ID_AssocRange_Iolets_vect;

										//-----------------------------------

										//----------------------------------------------------------------------
										// Start fluid ID index for each collision-streaming type
										site_t start_Index_Inlet_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
										site_t start_Index_InletWall_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
																+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);
										//----------------------------------------------------------------------
										// SiteCount for each collision-streaming type
										site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
										site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
										//----------------------------------------------------------------------

										//======================================================================
										// PreSend Fluid sites
										//======================================================================

										// I. Domain edge sites
										// I.1. Inlet_Edge
										n_LocalInlets = n_LocalInlets_mInlet_Edge;
										start_Fluid_ID_givenColStreamType = start_Index_Inlet_Edge;
										site_Count_givenColStreamType = site_Count_Inlet_Edge;
										//cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreSend_3;
										Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
										Iolet_ID_AssocRange_Iolets_vect = Iolets_Inlet_Edge;

										//
										/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge;
										GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_Inlet_Edge;
										GPUDataAddr_IoletsInfo = GPUDataAddr_Inlet_Edge;
										*/
										if(n_LocalInlets != 0){

											//if (mState->GetTimeStep() ==1)
											//	printf("Enters Inlet Edge - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_Inlet_Edge: %lld, site_Count_Inlet_Edge: %lld \n\n", mState->GetTimeStep(), myPiD, Inlet_Edge.n_local_iolets, n_LocalInlets_mInlet_Edge, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);


											for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

												//------------------------------------------------------------------
												// Details of the iolet obtained from the following
												int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
												site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
												site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

												// Number of fluid sites involved
												site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

												int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
												//------------------------------------------------------------------

												// Launch the GPU kernel here
												if (nBlocks_WallMom!=0)
													hemelb::GPU_WallMom_correction_File_prefactor_NoIoletIDSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_3>>>
																														( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge,
																														(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Edge,
																														(uint32_t*)GPUDataAddr_uint32_Iolet,
																														iolet_ID,
																														(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																														start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																														lower_Fluid_index, (max_Fluid_index+1),
																														mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																														);
											}
										} // Closes the if(n_LocalInlets != 0)
										//----------------------------------------------------------------------

										// I.2. InletWall Edge
										n_LocalInlets = n_LocalInlets_mInletWall_Edge;
										start_Fluid_ID_givenColStreamType = start_Index_InletWall_Edge;
										site_Count_givenColStreamType = site_Count_InletWall_Edge;
										// cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreSend_5;
										Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
										Iolet_ID_AssocRange_Iolets_vect = Iolets_InletWall_Edge;

										//
										/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge;
										GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_InletWall_Edge;
										GPUDataAddr_IoletsInfo = GPUDataAddr_InletWall_Edge;
										*/

										if(n_LocalInlets != 0){

											//if (mState->GetTimeStep() ==1)
											//	printf("Enters InletWall Edge - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_InletWall_Edge: %lld, site_Count_InletWall_Edge: %lld \n\n", mState->GetTimeStep(), myPiD, InletWall_Edge.n_local_iolets, n_LocalInlets_mInletWall_Edge, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);


											for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

												//------------------------------------------------------------------
												// Details of the iolet obtained from the following
												int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
												site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
												site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

												// Number of fluid sites involved
												site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

												int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
												//------------------------------------------------------------------

												// Launch the GPU kernel here
												if (nBlocks_WallMom!=0)
													hemelb::GPU_WallMom_correction_File_prefactor_NoIoletIDSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_5>>>
																														( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge,
																														(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Edge,
																														(uint32_t*)GPUDataAddr_uint32_Iolet,
																														iolet_ID,
																														(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																														start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																														lower_Fluid_index, (max_Fluid_index+1),
																														mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																														);
											}
										} // Closes the if(n_LocalInlets != 0)
										//======================================================================

								} // Ends void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch_PreSend()
						//=================================================================================================

						//=================================================================================================
								template<class LatticeType>
									void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch_PreReceive()
									{
										// Local rank
										const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
										int myPiD = rank_Com.Rank();

										//if (mState->GetTimeStep()==1)
										//	printf("Rank %d - Enters apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch \n\n", myPiD);

										//----------------------------------------------------------------------
										// Cuda kernel set-up
										int nThreadsPerBlock_WallMom_correct = 256;				//Number of threads per block for the evaluation of wall momentum correction terms
										dim3 nThreads_WallMom(nThreadsPerBlock_WallMom_correct);
										//----------------------------------------------------------------------

										//----------------------------------------------------------------------
										// Need the following as arguments to the function that will apply the BCs (replace what is below inlined when initial testing completed)
										// April 2023 - updated
										void *GPUDataAddr_wallMom_prefactor_correction; // prefactor term associated with wall momentum correction
										void *GPUDataAddr_IoletsInfo; // GPUDataAddr_Inlet_Edge
										void *GPUDataAddr_wallMom_correction;
										site_t start_Fluid_ID_givenColStreamType;
										site_t site_Count_givenColStreamType;
										int n_LocalInlets=0;

										std::vector<site_t> Iolet_ID_AssocRange_Iolets_vect;

										//-----------------------------------

										//----------------------------------------------------------------------
										// Start fluid ID index for each collision-streaming type
										site_t start_Index_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
										site_t start_Index_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
										//----------------------------------------------------------------------
										// SiteCount for each collision-streaming type
										site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
										site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
										//----------------------------------------------------------------------
										//======================================================================
										//  PreReceive Fluid sites
										//====================================================================
										// II. Inner domain
										// II.1. Inlet Inner

										n_LocalInlets = n_LocalInlets_mInlet;
										start_Fluid_ID_givenColStreamType = start_Index_Inlet_Inner;
										site_Count_givenColStreamType = site_Count_Inlet_Inner;
										//cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreRec_3;
										Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
										Iolet_ID_AssocRange_Iolets_vect = Iolets_Inlet_Inner;

										//
										/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner;
										GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_Inlet_Inner;
										GPUDataAddr_IoletsInfo = GPUDataAddr_Inlet_Inner;
										*/

										if(n_LocalInlets != 0){

											//if (mState->GetTimeStep() ==1)
											//	printf("Enters Inlet Inner - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_Inlet_Inner: %lld, site_Count_Inlet_Inner: %lld \n\n", mState->GetTimeStep(), myPiD, Inlet_Inner.n_local_iolets, n_LocalInlets_mInlet, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);

											for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

												//------------------------------------------------------------------
												// Details of the iolet obtained from the following
												int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
												site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
												site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

												// Number of fluid sites involved
												site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

												int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
												//------------------------------------------------------------------

												// Launch the GPU kernel here
												if (nBlocks_WallMom!=0)
													hemelb::GPU_WallMom_correction_File_prefactor_NoIoletIDSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_3>>>
																														( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner,
																														(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Inner,
																														(uint32_t*)GPUDataAddr_uint32_Iolet,
																														iolet_ID,
																														(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																														start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																														lower_Fluid_index, (max_Fluid_index+1),
																														mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																														);
											}
										} // Closes the if(n_LocalInlets != 0)
										//======================================================================


										//----------------------------------------------------------------------
										// II.2. InletWall_Inner
										n_LocalInlets = n_LocalInlets_mInletWall;
										start_Fluid_ID_givenColStreamType = start_Index_InletWall_Inner;
										site_Count_givenColStreamType = site_Count_InletWall_Inner;
										// cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreRec_5;
										Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
										Iolet_ID_AssocRange_Iolets_vect = Iolets_InletWall_Inner;

										//
										/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner;
										GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_InletWall_Inner;
										GPUDataAddr_IoletsInfo = GPUDataAddr_InletWall_Inner;
										*/
										if(n_LocalInlets != 0){

											//if (mState->GetTimeStep() ==1)
											//	printf("Enters InletWall Inner - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_InletWall_Inner: %lld, site_Count_InletWall_Inner: %lld \n\n", mState->GetTimeStep(), myPiD, InletWall_Inner.n_local_iolets, n_LocalInlets_mInletWall, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);

											for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

												//------------------------------------------------------------------
												// Details of the iolet obtained from the following
												int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
												site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
												site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

												// Number of fluid sites involved
												site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

												int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
												//------------------------------------------------------------------

												// Launch the GPU kernel here
												if (nBlocks_WallMom!=0)
													hemelb::GPU_WallMom_correction_File_prefactor_NoIoletIDSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_5>>>
																														( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner,
																														(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Inner,
																														(uint32_t*)GPUDataAddr_uint32_Iolet,
																														iolet_ID,
																														(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																														start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																														lower_Fluid_index, (max_Fluid_index+1),
																														mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																														);
											}
										} // Closes the if(n_LocalInlets != 0)
										//====================================================================
								} // Ends void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch_PreReceive()
						//=================================================================================================



//=================================================================================================
		template<class LatticeType>
			void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch()
			{
				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				//if (mState->GetTimeStep()==1)
				//	printf("Rank %d - Enters apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch \n\n", myPiD);

				//----------------------------------------------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_WallMom_correct = 256;				//Number of threads per block for the evaluation of wall momentum correction terms
				dim3 nThreads_WallMom(nThreadsPerBlock_WallMom_correct);
				//----------------------------------------------------------------------

				//----------------------------------------------------------------------
				// Need the following as arguments to the function that will apply the BCs (replace what is below inlined when initial testing completed)
				// April 2023 - updated
				void *GPUDataAddr_wallMom_prefactor_correction; // prefactor term associated with wall momentum correction
				void *GPUDataAddr_IoletsInfo; // GPUDataAddr_Inlet_Edge
				void *GPUDataAddr_wallMom_correction;
				site_t start_Fluid_ID_givenColStreamType;
				site_t site_Count_givenColStreamType;
				int n_LocalInlets=0;

				std::vector<site_t> Iolet_ID_AssocRange_Iolets_vect;

				//-----------------------------------

				//----------------------------------------------------------------------
				// Start fluid ID index for each collision-streaming type
				site_t start_Index_Inlet_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
				site_t start_Index_InletWall_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
										+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);

				site_t start_Index_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
				site_t start_Index_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
				//----------------------------------------------------------------------
				// SiteCount for each collision-streaming type
				site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
				site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
				site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
				site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
				//----------------------------------------------------------------------

				//======================================================================
				// PreSend Fluid sites
				//======================================================================

				// I. Domain edge sites
				// I.1. Inlet_Edge
				n_LocalInlets = n_LocalInlets_mInlet_Edge;
				start_Fluid_ID_givenColStreamType = start_Index_Inlet_Edge;
				site_Count_givenColStreamType = site_Count_Inlet_Edge;
				//cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreSend_3;
				Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
				Iolet_ID_AssocRange_Iolets_vect = Iolets_Inlet_Edge;

				//
				/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge;
				GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_Inlet_Edge;
				GPUDataAddr_IoletsInfo = GPUDataAddr_Inlet_Edge;
				*/
				if(n_LocalInlets != 0){

					//if (mState->GetTimeStep() ==1)
					//	printf("Enters Inlet Edge - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_Inlet_Edge: %lld, site_Count_Inlet_Edge: %lld \n\n", mState->GetTimeStep(), myPiD, Inlet_Edge.n_local_iolets, n_LocalInlets_mInlet_Edge, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);


					for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

						//------------------------------------------------------------------
						// Details of the iolet obtained from the following
						int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
						site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
						site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

						// Number of fluid sites involved
						site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

						int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
						//------------------------------------------------------------------

						// Launch the GPU kernel here
						if (nBlocks_WallMom!=0)
							hemelb::GPU_WallMom_correction_File_prefactor_NoIoletIDSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_3>>>
																								( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge,
																								(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Edge,
																								(uint32_t*)GPUDataAddr_uint32_Iolet,
																								iolet_ID,
																								(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																								start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																								lower_Fluid_index, (max_Fluid_index+1),
																								mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																								);
					}
				} // Closes the if(n_LocalInlets != 0)
				//----------------------------------------------------------------------

				// I.2. InletWall Edge
				n_LocalInlets = n_LocalInlets_mInletWall_Edge;
				start_Fluid_ID_givenColStreamType = start_Index_InletWall_Edge;
				site_Count_givenColStreamType = site_Count_InletWall_Edge;
				// cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreSend_5;
				Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
				Iolet_ID_AssocRange_Iolets_vect = Iolets_InletWall_Edge;

				//
				/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge;
				GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_InletWall_Edge;
				GPUDataAddr_IoletsInfo = GPUDataAddr_InletWall_Edge;
				*/

				if(n_LocalInlets != 0){

					//if (mState->GetTimeStep() ==1)
					//	printf("Enters InletWall Edge - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_InletWall_Edge: %lld, site_Count_InletWall_Edge: %lld \n\n", mState->GetTimeStep(), myPiD, InletWall_Edge.n_local_iolets, n_LocalInlets_mInletWall_Edge, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);


					for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

						//------------------------------------------------------------------
						// Details of the iolet obtained from the following
						int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
						site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
						site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

						// Number of fluid sites involved
						site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

						int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
						//------------------------------------------------------------------

						// Launch the GPU kernel here
						if (nBlocks_WallMom!=0)
							hemelb::GPU_WallMom_correction_File_prefactor_NoIoletIDSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreSend_5>>>
																								( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge,
																								(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Edge,
																								(uint32_t*)GPUDataAddr_uint32_Iolet,
																								iolet_ID,
																								(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																								start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																								lower_Fluid_index, (max_Fluid_index+1),
																								mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																								);
					}
				} // Closes the if(n_LocalInlets != 0)
				//======================================================================

				//======================================================================
				//  PreReceive Fluid sites
				//====================================================================
				// II. Inner domain
				// II.1. Inlet Inner

				n_LocalInlets = n_LocalInlets_mInlet;
				start_Fluid_ID_givenColStreamType = start_Index_Inlet_Inner;
				site_Count_givenColStreamType = site_Count_Inlet_Inner;
				//cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreRec_3;
				Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
				Iolet_ID_AssocRange_Iolets_vect = Iolets_Inlet_Inner;

				//
				/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner;
				GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_Inlet_Inner;
				GPUDataAddr_IoletsInfo = GPUDataAddr_Inlet_Inner;
				*/

				if(n_LocalInlets != 0){

					//if (mState->GetTimeStep() ==1)
					//	printf("Enters Inlet Inner - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_Inlet_Inner: %lld, site_Count_Inlet_Inner: %lld \n\n", mState->GetTimeStep(), myPiD, Inlet_Inner.n_local_iolets, n_LocalInlets_mInlet, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);

					for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

						//------------------------------------------------------------------
						// Details of the iolet obtained from the following
						int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
						site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
						site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

						// Number of fluid sites involved
						site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

						int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
						//------------------------------------------------------------------

						// Launch the GPU kernel here
						if (nBlocks_WallMom!=0)
							hemelb::GPU_WallMom_correction_File_prefactor_NoIoletIDSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_3>>>
																								( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner,
																								(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Inner,
																								(uint32_t*)GPUDataAddr_uint32_Iolet,
																								iolet_ID,
																								(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																								start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																								lower_Fluid_index, (max_Fluid_index+1),
																								mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																								);
					}
				} // Closes the if(n_LocalInlets != 0)
				//======================================================================


				//----------------------------------------------------------------------
				// II.2. InletWall_Inner
				n_LocalInlets = n_LocalInlets_mInletWall;
				start_Fluid_ID_givenColStreamType = start_Index_InletWall_Inner;
				site_Count_givenColStreamType = site_Count_InletWall_Inner;
				// cudaStream_t Collide_Stream_givenColStreamType = Collide_Stream_PreRec_5;
				Iolet_ID_AssocRange_Iolets_vect.resize(3*n_LocalInlets);
				Iolet_ID_AssocRange_Iolets_vect = Iolets_InletWall_Inner;

				//
				/*GPUDataAddr_wallMom_prefactor_correction = GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner;
				GPUDataAddr_wallMom_correction = GPUDataAddr_wallMom_correction_InletWall_Inner;
				GPUDataAddr_IoletsInfo = GPUDataAddr_InletWall_Inner;
				*/
				if(n_LocalInlets != 0){

					//if (mState->GetTimeStep() ==1)
					//	printf("Enters InletWall Inner - Time: %d, Rank: %d, Number of iolets involved - local = %d , n_LocalInlets = %d, start_Index_InletWall_Inner: %lld, site_Count_InletWall_Inner: %lld \n\n", mState->GetTimeStep(), myPiD, InletWall_Inner.n_local_iolets, n_LocalInlets_mInletWall, start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType);

					for (int index_iolet=0; index_iolet< n_LocalInlets; index_iolet++){ // Note the irregular numbering of iolets

						//------------------------------------------------------------------
						// Details of the iolet obtained from the following
						int iolet_ID = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet];
						site_t lower_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+1];
						site_t max_Fluid_index = Iolet_ID_AssocRange_Iolets_vect[3*index_iolet+2]; // INCLUDED fluid index

						// Number of fluid sites involved
						site_t site_Count_WallMom = max_Fluid_index - lower_Fluid_index + 1; // The +1 is because max_Fluid_index is INCLUDED in the range

						int nBlocks_WallMom = (site_Count_WallMom)/nThreadsPerBlock_WallMom_correct			+ ((site_Count_WallMom % nThreadsPerBlock_WallMom_correct > 0)         ? 1 : 0);
						//------------------------------------------------------------------

						// Launch the GPU kernel here
						if (nBlocks_WallMom!=0)
							hemelb::GPU_WallMom_correction_File_prefactor_NoIoletIDSearch <<<nBlocks_WallMom, nThreads_WallMom, 0, Collide_Stream_PreRec_5>>>
																								( (distribn_t*)GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner,
																								(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Inner,
																								(uint32_t*)GPUDataAddr_uint32_Iolet,
																								iolet_ID,
																								(distribn_t*)mLatDat->GPUDataAddr_Inlet_velocityTable,
																								start_Fluid_ID_givenColStreamType, site_Count_givenColStreamType,
																								lower_Fluid_index, (max_Fluid_index+1),
																								mState->GetTimeStep(), mState->GetTotalTimeSteps(), mState->GetInitTimeStep()
																								);
					}
				} // Closes the if(n_LocalInlets != 0)
				//====================================================================
		} // Ends void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch()
//=================================================================================================


		template<class LatticeType>
			bool LBM<LatticeType>::FinaliseGPU()
			{
				bool finalise_GPU_res = true;

				std::string hemeIoletBC_Inlet, hemeIoletBC_Outlet;
				get_Iolet_BCs(hemeIoletBC_Inlet, hemeIoletBC_Outlet);

				// Cuda Streams
				deviceStreamDestroy(Collide_Stream_PreSend_1);
				deviceStreamDestroy(Collide_Stream_PreSend_2);
				deviceStreamDestroy(Collide_Stream_PreSend_3);
				deviceStreamDestroy(Collide_Stream_PreSend_4);
				deviceStreamDestroy(Collide_Stream_PreSend_5);
				deviceStreamDestroy(Collide_Stream_PreSend_6);

				deviceStreamDestroy(Collide_Stream_PreRec_1);
				deviceStreamDestroy(Collide_Stream_PreRec_2);
				deviceStreamDestroy(Collide_Stream_PreRec_3);
				deviceStreamDestroy(Collide_Stream_PreRec_4);
				deviceStreamDestroy(Collide_Stream_PreRec_5);
				deviceStreamDestroy(Collide_Stream_PreRec_6);
				//	cudaStreamDestroy(stream_Read_distr_Data_GPU);

				deviceStreamDestroy(stream_Read_Data_GPU_Dens);

				deviceStreamDestroy(stream_ghost_dens_inlet);
				deviceStreamDestroy(stream_ghost_dens_outlet);
				deviceStreamDestroy(stream_ReceivedDistr);
				deviceStreamDestroy(stream_SwapOldAndNew);
				deviceStreamDestroy(stream_memCpy_CPU_GPU_domainEdge);

				deviceStreamDestroy(stability_check_stream);

				// Destroy the cuda stream created for the asynch. MemCopy DtH at the domain edges: created a stream in net::BaseNet object
				hemelb::net::Net& mNet_cuda_stream = *mNet;	// Access the mNet object
				mNet_cuda_stream.Destroy_stream_memCpy_GPU_CPU_domainEdge_new2(); // Which one is correct? Does it actually create the stream and then it imposes a barrier in net::BaseNet::Send

				//cudaStreamDestroy(stream_memCpy_GPU_CPU_domainEdge);

				// Free GPU memory
				bool status =  deviceFree(GPUDataAddr_dbl_MacroVars);
				if(!status){ fprintf(stderr, "deviceFree failed\n"); finalise_GPU_res=false; }

				status = deviceFree(GPUDataAddr_uint32_Wall);
				if(!status){ fprintf(stderr, "deviceFree failed\n"); finalise_GPU_res=false; }

				status = deviceFree(GPUDataAddr_uint32_Iolet);
				if(!status){ fprintf(stderr, "deviceFree failed\n"); finalise_GPU_res=false; }

				//----------------------------------------------------------------------
				// Iolets Info:
				//void *GPUDataAddr_Inlet_Edge, *GPUDataAddr_Outlet_Edge, *GPUDataAddr_InletWall_Edge, *GPUDataAddr_OutletWall_Edge;
				//void *GPUDataAddr_Inlet_Inner, *GPUDataAddr_Outlet_Inner, *GPUDataAddr_InletWall_Inner, *GPUDataAddr_OutletWall_Inner;

				/* // Fail to free the following - check using cudaPointerGetAttributes
				if(GPUDataAddr_Inlet_Edge){
					cudaStatus = deviceFree(GPUDataAddr_Inlet_Edge);
					if(!status){ fprintf(stderr, "deviceFree Iolets info (1) failed\n"); finalise_GPU_res=false; }
				}
				if(GPUDataAddr_InletWall_Edge){
					cudaStatus = deviceFree(GPUDataAddr_InletWall_Edge);
					if(!status){ fprintf(stderr, "deviceFree Iolets info (2) failed\n"); finalise_GPU_res=false; }
				}
				if(GPUDataAddr_Outlet_Edge){
					cudaStatus = deviceFree(GPUDataAddr_Outlet_Edge);
					if(!status){ fprintf(stderr, "deviceFree Iolets info (3) failed\n"); finalise_GPU_res=false; }
				}
				if(GPUDataAddr_OutletWall_Edge){
					cudaStatus = deviceFree(GPUDataAddr_OutletWall_Edge);
					if(!status){ fprintf(stderr, "deviceFree Iolets info (4) failed\n"); finalise_GPU_res=false; }
				}

				// Inner domain Iolets' info
				if(GPUDataAddr_Inlet_Inner){
					cudaStatus = deviceFree(GPUDataAddr_Inlet_Inner);
					if(!status){ fprintf(stderr, "deviceFree Iolets info (5) failed\n"); finalise_GPU_res=false; }
				}
				if(GPUDataAddr_InletWall_Inner){
					cudaStatus = deviceFree(GPUDataAddr_InletWall_Inner);
					if(!status){ fprintf(stderr, "deviceFree Iolets info (6) failed\n"); finalise_GPU_res=false; }
				}
				if(GPUDataAddr_Outlet_Inner){
					cudaStatus = deviceFree(GPUDataAddr_Outlet_Inner);
					if(!status){ fprintf(stderr, "deviceFree Iolets info (7) failed\n"); finalise_GPU_res=false; }
				}
				if(GPUDataAddr_OutletWall_Inner){
					cudaStatus = deviceFree(GPUDataAddr_OutletWall_Inner);
					if(!status){ fprintf(stderr, "deviceFree Iolets info (8) failed\n"); finalise_GPU_res=false; }
				}
				*/

				//----------------------------------------------------------------------
				// Vel BCs related

				// Prefactor Wall Momemtum Correction
				/*void *GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge;
				void *GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge;
				void *GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner;
				void *GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner;
				void *GPUDataAddr_wallMom_prefactor_correction_Outlet_Edge;
				void *GPUDataAddr_wallMom_prefactor_correction_OutletWall_Edge;
				void *GPUDataAddr_wallMom_prefactor_correction_Outlet_Inner;
				void *GPUDataAddr_wallMom_prefactor_correction_OutletWall_Inner;
				*/

				//----------------------------------------------------------------------
				if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
					status = deviceFree(d_ghostDensity);
					if(!status){ fprintf(stderr, "deviceFree ghost Density inlet failed\n"); finalise_GPU_res=false; }

					// Free up pinned Memory associated with the Pressure BCs
					if (h_ghostDensity_inlet != nullptr){
						status = deviceFreeHost(h_ghostDensity_inlet);
						if(!status){ fprintf(stderr, "deviceFreeHost h_ghostDensity_inlet failed ... \n"); finalise_GPU_res=false; }
						h_ghostDensity_inlet = nullptr;
					}
				}

				if (hemeIoletBC_Inlet == "LADDIOLET"){

					if(mLatDat->GPUDataAddr_Inlet_velocityTable != nullptr){
						status = deviceFree(mLatDat->GPUDataAddr_Inlet_velocityTable);
						if(!status){ fprintf(stderr, "deviceFree Velocity Table failed\n"); finalise_GPU_res=false; }
					}

					if(GPUDataAddr_wallMom_correction_Inlet_Edge != nullptr){
						status = deviceFree(GPUDataAddr_wallMom_correction_Inlet_Edge);
						if(!status){ fprintf(stderr, "deviceFree wall mom correction (1) inlet  failed\n"); finalise_GPU_res=false; }
					}

					if(GPUDataAddr_wallMom_correction_InletWall_Edge != nullptr){
						status  = deviceFree(GPUDataAddr_wallMom_correction_InletWall_Edge);
						if(!status){ fprintf(stderr, "deviceFree wall mom correction (2) inlet  failed\n"); finalise_GPU_res=false; }
					}

					if(GPUDataAddr_wallMom_correction_Inlet_Inner != nullptr){
						status = deviceFree(GPUDataAddr_wallMom_correction_Inlet_Inner);
						if(!status){ fprintf(stderr, "deviceFree wall mom correction (3) inlet  failed\n"); finalise_GPU_res=false; }
					}

					if(GPUDataAddr_wallMom_correction_InletWall_Inner != nullptr){
						status = deviceFree(GPUDataAddr_wallMom_correction_InletWall_Inner);
						if(!status){ fprintf(stderr, "deviceFree wall mom correction (4) inlet  failed\n"); finalise_GPU_res=false; }
					}

					// Only valid for the Vel Bcs Case: b. File
					if(GPUDataAddr_pp_Inlet_weightsTable_coord != nullptr){
						status = deviceFree(GPUDataAddr_pp_Inlet_weightsTable_coord);
						if(!status){ fprintf(stderr, "deviceFree pointer to pointers Coordinates in weights_table - inlet  failed\n"); finalise_GPU_res=false; }
					}

					if(GPUDataAddr_p_Inlet_weightsTable_wei != nullptr){
						status = deviceFree(GPUDataAddr_p_Inlet_weightsTable_wei);
						if(!status){ fprintf(stderr, "deviceFree pointer to pointers weights in weights_table - inlet  failed\n"); finalise_GPU_res=false; }
					}

					// Key value indices - Read these values from GPU Global mem instead of searching for the weight based on the key (xyz)
					if(GPUDataAddr_index_weightTable_Inlet_Edge != nullptr){
						status = deviceFree(GPUDataAddr_index_weightTable_Inlet_Edge);
						if(!status){ fprintf(stderr, "deviceFree map key value index in weights_table - inlet  failed\n"); finalise_GPU_res=false; }
					}

					if(GPUDataAddr_index_weightTable_InletWall_Edge != nullptr){
						status = deviceFree(GPUDataAddr_index_weightTable_InletWall_Edge);
						if(!status){ fprintf(stderr, "deviceFree map key value index in weights_table - inlet  failed\n"); finalise_GPU_res=false; }
					}

					if(GPUDataAddr_index_weightTable_Inlet_Inner != nullptr){
						status = deviceFree(GPUDataAddr_index_weightTable_Inlet_Inner);
						if(!status){ fprintf(stderr, "deviceFree map key value index in weights_table - inlet  failed\n"); finalise_GPU_res=false; }
					}
					if(GPUDataAddr_index_weightTable_InletWall_Inner != nullptr){
						status = deviceFree(GPUDataAddr_index_weightTable_InletWall_Inner);
						if(!status){ fprintf(stderr, "deviceFree map key value index in weights_table - inlet  failed\n"); finalise_GPU_res=false; }
					}

				} // Closes the if (hemeIoletBC_Inlet == "LADDIOLET")

				// TOFO: Free the following:
				/**
				void *GPUDataAddr_Coords_Inlet_Edge;
				void *GPUDataAddr_Coords_InletWall_Edge;
				void *GPUDataAddr_Coords_Inlet_Inner;
				void *GPUDataAddr_Coords_InletWall_Inner;
				void *GPUDataAddr_Coords_Outlet_Edge;
				void *GPUDataAddr_Coords_OutletWall_Edge;
				void *GPUDataAddr_Coords_Outlet_Inner;
				void *GPUDataAddr_Coords_OutletWall_Inner;
				*/



				if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
					status = deviceFree(d_ghostDensity_out);
					if(!status){ fprintf(stderr, "deviceFree ghost density outlet failed\n"); finalise_GPU_res=false; }

					// Free up pinned Memory associated with the Pressure BCs
					if (h_ghostDensity_outlet != nullptr){
						status = deviceFreeHost(h_ghostDensity_outlet);
						if(!status){ fprintf(stderr, "deviceFreeHost h_ghostDensity_outlet failed ... \n"); finalise_GPU_res=false; }
						h_ghostDensity_outlet = nullptr;
					}
				}

				if (hemeIoletBC_Outlet == "LADDIOLET"){
					/*
					void *GPUDataAddr_wallMom_correction_Outlet_Edge;
					void *GPUDataAddr_wallMom_correction_OutletWall_Edge;
					void *GPUDataAddr_wallMom_correction_Outlet_Inner;
					void *GPUDataAddr_wallMom_correction_OutletWall_Inner;
					*/
					status = deviceFree(GPUDataAddr_wallMom_correction_Outlet_Edge);
					if(!status){ fprintf(stderr, "deviceFree wall mom correction (1) outlet  failed\n"); finalise_GPU_res=false; }

					status = deviceFree(GPUDataAddr_wallMom_correction_OutletWall_Edge);
					if(!status){ fprintf(stderr, "deviceFree wall mom correction (2) outlet  failed\n"); finalise_GPU_res=false; }

					status = deviceFree(GPUDataAddr_wallMom_correction_Outlet_Inner);
					if(!status){ fprintf(stderr, "deviceFree wall mom correction (3) outlet  failed\n"); finalise_GPU_res=false; }

					status = deviceFree(GPUDataAddr_wallMom_correction_OutletWall_Inner);
					if(!status){ fprintf(stderr, "deviceFree wall mom correction (4) outlet  failed\n"); finalise_GPU_res=false; }
				}

				//----------------------------------------------------------------------

				status = deviceFree(d_inletNormal);
				if(!status){ fprintf(stderr, "deviceFree failed\n"); finalise_GPU_res=false; }

				status = deviceFree(d_outletNormal);
				if(!status){ fprintf(stderr, "deviceFree failed\n"); finalise_GPU_res=false; }

				status = deviceFree(mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat);
				if(!status){ fprintf(stderr, "deviceFree failed\n"); finalise_GPU_res=false; }

				status = deviceFree(mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat);
				if(!status){ fprintf(stderr, "deviceFree failed\n"); finalise_GPU_res=false; }

				status = deviceFree(GPUDataAddr_int64_Neigh_d);
				if(!status){ fprintf(stderr, "deviceFree failed\n"); finalise_GPU_res=false; }


				/**
					I need to free the following from the GPU global memory:
					// wall Momentum associated with Velocity BCs (LADDIOLET)
					void *GPUDataAddr_wallMom_Inlet_Edge;
					void *GPUDataAddr_wallMom_InletWall_Edge;
					void *GPUDataAddr_wallMom_Inlet_Inner;
					void *GPUDataAddr_wallMom_InletWall_Inner;
					void *GPUDataAddr_wallMom_Outlet_Edge;
					void *GPUDataAddr_wallMom_OutletWall_Edge;
					void *GPUDataAddr_wallMom_Outlet_Inner;
					void *GPUDataAddr_wallMom_OutletWall_Inner;
				*/

				//printf("deviceFree - Delete dynamically allocated memory on the GPU.\n\n");

				return finalise_GPU_res;
			}


// Note that it is called for all ranks except rank 0
template<class LatticeType>
			bool LBM<LatticeType>::Initialise_GPU(iolets::BoundaryValues* iInletValues,
					iolets::BoundaryValues* iOutletValues, const util::UnitConverter* iUnits)
			{

				bool initialise_GPU_res = true;

				// March 2023 - Bollean variable if exporting shear stress magnitude to disk
				bool save_wallShearStressMagn = true;
				//

				mInletValues = iInletValues;
				mOutletValues = iOutletValues;
				mUnits = iUnits;

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();
				// std::printf("Local Rank = %i and local fluid sites = %i \n\n", myPiD, mLatDat->GetLocalFluidSiteCount());

				//======================================================================
				// Preliminary check -
				// Compare: a) available GPU mem. and
				//			b) mem. requirements based on simulation domain

				// Available GPU memory


				// a. Available GPU mem.
				size_t avail_GPU_mem = deviceGetProperties(myPiD);

				// b. Rough estimate of the GPU Memory requested:
				// Total number of fluid sites and totSharedFs
				uint64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount(); // Actually GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)
				uint64_t totSharedFs = mLatDat->totalSharedFs;

				unsigned long long TotalMem_dbl_fOld = ( nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs)  * sizeof(distribn_t); // Total memory size for fOld
				unsigned long long TotalMem_dbl_MacroVars = (1+3) * nFluid_nodes  * sizeof(distribn_t); // Total memory size for macroVariables: density and Velocity n[nFluid_nodes], u[nFluid_nodes][3]
				unsigned long long TotalMem_int64_Neigh = ( nFluid_nodes * LatticeType::NUMVECTORS)  * sizeof(site_t); // Total memory size for neighbouring Indices
				// Reconsider the memory required for the following 2 (Wall / Iolet Intersection) - Needs to change... TODO!!!
				unsigned long long TotalMem_uint32_WallIntersect = nFluid_nodes * sizeof(uint32_t); // Total memory size
				unsigned long long TotalMem_uint32_IoletIntersect = nFluid_nodes * sizeof(uint32_t); // Total memory size
				unsigned long long TotalMem_int64_streamInd = totSharedFs * sizeof(site_t); // Total memory size for streamingIndicesForReceivedDistributions

				unsigned long long est_TotalMem_req = (TotalMem_dbl_fOld * 2 +  TotalMem_dbl_MacroVars + TotalMem_int64_Neigh + TotalMem_uint32_WallIntersect + TotalMem_uint32_IoletIntersect + TotalMem_int64_streamInd);

				if(est_TotalMem_req >= avail_GPU_mem){
					std::printf("Rank %i - Approx. estimate of GPU mem. required: %.1fGB - Available GPU mem. %.1fGB\n", myPiD, ((double)est_TotalMem_req/1073741824.0), ((double)avail_GPU_mem/1073741824.0));
					std::cout << "Warning: Not sufficient GPU memory!!! Increase the number of ranks or decrease system size" << std::endl;
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				//======================================================================


				// --------------------------------------------------------------------------------------------------------------------------------------------------------------
				// Distribution functions:
				// 	Option: Separately each element in memory
				//	f_old, f_new -	Comment: add the totalSharedFs values +1: Done!!!

				// Arrange the data in 2 ways - Done!!!
				//	a. Arrange by fluid index (as is oldDistributions), i.e f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]
				//	b. Arrange by index_LB, i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., fq[0 to (nFluid_nodes-1)]
				// KEEP ONLY Option (b) - Refer to earlier versions of the code for option (a).

				// Total number of fluid sites
				//uint64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount(); // Actually GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)
				//uint64_t totSharedFs = mLatDat->totalSharedFs;
				// std::printf("Proc# %i : Total Fluid nodes = %i, totalSharedFs = %i \n\n", myPiD, nFluid_nodes, totSharedFs);	// Test that I can access the value of totalSharedFs (protected member of class LatticeData (geometry/LatticeData.h) - declares class LBM as friend)

				TotalMem_dbl_fOld = ( nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs)  * sizeof(distribn_t); // Total memory size for fOld
				unsigned long long TotalMem_dbl_fNew = TotalMem_dbl_fOld;	// Total memory size for fNew
				TotalMem_dbl_MacroVars = (1+3) * nFluid_nodes  * sizeof(distribn_t); // Total memory size for macroVariables: density and Velocity n[nFluid_nodes], u[nFluid_nodes][3]

				//--------------------------------------------------------------------------------------------------
				// Alocate memory on the GPU for MacroVariables: density and Velocity
				// Number of elements (type double / distribn_t)
				// uint64_t nArray_MacroVars = nFluid_nodes; // uint64_t (unsigned long long int)

				bool status = deviceMalloc((void**)&GPUDataAddr_dbl_MacroVars, TotalMem_dbl_MacroVars);
				if(!status){
					fprintf(stderr, "GPU memory allocation MacroVariables failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// 12 April 2023
				// Check whether to allocate memory for the wall shear stress magnitude
				// Related to mSimConfig
				// Consider whether to send boolean output_shearStressMagn to GPU constant memory, or just pass as an argument to the kernels
				bool output_shearStressMagn = true;
				if (output_shearStressMagn){
					// Allocate mem for wall shear stress magnitude on the GPU
					// TODO...
				}
				//--------------------------------------------------------------------------------------------------

				//--------------------------------------------------------------------------------------------------
				// std::vector<distribn_t> oldDistributions; //! The distribution function fi's values for the current time step.
				// oldDistributions.resize(localFluidSites * latticeInfo.GetNumVectors() + 1 + totalSharedFs);  -  see src/geometry/LatticeData.h (line 422 in function void PopulateWithReadData)
				//--------------------------------------------------------------------------------------------------

				//--------------------------------------------------------------------------------------------------
				//	b. Arrange by index_LB, i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., f_(q-1)[0 to (nFluid_nodes-1)]
				distribn_t* Data_dbl_fOld_b = new distribn_t[nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs];	// distribn_t (type double)
				distribn_t* Data_dbl_fNew_b = new distribn_t[nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs];	// distribn_t (type double)

				if(!Data_dbl_fOld_b || !Data_dbl_fNew_b){
					std::cout << "Memory allocation error" << std::endl;
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// 	f_old - Done!!!
				#pragma omp parallel for collapse(2)
				for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++)
				{
					for (site_t i = 0; i < mLatDat->GetLocalFluidSiteCount(); i++)
					{
						*(&Data_dbl_fOld_b[l * mLatDat->GetLocalFluidSiteCount() + i]) = *(mLatDat->GetFOld(i * LatticeType::NUMVECTORS + l)); // distribn_t (type double) - Data_dbl_fOld contains the oldDistributions re-arranged
					}
				}

				// 	f_new - Done!!!
				#pragma omp parallel for collapse(2)
				for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++)
				{
					for (site_t i = 0; i < mLatDat->GetLocalFluidSiteCount(); i++)
					{
						*(&Data_dbl_fNew_b[l * mLatDat->GetLocalFluidSiteCount() + i]) = *(mLatDat->GetFNew(i * LatticeType::NUMVECTORS + l)); // distribn_t (type double) - Data_dbl_fNew contains the oldDistributions re-arranged
					}
				}
				//--------------------------------------------------------------------------------------------------

				//
				// Alocate memory on the GPU
				// Number of elements (type double/distribn_t) in oldDistributions and newDistributions
				// 	including the extra part (+1 + totalSharedFs) - Done!!!
				uint64_t nArray_Distr = nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs; // uint64_t (unsigned long long int)

				//--------------------------------------------------------------------------------------------------
				//	b. Arrange by index_LB
				//		i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., f_(q-1)[0 to (nFluid_nodes-1)]

				// Modified for the cuda-aware mpi option.
				// Access the pointer to global memory declared in class LatticeData (GPUDataAddr_dbl_fOld_b_mLatDat)
				// (geometry::LatticeData* mLatDat;)
				// Memory copy from host (Data_dbl_fOld_b) to Device (mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat)
				// cudaStatus = deviceMallocManaged((void**)&(mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat), nArray_Distr * sizeof(distribn_t));
				status = deviceMalloc((void**)&(mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat), nArray_Distr * sizeof(distribn_t));
				if(!status) {
					fprintf(stderr, "Device Malloc Failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
				}
				status = deviceMemcpy(mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																	Data_dbl_fOld_b, nArray_Distr * sizeof(distribn_t), memcpyHostToDevice);
				if(!status){
					fprintf(stderr, "GPU memory transfer Host To Device failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				//cudaStatus = deviceMallocManaged((void**)&(mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat), nArray_Distr * sizeof(distribn_t));
				status = deviceMalloc((void**)&(mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat), nArray_Distr * sizeof(distribn_t));

				if(!status) {
					fprintf(stderr, "Device Malloc2 Failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
				}

				status = deviceMemcpy(mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																	Data_dbl_fNew_b, nArray_Distr * sizeof(distribn_t), memcpyHostToDevice);
				if(!status){
					fprintf(stderr, "GPU memory transfer Host To Device failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Delete - Free-up memory here... (Not at the end of the function)
				//=================================================================================================================================



				//=================================================================================================================================
				// Neighbouring indices - necessary for the STREAMING STEP

				// 	The total size of the neighbouring indices should be: neighbourIndices.resize(latticeInfo.GetNumVectors() * localFluidSites); (see geometry/LatticeData.cc:369)
				// 	Keep only method (d): refer to the actual streaming index in f's array (arranged following method (b): Arrange by index_LB)

				//		Type site_t (units.h:28:		typedef int64_t site_t;)
				//		geometry/LatticeData.h:634:		std::vector<site_t> neighbourIndices; //! Data about neighbouring fluid sites.
				//	Memory requirements
				TotalMem_int64_Neigh = ( nFluid_nodes * LatticeType::NUMVECTORS)  * sizeof(site_t); // Total memory size for neighbouring Indices

				// -----------------------------------------------------------------------
				// d. Arrange by index_LB, i.e. neigh_0[0 to (nFluid_nodes-1)], neigh_1[0 to (nFluid_nodes-1)], ..., neigh_(q-1)[0 to (nFluid_nodes-1)]
				//		But instead of keeping the array index from HemeLB, convert to:
				//		1. the actual fluid ID and then to
				//		2. the actual address in global Memory
				site_t* Data_int64_Neigh_d = new site_t[TotalMem_int64_Neigh/sizeof(site_t)];	// site_t (type int64_t)
				if(!Data_int64_Neigh_d){
					std::cout << "Memory allocation error - Neigh. (d)" << std::endl;
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Re-arrange the neighbouring data - organised by index LB
				#pragma omp parallel for collapse(2)
				for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++)
				{
					for (site_t i = 0; i < mLatDat->GetLocalFluidSiteCount(); i++)
					{
						site_t neigh_Index_Heme = mLatDat->neighbourIndices[(int64_t)(LatticeType::NUMVECTORS)*i  + l]; // Refers to the address in hemeLB memory (CPU version - method a memory arrangement)

						// If the streaming Index (i.e. neighbour in LB_Dir = l) is within the simulation domain
						// Calculate its ACTUAL streaming fluid ID index
						// And then the corresponding address in global memory .
						if (neigh_Index_Heme < mLatDat->GetLocalFluidSiteCount() * LatticeType::NUMVECTORS )
						{
								site_t neigh_Fluid_Index = (neigh_Index_Heme - l)/LatticeType::NUMVECTORS;	// Evaluate the ACTUAL streaming fluid ID index
								site_t neigh_Address_Index = neigh_Fluid_Index + l * mLatDat->GetLocalFluidSiteCount();	// Evaluate the corresponding address in global memory (method b - memory arrangement)

								Data_int64_Neigh_d[(int64_t)l * mLatDat->GetLocalFluidSiteCount() + i] = neigh_Address_Index;
						}
						else{
								Data_int64_Neigh_d[(int64_t)l * mLatDat->GetLocalFluidSiteCount() + i] = neigh_Index_Heme;
						}

						/*
						// Investigate what is the neighbour index if wall link
						// It turns out that: For the sites next to walls, the corresponding neighbouring index is set to the maximum value based on the number of fluid sites on the Rank PLUS ONE,
						// i.e. this value is: mLatDat->GetLocalFluidSiteCount() * LatticeType::NUMVECTORS + 1
						geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(i);
						// For Debugging purposes -Remove later
    				bool test_bool_Wall_Intersect = site.HasWall(l);	// Boolean variable: if there is wall (True) - Compare with boolean variable site.HasWall(LB_Dir)
						if (test_bool_Wall_Intersect){
							if(myPiD==2) printf("Rank: %d, Site Index: %lld, Wall in LB_dir: %d, Neighbouring Index: %lld, Max Index: %lld \n\n", myPiD, i, l, neigh_Index_Heme, (int64_t)(mLatDat->GetLocalFluidSiteCount() * LatticeType::NUMVECTORS) );

						}
						*/
						//
						//std::printf("Memory allocation Data_int64_Neigh(b) successful from Proc# %i \n\n", myPiD);
					}
				}
				// ------------------------------------------------------------------------

				// Number of elements (type long long int/site_t) in neighbourIndices  - To do!!!
				uint64_t nArray_Neigh = nFluid_nodes * LatticeType::NUMVECTORS; // uint64_t (unsigned long long int)

				// ------------------------------------------------------------------------
				//	d. Arrange by index_LB, i.e. neigh_0[0 to (nFluid_nodes-1)], neigh_1[0 to (nFluid_nodes-1)], ..., neigh_(q-1)[0 to (nFluid_nodes-1)]
				//	 		But refer to ACTUAL address in Global memory (method b) for the FLUID ID index - TO BE USED ONLY when in PreReceive() - streaming in the simulation domain!!!
				status = deviceMalloc((void**)&GPUDataAddr_int64_Neigh_d, nArray_Neigh * sizeof(site_t));
				if(!status){
					fprintf(stderr, "GPU memory allocation for Neigh.(d) failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Memory copy from host (Data_int64_Neigh_b) to Device (GPUDataAddr_int64_Neigh_b)
				status = deviceMemcpy(GPUDataAddr_int64_Neigh_d, Data_int64_Neigh_d, nArray_Neigh * sizeof(site_t), memcpyHostToDevice);
				if(!status){
					fprintf(stderr, "GPU memory transfer Host To Device for Neigh.(d) failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Delete - Free-up memory here... (Not at the end of the function)
				//=================================================================================================================================


				//***********************************************************************************************************************************
				// Fluid-Wall links
				// Access the information for the fluid-wall links:
				//	function GetWallIntersectionData returns wallIntersection variable that we want...
				/** To do:
						1. Restrict to the number of fluid sites neighbouring wall sites. Get this info from the range of the corresponding collision-streaming kernels
						2. Probably remove entirely!!! Examine whether to keep this info:
								Fluid-Wall intersection can be accessed from the neighbouring fluid index above.
				*/

				TotalMem_uint32_WallIntersect = nFluid_nodes * sizeof(uint32_t); // Total memory size

				// Allocate memory on the host
				// Think about the following: Do I need to allocate nFluid_nodes or just the siteCount for this type of collision (check the limits for the mWallCollision). To do!!!
				uint32_t* Data_uint32_WallIntersect = new uint32_t[nFluid_nodes];	// distribn_t (type double)
				if(!Data_uint32_WallIntersect){
					std::cout << "Memory allocation error - Neigh." << std::endl;
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Fill the array Data_uint32_WallIntersect
				#pragma omp parallel for
				for (int64_t site_Index = 0; site_Index < mLatDat->GetLocalFluidSiteCount(); site_Index++) // for (int64_t site_Index = 0; site_Index < 10; site_Index++){
				{
					geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_Index);

					// Pass the value of test_Wall_Intersect (uint32_t ) to the GPU global memory - then compare with the value of mask for each LB direction to identify whether it is a wall-fluid link
					uint32_t test_Wall_Intersect = 0;
					test_Wall_Intersect = site.GetSiteData().GetWallIntersectionData(); // Returns the value of wallIntersection (type uint32_t)

					Data_uint32_WallIntersect[site_Index] = test_Wall_Intersect;

					/*
					// For debugging purposes - To check that test_Wall_Intersect was correctly set. Not needed later on
					for (unsigned int LB_Dir = 0; LB_Dir < LatticeType::NUMVECTORS; LB_Dir++)
					{
						//---------------------------------------------------------
						// This is for checking that test_Wall_Intersect can capture the wall-fluid info into a uint32_t value
						unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
						bool test_test_Wall = (test_Wall_Intersect & mask);
						//---------------------------------------------------------

						// For Debugging purposes -Remove later
						bool test_bool_Wall_Intersect = site.HasWall(LB_Dir);	// Boolean variable: if there is wall (True) - Compare with boolean variable site.HasWall(LB_Dir)

						if(test_bool_Wall_Intersect){
							if (!test_test_Wall) printf("Error: Expected Wall-fluid link \n\n!!!");
							//printf("Site: %lld - Dir: %d : Testing the comparison of test_Wall_Intersect and mask returns: %d \n\n", site_Index, LB_Dir, test_test_Wall);
						}
					} // Ends the for loop: Debugging purposes
					*/
				}
				// Ends the loop for Filling the array Data_uint32_WallIntersect

				// Alocate memory on the GPU
				status = deviceMalloc((void**)&GPUDataAddr_uint32_Wall, nFluid_nodes * sizeof(uint32_t));
				if(!status){
					fprintf(stderr, "GPU memory allocation for Wall-Fluid Intersection failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Memory copy from host (Data_uint32_WallIntersect) to Device (GPUDataAddr_uint32_Wall)
				status = deviceMemcpy(GPUDataAddr_uint32_Wall, Data_uint32_WallIntersect, nFluid_nodes * sizeof(uint32_t), memcpyHostToDevice);
				if(!status){
					fprintf(stderr, "GPU memory transfer Host To Device for Wall-Fluid Intersection failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Delete - Free-up memory here... (Not at the end of the function)
				delete[] Data_uint32_WallIntersect;

				//----------------------------------------------------------------------
				// March 2023
				/* Wall normals,
						see function: in geometry/Site.h: inline const util::Vector3D<distribn_t>& GetWallNormal() const
						TODO: Add this info and send to GPU global memory
				**/

				//----------------------------------------------------------------------
				//***********************************************************************************************************************************


				//***********************************************************************************************************************************
				// Fluid-Inlet links
				// Access the information for the fluid-inlet links:
				//	function GetIoletIntersectionData() returns ioletIntersection variable that we want...

				// Do we need nFluid_nodes elements of type uint32_t??? Think...
				// In PreSend() the site limits for mInletCollision:
				// offset = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
				// siteCount_iolet_PreSend = mLatDat->GetDomainEdgeCollisionCount(2);
				// Include the mInletWallCollision as well.

				// In PreReceive() the site limits for mInletCollision:
				// offset = 0 + mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
				// siteCount_iolet_PreReceive = mLatDat->GetMidDomainCollisionCount(2);
				// Include the mInletWallCollision as well.

				// To do:
				// 1. Allocate just for the fluid sites involved (siteCount_iolet_PreSend + siteCount_iolet_PreReceive + ...)

				TotalMem_uint32_IoletIntersect = nFluid_nodes * sizeof(uint32_t); // Total memory size

				// Allocate memory on the host
				uint32_t* Data_uint32_IoletIntersect = new uint32_t[nFluid_nodes];	// distribn_t (type double)
				if(!Data_uint32_IoletIntersect){
					std::cout << "Memory allocation error - iolet" << std::endl;
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Fill the array Data_uint32_IoletIntersect
				#pragma omp parallel for
				for (int64_t site_Index = 0; site_Index < mLatDat->GetLocalFluidSiteCount(); site_Index++) // for (int64_t site_Index = 0; site_Index < 10; site_Index++){
				{
					geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_Index);

					// Pass the value of test_Iolet_Intersect (uint32_t ) to the GPU global memory - then compare with the value of mask for each LB direction to identify whether it is a iolet-fluid link
					uint32_t test_Iolet_Intersect = 0;
					test_Iolet_Intersect = site.GetSiteData().GetIoletIntersectionData(); // Returns the value of ioletIntersection (type uint32_t)

					Data_uint32_IoletIntersect[site_Index] = test_Iolet_Intersect;

					/*
					// For debugging purposes - To check that test_Wall_Intersect was correctly set. Not needed later on
					for (unsigned int LB_Dir = 0; LB_Dir < LatticeType::NUMVECTORS; LB_Dir++)
					{
						//---------------------------------------------------------
						// This is for checking that test_Iolet_Intersect can capture the wall-fluid info into a uint32_t value
						unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Iolet_Intersect (To do: compare against test_bool_Wall_Intersect as well)
						bool test_test_Iolet = (test_Iolet_Intersect & mask);
						//---------------------------------------------------------

						// For Debugging purposes -Remove later
						bool test_bool_Iolet_Intersect = site.HasIolet(LB_Dir);	// Boolean variable: if there is Iolet (True) - Compare with boolean variable site.HasIolet(LB_Dir)

						if(test_bool_Iolet_Intersect){
							if (!test_test_Iolet) printf("Error: Expected Wall-fluid link \n\n!!!");
							//printf("Site: %lld - Dir: %d : Testing the comparison of test_Iolet_Intersect and mask returns: %d \n\n", site_Index, LB_Dir, test_test_Iolet);
						}
					}
					*/
				}
				// Ends the loop for Filling the array Data_uint32_IoletIntersect

				// Alocate memory on the GPU
				status = deviceMalloc((void**)&GPUDataAddr_uint32_Iolet, nFluid_nodes * sizeof(uint32_t));
				if(!status){
					fprintf(stderr, "GPU memory allocation for Iolet-Fluid Intersection failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
				status = deviceMemcpy(GPUDataAddr_uint32_Iolet, Data_uint32_IoletIntersect, nFluid_nodes * sizeof(uint32_t), memcpyHostToDevice);
				if(!status){
					fprintf(stderr, "GPU memory transfer Host To Device for Iolet failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Delete - Free-up memory here... (Not at the end of the function)
				//***********************************************************************************************************************************

				//***********************************************************************************************************************************
				// Iolets BCs:
				/** Before focusing on the type of Inlet / Outlet  BCs, i.e.
							a. Velocity (LADDIOLET)
							b. Pressure BCs (NASHZEROTHORDERPRESSUREIOLET)
						examine the iolets IDs and the corresponding fluid sites
				*/

				/**
				// 	IMPORTANT:
				//		Note that the value returned from GetTotalIoletCount() is the global iolet count, whereas the value returned from GetLocalIoletCount() is the local iolet count on the current RANK.
				// 		Function identify_Range_iolets_ID() returns: the local Iolet count and the Fluid sites range associated with each iolet and the corresponding iolet ID (and consequently ghost density)

				Get the local Iolet count and the Fluid sites range for the following:
					1. Inlets - Done!!!
							1.1. 	PreSend Collision-streaming (Domain Edges):  		collision-streaming Types 3 (mInletCollision) & 5 (mInletWallCollision)
												returns:
													std::vector<site_t> Iolets_Inlet_Edge; 			// vector with Inlet IDs and range associated with PreSend collision-streaming Type 3 (mInletCollision)
													int n_LocalInlets_mInlet_Edge; 							// number of local Inlets involved during the PreSend mInletCollision collision - Can include repeated iolet IDs
													int n_unique_LocalInlets_mInlet_Edge;				// number of unique local Inlets

													std::vector<site_t> Iolets_InletWall_Edge;	// vector with Inlet IDs and range associated with PreSend collision-streaming Type 5 (mInletWallCollision)
													int n_LocalInlets_mInletWall_Edge; 					// number of local Inlets involved during the PreSend mInletWallCollision collision
													int n_unique_LocalInlets_mInletWall_Edge;

							1.2. 	PreReceive Collision-streaming (Inner Domain):	collision-streaming Types 3 (mInletCollision) & 5 (mInletWallCollision)
												returns:
													std::vector<site_t> Iolets_Inlet_Inner;			// vector with Inlet IDs and range associated with PreReceive collision-streaming Types 3 (mInletCollision)
													int n_LocalInlets_mInlet; 									// number of local Inlets involved during the PreReceive mInletCollision collision

													std::vector<site_t> Iolets_InletWall_Inner;	// vector with Inlet IDs and range associated with PreReceive collision-streaming Types 5 (mInletWallCollision)
													int n_LocalInlets_mInletWall; 							// number of local Inlets involved during the PreReceive mInletWallCollision collision

					2. Outlets - Done!!!
							2.1. 	PreSend Collision-streaming (Domain Edges):  		collision-streaming Types 4 (mOutletCollision) & 6 (mOutletWallCollision)
												returns:
													std::vector<site_t> Iolets_Outlet_Edge;			// vector with Outlet IDs and range associated with PreSend collision-streaming Types 4 (mOutletCollision)
													int n_LocalOutlets_mOutlet_Edge; 						// number of local Outlets involved during the PreSend mOutletCollision collision

													std::vector<site_t> Iolets_OutletWall_Edge;	// vector with Outlet IDs and range associated with PreSend collision-streaming Types 6 (mOutletWallCollision)
													int n_LocalOutlets_mOutletWall_Edge; 				// number of local Outlets involved during the PreSend mOutletWallCollision collision

							2.2. 	PreReceive Collision-streaming (Inner Domain):	collision-streaming Types 4 (mOutletCollision) & 6 (mOutletWallCollision)
												returns:
												std::vector<site_t> Iolets_Outlet_Inner;			// vector with Outlet IDs and range associated with PreReceive collision-streaming Types 4 (mOutletCollision)
												int n_LocalOutlets_mOutlet; 									// number of local Outlets involved during the PreReceive mOutletCollision collision

												std::vector<site_t> Iolets_OutletWall_Inner;	// vector with Outlet IDs and range associated with PreReceive collision-streaming Types 6 (mOutletWallCollision)
												int n_LocalOutlets_mOutletWall; 							// number of local Outlets involved during the PreReceive mOutletWallCollision collision

				*/

				// TOTAL GLOBAL number of INLETS and OUTLETS
				int n_Inlets = mInletValues->GetTotalIoletCount();
				int n_Outlets = mOutletValues->GetTotalIoletCount();
				//printf("Rank: %d, Number of TOTAL (Local) inlets: %d (%d), TOTAL (Local) Outlets: %d (%d) \n\n", myPiD, n_Inlets, mInletValues->GetLocalIoletCount(), n_Outlets, mOutletValues->GetLocalIoletCount());

				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				//=============================================================================================================================================================
				/**		1. Inlets		**/
				//=============================================================================================================================================================
				/** 	1.1. 	PreSend Collision-streaming (Domain Edges):  		collision-streaming Types 3 (mInletCollision) & 5 (mInletWallCollision)	**/
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				//		Domain Edges:
				//		Limits of the inlet collision-streaming (i.e. Collision Types 3 & 5: mInletCollision, mInletWallCollision)
				//		1.1.a. 	Collision Type 3: mInletCollision
				site_t start_Index_Inlet_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
				site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);

				//		1.1.b. 	Collision Type 5: mInletWallCollision
				site_t start_Index_InletWall_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
                														+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);
				site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------

				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				// Loop over the site range involved in iolet collisions
				// Case 1.1.a. Collision Type 3: mInletCollision
				//std::vector<site_t> Iolets_Inlet_Edge;
				n_LocalInlets_mInlet_Edge = 0; // number of local Inlets involved during the mInletCollision collision- Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalInlets_mInlet_Edge = 0;
				if(site_Count_Inlet_Edge != 0){
					Iolets_Inlet_Edge = identify_Range_iolets_ID(start_Index_Inlet_Edge, (start_Index_Inlet_Edge + site_Count_Inlet_Edge),
																												&n_LocalInlets_mInlet_Edge, &n_unique_LocalInlets_mInlet_Edge);
					// TODO: Call function to prepare the struct object (with FAM for the array Iolets_ID_range):
					//				struct Iolets *createIolet(struct Iolets *iolet_member, int number_LocalIolets, int number_UniqueLocalIolets)
					Inlet_Edge.n_local_iolets = n_LocalInlets_mInlet_Edge;
					memcpy(&Inlet_Edge.Iolets_ID_range, &Iolets_Inlet_Edge[0], 3* n_LocalInlets_mInlet_Edge *sizeof(site_t));

					//
					// Debugging
					/*std::cout << "Rank: " << myPiD << " - Unique local Inlets: " << n_unique_LocalInlets_mInlet_Edge << " - Total local Inlets on current Rank (1st Round - mInlet_Edge): " << n_LocalInlets_mInlet_Edge << " with Inlet ID:";
					for (int index = 0; index < n_LocalInlets_mInlet_Edge; index++ )
						std::cout << ' ' << Iolets_Inlet_Edge[3*index];
					std::cout << "\n\n"; */
					//
					site_t MemSz = 3 * n_LocalInlets_mInlet_Edge *  sizeof(site_t);
					status = deviceMalloc((void**)&GPUDataAddr_Inlet_Edge, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Iolet: Inlet Edge failed...\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					status = deviceMemcpy(GPUDataAddr_Inlet_Edge, &Iolets_Inlet_Edge[0], MemSz, memcpyHostToDevice);
					if( !status ){
						fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Inlet Edge failed... \n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}
				}

				/* Debugging - Remove later
				// Works fine
				std::cout << "OUTSIDE Rank: " << myPiD << " - Unique local Inlets: " << n_unique_LocalInlets_mInlet_Edge << " - Total local Inlets on current Rank (1st Round - mInlet_Edge): " << n_LocalInlets_mInlet_Edge << " with Inlet ID:";
				for (int index = 0; index < n_LocalInlets_mInlet_Edge; index++ )
					std::cout << ' ' << Iolets_Inlet_Edge[3*index];
				std::cout << "\n\n";
				*/

				// Case 1.1.b. Collision Type 5: mInletWallCollision
				//std::vector<site_t> Iolets_InletWall_Edge;
				n_LocalInlets_mInletWall_Edge = 0; // number of local Inlets involved during the mInletWallCollision collision- Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalInlets_mInletWall_Edge = 0;
				if(site_Count_InletWall_Edge != 0){
					Iolets_InletWall_Edge = identify_Range_iolets_ID(start_Index_InletWall_Edge, (start_Index_InletWall_Edge + site_Count_InletWall_Edge), &n_LocalInlets_mInletWall_Edge, &n_unique_LocalInlets_mInletWall_Edge);

					InletWall_Edge.n_local_iolets = n_LocalInlets_mInletWall_Edge;
					memcpy(&InletWall_Edge.Iolets_ID_range, &Iolets_InletWall_Edge[0], 3* n_LocalInlets_mInletWall_Edge *sizeof(site_t));

					/*std::cout << "Rank: " << myPiD << " - Unique local Inlets: " << n_unique_LocalInlets_mInletWall_Edge << " - Total local Inlets on current Rank (1st Round - mInletWall_Edge): " << n_LocalInlets_mInletWall_Edge << " with Inlet ID:";
					for (int index = 0; index < n_LocalInlets_mInletWall_Edge; index++ )
						std::cout << ' ' << Iolets_InletWall_Edge[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalInlets_mInletWall_Edge *  sizeof(site_t);
					status = deviceMalloc((void**)&GPUDataAddr_InletWall_Edge, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Iolet: Inlet Wall Edge failed...\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					status = deviceMemcpy(GPUDataAddr_InletWall_Edge, &Iolets_InletWall_Edge[0], MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Inlet Wall Edge failed... \n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}
				}
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------

				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				/** 	1.2. 	PreReceive Collision-streaming (Inner Domain):	collision-streaming Types 3 (mInletCollision) & 5 (mInletWallCollision)	**/
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				// 		Inner Domain:
				//		Limits of the inlet collision-streaming (i.e. Collision Types 3 & 5: mInletCollision, mInletWallCollision)
				// 		1.2.a. 	Collision Type 3: mInletCollision
				site_t start_Index_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
				site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);

				// 		1.2.b. 	Collision Type 5: mInletWallCollision
				site_t start_Index_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
      	site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------

				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				// Case 1.2.a. 	Collision Type 3: mInletCollision
				//std::vector<site_t> Iolets_Inlet_Inner;
				n_LocalInlets_mInlet = 0; // number of local Inlets involved during the mInletCollision collision- Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalInlets_mInlet = 0;
				if(site_Count_Inlet_Inner != 0)
				{
					Iolets_Inlet_Inner = identify_Range_iolets_ID(start_Index_Inlet_Inner, (start_Index_Inlet_Inner + site_Count_Inlet_Inner), &n_LocalInlets_mInlet, &n_unique_LocalInlets_mInlet);

					Inlet_Inner.n_local_iolets = n_LocalInlets_mInlet;
					memcpy(&Inlet_Inner.Iolets_ID_range, &Iolets_Inlet_Inner[0], 3* n_LocalInlets_mInlet *sizeof(site_t));

					/*
					// Debugging
					// Prints the list of local iolets
					std::cout << "Rank: " << myPiD << " - Unique local Inlets: " << n_unique_LocalInlets_mInlet << " - Total local Inlets on current Rank (1st Round - mInlet): " << n_LocalInlets_mInlet << " with Inlet ID:";
					for (int index = 0; index < n_LocalInlets_mInlet; index++ )
						std::cout << ' ' << Iolets_Inlet_Inner[3*index];
					std::cout << "\n\n";
					*/
					//

					site_t MemSz = 3 * n_LocalInlets_mInlet *  sizeof(site_t);
					status = deviceMalloc((void**)&GPUDataAddr_Inlet_Inner, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Iolet: Inlet Inner failed...\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					status = deviceMemcpy(GPUDataAddr_Inlet_Inner, &Iolets_Inlet_Inner[0], MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Inlet Inner failed\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}

					//
					// Debugging
					/*// GPU results
					site_t* Data_Iolets_Inlet_Inner_GPU = new site_t[3 * n_LocalInlets_mInlet];
					status = deviceMemcpy(Data_Iolets_Inlet_Inner_GPU, GPUDataAddr_Inlet_Inner, MemSz, memcpyDeviceToHost);

					if(!status){
						const char * eStr = deviceGetErrorString();
						printf("GPU memory copy D2H for Data_Iolets_Inlet_Inner_GPU failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
						initialise_GPU_res = false; return initialise_GPU_res;
					}

					for(int index_inlet = 0; index_inlet<n_LocalInlets_mInlet; index_inlet++){
						printf("From GPU - Rank: %d, Iolet ID: %d - Lower limit : %ld - Upper limit : %ld \n", myPiD, Data_Iolets_Inlet_Inner_GPU[3*index_inlet], Data_Iolets_Inlet_Inner_GPU[3*index_inlet+1] ,Data_Iolets_Inlet_Inner_GPU[3*index_inlet+2]);
					}
					*/
					//
				}

				// Case 1.2.b. 	Collision Type 5: mInletWallCollision
				//std::vector<site_t> Iolets_InletWall_Inner;
				n_LocalInlets_mInletWall = 0; // number of local Inlets involved during the mInletWallCollision collision - Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalInlets_mInletWall = 0;
				if(site_Count_InletWall_Inner!=0)
				{
					Iolets_InletWall_Inner = identify_Range_iolets_ID(start_Index_InletWall_Inner, (start_Index_InletWall_Inner + site_Count_InletWall_Inner), &n_LocalInlets_mInletWall, &n_unique_LocalInlets_mInletWall);

					InletWall_Inner.n_local_iolets = n_LocalInlets_mInletWall;
					memcpy(&InletWall_Inner.Iolets_ID_range, &Iolets_InletWall_Inner[0], 3* n_LocalInlets_mInletWall *sizeof(site_t));
					/*
					// Debugging
					std::cout << "Rank: " << myPiD << " - Unique local Inlets: " << n_unique_LocalInlets_mInletWall << " - Total local Inlets on current Rank (1st Round - mInletWall): " << n_LocalInlets_mInletWall << " with Inlet ID:";
					for (int index = 0; index < n_LocalInlets_mInletWall; index++ )
						std::cout << ' ' << Iolets_InletWall_Inner[3*index];
					std::cout << "\n\n";
					*/
					//

					site_t MemSz = 3 * n_LocalInlets_mInletWall *  sizeof(site_t);
					status = deviceMalloc((void**)&GPUDataAddr_InletWall_Inner, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Iolet: Inlet Wall Inner failed...\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					status = deviceMemcpy(GPUDataAddr_InletWall_Inner, &Iolets_InletWall_Inner[0], MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Inlet Wall Inner failed... \n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}
				}
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				//=============================================================================================================================================================


				//=============================================================================================================================================================
				/**		2. Outlets	**/
				//=============================================================================================================================================================
				/**		2.1. 	PreSend Collision-streaming (Domain Edges):  		collision-streaming Types 4 (mOutletCollision) & 6 (mOutletWallCollision) **/
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				//		Domain Edges:
				//		Limits of the outlet collision-streaming (i.e. Collision Types 4 & 6: mOutletCollision, mOutletWallCollision)
				//		2.1.a. 	Collision Type 4: mOutletCollision
				site_t start_Index_Outlet_Edge = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1) + mLatDat->GetDomainEdgeCollisionCount(2);
				site_t site_Count_Outlet_Edge = mLatDat->GetDomainEdgeCollisionCount(3);

				// 	2.1.b. 	Collision Type 6: mOutletWallCollision
				site_t start_Index_OutletWall_Edge = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
              															+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3) + mLatDat->GetDomainEdgeCollisionCount(4);
      	site_t site_Count_OutletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(5);
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------

				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				// Loop over the site range involved in iolet collisions
				// Case 2.1.a. Collision Type 4: mOutletCollision
				//std::vector<site_t> Iolets_Outlet_Edge;
				n_LocalOutlets_mOutlet_Edge = 0; // number of local Outlets involved during the mOutletCollision collision- Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalOutlets_mOutlet_Edge = 0;
				if (site_Count_Outlet_Edge!=0){
					Iolets_Outlet_Edge = identify_Range_iolets_ID(start_Index_Outlet_Edge, (start_Index_Outlet_Edge + site_Count_Outlet_Edge), &n_LocalOutlets_mOutlet_Edge, &n_unique_LocalOutlets_mOutlet_Edge);

					Outlet_Edge.n_local_iolets = n_LocalOutlets_mOutlet_Edge;
					memcpy(&Outlet_Edge.Iolets_ID_range, &Iolets_Outlet_Edge[0], 3* n_LocalOutlets_mOutlet_Edge *sizeof(site_t));

					/*std::cout << "Rank: " << myPiD << " - Unique local Outlets: " << n_unique_LocalOutlets_mOutlet_Edge << " - Total local Outlets on current Rank (1st Round - mOutlet_Edge): " << n_LocalOutlets_mOutlet_Edge << " with Inlet ID:";
					for (int index = 0; index < n_LocalOutlets_mOutlet_Edge; index++ )
						std::cout << ' ' << Iolets_Outlet_Edge[3*index];
					std::cout << "\n\n";*/
					site_t MemSz = 3 * n_LocalOutlets_mOutlet_Edge *  sizeof(site_t);
					status = deviceMalloc((void**)&GPUDataAddr_Outlet_Edge, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Iolet: Outlet Edge failed...\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					status = deviceMemcpy(GPUDataAddr_Outlet_Edge, &Iolets_Outlet_Edge[0], MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Outlet Edge failed... \n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}
				}

				// Case 2.1.b. Collision Type 6: mOutletWallCollision
				//std::vector<site_t> Iolets_OutletWall_Edge;
				n_LocalOutlets_mOutletWall_Edge = 0; // number of local Outlets involved during the mOutletCollision collision- Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalOutlets_mOutletWall_Edge = 0;
				if (site_Count_OutletWall_Edge!=0){
					Iolets_OutletWall_Edge = identify_Range_iolets_ID(start_Index_OutletWall_Edge, (start_Index_OutletWall_Edge + site_Count_OutletWall_Edge), &n_LocalOutlets_mOutletWall_Edge, &n_unique_LocalOutlets_mOutletWall_Edge);

					OutletWall_Edge.n_local_iolets = n_LocalOutlets_mOutletWall_Edge;
					memcpy(&OutletWall_Edge.Iolets_ID_range, &Iolets_OutletWall_Edge[0], 3* n_LocalOutlets_mOutletWall_Edge *sizeof(site_t));

					/*std::cout << "Rank: " << myPiD << " - Unique local Outlets: " << n_unique_LocalOutlets_mOutletWall_Edge << " - Total local Outlets on current Rank (1st Round - mOutletWall_Edge): " << n_LocalOutlets_mOutletWall_Edge << " with Inlet ID:";
					for (int index = 0; index < n_LocalOutlets_mOutletWall_Edge; index++ )
						std::cout << ' ' << Iolets_OutletWall_Edge[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalOutlets_mOutletWall_Edge *  sizeof(site_t);
					status = deviceMalloc((void**)&GPUDataAddr_OutletWall_Edge, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Iolet: Outlet Wall Edge failed...\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					status = deviceMemcpy(GPUDataAddr_OutletWall_Edge, &Iolets_OutletWall_Edge[0], MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Outlet Wall Edge failed... \n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}
				}
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------

				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				/**	2.2. 	PreReceive Collision-streaming (Inner Domain):	collision-streaming Types 4 (mOutletCollision) & 6 (mOutletWallCollision)	**/
				// 		Inner Domain:
				//		Limits of the outlet collision-streaming (i.e. Collision Types 4 & 6: mOutletCollision, mOutletWallCollision)
				// 		2.2.a. 	Collision Type 4: mOutletCollision
				site_t start_Index_Outlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2);
				site_t site_Count_Outlet_Inner = mLatDat->GetMidDomainCollisionCount(3);

				// 		2.2.b. 	Collision Type 6: mOutletWallCollision
				site_t start_Index_OutletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2)
                															+ mLatDat->GetMidDomainCollisionCount(3) + mLatDat->GetMidDomainCollisionCount(4);
      	site_t site_Count_OutletWall_Inner = mLatDat->GetMidDomainCollisionCount(5);
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------

				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				// Case 2.2.a. 	Collision Type 4: mOutletCollision
				//std::vector<site_t> Iolets_Outlet_Inner;
				n_LocalOutlets_mOutlet = 0; // number of local Outlets involved during the mOutletCollision collision- Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalOutlets_mOutlet = 0;
				if (site_Count_Outlet_Inner!=0){
					Iolets_Outlet_Inner = identify_Range_iolets_ID(start_Index_Outlet_Inner, (start_Index_Outlet_Inner + site_Count_Outlet_Inner), &n_LocalOutlets_mOutlet, &n_unique_LocalOutlets_mOutlet);

					Outlet_Inner.n_local_iolets = n_LocalOutlets_mOutlet;
					memcpy(&Outlet_Inner.Iolets_ID_range, &Iolets_Outlet_Inner[0], 3* n_LocalOutlets_mOutlet *sizeof(site_t));

					/*std::cout << "Rank: " << myPiD << " - Unique local Outlets: " << n_unique_LocalOutlets_mOutlet << " - Total local Outlets on current Rank (1st Round - mOutlet): " << n_LocalOutlets_mOutlet << " with Inlet ID:";
					for (int index = 0; index < n_LocalOutlets_mOutlet; index++ )
						std::cout << ' ' << Iolets_Outlet_Inner[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalOutlets_mOutlet *  sizeof(site_t);
					status = deviceMalloc((void**)&GPUDataAddr_Outlet_Inner, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Iolet: Outlet Inner failed...\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					status = deviceMemcpy(GPUDataAddr_Outlet_Inner, &Iolets_Outlet_Inner[0], MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Outlet Inner failed... \n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}
				}

				// Case 2.2.b. 	Collision Type 6: mOutletWallCollision
				//std::vector<site_t> Iolets_OutletWall_Inner;
				n_LocalOutlets_mOutletWall = 0; // number of local Outlets involved during the mOutletWallCollision collision - Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalOutlets_mOutletWall = 0;
				if (site_Count_OutletWall_Inner!=0){
					Iolets_OutletWall_Inner = identify_Range_iolets_ID(start_Index_OutletWall_Inner, (start_Index_OutletWall_Inner + site_Count_OutletWall_Inner), &n_LocalOutlets_mOutletWall, &n_unique_LocalOutlets_mOutletWall);

					OutletWall_Inner.n_local_iolets = n_LocalOutlets_mOutletWall;
					memcpy(&OutletWall_Inner.Iolets_ID_range, &Iolets_OutletWall_Inner[0], 3* n_LocalOutlets_mOutletWall *sizeof(site_t));

					/*std::cout << "Rank: " << myPiD << " - Unique local Outlets: " << n_unique_LocalOutlets_mOutletWall << " - Total local Outlets on current Rank (1st Round - mOutletWall): " << n_LocalOutlets_mOutletWall << " with Inlet ID:";
					for (int index = 0; index < n_LocalOutlets_mOutletWall; index++ )
						std::cout << ' ' << Iolets_OutletWall_Inner[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalOutlets_mOutletWall *  sizeof(site_t);
					status = deviceMalloc((void**)&GPUDataAddr_OutletWall_Inner, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Iolet: Outlet Wall Inner failed...\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					status = deviceMemcpy(GPUDataAddr_OutletWall_Inner, &Iolets_OutletWall_Inner[0], MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Outlet Wall Inner failed... \n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}
				}
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------


				printf("================================================================\n");
				printf("Rank: %d, n_Inlets_Inner: %d, n_InletsWall_Inner: %d, n_Inlets_Edge: %d, n_InletsWall_Edge: %d \n", myPiD, n_LocalInlets_mInlet, n_LocalInlets_mInletWall, n_LocalInlets_mInlet_Edge, n_LocalInlets_mInletWall_Edge);
				printf("Rank: %d, n_Outlets_Inner: %d, n_OutletsWall_Inner: %d, n_Outlets_Edge: %d, n_OutletsWall_Edge: %d \n", myPiD, n_LocalOutlets_mOutlet, n_LocalOutlets_mOutletWall, n_LocalOutlets_mOutlet_Edge, n_LocalOutlets_mOutletWall_Edge);
				printf("================================================================\n");

				//=============================================================================================================================================================


				//=============================================================================================================================================================
				// Examine the type of Iolets BCs:
				//----------------------------------------------------------------------
				// ***** Velocity BCs ***** Option: "LADDIOLET" - CMake file
				//	Possible options: a. parabolic, b. file, c. womersley
				// 		See SimConfig.cc

				/**
				 Possible approach:
					Allocate the memory on the GPU for the desired/specified velocity (wallMom[3*NUMVECTORS]) at each of the iolets with Velocity BCs
					Parameters needed:
						a. Number of fluid sites for each iolet (known iolet ID, ACTUALLY this is not needed - treat as a whole)
						b. Value for wallMom[3*NUMVECTORS] at each fluid site,
										which will be used to evaluate the correction to the bounced back post collision distr. function
									LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));

						TODO: 1. Is the values wallMom[3*NUMVECTORS] at each iolet fluid site time-dependent?
												If YES, then calculate these values and memcpy to GPU before calling the GPU collision-streaming Kernels
												If NO,	then calculate these values once in Initialise_GPU and memcpy to GPU.
									2. Flag to use to call the appropriate GPU collision-streaming kernel (Vel Vs Pressure BCs)
				*/
				//----------------------------------------------------------------------

				/*
				// Get the type of Iolet BCs from the CMake file compiling options
				std::string hemeIoletBC_Inlet, hemeIoletBC_Outlet;
				get_Iolet_BCs(hemeIoletBC_Inlet, hemeIoletBC_Outlet);
				*/
				// Check If I can get the type of Iolet BCs from the CMake file
				#define QUOTE_RAW(x) #x
				#define QUOTE_CONTENTS(x) QUOTE_RAW(x)

				std::string hemeIoletBC_Inlet, hemeIoletBC_Outlet;
				hemeIoletBC_Inlet = QUOTE_CONTENTS(HEMELB_INLET_BOUNDARY);
				//printf("Function Call: Inlet Type of BCs: hemeIoletBC_Inlet: %s \n", hemeLB_IoletBC_Inlet.c_str()); //note the use of c_str

				hemeIoletBC_Outlet = QUOTE_CONTENTS(HEMELB_OUTLET_BOUNDARY);
				//printf("Function Call: Outlet Type of BCs: hemeIoletBC_Outlet: %s \n\n", hemeLB_IoletBC_Outlet.c_str()); //note the use of c_str


				/*
				Oct 2022
				Change the Vel. BCs Implementation approach. Everything on the GPU...
					Boolean variable useWeightsFromFile when
						HEMELB_USE_VELOCITY_WEIGHTS_FILE is set to ON
				*/
				bool useWeightsFromFile = false;
				#ifdef HEMELB_USE_VELOCITY_WEIGHTS_FILE
        				useWeightsFromFile = true;
        #endif
				//printf("Boolean variable useWeightsFromFile: %d \n", useWeightsFromFile);

				//----------------------------------------------------------------------
				// Allocate memory on the GPU for each case:
				// a. Vel BCs - Wall momentum correction terms
				// b. Pres BCs - Ghost density
				//----------------------------------------------------------------------
				// ***** Pressure BCs ***** Option: "NASHZEROTHORDERPRESSUREIOLET" - CMake file
				// 	Set the Ghost Density if Inlet/Outlet BCs is set to NashZerothOrderPressure
				// 	Just allocate the memory as the ghostDensity can change as a function of time. MemCopies(host-to-device) before the gpu inlet/outlet collision kernels
				//----------------------------------------------------------------------

				/** New - Oct 2022:
				 For the case of Vel BCs and everything (coordinates, weights etc) on the GPU, we still need to allocate the memory on the GPU
				 except if we calculate the correction term on the GPU, but we don't need to save it.
				 	We do need to save it, as the correction term calculations will precede
					the corresponding GPU collision-streaming kernel

						Option 1: choose this one
											Needs saving (kernel at the begining of the PreSend step or some initial stage of the time-step)
											and then read the values at the iolets kernels
							Or
						Option 2: Calculate the correction at the same time of executing the kernels (some inline function)
				*/
				site_t MemSz; 		// Variable: Memory Size to be allocated

				// Inlets BCs
				if(hemeIoletBC_Inlet == "LADDIOLET"){
					//printf("INDEED the Inlet Type of BCs is: %s \n", hemeIoletBC_Inlet.c_str()); //note the use of c_str

					// Allocate memory on the GPU for the case of Inlet Velocity BCs
					// Wall Momentum correction term (one value for each LB direction (except LB_Dir=0) and lattice site at iolets)

					site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
					if(site_Count_Inlet_Edge!=0){
						// Correction term
						MemSz = site_Count_Inlet_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_correction_Inlet_Edge, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom - Inlet Edge failed\n");
							initialise_GPU_res = false;
							return initialise_GPU_res; //return false;
						}
					}

					site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
					if(site_Count_InletWall_Edge!=0){
						// Correction term
						MemSz = site_Count_InletWall_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_correction_InletWall_Edge, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom - InletWall Edge failed\n");
							initialise_GPU_res = false;
							return initialise_GPU_res; //return false;
						}
					}

					site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
					if(site_Count_Inlet_Inner!=0){
						// Correction term
						MemSz = site_Count_Inlet_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_correction_Inlet_Inner, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom - Inlet Inner failed\n");
							initialise_GPU_res = false;
							return initialise_GPU_res; //return false;
					}
				}

					site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
					if(site_Count_InletWall_Inner!=0){
						// Correction term
						MemSz = site_Count_InletWall_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_correction_InletWall_Inner, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom - InletWall Inner failed\n");
							initialise_GPU_res = false;
							return initialise_GPU_res; //return false;
					}
				}

				}
				else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
					// printf("INDEED the Inlet Type of BCs is: %s \n", hemeIoletBC_Inlet.c_str()); //note the use of c_str

					// Allocate memory on the GPU for the case of Inlet Pressure BCs
					// Ghost Density Inlet
					status = deviceMalloc((void**)&d_ghostDensity, n_Inlets * sizeof(distribn_t));
					if(!status){
						fprintf(stderr, "GPU memory allocation ghostDensity - Inlets failed\n");
						initialise_GPU_res = false;
						return initialise_GPU_res; 	//return false;
					}
				}
				//----------------------------------------------------------------------

				//----------------------------------------------------------------------
				// Outlets BCs
				if(hemeIoletBC_Outlet == "LADDIOLET"){
					//printf("INDEED the Outlet Type of BCs is: %s \n", hemeIoletBC_Outlet.c_str()); //note the use of c_str

					// Allocate memory on the GPU for the case of Outlet Velocity BCs
					site_t site_Count_Outlet_Edge = mLatDat->GetDomainEdgeCollisionCount(3);
					if(site_Count_Outlet_Edge!=0){
						// Correction term
						MemSz = site_Count_Outlet_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_correction_Outlet_Edge, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom - Outlet Edge failed\n");
							initialise_GPU_res = false;
							return initialise_GPU_res; //return false;
						}
					}

					site_t site_Count_OutletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(5);
					if(site_Count_OutletWall_Edge!=0){
						// Correction term
						MemSz = site_Count_OutletWall_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_correction_OutletWall_Edge, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom - OutletWall Edge failed\n");
							initialise_GPU_res = false;
							return initialise_GPU_res; //return false;
						}
					}

					site_t site_Count_Outlet_Inner = mLatDat->GetMidDomainCollisionCount(3);
					if(site_Count_Outlet_Inner!=0){
						// Correction term
						MemSz = site_Count_Outlet_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_correction_Outlet_Inner, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom - Outlet Inner failed\n");
							initialise_GPU_res = false;
							return initialise_GPU_res; //return false;
						}
					}

					site_t site_Count_OutletWall_Inner = mLatDat->GetMidDomainCollisionCount(5);
					if(site_Count_OutletWall_Inner!=0){
						// Correction term
						MemSz = site_Count_OutletWall_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_correction_OutletWall_Inner, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom - OutletWall Inner failed\n");
							initialise_GPU_res = false;
							return initialise_GPU_res; //return false;
						}
					}

				}
				else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
					//printf("INDEED the Outlet Type of BCs is: %s \n", hemeIoletBC_Outlet.c_str()); //note the use of c_str

					// Allocate memory on the GPU for the case of Outlet Pressure BCs
					// Ghost Density Outlet
					status = deviceMalloc((void**)&d_ghostDensity_out, n_Outlets * sizeof(distribn_t));
					if(!status){
						fprintf(stderr, "GPU memory allocation ghostDensity - Outlets failed\n");
						initialise_GPU_res = false;
						return initialise_GPU_res; //return false;
					}
				}
				//----------------------------------------------------------------------



				//----------------------------------------------------------------------
				// Normal vectors to Iolets
				// Inlets:
				float* h_inletNormal = new float[3*n_Inlets]; 	// x,y,z components

				for (int i=0; i<n_Inlets; i++){
				//for (int i=0; i<mInletValues->GetLocalIoletCount(); i++){
					//util::Vector3D<float> ioletNormal = mInletValues->GetLocalIolet(i)->GetNormal();
					util::Vector3D<float> ioletNormal = mInletValues->GetIolets()[i]->GetNormal();
						h_inletNormal[3*i] = ioletNormal.x;
						h_inletNormal[3*i+1] = ioletNormal.y;
						h_inletNormal[3*i+2] = ioletNormal.z;
						//std::cout << "Cout: ioletNormal.x : " <<  h_inletNormal[i] << " - ioletNormal.y : " <<  h_inletNormal[i+1] << " - ioletNormal.z : " <<  h_inletNormal[i+2] << std::endl;
					}

					status = deviceMalloc((void**)&d_inletNormal, 3*n_Inlets * sizeof(float));
					if(!status){
						fprintf(stderr, "GPU memory allocation inletNormal failed\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}
					// Memory copy from host (h_inletNormal) to Device (d_inletNormal)
					status = deviceMemcpy(d_inletNormal, h_inletNormal, 3*n_Inlets * sizeof(float), memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer (inletNormal) Host To Device failed\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
					}


				// Outlets:
				float* h_outletNormal = new float[3*n_Outlets]; 	// x,y,z components

				for (int i=0; i<n_Outlets; i++){
				//	for (int i=0; i<mOutletValues->GetLocalIoletCount(); i++){
						//util::Vector3D<float> ioletNormal = mOutletValues->GetLocalIolet(i)->GetNormal();
						util::Vector3D<float> ioletNormal = mOutletValues->GetIolets()[i]->GetNormal();
						h_outletNormal[3*i] = ioletNormal.x;
						h_outletNormal[3*i+1] = ioletNormal.y;
						h_outletNormal[3*i+2] = ioletNormal.z;
						//std::cout << "Cout: ioletNormal.x : " <<  h_outletNormal[3*i] << " - ioletNormal.y : " <<  h_outletNormal[3*i+1] << " - ioletNormal.z : " <<  h_outletNormal[3*i+2] << std::endl;
					}

					status = deviceMalloc((void**)&d_outletNormal, 3*n_Outlets * sizeof(float));
					if(!status){
						fprintf(stderr, "GPU memory allocation outletNormal failed\n");
						initialise_GPU_res = false;
						return initialise_GPU_res; //return false;
					}
					// Memory copy from host (h_outletNormal) to Device (d_outletNormal)
					status = deviceMemcpy(d_outletNormal, h_outletNormal, 3*n_Outlets * sizeof(float), memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer (outletNormal) Host To Device failed\n");
						initialise_GPU_res = false;
						return initialise_GPU_res; //return false;
					}

				//----------------------------------------------------------------------
				//***********************************************************************************************************************************


				//***********************************************************************************************************************************
				/** Vel. BCs case - Copy everything needed on the GPU:
				A. velocityTable for each iolet
						Done!!!

				B. Coordinates of fluid sites at each iolet
						Done!!!
						Not needed if we use the kernel GPU_WallMom_correction_File_Weights_NoSearch

				C. weights_table for all the points (coordinates) assign a certain weight (needs to read this from a file)
						Done!!!

				D. Information for the iolets:
					 	D.1. position of each iolet (is this the centre of the iolet?)
				 						from the input file (.xml): <position units="lattice" value="(12.9992,12.9992,3)"/>
						D.2. radius of each iolet
				 						from the input file (.xml): <radius value="0.999917" units="m"/>
						TODO: Check for which subtype this is useful?

				E. TODO: Generate a map container with the fluid ID and the index in the weights_table to get the
							appropriate weight.
							Generate 4 such maps for Inlets/Outlets (Inlet_Edge, InletWall_Edge, Inlet_Inner, InletWall_Inner, etc
								and pass the corresponding array to the GPU
							with the array shifted index by the starting Fluid ID for that corresponding type of collision-streaming
								i.e. arrays:
									index_weightTable_Inlet_Edge
									index_weightTable_InletWall_Edge
									index_weightTable_Inlet_Inner
									index_weightTable_InletWall_Inner

				*/

				// Inlets' BCs
				if(hemeIoletBC_Inlet == "LADDIOLET"){
					// printf("Rank: %d, Entering the LaddIolet loop for storing iolet coordinates on the GPU etc \n\n", myPiD);

					//----------------------------------------------------------------------
					// Start fluid ID index for each collision-streaming type
					site_t start_Index_Inlet_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
					site_t start_Index_InletWall_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
																							+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);

					site_t start_Index_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
					site_t start_Index_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
					//----------------------------------------------------------------------
					// SiteCount for each collision-streaming type
					site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
					site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
					site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
					site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
					//----------------------------------------------------------------------

					//====================================================================
					// A. velocityTable
					/** 	Only useful for the Case: b. File
					 			TODO:
								1. In the future include what is below in a function (with a virtual function in lb/iolets/InOutLetVelocity) that will be called
										only when appropriate, i.e.  depending what is in the input file	<condition type="velocity" subtype="file">
								2. Moreover, if there is periodicity in the values, save memory instead of allocating as
								 		Data_dbl_Inlet_velocityTable = new distribn_t[n_Inlets * (total_TimeSteps+1)];

									see void InOutLetFileVelocity::CalculateTable(LatticeTimeStep totalTimeSteps, PhysicalTime timeStepLength)
									From 	lb/iolets/InOutLetFile.cc (h) - Generates densityTable (std::vector<LatticeDensity> densityTable;)
										and
									lb/iolets/InOutLetFileVelocity.cc (h) - Generates velocityTable (std::vector<LatticeSpeed> velocityTable;)

	        	lb/iolets/InOutLetFileVelocity.cc generates the
							std::vector<LatticeSpeed> velocityTable;
	        	which is of size
	          	velocityTable.resize(totalTimeSteps + 1);

	        	just need to copy the table...
					*/

					// Call this only on MPI ranks with local iolets
					if(n_unique_LocalInlets_mInlet_Edge!=0 || n_unique_LocalInlets_mInletWall_Edge !=0
						 || n_unique_LocalInlets_mInlet !=0 || n_unique_LocalInlets_mInletWall !=0){

						// Get the total timeSteps of the simulation (size of velocityTable is totalTimeSteps +1, NOTE: for each iolet!!!)
						int total_TimeSteps = mState->GetTotalTimeSteps(); // Type: units.h:  typedef unsigned long LatticeTimeStep
						//double timeStepLength = mState->GetTimeStepLength();
						//printf("Total timeSteps of the simulation : %d and timeStepLength : %e \n", total_TimeSteps, timeStepLength);


						/** We need to access velocityTable which is a
								private member of class lb::iolets::InOutLetFileVelocity : public InOutLetVelocity

								It seems that there is one such table for each iolet, hence we need to have as many copies of these velocityTables
								- velocityTable is generated with the constructor of class BoundaryValues, which is taking place
										in SimulationMaster.cu:
										inletValues = new hemelb::lb::iolets::BoundaryValues(hemelb::geometry::INLET_TYPE, ...
										outletValues = new hemelb::lb::iolets::BoundaryValues(hemelb::geometry::OUTLET_TYPE, ...
								- Consider storing all the data in a single array in GPU global memory
						*/

						//
						// Just checking the number of inlets that the local MPI rank can "see" - These are the total inlets of the geometry AND NOT the total local inlets!!!
						// TODO: Reduce the memory requirements - Check whether there is a repeating pattern

						// std::cout << "Rank: " <<  myPiD <<  " - Number of inlets:" << n_Inlets << std::endl;
						Data_dbl_Inlet_velocityTable = new distribn_t[n_Inlets * (total_TimeSteps+1)];
						unsigned long long TotalMem_Inlet_velocityTable = n_Inlets * (total_TimeSteps+1) * sizeof(distribn_t);

						if(!Data_dbl_Inlet_velocityTable){
							std::cout << "Memory allocation error for velocityTable" << std::endl;
							initialise_GPU_res = false; return initialise_GPU_res;
						}

						// Read the data and copy in Data_dbl_Inlet_velocityTable
						// TODO: Consider running the loop ONLY over the number of local iolets
						//				That would be the sum of the following (NO!!! - This is Wrong as there can be overlap of the same iolets appearing in the 4 different types below)
						//						n_unique_LocalInlets_mInlet_Edge, n_unique_LocalInlets_mInletWall_Edge, n_unique_LocalInlets_mInlet, n_unique_LocalInlets_mInletWall
						//				or maybe just run the function identify_Range_iolets_ID with the range of fluid sites on the local rank
						for (int index_inlet = 0; index_inlet < n_Inlets; index_inlet++)
		        {
							//printf("Copying velocityTable for iolet ID: %d \n", index_inlet);
							//iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetLocalIolet(index_inlet));
							iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetIolets()[index_inlet]);

							if (iolet != nullptr){
								//----------------
								// Feb 2024 - IZ (case checkpointing) - Get the initial time of the simulation
								uint64_t t_start=mState->GetInitTimeStep();
								//
								//printf("From lb.hpp - Initial Time %d \n", t_start);
								//for (int timeStep=0; timeStep<total_TimeSteps+1; timeStep++)
								for (int timeStep=t_start; timeStep<(t_start + total_TimeSteps+1); timeStep++)
								{
									distribn_t velTemp = *(iolet->return_VelocityTable(timeStep)); 	// Read the values from velocityTable
									// Shift the velocity values according to the start time (t_start)
									Data_dbl_Inlet_velocityTable[index_inlet*(total_TimeSteps+1)+timeStep-t_start] = velTemp;
									//Initially: Data_dbl_Inlet_velocityTable[index_inlet*(total_TimeSteps+1)+timeStep] = velTemp;
								}
								//----------------
							}
						}

						status = deviceMalloc((void**)&(mLatDat->GPUDataAddr_Inlet_velocityTable),  TotalMem_Inlet_velocityTable);
						if(!status){
							fprintf(stderr, "GPU memory allocation for Velocity Table failed\n");
							initialise_GPU_res = false; return initialise_GPU_res;
						}

						// Memory copy from host (Data_dbl_Inlet_velocityTable) to Device (GPUDataAddr_Inlet_velocityTable)
						status = deviceMemcpy(mLatDat->GPUDataAddr_Inlet_velocityTable,
																		Data_dbl_Inlet_velocityTable,  TotalMem_Inlet_velocityTable, memcpyHostToDevice);
						if(!status){
							fprintf(stderr, "GPU memory transfer Host To Device for Velocity Table failed\n");
							initialise_GPU_res = false; return initialise_GPU_res;
						}

						delete[] Data_dbl_Inlet_velocityTable;
					}
					//====================================================================

#ifdef HEMELB_USE_VEL_WEIGHTS_ON_GPU
					//====================================================================
					/** B. Coordinates of fluid sites at each iolet Section
					 		Useful irrespective of the subtype - i.e. whether
								1. parabolic
          			2. From file
          			3. Womersley
							which is specified by the following line in the input file: <condition type="velocity" subtype="file">

							Function memCpy_HtD_GPUmem_Coords_Iolets
							***There is a bug in the function - Does not save the coordinates on the GPU global memory successfully ***
							*** TODO: Fix the error ***
							*** Manually doing what the function was supposed to be doing ***
								Performs the memcpy HtD for the coordinates of the iolets points
									Format (int64_t), x,y,z coordinates in consecutive manner
										for the fluid sites in the range Fluid index:[firstIndex, (firstIndex + siteCount) )
								The data on the GPU global memory reside in memory pointed by the following:
										void *GPUDataAddr_Coords_Inlet_Edge;
										void *GPUDataAddr_Coords_InletWall_Edge;
										void *GPUDataAddr_Coords_Inlet_Inner;
										void *GPUDataAddr_Coords_InletWall_Inner;
										void *GPUDataAddr_Coords_Outlet_Edge;
										void *GPUDataAddr_Coords_OutletWall_Edge;
										void *GPUDataAddr_Coords_Outlet_Inner;
										void *GPUDataAddr_Coords_OutletWall_Inner;

								with shifted indices:
									i.e. int64_t shifted_Fluid_Ind = siteIndex - firstIndex;

								and keeping in consecutive order the x,y,z coords
									i.e.
									Data_int64_Coords_iolets[shifted_Fluid_Ind*3] = x_coord;
									Data_int64_Coords_iolets[shifted_Fluid_Ind*3+1] = y_coord;
									Data_int64_Coords_iolets[shifted_Fluid_Ind*3+2] = z_coord;
					*/

					//--------------------------------------------------------------------
					// Domain Edge
					// Collision Type 3 (mInletCollision):
					//site_t start_Index_Inlet_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
					//site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
					if (site_Count_Inlet_Edge!=0){
						//memCpy_HtD_GPUmem_Coords_Iolets(start_Index_Inlet_Edge, site_Count_Inlet_Edge, GPUDataAddr_Coords_Inlet_Edge);

						//==================================================================
						// Test - inlining the above function...
						// Remove later
						site_t nArr_Coords_iolets = 3 * site_Count_Inlet_Edge; // Number of elements (the fluid index can be used to get the x,y,z Coords) of type int64_t (units.h:  typedef int64_t site_t;)
			  		site_t* Data_int64_Coords_iolets = new site_t[nArr_Coords_iolets];	// site_t (type int64_t)

						// Allocate memory on the GPU (global memory)
						site_t MemSz = nArr_Coords_iolets *  sizeof(int64_t); 	// site_t (int64_t) Check that will remain like this in the future
						status = deviceMalloc((void**)&GPUDataAddr_Coords_Inlet_Edge, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation - Coords for Iolets (Inlet_Edge) EXPANDED version - failed...\n");
						}
						/*else{ printf("GPU memory allocation - Coords for Iolets (Inlet_Edge) EXPANDED version - Bytes: %lld - SUCCESS from Rank: %d \n", MemSz, myPiD);
					}*/

						site_t firstIndex = start_Index_Inlet_Edge;
						site_t siteCount = site_Count_Inlet_Edge;

						#pragma omp parallel for
						for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
			  		{
							// Save the coords to Data_int64_Coords_iolets
							int64_t shifted_Fluid_Ind = siteIndex - firstIndex;
							geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

							int64_t x_coord = site.GetGlobalSiteCoords().x;
							int64_t y_coord = site.GetGlobalSiteCoords().y;
							int64_t z_coord = site.GetGlobalSiteCoords().z;

							Data_int64_Coords_iolets[shifted_Fluid_Ind*3] = x_coord;
							Data_int64_Coords_iolets[shifted_Fluid_Ind*3+1] = y_coord;
							Data_int64_Coords_iolets[shifted_Fluid_Ind*3+2] = z_coord;
						}
						// Perform a HtD memcpy (from Data_int64_Coords_iolets to GPUDataAddr_Coords_iolets)
			 			status = deviceMemcpy(GPUDataAddr_Coords_Inlet_Edge,
				 													&Data_int64_Coords_iolets[0], MemSz, memcpyHostToDevice);

						if(!status){
		     			const char * eStr = deviceGetErrorString();
		     			printf("GPU memory copy for IOLETS (Inlet_Edge) coordinates failed with error: \"%s\" at proc# %i - SiteCount: %lld \n", eStr, myPiD, siteCount);
							initialise_GPU_res = false; return initialise_GPU_res;
		   			}
			 			//cudaDeviceSynchronize(); // Does not need this actually as the CPU blocks until memory is copied
			 			delete[] Data_int64_Coords_iolets;
						//==================================================================
						/*// Test reading the coordinates
						int64_t lower_Fluid_index = start_Index_Inlet_Edge;
						int64_t max_Fluid_index = lower_Fluid_index + site_Count_Inlet_Edge -1; // Value included
						hemelb::GPU_Check_Coordinates<<<1,1>>>(
																(int64_t*)GPUDataAddr_Coords_Inlet_Edge,
																start_Index_Inlet_Edge,
																lower_Fluid_index, max_Fluid_index
															);
					 	*/
						//==================================================================
					}
					//--------------------------------

					// Collision Type 5 (mInletWallCollision):
					//site_t start_Index_InletWall_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
					//                                    + mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);
					//site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
					if (site_Count_InletWall_Edge!=0){
							//memCpy_HtD_GPUmem_Coords_Iolets(start_Index_InletWall_Edge, site_Count_InletWall_Edge, GPUDataAddr_Coords_InletWall_Edge);

							//==================================================================
							// Test manually making the work of the above function...
							// Remove later
							site_t nArr_Coords_iolets = 3 * site_Count_InletWall_Edge; // Number of elements (the fluid index can be used to get the x,y,z Coords) of type int64_t (units.h:  typedef int64_t site_t;)
				  		site_t* Data_int64_Coords_iolets = new site_t[nArr_Coords_iolets];	// site_t (type int64_t)

							// Allocate memory on the GPU (global memory)
							site_t MemSz = nArr_Coords_iolets *  sizeof(int64_t); 	// site_t (int64_t) Check that will remain like this in the future
							status = deviceMalloc((void**)&GPUDataAddr_Coords_InletWall_Edge, MemSz);
							if(!status){
								fprintf(stderr, "GPU memory allocation - Coords for Iolets (InletWall_Edge) EXPANDED version - failed...\n");
							}
							/*else{ printf("GPU memory allocation - Coords for Iolets (InletWall_Edge) EXPANDED version - Bytes: %lld - SUCCESS from Rank: %d \n", MemSz, myPiD);
							}*/

							site_t firstIndex = start_Index_InletWall_Edge;
							site_t siteCount = site_Count_InletWall_Edge;

							#pragma omp parallel for
							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
				  		{
								// Save the coords to Data_int64_Coords_iolets
								int64_t shifted_Fluid_Ind = siteIndex - firstIndex;
								geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

								int64_t x_coord = site.GetGlobalSiteCoords().x;
								int64_t y_coord = site.GetGlobalSiteCoords().y;
								int64_t z_coord = site.GetGlobalSiteCoords().z;

								Data_int64_Coords_iolets[shifted_Fluid_Ind*3] = x_coord;
								Data_int64_Coords_iolets[shifted_Fluid_Ind*3+1] = y_coord;
								Data_int64_Coords_iolets[shifted_Fluid_Ind*3+2] = z_coord;
							}
							// Perform a HtD memcpy (from Data_int64_Coords_iolets to GPUDataAddr_Coords_iolets)
				 			status = deviceMemcpy(GPUDataAddr_Coords_InletWall_Edge,
					 													&Data_int64_Coords_iolets[0], MemSz, memcpyHostToDevice);

							if(!status){
			     			const char * eStr = deviceGetErrorString();
			     			printf("GPU memory copy for IOLETS (InletWall_Edge) coordinates failed with error: \"%s\" at proc# %i - SiteCount: %lld \n", eStr, myPiD, siteCount);
								initialise_GPU_res = false; return initialise_GPU_res;
			   			}
				 			//cudaDeviceSynchronize(); // Does not need this actually as the CPU blocks until memory is copied
				 			delete[] Data_int64_Coords_iolets;
							//==================================================================
							/*// Test reading the coordinates
							int64_t lower_Fluid_index = start_Index_InletWall_Edge;
							int64_t max_Fluid_index = lower_Fluid_index + site_Count_InletWall_Edge -1; // Value included
							hemelb::GPU_Check_Coordinates<<<1,1>>>(
																	(int64_t*)GPUDataAddr_Coords_InletWall_Edge,
																	start_Index_InletWall_Edge,
																	lower_Fluid_index, max_Fluid_index
																);*/
							//==================================================================

					}
					//--------------------------------
					// Ends the Domain edge
					//--------------------------------------------------------------------

					//--------------------------------------------------------------------
					// Inner domain
					// Collision Type 3 (mInletCollision):
					//site_t start_Index_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
					//site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
					if (site_Count_Inlet_Inner!=0){
						//memCpy_HtD_GPUmem_Coords_Iolets(start_Index_Inlet_Inner, site_Count_Inlet_Inner, GPUDataAddr_Coords_Inlet_Inner);

						//==================================================================
						// Test - manually making the work of the above function...
						// Remove later
						site_t nArr_Coords_iolets = 3 * site_Count_Inlet_Inner; // Number of elements (the fluid index can be used to get the x,y,z Coords) of type int64_t (units.h:  typedef int64_t site_t;)
			  		site_t* Data_int64_Coords_iolets = new site_t[nArr_Coords_iolets];	// site_t (type int64_t)

						// Allocate memory on the GPU (global memory)
						site_t MemSz = nArr_Coords_iolets *  sizeof(int64_t); 	// site_t (int64_t) Check that will remain like this in the future
						status = deviceMalloc((void**)&GPUDataAddr_Coords_Inlet_Inner, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation - Coords for Iolets EXPANDED version - failed...\n");
						}
						/*else{ printf("GPU memory allocation - Coords for Iolets EXPANDED version - Bytes: %lld - SUCCESS from Rank: %d \n", MemSz, myPiD);
						}*/

						site_t firstIndex = start_Index_Inlet_Inner;
						site_t siteCount = site_Count_Inlet_Inner;

						#pragma omp parallel for
						for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
			  		{
							// Save the coords to Data_int64_Coords_iolets
							int64_t shifted_Fluid_Ind = siteIndex - firstIndex;
							geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

							int64_t x_coord = site.GetGlobalSiteCoords().x;
							int64_t y_coord = site.GetGlobalSiteCoords().y;
							int64_t z_coord = site.GetGlobalSiteCoords().z;

							Data_int64_Coords_iolets[shifted_Fluid_Ind*3] = x_coord;
							Data_int64_Coords_iolets[shifted_Fluid_Ind*3+1] = y_coord;
							Data_int64_Coords_iolets[shifted_Fluid_Ind*3+2] = z_coord;
						}
						// Perform a HtD memcpy (from Data_int64_Coords_iolets to GPUDataAddr_Coords_iolets)
			 			status = deviceMemcpy(GPUDataAddr_Coords_Inlet_Inner,
				 													&Data_int64_Coords_iolets[0], MemSz, memcpyHostToDevice);

						if(!status){
		     			const char * eStr = deviceGetErrorString();
		     			printf("GPU memory copy for IOLETS (Inlet_Inner) coordinates failed with error: \"%s\" at proc# %i - SiteCount: %lld \n", eStr, myPiD, siteCount);
							initialise_GPU_res = false; return initialise_GPU_res;
		   			}
			 			//cudaDeviceSynchronize(); // Does not need this actually as the CPU blocks until memory is copied
			 			delete[] Data_int64_Coords_iolets;
						//==================================================================
						/*
						// Test reading the coordinates
						int64_t lower_Fluid_index = start_Index_Inlet_Inner;
						int64_t max_Fluid_index = lower_Fluid_index + site_Count_Inlet_Inner -1; // Value included
						hemelb::GPU_Check_Coordinates<<<1,1>>>(
																(int64_t*)GPUDataAddr_Coords_Inlet_Inner,
																start_Index_Inlet_Inner,
																lower_Fluid_index, max_Fluid_index
															);
						*/
						//==================================================================
					}
					//--------------------------------

					// Collision Type 5 (mInletWallCollision):
					//site_t start_Index_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
					//site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
					if (site_Count_InletWall_Inner!=0){
						//memCpy_HtD_GPUmem_Coords_Iolets(start_Index_InletWall_Inner, site_Count_InletWall_Inner, &GPUDataAddr_Coords_InletWall_Inner);

						//==================================================================
						// Test manually making the work of the above function...
						// Remove later
						site_t nArr_Coords_iolets = 3 * site_Count_InletWall_Inner; // Number of elements (the fluid index can be used to get the x,y,z Coords) of type int64_t (units.h:  typedef int64_t site_t;)
						site_t* Data_int64_Coords_iolets = new site_t[nArr_Coords_iolets];	// site_t (type int64_t)

						// Allocate memory on the GPU (global memory)
						MemSz = nArr_Coords_iolets *  sizeof(int64_t); 	// site_t (int64_t) Check that will remain like this in the future
						status = deviceMalloc((void**)&GPUDataAddr_Coords_InletWall_Inner, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation - Coords for Iolets InletWall Inner EXPANDED version - failed...\n");
						}
						/*else{ printf("GPU memory allocation - Coords for Iolets InletWall Inner EXPANDED version - Bytes: %lld - SUCCESS from Rank: %d \n", MemSz, myPiD);
						}*/

						site_t firstIndex = start_Index_InletWall_Inner;
						site_t siteCount = site_Count_InletWall_Inner;

						#pragma omp parallel for
						for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
						{
							// Save the coords to Data_int64_Coords_iolets
							int64_t shifted_Fluid_Ind = siteIndex - firstIndex;
							geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

							int64_t x_coord = site.GetGlobalSiteCoords().x;
							int64_t y_coord = site.GetGlobalSiteCoords().y;
							int64_t z_coord = site.GetGlobalSiteCoords().z;

							Data_int64_Coords_iolets[shifted_Fluid_Ind*3] = x_coord;
							Data_int64_Coords_iolets[shifted_Fluid_Ind*3+1] = y_coord;
							Data_int64_Coords_iolets[shifted_Fluid_Ind*3+2] = z_coord;
						}
						// Perform a HtD memcpy (from Data_int64_Coords_iolets to GPUDataAddr_Coords_iolets)
						status = deviceMemcpy(GPUDataAddr_Coords_InletWall_Inner,
																	&Data_int64_Coords_iolets[0], MemSz, memcpyHostToDevice);
						if(!status){
							const char * eStr = deviceGetErrorString();
							printf("GPU memory copy for IOLETS (Inlet_Inner) coordinates failed with error: \"%s\" at proc# %i - SiteCount: %lld \n", eStr, myPiD, siteCount);
							initialise_GPU_res = false; return initialise_GPU_res;
						}
						//cudaDeviceSynchronize(); // Does not need this actually as the CPU blocks until memory is copied
						delete[] Data_int64_Coords_iolets;
						//==================================================================
						/*
						// Test reading the coordinates
						int64_t lower_Fluid_index = start_Index_InletWall_Inner;
						int64_t max_Fluid_index = lower_Fluid_index + site_Count_InletWall_Inner -1; // Value included
						hemelb::GPU_Check_Coordinates<<<1,1>>>(
																(int64_t*)GPUDataAddr_Coords_InletWall_Inner,
																start_Index_InletWall_Inner,
																lower_Fluid_index, max_Fluid_index
															);
						*/
						//==================================================================
					}
					//====================================================================

					//====================================================================
					/** C. weights_table - only used for the case of Vel. BCs - File (subtype)
									private member of class InOutLetFileVelocity : public InOutLetVelocity
           						std::map<std::vector<int>, double> weights_table;
							Questions:
							C.1. How mamy weights_table there are?
											1 for each iolet (inlet/outlet) (YES!!!) or

							NOTE:
							Pointer to pointer, see
							https://stackoverflow.com/questions/26111794/how-to-use-pointer-to-pointer-in-cuda
					*/

					// Create an array of pointers for the coordinates and the weights
					int64_t *Data_int_Inlet_weightsTable_coord[n_Inlets];
					distribn_t *Data_dbl_Inlet_weightsTable_wei[n_Inlets];
					arr_elementsInEachInlet=new int[n_Inlets];

					if (useWeightsFromFile) {

						// Ensure that this is called only on the MPI ranks with local iolets
						if(n_unique_LocalInlets_mInlet_Edge!=0 || n_unique_LocalInlets_mInletWall_Edge !=0
							 || n_unique_LocalInlets_mInlet !=0 || n_unique_LocalInlets_mInletWall !=0){
						//------------------------------------------------------------------
						// 1. Read first the coordinates and the appropriate weight from the weights_table files
						// 		Note: We are doing a loop here over all the inlets of the whole domain
						//						NOT over the local inlets on this MPI rank
						// 					n_Inlets: total inlets on whole domain
						//					n_unique_LocalInlets... is the appropriate variable if we consider just the local inlets
						for (int index_inlet = 0; index_inlet < n_Inlets; index_inlet++)
		        {
							//printf("Copying velocityTable for iolet ID: %d \n", index_inlet);
							//iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetLocalIolet(index_inlet));
							iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetIolets()[index_inlet]);

							std::string get_velocityFilePath = iolet->GetFilePath(); // returns velocityFilePath
							//std::cout << "Velocity File Path: " << get_velocityFilePath << std::endl;

							const std::string in_name = get_velocityFilePath + ".weights.txt";
							util::check_file(in_name.c_str());
							//std::cout << "Velocity weights File Path: " << in_name << std::endl;

							//-----------------------------------------------------
							// Read the weights table from the file provided
							// load and read file
							std::fstream myfile;
							myfile.open(in_name.c_str(), std::ios_base::in);

							// log::Logger::Log<log::Warning, log::OnePerCore>(" ----> loading weights file: %s",in_name.c_str());

							std::string input_line;

							std::map<std::vector<int>, double> weights_table;
							/* input files are in ASCII, in format:
							 * coord_x coord_y coord_z weights_value */

							// Assuming that each inlet has a different weights table with a different number of elements
							int count_lines_weights_table=0;
							while (myfile.good())
							{
								// TODO: Consider this should be of type int64_t?
								// 				These values are passed later as int64_t for the GPU global memory
								int x, y, z;
								double v;
								myfile >> x >> y >> z >> v;

								std::vector<int> xyz;
								xyz.push_back(x);
								xyz.push_back(y);
								xyz.push_back(z);
								weights_table[xyz] = v;

								count_lines_weights_table++;
								//std::cout << "Coordinates: " << x << ", " << y << ", " << z << " Weight: " << v << std::endl;
								/*
								log::Logger::Log<log::Trace, log::OnePerCore>("%lld %lld %lld %f",
										x,
										y,
										z,
										weights_table[xyz]);*/
							}
							myfile.close();

							arr_elementsInEachInlet[index_inlet] = count_lines_weights_table -1; // -1 as it goes one more time into the while loop (while (myfile.good()))
							//printf("Rank: %d, nElements (lines) : %d read from file weights_table at inlet ID : %d \n", myPiD, arr_elementsInEachInlet[index_inlet], index_inlet);

							// Memory allocation for each array with the coordinates of the inlet points in the weights_table
							Data_int_Inlet_weightsTable_coord[index_inlet] = new int64_t[3 * arr_elementsInEachInlet[index_inlet]]; // to keep all 3 components

							Data_dbl_Inlet_weightsTable_wei[index_inlet] = new distribn_t[arr_elementsInEachInlet[index_inlet]];

							// Copy the data in arrays:
							// Data_int_Inlet_weightsTable_coord[index_inlet] and
							// Data_dbl_Inlet_weightsTable_wei[index_inlet]
							std::map<std::vector<int>, double>::iterator itr;
							for (itr = weights_table.begin(); itr != weights_table.end(); ++itr) {
								int current_pos = std::distance(weights_table.begin(),itr);

								// TODO:
								//		Consider which data-layout is better (enable easier search later in the GPU kernel for calculating correction term)
								/*// Data-layout option 1.1. x,y,z for each map-key together
								Data_int_Inlet_weightsTable_coord[index_inlet][current_pos*3]= (int64_t)itr->first[0];
								Data_int_Inlet_weightsTable_coord[index_inlet][current_pos*3+1]= (int64_t)itr->first[1];
								Data_int_Inlet_weightsTable_coord[index_inlet][current_pos*3+2]= (int64_t)itr->first[2];
								*/
								// Data-layout option 1.2. x,y,z for each map-key separate
								// 	first the x-coordinates [0 to (count_lines_weights_table-2)],
								//	then the y-ccord [0 to (count_lines_weights_table-2)] and z-coord [0 to (count_lines_weights_table-2)] for all points in the table
								int n_count = arr_elementsInEachInlet[index_inlet];
								Data_int_Inlet_weightsTable_coord[index_inlet][current_pos]= (int64_t)itr->first[0];
								Data_int_Inlet_weightsTable_coord[index_inlet][1*n_count + current_pos]= (int64_t)itr->first[1];
								Data_int_Inlet_weightsTable_coord[index_inlet][2*n_count + current_pos]= (int64_t)itr->first[2];

								Data_dbl_Inlet_weightsTable_wei[index_inlet][current_pos]= itr->second;
	    	 			} // Ends copying the data in the 2 arrays

						} 	// Ends the loop over the inlets
						//------------------------------------------------------------------

						//------------------------------------------------------------------
						// Pointer to (n_Inlets) number of pointers
						// printf("Rank = %d - Number of inlets = %d \n\n", myPiD, n_Inlets ); // correct - n_Inlets=1
						status = deviceMalloc((void**)&GPUDataAddr_pp_Inlet_weightsTable_coord, sizeof(int64_t*) * n_Inlets);   		// points to n_Inlets pointers
						status = deviceMalloc((void**)&GPUDataAddr_pp_Inlet_weightsTable_wei, sizeof(distribn_t*) * n_Inlets);	// points to n_Inlets pointers

						// 2. Move the data to the GPU - Loop over each one of the inlets
						for (int index_inlet = 0; index_inlet < n_Inlets; index_inlet++)
		        {
							int TotalMem_Inlet_weightsTable_coord = 3 * arr_elementsInEachInlet[index_inlet] *sizeof(int64_t);
							int TotalMem_Inlet_weightsTable_wei = arr_elementsInEachInlet[index_inlet] * sizeof(distribn_t);

							//----------------------------------------------------------------
							// a. Coordinates of the points in the weights_table as in Data_int_Inlet_weightsTable_coord
							// Allocates memory on the device
							status = deviceMalloc((void**)&GPUDataAddr_p_Inlet_weightsTable_coord, 3 * arr_elementsInEachInlet[index_inlet] * sizeof(int64_t));
							if(!status){
								fprintf(stderr, "GPU memory allocation for coordinates in the weights_table failed...\n");
								initialise_GPU_res = false; return initialise_GPU_res;
							}

							// Fill the values
							// Note:
							//	1) &GPUDataAddr_p_Inlet_weightsTable_coord is a HOST pointer (contains the data of the coordinates)
							//	2) If such arrays of addresses need to be passed to a kernel, they can be initialized in the
							//			pinned memory which the device can access directly.
							//------------------------
							//printf("Rank: %d, nElements (lines) : %d After reading from file weights_table at inlet ID : %d \n", myPiD, arr_elementsInEachInlet[index_inlet], index_inlet);

							deviceMemcpy(GPUDataAddr_p_Inlet_weightsTable_coord,  // destination Device
				                 Data_int_Inlet_weightsTable_coord[index_inlet],     // source Host!!! &GPUDataAddr_p_Inlet_weightsTable_coord is a host pointer
				                 3 * arr_elementsInEachInlet[index_inlet] * sizeof(int64_t), memcpyHostToDevice);
							//------------------------

							// &GPUDataAddr_p_Inlet_weightsTable_coord is a host pointer
							status = deviceMemcpy(&GPUDataAddr_pp_Inlet_weightsTable_coord[index_inlet],  // destination Device
				                  						&GPUDataAddr_p_Inlet_weightsTable_coord,     // source Host!!! &GPUDataAddr_p_Inlet_weightsTable_coord is a host pointer
				                  						sizeof(int64_t*), memcpyHostToDevice);
							if(!status){
								fprintf(stderr, "GPU memory copy H2D for coordinates in the weights_table failed...\n");
								initialise_GPU_res = false; return initialise_GPU_res;
							}
							//----------------------------------------------------------------

							//----------------------------------------------------------------
							// b. weights in the weights_table as in Data_dbl_Inlet_weightsTable_wei
							deviceMalloc(&GPUDataAddr_p_Inlet_weightsTable_wei, sizeof(distribn_t) * arr_elementsInEachInlet[index_inlet]);
							// &GPUDataAddr_p_Inlet_weightsTable_wei is a host pointer - points to Data_dbl_Inlet_weightsTable_wei[index_inlet]

							deviceMemcpy(GPUDataAddr_p_Inlet_weightsTable_wei,  // destination Device
				                 Data_dbl_Inlet_weightsTable_wei[index_inlet],     // source Host!!! &GPUDataAddr_p_Inlet_weightsTable_coord is a host pointer
				                 arr_elementsInEachInlet[index_inlet] * sizeof(distribn_t), memcpyHostToDevice);

							deviceMemcpy(&GPUDataAddr_pp_Inlet_weightsTable_wei[index_inlet],  	// destination Device
				                  &GPUDataAddr_p_Inlet_weightsTable_wei,     						// source Host
				                  sizeof(distribn_t*), memcpyHostToDevice);
						//------------------------------------------------------------------

							// After the MemCopy to the GPU has completed - delete/free memory
							// TODO: Check whether to keep
							//delete[] Data_int_Inlet_weightsTable_coord[index_inlet];
							//delete[] Data_dbl_Inlet_weightsTable_wei[index_inlet];

						} // Ends the loop over the inlets
						//------------------------------------------------------------------

						} // Ends the loop if there are any iolets in the current MPI RANK


						//==================================================================
						/**
						E. Generate 4 vectors for Inlets/Outlets (Inlet_Edge, InletWall_Edge, Inlet_Inner, InletWall_Inner, etc
										and pass the corresponding info to the GPU (index of the weight)
										So that we do not have to do the search on the GPU for the appropriate weight to be used (depending on the local coordinates)

										Maybe in the future consider using thrust vectors...
										see https://stackoverflow.com/questions/11113485/how-to-cast-thrustdevice-vectorint-to-raw-pointer

									with the array shifted index by the starting Fluid ID for that corresponding type of collision-streaming
										i.e. std::vector<int64_t> index_weightTable:
											index_weightTable_Inlet_Edge
											index_weightTable_InletWall_Edge
											index_weightTable_Inlet_Inner
											index_weightTable_InletWall_Inner

									21 April 2023
									BUG in the above approach
									New approach - Save directly the vel. weight instead of the index in the weight table.
									It will reduce the memory requirements on the GPU as well (NO need to save index and weights tables - Just the Vel. weight)
									So, now we will have the following:
									 	weightTable_Inlet_Edge
										weightTable_InletWall_Edge
										weightTable_Inlet_Inner
										weightTable_InletWall_Inner

									We generate the following: a.Indices and b. vel weights for each site and LB_dir
										e.g. 	GPUDataAddr_index_weightTable_InletWall_Inner
													GPUDataAddr_weightTable_InletWall_Inner

									Still the same error/bug appears...

									25 April 2023
									Reconsider the above approach..
									use a wall momentum prefactor correction term.
						*/
						//------------------------------------------------------------------
						// 4 January 2023 - Completed!
						//	E.1. Inlet_Edge
						// 		Info (map key index) to be placed in: index_weightTable_Inlet_Edge
						if (site_Count_Inlet_Edge!=0){

							site_t firstIndex = start_Index_Inlet_Edge;
							site_t siteCount = site_Count_Inlet_Edge;

							std::vector<int64_t> index_weightTable;
							std::vector<distribn_t> vel_weightTable;

							int n_elements_reserved=siteCount*(LatticeType::NUMVECTORS-1);
							index_weightTable.reserve(n_elements_reserved);
							vel_weightTable.reserve(n_elements_reserved);

							// Loop to get the index of the appropriate weight
							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								// Save the coords to Data_int64_Coords_iolets
								int64_t shifted_Fluid_Ind = siteIndex - firstIndex;

								geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

								int boundaryId = site.GetIoletId();
            		//iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetLocalIolet(boundaryId));
								iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetIolets()[boundaryId]);

            		LatticePosition sitePos(site.GetGlobalSiteCoords());

								// No need to include the LB_Dir=0
								for (int LB_Dir = 1; LB_Dir < LatticeType::NUMVECTORS; LB_Dir++) {

									if (site.HasIolet(LB_Dir)){
										LatticePosition halfWay(sitePos);
				            halfWay.x += 0.5 * LatticeType::CX[LB_Dir];
				            halfWay.y += 0.5 * LatticeType::CY[LB_Dir];
				            halfWay.z += 0.5 * LatticeType::CZ[LB_Dir];

										//int index_weight = *(iolet->return_index_weight_VelocityTable(halfWay));
										int index_weightTable_received;
										distribn_t vel_weight_received;
										iolet->return_index_weight_VelocityTable(halfWay, &index_weightTable_received, &vel_weight_received); // Note that any dependency on the LB_Dir is absorbed in halfWay

										//if (siteIndex>=10025){
										//	printf("Fluid ID: %d LB-Dir: %d index: %d \n", siteIndex, LB_Dir, index_weightTable_received);
										//}
										index_weightTable.push_back(index_weightTable_received);
										vel_weightTable.push_back(vel_weight_received);
									}
									else{
										int index_weightTable_received = INT_MAX-1; // Should never access this value in the GPU kernel as it refers to noIolet links. Used for debugging purposes - checking for errors
										distribn_t vel_weight_received = 0.0;
										index_weightTable.push_back(index_weightTable_received);
										vel_weightTable.push_back(vel_weight_received);
									}

								}

							} // Ends the loop over the fluid sites
							index_weightTable_Inlet_Edge = index_weightTable;
							weightTable_Inlet_Edge = vel_weightTable;

							// Allocate memory on the GPU
							bool status_1, status_2, status_3, status_4;
							status_1 = deviceMalloc((void**)&GPUDataAddr_index_weightTable_Inlet_Edge, n_elements_reserved * sizeof(int64_t));
							status_2 = deviceMalloc((void**)&GPUDataAddr_weightTable_Inlet_Edge, n_elements_reserved * sizeof(distribn_t));

							// Host-to-Device memory copy
							status_3 = deviceMemcpy(GPUDataAddr_index_weightTable_Inlet_Edge,
																	&index_weightTable[0], n_elements_reserved * sizeof(int64_t), memcpyHostToDevice);
							status_4 = deviceMemcpy(GPUDataAddr_weightTable_Inlet_Edge,
																											&vel_weightTable[0], n_elements_reserved * sizeof(distribn_t), memcpyHostToDevice);

							if(status_1 != cudaSuccess || status_2 != cudaSuccess || status_3 != cudaSuccess || status_4 != cudaSuccess ){
								fprintf(stderr, "GPU memory allocation and transfer Host To Device - key value indices and/or WeightsTable failed\n");
								initialise_GPU_res = false;
								return initialise_GPU_res;
							}

						} // Ends the if (site_Count_Inlet_Edge!=0)

						//------------------------------------------------------------------
						// E.2. InletWall_Edge
						// 		Info (map key index) to be placed in: index_weightTable_InletWall_Edge
						if (site_Count_InletWall_Edge!=0){

							site_t firstIndex = start_Index_InletWall_Edge;
							site_t siteCount = site_Count_InletWall_Edge;

							std::vector<int64_t> index_weightTable;
							std::vector<distribn_t> vel_weightTable;

							int n_elements_reserved=siteCount*(LatticeType::NUMVECTORS-1);
							index_weightTable.reserve(n_elements_reserved);
							vel_weightTable.reserve(n_elements_reserved);

							// Loop to get the index of the appropriate weight
							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								// Save the coords to Data_int64_Coords_iolets
								int64_t shifted_Fluid_Ind = siteIndex - firstIndex;

								geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

								int boundaryId = site.GetIoletId();
								//iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetLocalIolet(boundaryId));
								iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetIolets()[boundaryId]);

								LatticePosition sitePos(site.GetGlobalSiteCoords());

								// No need to include the LB_Dir=0
								for (int LB_Dir = 1; LB_Dir < LatticeType::NUMVECTORS; LB_Dir++) {
									//
									if (site.HasIolet(LB_Dir)){
										LatticePosition halfWay(sitePos);
				            halfWay.x += 0.5 * LatticeType::CX[LB_Dir];
				            halfWay.y += 0.5 * LatticeType::CY[LB_Dir];
				            halfWay.z += 0.5 * LatticeType::CZ[LB_Dir];

										//int index_weight = *(iolet->return_index_weight_VelocityTable(halfWay));
										int index_weightTable_received;
										distribn_t vel_weight_received;
										iolet->return_index_weight_VelocityTable(halfWay, &index_weightTable_received, &vel_weight_received); // Note that any dependency on the LB_Dir is absorbed in halfWay

										//if (siteIndex>=10025){
										//	printf("Fluid ID: %d LB-Dir: %d index: %d \n", siteIndex, LB_Dir, index_weightTable_received);
										//}
										index_weightTable.push_back(index_weightTable_received);
										vel_weightTable.push_back(vel_weight_received);
									}
									else{
										int index_weightTable_received = INT_MAX-1; // Should never access this value in the GPU kernel. Used for checking errors
										distribn_t vel_weight_received = 0.0;
										index_weightTable.push_back(index_weightTable_received);
										vel_weightTable.push_back(vel_weight_received);
									}
									//
								}
							} // Ends the loop over the fluid sites
							index_weightTable_InletWall_Edge=index_weightTable;
							weightTable_InletWall_Edge = vel_weightTable;

							// Allocate memory on the GPU
							bool status_1, status_2, status_3, status_4;
							status_1 = deviceMalloc((void**)&GPUDataAddr_index_weightTable_InletWall_Edge, n_elements_reserved * sizeof(int64_t));
							status_2 = deviceMalloc((void**)&GPUDataAddr_weightTable_InletWall_Edge, n_elements_reserved * sizeof(distribn_t));


							// Host-to-Device memory copy
							status_3 = deviceMemcpy(GPUDataAddr_index_weightTable_InletWall_Edge,
																	&index_weightTable[0], n_elements_reserved * sizeof(int64_t), memcpyHostToDevice);
							status_4 = deviceMemcpy(GPUDataAddr_weightTable_InletWall_Edge,
																											&vel_weightTable[0], n_elements_reserved * sizeof(distribn_t), memcpyHostToDevice);

							if(status_1 != cudaSuccess || status_2 != cudaSuccess || status_3 != cudaSuccess || status_4 != cudaSuccess ){
								fprintf(stderr, "GPU memory allocation and transfer Host To Device - key value indices and/or WeightsTable failed\n");
								initialise_GPU_res = false;
								return initialise_GPU_res;
							}

						} // Ends the if (site_Count_InletWall_Edge!=0)

						//------------------------------------------------------------------
						// E.3. Inlet_Inner
						// 		Info (map key index) to be placed in: index_weightTable_Inlet_Inner
						if (site_Count_Inlet_Inner!=0){

							site_t firstIndex = start_Index_Inlet_Inner;
							site_t siteCount = site_Count_Inlet_Inner;

							std::vector<int64_t> index_weightTable;
							std::vector<distribn_t> vel_weightTable;

							int n_elements_reserved=siteCount*(LatticeType::NUMVECTORS-1);
							index_weightTable.reserve(n_elements_reserved);
							vel_weightTable.reserve(n_elements_reserved);

							// Loop to get the index of the appropriate weight
							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								// Save the coords to Data_int64_Coords_iolets
								int64_t shifted_Fluid_Ind = siteIndex - firstIndex;

								geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

								int boundaryId = site.GetIoletId();
            		//iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetLocalIolet(boundaryId));
								iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetIolets()[boundaryId]);

            		LatticePosition sitePos(site.GetGlobalSiteCoords());

								// No need to include the LB_Dir=0
								for (int LB_Dir = 1; LB_Dir < LatticeType::NUMVECTORS; LB_Dir++) {
									//
									if (site.HasIolet(LB_Dir)){
										LatticePosition halfWay(sitePos);
				            halfWay.x += 0.5 * LatticeType::CX[LB_Dir];
				            halfWay.y += 0.5 * LatticeType::CY[LB_Dir];
				            halfWay.z += 0.5 * LatticeType::CZ[LB_Dir];

										//int index_weight = *(iolet->return_index_weight_VelocityTable(halfWay));
										int index_weightTable_received;
										distribn_t vel_weight_received;
										iolet->return_index_weight_VelocityTable(halfWay, &index_weightTable_received, &vel_weight_received); // Note that any dependency on the LB_Dir is absorbed in halfWay

										//if (siteIndex>=10025){
										//printf("Fluid ID: %d LB-Dir: %d index: %d - Vel-Weight : %f \n", siteIndex, LB_Dir, index_weightTable_received, vel_weight_received);
										//}
										index_weightTable.push_back(index_weightTable_received);
										vel_weightTable.push_back(vel_weight_received);
									}
									else{
										int index_weightTable_received = INT_MAX-1; // Should never access this value in the GPU kernel. Used for checking errors
										distribn_t vel_weight_received = 0.0;
										index_weightTable.push_back(index_weightTable_received);
										vel_weightTable.push_back(vel_weight_received);
									}
									//
								}
							} // Ends the loop over the fluid sites
							index_weightTable_Inlet_Inner=index_weightTable;
							weightTable_Inlet_Inner = vel_weightTable;

							// Allocate memory on the GPU
							bool status_1, status_2, status_3, status_4;
							status_1 = deviceMalloc((void**)&GPUDataAddr_index_weightTable_Inlet_Inner, n_elements_reserved * sizeof(int64_t));
							status_2 = deviceMalloc((void**)&GPUDataAddr_weightTable_Inlet_Inner, n_elements_reserved * sizeof(distribn_t));

							// Host-to-Device memory copy
							status_3 = deviceMemcpy(GPUDataAddr_index_weightTable_Inlet_Inner,
																	&index_weightTable[0], n_elements_reserved * sizeof(int64_t), memcpyHostToDevice);
							status_4 = deviceMemcpy(GPUDataAddr_weightTable_Inlet_Inner,
																											&vel_weightTable[0], n_elements_reserved * sizeof(distribn_t), memcpyHostToDevice);

							if(status_1 != cudaSuccess || status_2 != cudaSuccess || status_3 != cudaSuccess || status_4 != cudaSuccess ){
								fprintf(stderr, "GPU memory allocation and transfer Host To Device - key value indices and/or WeightsTable failed\n");
								initialise_GPU_res = false;
								return initialise_GPU_res;
							}

						} // Ends the if (site_Count_Inlet_Inner!=0)

						//------------------------------------------------------------------
						// E.4. InletWall_Inner
						// 		Info (map key index) to be placed in: index_weightTable_InletWall_Inner
						if (site_Count_InletWall_Inner!=0){

							site_t firstIndex = start_Index_InletWall_Inner;
							site_t siteCount = site_Count_InletWall_Inner;

							std::vector<int64_t> index_weightTable;
							std::vector<distribn_t> vel_weightTable;

							int n_elements_reserved=siteCount*(LatticeType::NUMVECTORS-1);
							index_weightTable.reserve(n_elements_reserved);
							vel_weightTable.reserve(n_elements_reserved);

							// Loop to get the index of the appropriate weight
							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								// Save the coords to Data_int64_Coords_iolets
								int64_t shifted_Fluid_Ind = siteIndex - firstIndex;

								geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);

								int boundaryId = site.GetIoletId();
            		//iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetLocalIolet(boundaryId));
								iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetIolets()[boundaryId]);

            		LatticePosition sitePos(site.GetGlobalSiteCoords());

								// No need to include the LB_Dir=0
								for (int LB_Dir = 1; LB_Dir < LatticeType::NUMVECTORS; LB_Dir++) {

									if (site.HasIolet(LB_Dir)){
										LatticePosition halfWay(sitePos);
				            halfWay.x += 0.5 * LatticeType::CX[LB_Dir];
				            halfWay.y += 0.5 * LatticeType::CY[LB_Dir];
				            halfWay.z += 0.5 * LatticeType::CZ[LB_Dir];

										//int index_weight = *(iolet->return_index_weight_VelocityTable(halfWay));
										int index_weightTable_received;
										distribn_t vel_weight_received;
										iolet->return_index_weight_VelocityTable(halfWay, &index_weightTable_received, &vel_weight_received); // Note that any dependency on the LB_Dir is absorbed in halfWay

										//if (siteIndex>=10025){
										//	printf("Fluid ID: %d LB-Dir: %d index: %d \n", siteIndex, LB_Dir, index_weightTable_received);
										//}
										index_weightTable.push_back(index_weightTable_received);
										vel_weightTable.push_back(vel_weight_received);
									}
									else{
										int index_weightTable_received = INT_MAX-1; // Should never access this value in the GPU kernel. Used for checking errors
										distribn_t vel_weight_received = 0.0;
										index_weightTable.push_back(index_weightTable_received);
										vel_weightTable.push_back(vel_weight_received);
									}

								}

							} // Ends the loop over the fluid sites
							index_weightTable_InletWall_Inner=index_weightTable;
							weightTable_InletWall_Inner = vel_weightTable;

							// Allocate memory on the GPU
							bool status_1, status_2, status_3, status_4;
							status_1 = deviceMalloc((void**)&GPUDataAddr_index_weightTable_InletWall_Inner, n_elements_reserved * sizeof(int64_t));
							status_2 = deviceMalloc((void**)&GPUDataAddr_weightTable_InletWall_Inner, n_elements_reserved * sizeof(distribn_t));

							// Host-to-Device memory copy
							status_3 = deviceMemcpy(GPUDataAddr_index_weightTable_InletWall_Inner,
																	&index_weightTable[0], n_elements_reserved * sizeof(int64_t), memcpyHostToDevice);
							status_4 = deviceMemcpy(GPUDataAddr_weightTable_InletWall_Inner,
																											&vel_weightTable[0], n_elements_reserved * sizeof(distribn_t), memcpyHostToDevice);

							if(status_1 != cudaSuccess || status_2 != cudaSuccess || status_3 != cudaSuccess || status_4 != cudaSuccess ){
								fprintf(stderr, "GPU memory allocation and transfer Host To Device - key value indices and/or WeightsTable failed\n");
								initialise_GPU_res = false;
								return initialise_GPU_res;
							}

						} // Ends the if (site_Count_InletWall_Inner!=0)
						//==================================================================
						// Ends the section on getting the index for the appropriate weight

					} // Ends the loop if (useWeightsFromFile)
					//====================================================================
#endif

					//====================================================================
					// D. Information regarding the Inlets:
					//		D.1. Positions of the iolets
					//		D.2. Radius of the iolets

					// Ensure that this is called only on the MPI ranks with local iolets
					if(n_unique_LocalInlets_mInlet_Edge!=0 || n_unique_LocalInlets_mInletWall_Edge !=0
						 || n_unique_LocalInlets_mInlet !=0 || n_unique_LocalInlets_mInletWall !=0){

					// vectors to hold the iolets position AND radius (Send this later to the GPU): iolet ID, x,y,z coords
					std::vector<double> inlets_position;
					std::vector<double> inlets_radius;
					for (int index_inlet = 0; index_inlet < n_Inlets; index_inlet++)
					{
						//printf("Copying velocityTable for iolet ID: %d \n", index_inlet);
						//iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetLocalIolet(index_inlet));
						iolets::InOutLetFileVelocity* iolet = dynamic_cast<iolets::InOutLetFileVelocity*>(mInletValues->GetIolets()[index_inlet]);

						//-----------------------------------------------------
						// Returns the position as reported in the input file.
						// TODO: Check the subtype of Vel BCs that needs this info
						LatticePosition position_iolet = iolet->GetPosition();						// LatticePosition: Vector3D<double>
						LatticeDistance local_radius_iolet = iolet->GetRadius();					// double
						// printf("Iolet %d position: (x,y,z) = (%f, %f, %f) \n", index_inlet, position_iolet.x, position_iolet.y, position_iolet.z );
						// printf("Iolet %d radius: %f \n", index_inlet, local_radius_iolet);	// Prints the radius divided by the resolution (Why???)

						inlets_position.push_back((double)index_inlet);
						inlets_position.push_back(position_iolet.x);
						inlets_position.push_back(position_iolet.y);
						inlets_position.push_back(position_iolet.z);

						inlets_radius.push_back((double)index_inlet);
						inlets_radius.push_back(local_radius_iolet);
						// Store the position of the iolets (x,y,z ) = (position_iolet.x, position_iolet.y, position_iolet.z) - type double
						// TODO!! Think how to do it: IOLET ID, position_iolet.x, position_iolet.y, position_iolet.z
						//-----------------------------------------------------
					} // Ends the loop over the inlets

					inlets_position.resize(n_Inlets*4);
					inlets_radius.resize(n_Inlets*2);

					/*std::vector<double>::iterator it;
					// Debugging - Print the values:
					std::cout << "Rank: " << myPiD << " - Inlet positions vector contains:";
					for (it=inlets_position.begin(); it!=inlets_position.end(); ++it)
						std::cout << ' ' << *it << '\n';
						 //std::cout << ' ' << *it << " Coordinates: " <<  *(it+1) << ' ' << *(it+2) << ' ' << *(it+3) << '\n';
					 std::cout << '\n\n';
					 */

					 //--------------------------------
					 // D.1. Positions of the iolets
					 // Allocate memory on the GPU (global memory)
					 site_t MemSz = 4 * n_Inlets * sizeof(double); 	// site_t (int64_t) Check that will remain like this in the future
					 status = deviceMalloc((void**)&GPUDataAddr_inlets_position, MemSz);
					 if(!status){ fprintf(stderr, "GPU memory allocation - Inlets position - failed...\n");
						 initialise_GPU_res = false; }

					 // Perform a HtD memcpy (from  inlets_position to GPUDataAddr_inlets_position)
					 status = deviceMemcpy(GPUDataAddr_inlets_position, &inlets_position, MemSz, memcpyHostToDevice);

					 if(!status){
						 const char * eStr = deviceGetErrorString();
						 printf("GPU memory copy - Inlets position failed with error: \"%s\" at proc# %i \n", eStr, myPiD);
						 initialise_GPU_res = false;
					 }
					 //--------------------------------
					 //--------------------------------
					 // D.2. Radius of the iolets
					 // Allocate memory on the GPU (global memory)
					 MemSz = 2 * n_Inlets * sizeof(double); 	// site_t (int64_t) Check that will remain like this in the future
					 status = deviceMalloc((void**)&GPUDataAddr_inlets_radius, MemSz);
					 if(!status){ fprintf(stderr, "GPU memory allocation - Inlets radius - failed...\n");
						 initialise_GPU_res = false; }

					 // Perform a HtD memcpy (from  inlets_position to GPUDataAddr_inlets_position)
					 status = deviceMemcpy(GPUDataAddr_inlets_radius, &inlets_radius, MemSz, memcpyHostToDevice);

					 if(!status){
						 const char * eStr = deviceGetErrorString();
						 printf("GPU memory copy - Inlets radius failed with error: \"%s\" at proc# %i \n", eStr, myPiD);
						 initialise_GPU_res = false;
					 }
					 //--------------------------------
				 }
					//====================================================================

					//
					//====================================================================
					/** April 2023
					 	New approach for the Vel BCs case - subtype File (use weights)
								Prefactor associated with wall momentum (purely geometry dependent  - containing the appropriate weights)

							Send these values to the GPU global memory in
								  GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge
									GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge
								 	GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner
									GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner
					*/

					//------------------------------------------------
					/*
					// I. Allocate memory on the GPU for the following:
							void *GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge;
							void *GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge;
							void *GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner;
							void *GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner;

							// Do not Implement for this at the moment
							void *GPUDataAddr_wallMom_prefactor_correction_Outlet_Edge;
							void *GPUDataAddr_wallMom_prefactor_correction_OutletWall_Edge;
							void *GPUDataAddr_wallMom_prefactor_correction_Outlet_Inner;
							void *GPUDataAddr_wallMom_prefactor_correction_OutletWall_Inner;
					*/

					// Memory requirements
					site_t Total_Mem_WallMom_prefactor_Inlet = (site_Count_Inlet_Edge + site_Count_InletWall_Edge + site_Count_Inlet_Inner + site_Count_InletWall_Inner)
																								* (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);

					if(site_Count_Inlet_Edge!=0){
						MemSz = site_Count_Inlet_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom prefactor_correction - Inlet Edge failed\n");
							initialise_GPU_res = false; return initialise_GPU_res; //return false;
						}
					}

					if(site_Count_InletWall_Edge!=0){
						MemSz = site_Count_InletWall_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom prefactor_correction - InletWall Edge failed\n");
							initialise_GPU_res = false; return initialise_GPU_res; //return false;
						}
					}

					if(site_Count_Inlet_Inner!=0){
						MemSz = site_Count_Inlet_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom prefactor_correction - Inlet Inner failed\n");
							initialise_GPU_res = false; return initialise_GPU_res; //return false;
						}
					}

					if(site_Count_InletWall_Inner!=0){
						MemSz = site_Count_InletWall_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						status = deviceMalloc((void**)&GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation wallMom prefactor_correction - InletWall Inner failed\n");
							initialise_GPU_res = false; return initialise_GPU_res; //return false;
						}
					}

					//------------------------------------------------

					// II. Send the prefactors to the GPU
					// Domain Edge
					// Collision Type 3 (mInletCollision):
					if (site_Count_Inlet_Edge!=0){
						// Get the prefactor associated with the wall momentum correction term
						GetWallMom_prefactor_correction_Direct(mInletCollision, start_Index_Inlet_Edge, site_Count_Inlet_Edge, propertyCache, wallMom_prefactor_correction_Inlet_Edge);
						// Note that this contains also the LB_dir 0 (it shouldn't, but take that into account)
						wallMom_prefactor_correction_Inlet_Edge.resize(site_Count_Inlet_Edge*LatticeType::NUMVECTORS);

						//-------------------------------------------------
						/*// Debugging
						for (int site_i=0; site_i<site_Count_Inlet_Edge; site_i++){
							for (int LB_dir=0; LB_dir<19; LB_dir++){
								double wallMom_prefactor_correction = wallMom_prefactor_correction_Inlet_Edge[site_i * LatticeType::NUMVECTORS + LB_dir];
								if(wallMom_prefactor_correction != 0.0 ){
									printf("Section 3.1 : wallMom_prefactor_correction_Inlet_Edge / site_i: %lu, Dir: %d, Wall Mom pref. Correction: %.5e \n", (site_i+start_Index_Inlet_Edge),
																																LB_dir, wallMom_prefactor_correction);
								}
							}

						}*/
						//-------------------------------------------------

						// Function to allocate memory on the GPU's global memory for the prefactor associated with wallMom
						memCpy_HtD_GPUmem_WallMom_prefactor_correction(start_Index_Inlet_Edge, site_Count_Inlet_Edge, wallMom_prefactor_correction_Inlet_Edge,
																														GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge);
					}

					// Collision Type 5 (mInletWallCollision):
					//printf("Rank: %d - site_Count_InletWall_Edge : %ld \n", myPiD, site_Count_InletWall_Edge);
					if (site_Count_InletWall_Edge!=0){
						// Get the prefactor associated with the wall momentum correction term
						GetWallMom_prefactor_correction_Direct(mInletWallCollision, start_Index_InletWall_Edge, site_Count_InletWall_Edge, propertyCache, wallMom_prefactor_correction_InletWall_Edge);
						// Note that this contains also the LB_dir 0 (it shouldn't but take that into account)
						wallMom_prefactor_correction_InletWall_Edge.resize(site_Count_InletWall_Edge*LatticeType::NUMVECTORS);

						//-------------------------------------------------
						/*// Debugging
						for (int site_i=0; site_i<site_Count_InletWall_Edge; site_i++){
							for (int LB_dir=0; LB_dir<19; LB_dir++){
								double wallMom_prefactor_correction = wallMom_prefactor_correction_InletWall_Edge[site_i * LatticeType::NUMVECTORS + LB_dir];
								if(wallMom_prefactor_correction != 0.0 ){
									printf("Section 3.2 : wallMom_prefactor_correction_InletWall_Edge / site_i: %lu, Dir: %d, Wall Mom pref. Correction: %.5e \n", (site_i+start_Index_InletWall_Edge),
																																LB_dir, wallMom_prefactor_correction);
								}
							}

						}*/
						//-------------------------------------------------

						// Function to allocate memory on the GPU's global memory for the prefactor associated with wallMom
						memCpy_HtD_GPUmem_WallMom_prefactor_correction(start_Index_InletWall_Edge, site_Count_InletWall_Edge, wallMom_prefactor_correction_InletWall_Edge,
																														GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge);
					}

					// Inner domain
					// Collision Type 3 (mInletCollision):
					// site_t start_Index_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
					// site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
					if (site_Count_Inlet_Inner!=0){
						// Get the prefactor associated with the wall momentum correction term
						GetWallMom_prefactor_correction_Direct(mInletCollision, start_Index_Inlet_Inner, site_Count_Inlet_Inner, propertyCache, wallMom_prefactor_correction_Inlet_Inner);
						// Note that this contains also the LB_dir 0 (it shouldn't but take that into account)
						wallMom_prefactor_correction_Inlet_Inner.resize(site_Count_Inlet_Inner*LatticeType::NUMVECTORS);

						//-------------------------------------------------
						/*
						// Debugging
						for (int site_i=0; site_i<site_Count_Inlet_Inner; site_i++){
							for (int LB_dir=0; LB_dir<19; LB_dir++){
								double wallMom_prefactor_correction = wallMom_prefactor_correction_Inlet_Inner[site_i * LatticeType::NUMVECTORS + LB_dir];
								if(wallMom_prefactor_correction != 0.0 ){
									printf("Section 3.3 : wallMom_prefactor_correction_Inlet_Inner / site_i: %lu, Dir: %d, Wall Mom pref. Correction: %.5e \n", (site_i+start_Index_Inlet_Inner),
																																LB_dir, wallMom_prefactor_correction);
								}
							}
						}
						*/
						//-------------------------------------------------

						// Function to allocate memory on the GPU's global memory for the prefactor associated with wallMom
						memCpy_HtD_GPUmem_WallMom_prefactor_correction(start_Index_Inlet_Inner, site_Count_Inlet_Inner, wallMom_prefactor_correction_Inlet_Inner,
																														GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner);
					}

					// Collision Type 5 (mInletWallCollision):
					//site_t start_Index_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
					//site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
					if (site_Count_InletWall_Inner!=0){
						// Get the prefactor associated with the wall momentum correction term
						GetWallMom_prefactor_correction_Direct(mInletWallCollision, start_Index_InletWall_Inner, site_Count_InletWall_Inner, propertyCache, wallMom_prefactor_correction_InletWall_Inner);
						// Note that this contains also the LB_dir 0 (it shouldn't baut take that into account)
						wallMom_prefactor_correction_InletWall_Inner.resize(site_Count_InletWall_Inner*LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the prefactor associated with wallMom
						memCpy_HtD_GPUmem_WallMom_prefactor_correction(start_Index_InletWall_Inner, site_Count_InletWall_Inner, wallMom_prefactor_correction_InletWall_Inner,
																														GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner);
					}
					//====================================================================
					//

				} // Ends the loop if(hemeIoletBC_Inlet == "LADDIOLET")
				else if (hemeIoletBC_Outlet == "LADDIOLET"){
					// Add here the corresponding code for the outlet
					// A.2. Outlets' BCs
					// Place here the part for sending the coordinates for the outlets
					// TODO!!!
					//
				}
				//***********************************************************************************************************************************



				//***********************************************************************************************************************************
				// Allocate memory for streamingIndicesForReceivedDistributions on the GPU constant Memory
				// From geometry/LatticeData.h:	std::vector<site_t> streamingIndicesForReceivedDistributions; //! The indices to stream to for distributions received from other processors.

				TotalMem_int64_streamInd = totSharedFs * sizeof(site_t); // Total memory size for streamingIndicesForReceivedDistributions
				site_t* Data_int64_streamInd = new site_t[totSharedFs];	// site_t (type int64_t)

				if(!Data_int64_streamInd){
					std::cout << "Memory allocation error - streamingIndicesForReceivedDistributions" << std::endl;
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				Data_int64_streamInd = &(mLatDat->streamingIndicesForReceivedDistributions[0]);  // Data_int64_streamInd points to &(mLatDat->streamingIndicesForReceivedDistributions[0])

				// Debugging
				/* for (site_t i = 0; i < totSharedFs; i++){
					 site_t streamIndex = Data_int64_streamInd[i];
					 printf("Index = %lld, Streamed Index = %lld \n\n", i, streamIndex);
				}*/

				// Alocate memory on the GPU
				status = deviceMalloc((void**)&GPUDataAddr_int64_streamInd, totSharedFs * sizeof(site_t));
				if(!status){
					fprintf(stderr, "GPU memory allocation for streamingIndicesForReceivedDistributions failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// Memory copy from host (Data_int64_streamInd) to Device (GPUDataAddr_int64_Neigh)
				status = deviceMemcpy(GPUDataAddr_int64_streamInd, Data_int64_streamInd, totSharedFs * sizeof(site_t), memcpyHostToDevice);
				if(!status){
					fprintf(stderr, "GPU memory transfer Host To Device for streamingIndicesForReceivedDistributions failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}
				//***********************************************************************************************************************************

				//***********************************************************************************************************************************
				// Flag for the stability of the Code
				// Set up the stability value as UndefinedStability (value -1) and modify
				//  to value Unstable (value 0), see SimulationState.h

				//int* d_Stability_GPU;
				mLatDat->h_Stability_GPU_mLatDat=-1;
				h_Stability_GPU=-1;
				//h_Stability_GPU[0] = -1; // Same value as the one used to denote UndefinedStability (value -1)
				status = deviceMalloc((void**)&d_Stability_GPU, sizeof(int));
				status = deviceMalloc((void**)&(mLatDat->d_Stability_GPU_mLatDat), sizeof(int));

				status = deviceMemcpy(d_Stability_GPU, &h_Stability_GPU, sizeof(int), memcpyHostToDevice);
				if(!status){
					fprintf(stderr, "GPU memory transfer Host To Device for Stability param. failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				status = deviceMemcpy(mLatDat->d_Stability_GPU_mLatDat, &(mLatDat->h_Stability_GPU_mLatDat), sizeof(int), memcpyHostToDevice);
				if(!status){
					fprintf(stderr, "GPU memory transfer Host To Device for Stability param. failed\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}
				//***********************************************************************************************************************************

				//***********************************************************************************************************************************
				/** Case - Require to save wall Shear Stress Magnitude
						Need to allocate memory on the GPU for the following:
							a. wall shear stress magnitude
									This will be required for the sites with wall links, i.e.
									total_numElements = mLatDat->GetDomainEdgeCollisionCount(1) + mLatDat->GetDomainEdgeCollisionCount(4) + mLatDat->GetDomainEdgeCollisionCount(5)
																		+ mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(4) + mLatDat->GetMidDomainCollisionCount(5);

									Domain Edges
									a.1.	Collision Type 2: mWallCollision
													offset = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0);
													site_Count = mLatDat->GetDomainEdgeCollisionCount(1);

									a.2.	Collision Type 5 (Inlet - Walls): mInletWallCollision
										      offset = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1) +
										                mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);
										      site_Count = mLatDat->GetDomainEdgeCollisionCount(4);

									a.3. 	Collision Type 6 (Outlet - Walls): mOutletWallCollision
										      offset = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
										              + mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3) + mLatDat->GetDomainEdgeCollisionCount(4);
										      site_Count = mLatDat->GetDomainEdgeCollisionCount(5);

									Inner Domain
									a.4. 	Collision Type 2: mWallCollision
									      	offset = mLatDat->GetMidDomainCollisionCount(0);
									      	site_Count = mLatDat->GetMidDomainCollisionCount(1);

									a.5. 	Collision Type 5 (Inlet - Walls): mInletWallCollision
									      	offset = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
									      	site_Count = mLatDat->GetMidDomainCollisionCount(4);

									a.6. 	Collision Type 6 (Outlet - Walls): mOutletWallCollision
										      offset = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2)
										                + mLatDat->GetMidDomainCollisionCount(3)
										                + mLatDat->GetMidDomainCollisionCount(4);
										      site_Count = mLatDat->GetMidDomainCollisionCount(5);

							b. ???
				*/
				if (save_wallShearStressMagn==true) {
					bool init_res_wallShearStress = initialise_GPU_WallShearStressMagn(mInletValues, mOutletValues, mUnits);
					//printf("Rank: %d - Initialise_GPU_WallShearStress: %d \n", myPiD, init_res_wallShearStress);
					if (!init_res_wallShearStress){ initialise_GPU_res = false; return initialise_GPU_res; }
				}
				//======================================================================

				//======================================================================
				// July 2024
				// Sponge Layer - LES implementation
				// If the kernel is set to LBGKSL
				// Required element for LBGKSpongeLayer
				const std::string hemeKernel = QUOTE_CONTENTS(HEMELB_KERNEL);

				if (hemeKernel == "LBGKSL")
				{
					//printf("Initialisation necessary for the LBGKSL - LES implementation!!! \n\n");
					//--------------------------------------------------------------------
					bool init_res_LBGKSL = initialise_GPU_LBGKSL();
					//printf("Rank: %d - initialise_GPU_LBGKSL: %d \n", myPiD, init_res_LBGKSL);
					if (!init_res_LBGKSL){
						printf("Rank: %d - Failure to initialise the GPU for the LBGKSL... \n", myPiD);
						initialise_GPU_res = false; return initialise_GPU_res;
					}
					//--------------------------------------------------------------------
					/*if(myPiD!=0){
						// Number of fluid sites
						site_t nFluid_sites = mLatDat->GetLocalFluidSiteCount();
						//printf("Number of fluid sites: %ld \n", nFluid_sites);

						// Memory Size required
						site_t MemSz = nFluid_sites * sizeof(distribn_t);

						// Mem.copy vTau to the GPU (GPUDataAddr_vTau)
						// 1. Allocate memory on the GPU
						bool status = deviceMalloc((void**)&GPUDataAddr_vTau, MemSz);
						if(!status){
							fprintf(stderr, "GPU memory allocation vTau failed...\n");
							bool InitState_SpongeLayer_GPU_res = false; return InitState_SpongeLayer_GPU_res;
						}

						// 2. Memory copy from host (Data_dbl_WallNormal_Edge_Type2) to Device (GPUDataAddr_WallNormal_Edge_Type2)
						// Access vTau from the LBGKSpongeLayer class using the static GetvTau
						const distribn_t* vTauValue = hemelb::lb::kernels::LBGKSpongeLayer<LatticeType>::GetvTau(0);

						status = deviceMemcpy(GPUDataAddr_vTau, &vTauValue[0], MemSz, memcpyHostToDevice);
						if(!status){
							fprintf(stderr, "GPU memory transfer vTau Host To Device failed\n");
							bool InitState_SpongeLayer_GPU_res = false; return InitState_SpongeLayer_GPU_res;
						}
					}*/
				}
				//======================================================================

				//***********************************************************************************************************************************

				//***********************************************************************************************************************************
				// Check the total memory requirements
				// Change this in the future as roughly half of this (either memory arrangement (a) or (b)) will be needed. To do!!!
				// Add a check whether the memory on the GPU global memory is sufficient!!! Abort if not or split the domain into smaller subdomains and pass info gradually! To do!!!
				// unsigned long long TotalMem_req = (TotalMem_dbl_fOld * 4 +  TotalMem_dbl_MacroVars + TotalMem_int64_Neigh *4 + TotalMem_uint32_WallIntersect + TotalMem_uint32_IoletIntersect + TotalMem_int64_streamInd); //
				unsigned long long TotalMem_req = (TotalMem_dbl_fOld * 2 +  TotalMem_dbl_MacroVars + TotalMem_int64_Neigh + TotalMem_uint32_WallIntersect + TotalMem_uint32_IoletIntersect + TotalMem_int64_streamInd); //
				// printf("Rank: %d - Total requested global memory %.2fGB \n\n", myPiD, ((double)TotalMem_req/1073741824.0));
				//***********************************************************************************************************************************

				//=================================================================================================================================
				// Copy constants to the GPU memory - Limit is 64 kB
				//	2. Constants:
				//		a. weights for the equilibrium distr. functions
				//		b. Number of vectors: LatticeType::NUMVECTORS
				//		c. INVERSEDIRECTIONS for the bounce simple back simple (Wall BCs): LatticeType::INVERSEDIRECTIONS
				//		d. Lattice Velocity directions CX[DmQn::NUMVECTORS], CY[DmQn::NUMVECTORS], CZ[DmQn::NUMVECTORS]
				//		e. Relaxation Time tau
				//		f. Cs2
				//		g. useWeightsFromFile - Case of Vel BCs

				// 2.a. Weight coefficients for the equilibrium distr. functions
				status = deviceMemcpyToSymbol(hemelb::_EQMWEIGHTS_19, LatticeType::EQMWEIGHTS, LatticeType::NUMVECTORS*sizeof(double), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (1)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
					//goto Error;
				}

				// 2.b. Number of vectors: LatticeType::NUMVECTORS
				static const unsigned int num_Vectors = LatticeType::NUMVECTORS;
				status = deviceMemcpyToSymbol(&hemelb::_NUMVECTORS, &num_Vectors, sizeof(num_Vectors), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (2)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
					//goto Error;
				}

				// 2.c. Inverse directions for the bounce back LatticeType::INVERSEDIRECTIONS[direction]
				status = deviceMemcpyToSymbol(hemelb::_InvDirections_19, LatticeType::INVERSEDIRECTIONS, LatticeType::NUMVECTORS*sizeof(int), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (3)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
					//goto Error;
				}

				// 2.d. Lattice Velocity directions CX[DmQn::NUMVECTORS], CY[DmQn::NUMVECTORS], CZ[DmQn::NUMVECTORS]
				status = deviceMemcpyToSymbol(hemelb::_CX_19, LatticeType::CX, LatticeType::NUMVECTORS*sizeof(int), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (4)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}
				status = deviceMemcpyToSymbol(hemelb::_CY_19, LatticeType::CY, LatticeType::NUMVECTORS*sizeof(int), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (5)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}
				status = deviceMemcpyToSymbol(hemelb::_CZ_19, LatticeType::CZ, LatticeType::NUMVECTORS*sizeof(int), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (6)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// 2.e. Relaxation Time tau
				//static const int num_Vectors = LatticeType::NUMVECTORS;
				// mParams object of type hemelb::lb::LbmParameters (struct LbmParameters)
				double tau = mParams.GetTau();
				if(myPiD==1) printf("Relaxation Time = %.5f\n\n", tau);
				double minus_inv_tau = mParams.GetOmega();	// printf("Minus Inv. Relaxation Time = %.5f\n\n", minus_inv_tau);

				status = deviceMemcpyToSymbol(&hemelb::dev_tau, &tau, sizeof(tau), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (7)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				status = deviceMemcpyToSymbol(&hemelb::dev_minusInvTau, &minus_inv_tau, sizeof(minus_inv_tau), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (8)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				status = deviceMemcpyToSymbol(&hemelb::_Cs2, &Cs2, sizeof(Cs2), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (9)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
					//return false;
				}

				// 2g. useWeightsFromFile - Case of Vel BCs
				status = deviceMemcpyToSymbol(&hemelb::_useWeightsFromFile, &useWeightsFromFile, sizeof(useWeightsFromFile), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (10)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
				}


				// 2h. stress parameter (for the case of evaluating wall shear stress magnitude)
        distribn_t iStressParameter = mParams.GetStressParameter();
				//printf("StressParameter : %f\n", iStressParameter);
				status = deviceMemcpyToSymbol(&hemelb::_iStressParameter, &iStressParameter, sizeof(iStressParameter), 0, memcpyHostToDevice);
				if (!status) {
					fprintf(stderr, "GPU constant memory copy failed (11)\n");
					initialise_GPU_res = false;
					return initialise_GPU_res;
				}

				//=================================================================================================================================
				/*
				// Pinned memory for totalSharedFs
				//	a. for the D2H memcpy: Data_D2H_memcpy_totalSharedFs
				//	b. for the H2D memcpy: Data_H2D_memcpy_totalSharedFs
				//distribn_t* Data_H2D_memcpy_totalSharedFs;

				//a. for the D2H memcpy
				MemSz = (1+totSharedFs) * sizeof(distribn_t);
				status = deviceMallocHost((void**)&Data_D2H_memcpy_totalSharedFs, MemSz);
				if(!status){ fprintf(stderr, "deviceMallocHost for Data_D2H_memcpy_totalSharedFs failed... Rank = %d, Time = %d \n",myPiD, mState->GetTimeStep()); }
				// memset(Data_D2H_memcpy_totalSharedFs, 0, MemSz);

				//b. for the H2D memcpy
				MemSz = totSharedFs * sizeof(distribn_t);

				status = deviceMallocHost((void**)&Data_H2D_memcpy_totalSharedFs, MemSz);
				if(!status){ fprintf(stderr, "deviceMallocHost for Data_H2D_memcpy_totalSharedFs failed... Rank = %d, Time = %d \n",myPiD, mState->GetTimeStep()); }
				// memset(Data_H2D_memcpy_totalSharedFs, 0, MemSz);
				*/
				//=================================================================================================================================

				//cudaDeviceSynchronize();

				// Create the Streams here
				deviceStreamCreate(&Collide_Stream_PreSend_1);
				deviceStreamCreate(&Collide_Stream_PreSend_2);
				deviceStreamCreate(&Collide_Stream_PreSend_3);
				deviceStreamCreate(&Collide_Stream_PreSend_4);
				deviceStreamCreate(&Collide_Stream_PreSend_5);
				deviceStreamCreate(&Collide_Stream_PreSend_6);

				deviceStreamCreate(&Collide_Stream_PreRec_1);
				deviceStreamCreate(&Collide_Stream_PreRec_2);
				deviceStreamCreate(&Collide_Stream_PreRec_3);
				deviceStreamCreate(&Collide_Stream_PreRec_4);
				deviceStreamCreate(&Collide_Stream_PreRec_5);
				deviceStreamCreate(&Collide_Stream_PreRec_6);

				deviceStreamCreate(&stream_ghost_dens_inlet);
				deviceStreamCreate(&stream_ghost_dens_outlet);

				deviceStreamCreate(&stream_ReceivedDistr);
				deviceStreamCreate(&stream_SwapOldAndNew);
				deviceStreamCreate(&stream_memCpy_CPU_GPU_domainEdge);

				deviceStreamCreate(&stream_Read_Data_GPU_Dens);
				deviceStreamCreate(&stability_check_stream);

				//----------------------------------------------------------------------
				// Create the cuda stream for the asynch. MemCopy DtH at the domain edges: creates a stream in net::BaseNet object
				hemelb::net::Net& mNet_cuda_stream = *mNet;	// Needs the constructor and be initialised
				mNet_cuda_stream.Create_stream_memCpy_GPU_CPU_domainEdge_new2(); // create the stream and then impose a synch barrier in net::BaseNet::Send
				/**
					Syncronisation barrier then placed in net/phased/NetConcern.h (before sending the data to be exchanged at domain boundaries):
						in net/phased/NetConcern.h:
							Synchronisation barrier - Barrier for stream created for the asynch. memcpy at domain edges
							net.Synchronise_memCpy_GPU_CPU_domainEdge();
				*/
				//----------------------------------------------------------------------

				// Delete allocated host memory that is no longer needed
				delete[] Data_dbl_fOld_b;
				delete[] Data_dbl_fNew_b;
				delete[] Data_int64_Neigh_d;
				//delete[] Data_uint32_WallIntersect;
				delete[] Data_uint32_IoletIntersect;
				delete[] h_inletNormal;
			        delete[] h_outletNormal;

				if (hemeIoletBC_Inlet == "LADDIOLET") {
					// if subtype Case: b. File
					// Delete that at an earlier point already - Remove!
					//delete[] Data_dbl_Inlet_velocityTable;
				}
				if (hemeIoletBC_Outlet == "LADDIOLET") {
					// if subtype Case: b. File
					//delete[] Data_dbl_Outlet_velocityTable;
				}

				return initialise_GPU_res;
			}


			//------------------------------------------------------------------------------
			// IZ - August 2024
			/** 	LBGK Sponge Layer - LES related - Initialise_GPU
							Note that it is called for all ranks except rank 0

						1. Allocate memory associated with the following:
								void *GPUDataAddr_vTau;
						2. MemCpy D2H (vTau to GPUDataAddr_vTau)
			*/
			//------------------------------------------------------------------------------
	template<class LatticeType>
			bool LBM<LatticeType>::initialise_GPU_LBGKSL()
			{
				bool initialise_GPU_LBGKSL_res = true;
				bool status;

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				if(myPiD!=0){
					// Number of fluid sites
					site_t nFluid_sites = mLatDat->GetLocalFluidSiteCount();

					//
					// Debugging - Testing
					//printf("Number of fluid sites: %ld \n", nFluid_sites);
					//

					// Memory Size required
					site_t MemSz = nFluid_sites * sizeof(distribn_t);

					// Mem.copy vTau to the GPU (GPUDataAddr_vTau)
					// 1. Allocate memory on the GPU
					bool status = deviceMalloc((void**)&GPUDataAddr_vTau, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation vTau failed...\n");
						initialise_GPU_LBGKSL_res = false;
						return initialise_GPU_LBGKSL_res;
					}

					// 2. Memory copy from host (Data_dbl_WallNormal_Edge_Type2) to Device (GPUDataAddr_WallNormal_Edge_Type2)
					// Access vTau from the LBGKSpongeLayer class using the static GetvTau
					const distribn_t* vTauValue = hemelb::lb::kernels::LBGKSpongeLayer<LatticeType>::GetvTau(0);

					/*
					// Debugging
					std::cout << "vTauValue:" << std::endl;
					for(int index=0; index<nFluid_sites;index++)
					{
						std::cout << "vTauValue[" << index << "]: " << vTauValue[index] <<std::endl;
					}*/

					status = deviceMemcpy(GPUDataAddr_vTau, &vTauValue[0], MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer vTau Host To Device failed\n");
						initialise_GPU_LBGKSL_res = false;
						return initialise_GPU_LBGKSL_res;
					}

					//====================================================================
					// Copy constants to the GPU memory - Limit is 64 kB
					double smag_cnst = mSimConfig->GetCSmagorinsky();//mParams.Smagorinsky_const;
					//if(myPiD==1) printf("Smagorinsky constant = %.2f \n\n", smag_cnst);

					status = deviceMemcpyToSymbol(&hemelb::dev_smag_cnst, &smag_cnst, sizeof(smag_cnst), 0, memcpyHostToDevice);
					if (!status) {
						fprintf(stderr, "GPU memory transfer Smagorinsky const Host To Device failed\n");
						initialise_GPU_LBGKSL_res = false;
						return initialise_GPU_LBGKSL_res;
					}
					//====================================================================
				}

				return initialise_GPU_LBGKSL_res;
			}


//------------------------------------------------------------------------------
// IZ - April 2023
/** 	Wall Shear Stress related - Initialise_GPU
				Note that it is called for all ranks except rank 0

			1. Allocate memory associated with the following (shear stress magnitude):
					void *GPUDataAddr_WallShearStressMagn_Edge_Type2;
					void *GPUDataAddr_WallShearStressMagn_Edge_Type5;
					void *GPUDataAddr_WallShearStressMagn_Edge_Type6;
					void *GPUDataAddr_WallShearStressMagn_Inner_Type2;
					void *GPUDataAddr_WallShearStressMagn_Inner_Type5;
					void *GPUDataAddr_WallShearStressMagn_Inner_Type6;
			2.
*/
//------------------------------------------------------------------------------
template<class LatticeType>
			bool LBM<LatticeType>::initialise_GPU_WallShearStressMagn(iolets::BoundaryValues* iInletValues,
					iolets::BoundaryValues* iOutletValues, const util::UnitConverter* iUnits)
			{

				bool initialise_GPU_WallShearStress_res = true;
				bool status;

				mInletValues = iInletValues;
				mOutletValues = iOutletValues;
				mUnits = iUnits;

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				//======================================================================
				// Number of sites requiring wall shear stress calculations
				site_t total_numElements = mLatDat->GetDomainEdgeCollisionCount(1) + mLatDat->GetDomainEdgeCollisionCount(4) + mLatDat->GetDomainEdgeCollisionCount(5)
													+ mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(4) + mLatDat->GetMidDomainCollisionCount(5);

				site_t TotalMem_dbl_WallShearStressMagn = total_numElements * sizeof(distribn_t);

				//----------------------------------------
				// Site Count and Starting Indices
				// Domain Edge (walls (type2) - Inlets with walls (type5) - Outlets with walls(type6))
				site_t start_Index_Edge_Type2 = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0);
				site_t total_numElements_Edge_Type2 = mLatDat->GetDomainEdgeCollisionCount(1);

				site_t start_Index_Edge_Type5 = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
                												+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);
				site_t total_numElements_Edge_Type5 = mLatDat->GetDomainEdgeCollisionCount(4);

				site_t start_Index_Edge_Type6 = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
              												+ mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3) + mLatDat->GetDomainEdgeCollisionCount(4);
				site_t total_numElements_Edge_Type6 = mLatDat->GetDomainEdgeCollisionCount(5);

				// Inner Domain
				site_t start_Index_Inner_Type2 = mLatDat->GetMidDomainCollisionCount(0);
				site_t total_numElements_Inner_Type2 = mLatDat->GetMidDomainCollisionCount(1);

				site_t start_Index_Inner_Type5 = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
				site_t total_numElements_Inner_Type5 = mLatDat->GetMidDomainCollisionCount(4);

				site_t start_Index_Inner_Type6 = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2)
                											+ mLatDat->GetMidDomainCollisionCount(3) + mLatDat->GetMidDomainCollisionCount(4);
				site_t total_numElements_Inner_Type6 = mLatDat->GetMidDomainCollisionCount(5);
				//----------------------------------------

				//======================================================================
				//----------------------------------------
				// Wall shear stress magnitudes
				// Allocate memory on the GPU for each one of the wall shear stress magnitudes (based on the collision type) separately
				// 1.
				site_t site_count = total_numElements_Edge_Type2;
				if (site_count!=0){
					site_t MemSz = site_count * sizeof(distribn_t);

					status = deviceMalloc((void**)&GPUDataAddr_WallShearStressMagn_Edge_Type2, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Wall Shear Stress magnitude: Wall Edge failed...\n");
						initialise_GPU_WallShearStress_res = false;
						return initialise_GPU_WallShearStress_res;
					}
				}

				// 2.
				site_count = total_numElements_Edge_Type5;
				if (site_count!=0){
					site_t MemSz = site_count * sizeof(distribn_t);

					status = deviceMalloc((void**)&GPUDataAddr_WallShearStressMagn_Edge_Type5, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Wall Shear Stress magnitude: Inlet Wall Edge failed...\n");
						initialise_GPU_WallShearStress_res = false;
						return initialise_GPU_WallShearStress_res;
					}
				}

				// 3.
				site_count = total_numElements_Edge_Type6;
				if (site_count!=0){
					site_t MemSz = site_count * sizeof(distribn_t);

					status = deviceMalloc((void**)&GPUDataAddr_WallShearStressMagn_Edge_Type6, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Wall Shear Stress magnitude: Outlet Wall Edge failed...\n");
						initialise_GPU_WallShearStress_res = false;
						return initialise_GPU_WallShearStress_res;
					}
				}

				// 4.
				site_count = total_numElements_Inner_Type2;
				if (site_count!=0){
					site_t MemSz = site_count * sizeof(distribn_t);

					status = deviceMalloc((void**)&GPUDataAddr_WallShearStressMagn_Inner_Type2, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Wall Shear Stress magnitude: Wall Inner failed...\n");
						initialise_GPU_WallShearStress_res = false;
						return initialise_GPU_WallShearStress_res;
					}
				}

				// 5.
				site_count = total_numElements_Inner_Type5;
				if (site_count!=0){
					site_t MemSz = site_count * sizeof(distribn_t);

					status = deviceMalloc((void**)&GPUDataAddr_WallShearStressMagn_Inner_Type5, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Wall Shear Stress magnitude: Inlet Wall Inner failed...\n");
						initialise_GPU_WallShearStress_res = false;
						return initialise_GPU_WallShearStress_res;
					}
				}

				// 6.
				site_count = total_numElements_Inner_Type6;
				if (site_count!=0){
					site_t MemSz = site_count * sizeof(distribn_t);

					status = deviceMalloc((void**)&GPUDataAddr_WallShearStressMagn_Inner_Type6, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation Wall Shear Stress magnitude: Outlet Wall Inner failed...\n");
						initialise_GPU_WallShearStress_res = false;
						return initialise_GPU_WallShearStress_res;
					}
				}
				//----------------------------------------
				//======================================================================

				//======================================================================
				//----------------------------------------
				// Wall normal vectors
				distribn_t *Data_dbl_WallNormal_Edge_Type2, *Data_dbl_WallNormal_Edge_Type5, *Data_dbl_WallNormal_Edge_Type6;
				distribn_t *Data_dbl_WallNormal_Inner_Type2, *Data_dbl_WallNormal_Inner_Type5, *Data_dbl_WallNormal_Inner_Type6;


				// 1.
				site_t start_index = start_Index_Edge_Type2;
				site_count = total_numElements_Edge_Type2;

				if(site_count!=0){
					Data_dbl_WallNormal_Edge_Type2 = new distribn_t[3*site_count];

					for (site_t site_index=start_index; site_index < start_index + site_count; site_index++){

						// Get the components of the normal vector to the wall surface
						geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_index);
						util::Vector3D<distribn_t> normalToWall = site.GetWallNormal();

						site_t shifted_Ind = site_index-start_index;
						Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind] = normalToWall.x;
						Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind +1] = normalToWall.y;
						Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind +2] = normalToWall.z;
						//std::cout << "Cout: WallNormal.x : " <<  Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind] << " - WallNormal.y : " <<  Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind +1] << " - WallNormal.z : " <<  Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind +2] << std::endl;
					}

					// Allocate memory on the GPU (global memory)
					site_t MemSz = 3*site_count *  sizeof(distribn_t); 	// site_t (int64_t) Check that will remain like this in the future
					status = deviceMalloc((void**)&GPUDataAddr_WallNormal_Edge_Type2, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation - Wall Normal - Wall Edge failed ...\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Memory copy from host (Data_dbl_WallNormal_Edge_Type2) to Device (GPUDataAddr_WallNormal_Edge_Type2)
					status = deviceMemcpy(GPUDataAddr_WallNormal_Edge_Type2, Data_dbl_WallNormal_Edge_Type2, MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer (Wall Normal - Wall Edge) Host To Device failed\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Free-up memory
					delete[] Data_dbl_WallNormal_Edge_Type2;
				}
				//----------------------------------------

				// 2.
				start_index = start_Index_Edge_Type5;
				site_count = total_numElements_Edge_Type5;

				if(site_count!=0){
					Data_dbl_WallNormal_Edge_Type5 = new distribn_t[3*site_count];

					for (site_t site_index=start_index; site_index < start_index + site_count; site_index++){

						// Get the components of the normal vector to the wall surface
						geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_index);
						util::Vector3D<distribn_t> normalToWall = site.GetWallNormal();

						site_t shifted_Ind = site_index-start_index;
						Data_dbl_WallNormal_Edge_Type5[3*shifted_Ind] = normalToWall.x;
						Data_dbl_WallNormal_Edge_Type5[3*shifted_Ind +1] = normalToWall.y;
						Data_dbl_WallNormal_Edge_Type5[3*shifted_Ind +2] = normalToWall.z;
						//std::cout << "Cout: WallNormal.x : " <<  Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind] << " - WallNormal.y : " <<  Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind +1] << " - WallNormal.z : " <<  Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind +2] << std::endl;
					}

					// Allocate memory on the GPU (global memory)
					site_t MemSz = 3*site_count *  sizeof(distribn_t);
					status = deviceMalloc((void**)&GPUDataAddr_WallNormal_Edge_Type5, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation - Wall Normal - Inlet-Wall Edge failed ...\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Memory copy from host (Data_dbl_WallNormal_Edge_Type2) to Device (GPUDataAddr_WallNormal_Edge_Type2)
					status = deviceMemcpy(GPUDataAddr_WallNormal_Edge_Type5, Data_dbl_WallNormal_Edge_Type5, MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer (Wall Normal - Inlet-Wall Edge) Host To Device failed\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Free-up memory
					delete[] Data_dbl_WallNormal_Edge_Type5;
				}
				//----------------------------------------

				// 3.
				start_index = start_Index_Edge_Type6;
				site_count = total_numElements_Edge_Type6;

				if(site_count!=0){
					Data_dbl_WallNormal_Edge_Type6 = new distribn_t[3*site_count];

					for (site_t site_index=start_index; site_index < start_index + site_count; site_index++){

						// Get the components of the normal vector to the wall surface
						geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_index);
						util::Vector3D<distribn_t> normalToWall = site.GetWallNormal();

						site_t shifted_Ind = site_index-start_index;
						Data_dbl_WallNormal_Edge_Type6[3*shifted_Ind] = normalToWall.x;
						Data_dbl_WallNormal_Edge_Type6[3*shifted_Ind +1] = normalToWall.y;
						Data_dbl_WallNormal_Edge_Type6[3*shifted_Ind +2] = normalToWall.z;
						//std::cout << "Cout: WallNormal.x : " <<  Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind] << " - WallNormal.y : " <<  Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind +1] << " - WallNormal.z : " <<  Data_dbl_WallNormal_Edge_Type2[3*shifted_Ind +2] << std::endl;
					}

					// Allocate memory on the GPU (global memory)
					site_t MemSz = 3*site_count *  sizeof(distribn_t);
					status = deviceMalloc((void**)&GPUDataAddr_WallNormal_Edge_Type6, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation - Wall Normal - Outlet-Wall Edge failed ...\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Memory copy from host (Data_dbl_WallNormal_Edge_Type6) to Device (GPUDataAddr_WallNormal_Edge_Type6)
					status = deviceMemcpy(GPUDataAddr_WallNormal_Edge_Type6, Data_dbl_WallNormal_Edge_Type6, MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer (Wall Normal - Outlet-Wall Edge) Host To Device failed\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Free-up memory
					delete[] Data_dbl_WallNormal_Edge_Type6;
				}
				//----------------------------------------

				// 4.
				start_index = start_Index_Inner_Type2;
				site_count = total_numElements_Inner_Type2;

				if(site_count!=0){
					Data_dbl_WallNormal_Inner_Type2 = new distribn_t[3*site_count];

					for (site_t site_index=start_index; site_index < start_index + site_count; site_index++){

						// Get the components of the normal vector to the wall surface
						geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_index);
						util::Vector3D<distribn_t> normalToWall = site.GetWallNormal();

						site_t shifted_Ind = site_index-start_index;
						Data_dbl_WallNormal_Inner_Type2[3*shifted_Ind] = normalToWall.x;
						Data_dbl_WallNormal_Inner_Type2[3*shifted_Ind +1] = normalToWall.y;
						Data_dbl_WallNormal_Inner_Type2[3*shifted_Ind +2] = normalToWall.z;

					}

					// Allocate memory on the GPU (global memory)
					site_t MemSz = 3*site_count *  sizeof(distribn_t);
					status = deviceMalloc((void**)&GPUDataAddr_WallNormal_Inner_Type2, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation - Wall Normal - Wall Inner failed ...\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Memory copy from host (Data_dbl_WallNormal_Edge_Type6) to Device (GPUDataAddr_WallNormal_Edge_Type6)
					status = deviceMemcpy(GPUDataAddr_WallNormal_Inner_Type2, Data_dbl_WallNormal_Inner_Type2, MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer (Wall Normal - Wall Inner) Host To Device failed\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Free-up memory
					delete[] Data_dbl_WallNormal_Inner_Type2;
				}
				//----------------------------------------

				// 5.
				start_index = start_Index_Inner_Type5;
				site_count = total_numElements_Inner_Type5;

				if(site_count!=0){
					Data_dbl_WallNormal_Inner_Type5 = new distribn_t[3*site_count];

					for (site_t site_index=start_index; site_index < start_index + site_count; site_index++){

						// Get the components of the normal vector to the wall surface
						geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_index);
						util::Vector3D<distribn_t> normalToWall = site.GetWallNormal();

						site_t shifted_Ind = site_index-start_index;
						Data_dbl_WallNormal_Inner_Type5[3*shifted_Ind] = normalToWall.x;
						Data_dbl_WallNormal_Inner_Type5[3*shifted_Ind +1] = normalToWall.y;
						Data_dbl_WallNormal_Inner_Type5[3*shifted_Ind +2] = normalToWall.z;
					}

					// Allocate memory on the GPU (global memory)
					site_t MemSz = 3*site_count *  sizeof(distribn_t);
					status = deviceMalloc((void**)&GPUDataAddr_WallNormal_Inner_Type5, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation - Wall Normal - Inlet-Wall Inner failed ...\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Memory copy from host (Data_dbl_WallNormal_Edge_Type6) to Device (GPUDataAddr_WallNormal_Edge_Type6)
					status = deviceMemcpy(GPUDataAddr_WallNormal_Inner_Type5, Data_dbl_WallNormal_Inner_Type5, MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer (Wall Normal - Inlet-Wall Inner) Host To Device failed\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Free-up memory
					delete[] Data_dbl_WallNormal_Inner_Type5;
				}
				//----------------------------------------

				// 6.
				start_index = start_Index_Inner_Type6;
				site_count = total_numElements_Inner_Type6;

				if(site_count!=0){
					Data_dbl_WallNormal_Inner_Type6 = new distribn_t[3*site_count];

					for (site_t site_index=start_index; site_index < start_index + site_count; site_index++){

						// Get the components of the normal vector to the wall surface
						geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_index);
						util::Vector3D<distribn_t> normalToWall = site.GetWallNormal();

						site_t shifted_Ind = site_index-start_index;
						Data_dbl_WallNormal_Inner_Type6[3*shifted_Ind] = normalToWall.x;
						Data_dbl_WallNormal_Inner_Type6[3*shifted_Ind +1] = normalToWall.y;
						Data_dbl_WallNormal_Inner_Type6[3*shifted_Ind +2] = normalToWall.z;
					}

					// Allocate memory on the GPU (global memory)
					site_t MemSz = 3*site_count *  sizeof(distribn_t);
					status = deviceMalloc((void**)&GPUDataAddr_WallNormal_Inner_Type6, MemSz);
					if(!status){
						fprintf(stderr, "GPU memory allocation - Wall Normal - Outlet-Wall Inner failed ...\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Memory copy from host (Data_dbl_WallNormal_Edge_Type6) to Device (GPUDataAddr_WallNormal_Edge_Type6)
					status = deviceMemcpy(GPUDataAddr_WallNormal_Inner_Type6, Data_dbl_WallNormal_Inner_Type6, MemSz, memcpyHostToDevice);
					if(!status){
						fprintf(stderr, "GPU memory transfer (Wall Normal - Outlet-Wall Inner) Host To Device failed\n");
						initialise_GPU_WallShearStress_res = false; return initialise_GPU_WallShearStress_res;
					}

					// Free-up memory
					delete[] Data_dbl_WallNormal_Inner_Type6;
				}
				//----------------------------------------

				//======================================================================

				return initialise_GPU_WallShearStress_res;
			}



		template<class LatticeType>
			void LBM<LatticeType>::count_Iolet_ID_frequency( std::vector<int> &vect , int Iolet_ID_index, int* frequency_ret)
			{
				int count_elements = std::count (vect.begin(), vect.end(), Iolet_ID_index);
				*frequency_ret = count_elements;
			}


			//========================================================================
			// Function to return:
			// a. Number of local iolets on current processor
			// b. Vector (size = n_local_Iolets * 3) with the following elements:
			// 	b.1. Local Iolet ID
			//	b.2. Range of fluid sites associated with each one of these iolets.
			//				[min_index, max_index] : NOTE INCLUDING THE max_index !!!
			// 	i.e. [local Iolet ID #0, min_index #0, max_index #0, local Iolet ID #1, min_index #1, max_index #1, ..., local Iolet ID #(number_elements_1), min_index #(number_elements_1), max_index #(number_elements_1)]
			//				where number_elements_1 is the number of different iolets with consecutive fluid ID numbering - NOT the unique local iolet count, see how to distinguish: (a) n_LocalInlets... Vs  (b) n_unique_LocalInlets...
			//				For example, there may be a case where iolets proceed like this (with increasing fluid ID): boundary_Iolet_ID: 0 1 2 1 2 3 2 	(Irregular numbering of iolets)
			//========================================================================
		template<class LatticeType>
			std::vector<site_t> LBM<LatticeType>::identify_Range_iolets_ID(site_t first_index, site_t upper_index, int* n_local_IoletsForRange, int* n_unique_local_Iolets)
			{
				std::vector<int64_t> result_locIolet_Info;

			  // Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				std::vector<site_t> fluid_ID_sites; // vector to hold the fluid sites ID;
				std::vector<int> boundary_Iolet_ID; // vector to hold the corresponding Iolet ID;

				//======================================================================
				// 1. Loop over the fluid sites in the range [first_index, upper_index)
				// 2. Store the inlet/outlet ID (parameter boundaryId) in the vector boundary_Iolet_ID
				// 3. Store the fluid index in vector fluid_ID_sites
				for (site_t Index_Iolet_Fluid = first_index; Index_Iolet_Fluid < upper_index; Index_Iolet_Fluid++ )
				{
			    geometry::Site<geometry::LatticeData> site =mLatDat->GetSite(Index_Iolet_Fluid);
			    int boundaryId = site.GetIoletId(); // It refers to the inlet/outlet ID, e.g. for the pipe case will return boundaryId=0

					boundary_Iolet_ID.push_back(boundaryId);
					fluid_ID_sites.push_back(Index_Iolet_Fluid);

					//double ghost_dens = mOutletValues->GetBoundaryDensity(site.GetIoletId());
					//if(myPiD==3) printf("Rank: %d, site_ID = %lld, boundaryId = %d, ghost_dens = %.5f \n", myPiD, Index_Iolet_Fluid, boundaryId, ghost_dens);
				}
				//======================================================================

				//======================================================================
				std::vector<int> boundary_Iolet_ID_cp = boundary_Iolet_ID;	// boundary_Iolet_ID_cp contains the original Iolet IDs (number equal to the number of fluid sites)

			  std::vector<int>::iterator it;
			  /*
			  // Print the values:
			  std::cout << "Initial boundary_Iolet_ID contains:";
			  for (it=boundary_Iolet_ID.begin(); it!=boundary_Iolet_ID.end(); ++it)
			      std::cout << ' ' << *it;
			  std::cout << '\n';
			  */
				//----------------------------------------------------------------------
				// 1st Reduction:
				// 	Eliminates all except the first element from every consecutive group of equivalent elements from the range [first, last) and returns a past-the-end iterator for the new logical end of the range.
			  it = std::unique (boundary_Iolet_ID.begin(), boundary_Iolet_ID.end());

			  // Resizing the vector so as to remove the undefined terms (the terms ?)
			  boundary_Iolet_ID.resize( std::distance(boundary_Iolet_ID.begin(),it) );	// NOTE: It may still contain duplicates of some iolets IDs
			  int number_elements_1 = boundary_Iolet_ID.size(); // Number of local iolets (on current Rank)
			  //printf("Rank: %d, Number of elements(1st unique call): %d \n\n", myPiD, number_elements_1);
				// print out content:
				/*std::cout << "Rank: " << myPiD << " - Total local Iolets on current Rank (1st Round): " << number_elements_1 << " with boundary_Iolet_ID:";
				for (it=boundary_Iolet_ID.begin(); it!=boundary_Iolet_ID.end(); ++it)
						std::cout << ' ' << *it;
				std::cout << "\n\n";
				*/

				//----------------------------------------------------------------------
				// 2nd Reduction:
				// 	Important Check For the iolets ID:
			  // 		Check whether the Fluid IDs proceed in such a way that larger Fluid ID corresponds to a larger ONLY iolet ID, without repeating previously seen Iolet ID...
			  // 		i.e. the resized boundary_Iolet_ID proceeds in ascending order. (i.e. Completes the numbering of the fluid sites IDs before changing iolet)

				// Before beginning copy the elements contained in boundary_Iolet_ID after the first reduction (removal of continuous repeated Iolet values - may contain repeating Iolet IDs)
				std::vector<int> boundary_Iolet_ID_cp_1 = boundary_Iolet_ID; // boundary_Iolet_ID_cp_1 contains the Iolet IDs after the first reduction

				// a. Sort first the values in the resized vector
			  std::sort(boundary_Iolet_ID.begin(), boundary_Iolet_ID.end());

				// b. followed by unique and resize to remove all duplicates
			  it = std::unique(boundary_Iolet_ID.begin(), boundary_Iolet_ID.end());
			  // Resizing the vector so as to remove the undefined terms (the terms ?)
			  boundary_Iolet_ID.resize( std::distance(boundary_Iolet_ID.begin(),it) );
			  int n_unique_iolet_IDs = boundary_Iolet_ID.size(); // If ascending numbering of Fluid IDs and ioler IDs then n_unique_iolet_IDs = number_elements_1
			  //printf("Rank: %d, Number of elements(2nd unique call - after sort call): %d \n\n", myPiD, n_unique_iolet_IDs);
				/*std::cout << "Rank: " << myPiD << " - Total unique local Iolets on current Rank (2nd Round): " << n_unique_iolet_IDs << " with boundary_Iolet_ID:";
				for (it=boundary_Iolet_ID.begin(); it!=boundary_Iolet_ID.end(); ++it)
						std::cout << ' ' << *it;
				std::cout << "\n\n";
				*///----------------------------------------------------------------------

				// Case 1: Irregular numbering of iolets
				if(number_elements_1!=n_unique_iolet_IDs){
			    // printf("Fluid ID numbering jumps to a different iolet and returns ... Think about it... \n\n");

					// Need to search for the elements (Iolet IDs) contained in the vector after the 1st reduction
					// 	as there is a repetition of Iolet IDs, something like: Iolet IDs: 0, 1, 2, 1, 2, 3, 2
					// 	Hence, look for the elements in boundary_Iolet_ID_cp_1
					// 	and the fluid sites range in the original vector, i.e. boundary_Iolet_ID_cp

					// Looks through boundary_Iolet_ID_cp_1 using the ordered map of the Iolet IDs (as appears in boundary_Iolet_ID)
					std::vector<int> frequency_Iolets;
					for (int Iolet_ID_index = 0; Iolet_ID_index < n_unique_iolet_IDs; Iolet_ID_index++ ){
						int frequency=0;
						count_Iolet_ID_frequency( boundary_Iolet_ID_cp_1 , boundary_Iolet_ID[Iolet_ID_index], &frequency);
						printf("Rank: %d, Iolet ID: %d, Frequency: %d \n", myPiD, boundary_Iolet_ID[Iolet_ID_index], frequency);
						frequency_Iolets.push_back(frequency);
					}
					// Debugging:
					/*
					for (int index = 0; index < n_unique_iolet_IDs; index++){
						printf("Rank: %d, Iolet ID: %d occurs %d times \n", myPiD, boundary_Iolet_ID[index], frequency_Iolets[index]);
					}*/

					//int it_min_arr[number_elements_1]={0}; // contains the first index in the original vector boundary_Iolet_ID_cp, for each element in boundary_Iolet_ID_cp_1


					// Get the first index of each element (iolet ID) in the vector boundary_Iolet_ID_cp
					int count_shift = 0;
			    for (int i_local_iolet=0; i_local_iolet<number_elements_1; i_local_iolet++)
			    {
			      int value_search = boundary_Iolet_ID_cp_1[i_local_iolet]; // Value to search
						//printf("Rank: %d, boundary_Iolet_ID to search= %d \n", myPiD, value_search);

						// Needs to shift the beginning
						std::vector<int>::iterator it_min = std::find(boundary_Iolet_ID_cp.begin() + count_shift, boundary_Iolet_ID_cp.end(), value_search); // If element is found then it returns an iterator to the first element in the given range that’s equal to the given element, else it returns an end of the list.

						result_locIolet_Info.push_back((int64_t)value_search);

						int index_min, index_max;	// Index in the vector

					 	if (it_min != boundary_Iolet_ID_cp.end()){
						 	//std::cout << "Element Found" << std::endl;
						 	// Get index of element from iterator
						 	index_min = std::distance(boundary_Iolet_ID_cp.begin() , it_min);
						 	//printf("Rank: %d, Index_Min :%d, Fluid ID: %ld \n", myPiD, index_min, fluid_ID_sites[index_min]);

						 	// Store the info for the fluid ID in the vector to be returned
						 	result_locIolet_Info.push_back(fluid_ID_sites[index_min]);
					 	}
					 	else{
						 	std::cout << "Element Not Found" << std::endl;
						 	continue;
					 	}


						// Search for the next in line Iolet ID, as it appears in boundary_Iolet_ID_cp_1
						// Search for the upper index (element with the highest index having the boundaryId value)
			      if(i_local_iolet < (number_elements_1-1)) { // So that it can search for element boundary_Iolet_ID[i+1]
			        // Get the upper index  - Find the index of the next element (i.e. boundary_Iolet_ID_cp_1[i+1] - if it exists!!!) in the vector boundary_Iolet_ID_cp
			        std::vector<int>::iterator it_min_next = std::find(boundary_Iolet_ID_cp.begin() + count_shift, boundary_Iolet_ID_cp.end(), boundary_Iolet_ID_cp_1[i_local_iolet + 1]);
			        if (it_min_next != boundary_Iolet_ID_cp.end())
			        {
			          // Get index of element from iterator
			          index_max = index_min + std::distance(it_min,it_min_next) -1; // index_max included in the range, i.e. [index_min, index_max]
			          //printf("Rank: %d, Index_Max :%d, Fluid ID: %ld \n\n", myPiD, index_max, fluid_ID_sites[index_max]);

								// Store the info for the fluid ID in the vector to be returned
								result_locIolet_Info.push_back(fluid_ID_sites[index_max]);
			        }
			        else{
			          std::cout << "Element Not Found" << std::endl;
			        }
			      }
			      else{
			        int index_max = index_min + std::distance(it_min,boundary_Iolet_ID_cp.end()) -1;
			        //printf("Rank: %d, Single/Last element!!! Index_Max :%d, Fluid ID: %ld \n\n", myPiD, index_max, fluid_ID_sites[index_max]);
							// Store the info for the fluid ID in the vector to be returned
							result_locIolet_Info.push_back(fluid_ID_sites[index_max]);
			      }

						// Needs to update the count_shift, so that we skip the repeated values that we searched already
						count_shift += (index_max - index_min) +1;
						// printf("count shift = %d \n\n", count_shift);
					}


				} // Closes the if(number_elements_1!=n_unique_iolet_IDs) - Irregular numbering (indexing) of iolets
				else{
			    // The fluid ID numbering increases with increasing iolet ID - Regular numbering (indexing) of iolets
			    // Find the FLUID ID range associated with each iolet ID: [min_index, max_index] : NOTE INCLUDING THE max_index !!!

					/*// print out content:
			    std::cout << "Rank: " << myPiD << " - Total local Iolets on current Rank: " << number_elements_1 << " with boundary_Iolet_ID:";
			    for (it=boundary_Iolet_ID.begin(); it!=boundary_Iolet_ID.end(); ++it)
			        std::cout << ' ' << *it;
			    std::cout << "\n\n";
					*/

			    // Get the first index of each element (iolet ID) in the vector boundary_Iolet_ID_cp
			    for (int i_local_iolet=0; i_local_iolet<number_elements_1; i_local_iolet++)
			    {

			      int value_search = boundary_Iolet_ID[i_local_iolet]; // Value to search
						//printf("Rank: %d, boundary_Iolet_ID = %d \n", myPiD, value_search);

			      std::vector<int>::iterator it_min = std::find(boundary_Iolet_ID_cp.begin(), boundary_Iolet_ID_cp.end(), value_search); // If element is found then it returns an iterator to the first element in the given range that’s equal to the given element, else it returns an end of the list.

						result_locIolet_Info.push_back((site_t)value_search);

			      int index_min, index_max;	// Index in the vector

			      if (it_min != boundary_Iolet_ID_cp.end()){
			        //std::cout << "Element Found" << std::endl;
			        // Get index of element from iterator
			        index_min = std::distance(boundary_Iolet_ID_cp.begin(), it_min);
			        //printf("Rank: %d, Index_Min :%d, Fluid ID: %lld \n", myPiD, index_min, fluid_ID_sites[index_min]);

							// Store the info for the fluid ID in the vector to be returned
							result_locIolet_Info.push_back(fluid_ID_sites[index_min]);
			      }
			      else{
			        std::cout << "Element Not Found" << std::endl;
			        continue;
			      }


						// Search for the upper index (element with the highest index having the boundaryId value)
			      if(i_local_iolet < (number_elements_1-1)) { // So that it can search for element boundary_Iolet_ID[i+1]
			        // Get the upper index  - Find the index of the next element (i.e. boundary_Iolet_ID[i+1] - if it exists!!!) in the vector boundary_Iolet_ID_cp
			        std::vector<int>::iterator it_min_next = std::find(boundary_Iolet_ID_cp.begin(), boundary_Iolet_ID_cp.end(), boundary_Iolet_ID[i_local_iolet + 1]);
			        if (it_min_next != boundary_Iolet_ID_cp.end())
			        {
			          // Get index of element from iterator
			          index_max = index_min + std::distance(it_min,it_min_next) - 1;
			          //printf("Rank: %d, Index_Max :%d, Fluid ID: %lld \n\n", myPiD, index_max, fluid_ID_sites[index_max]);

								// Store the info for the fluid ID in the vector to be returned
								result_locIolet_Info.push_back(fluid_ID_sites[index_max]);
			        }
			        else{
			          std::cout << "Element Not Found" << std::endl;
			        }
			      }
			      else{
			        int index_max = index_min + std::distance(it_min,boundary_Iolet_ID_cp.end()) - 1;
			        //printf("Rank: %d, Single/Last element!!! Index_Max :%d, Fluid ID: %lld \n\n", myPiD, index_max, fluid_ID_sites[index_max]);
							// Store the info for the fluid ID in the vector to be returned
							result_locIolet_Info.push_back(fluid_ID_sites[index_max]);
			      }
			    } //Closes the loop over the unique number of elements in boundary_Iolet_ID

			  } // Closes the case of fluid ID numbering increases with increasing iolet ID
				//======================================================================

				/*// Code development/debugging phase - Remove later
				// print out content:
				std::vector<site_t>::iterator it_ret;
				printf("==============================================================\n");
				std::cout << "Rank: " << myPiD << " - Contents of returned vector: ";
				for (it_ret=result_locIolet_Info.begin(); it_ret!=result_locIolet_Info.end(); ++it_ret)
						std::cout << ' ' << *it_ret;
				std::cout << "\n\n";
				printf("==============================================================\n");
				*/


				// The value returned here should not be the unique number of Iolets on the RANK (this is the value n_unique_iolet_IDs)
				*n_local_IoletsForRange = number_elements_1;

				// The value of unique local Iolets
				*n_unique_local_Iolets = n_unique_iolet_IDs;

				return result_locIolet_Info;
			} // Ends the function



#endif



	// IZ- Nov 2023 - Added for the Checkpointing functionality
	template<class LatticeType>
		void LBM<LatticeType>::SetInitialConditions(const net::IOCommunicator& ioComms)
		{
			//InitialCondition icond = InitialCondition::FromConfig(mSimConfig->GetInitialCondition());
			auto icond = InitialCondition::FromConfig(mSimConfig->GetInitialCondition());
			icond.SetFs<LatticeType>(mLatDat, ioComms, mState);
			//icond.SetTime(mState);

			//---------------
			// Testing - Remove later
			// uint64_t time_currentStep = mState->GetTimeStep();
			// printf("Current Time-Step as set in SetInitialConditions (lb.hpp) %ld \n", time_currentStep);
			//---------------
		}

/** JM Method before trying to bring Checkpointing in
		template<class LatticeType>
			void LBM<LatticeType>::SetInitialConditions()
			{
				distribn_t density = mUnits->ConvertPressureToLatticeUnits(mSimConfig->GetInitialPressure()) / Cs2;

				for (site_t i = 0; i < mLatDat->GetLocalFluidSiteCount(); i++)
				{
					distribn_t f_eq[LatticeType::NUMVECTORS];

					LatticeType::CalculateFeq(density, 0.0, 0.0, 0.0, f_eq);

					distribn_t* f_old_p = mLatDat->GetFOld(i * LatticeType::NUMVECTORS);
					distribn_t* f_new_p = mLatDat->GetFNew(i * LatticeType::NUMVECTORS);

				//	if(i==mLatDat->GetLocalFluidSiteCount()-1) std::printf("Fluid Site %d \n", i);

					for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++)
					{
						f_new_p[l] = f_old_p[l] = f_eq[l];
				//		if(i==mLatDat->GetLocalFluidSiteCount()-1) std::printf("Distribution Functions: f_new_p[%d] = %.5f , f_old_p = %.5f, f_eq = %.5f \n\n", l, f_new_p[l], f_old_p[l], f_eq[l]);
					}
				}
			}
**/


		template<class LatticeType>
			void LBM<LatticeType>::RequestComms()
			{
				timings[hemelb::reporting::Timers::lb].Start();

				// Delegate to the lattice data object to post the asynchronous sends and receives
				// (via the Net object).
				// NOTE that this doesn't actually *perform* the sends and receives, it asks the Net
				// to include them in the ISends and IRecvs that happen later.
				mLatDat->SendAndReceive(mNet);
/*
#ifdef HEMELB_USE_GPU
				// Calculate density and momentum (velocity) from the distr. functions
				// Ensure that the Swap operation at the end of the previous time-step has completed
				// Synchronisation barrier or maybe use the same cuda stream (stream_ReceivedDistr)
				//bool status;


				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				//if (myPiD!=0) deviceStreamSynchronize(stream_ReceivedDistr);

				//----------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_CalcMacroVars = 128;				//Number of threads per block for calculating MacroVariables
				dim3 nThreadsCalcMacroVars(nThreadsPerBlock_CalcMacroVars);

				// Number of fluid nodes:
				site_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount();
				site_t first_Index = 0;
				site_t site_Count = nFluid_nodes;

				int nBlocksCalcMacroVars = nFluid_nodes/nThreadsPerBlock_CalcMacroVars			+ (( nFluid_nodes % nThreadsPerBlock_CalcMacroVars > 0)         ? 1 : 0);
				//----------------------------------


				// To access the data in GPU global memory nArr_dbl is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)
				// nArr_dbl = (mLatDat->GetLocalFluidSiteCount()) = nFluid_nodes
				if(nBlocksCalcMacroVars!=0)
					hemelb::GPU_CalcMacroVars <<<nBlocksCalcMacroVars, nThreadsCalcMacroVars, 0, stream_ReceivedDistr>>> ( 	(distribn_t*)GPUDataAddr_dbl_fOld_b,
																																																									(distribn_t*)GPUDataAddr_dbl_MacroVars,
																																																									nFluid_nodes, first_Index, (first_Index + site_Count)); //

#endif
*/


				timings[hemelb::reporting::Timers::lb].Stop();
			}


		template<class LatticeType>
			void LBM<LatticeType>::PreSend()
			{
				timings[hemelb::reporting::Timers::lb].Start();
				timings[hemelb::reporting::Timers::lb_calc].Start();

				/**
				 * In the PreSend phase, we do LB on all the sites that need to have results sent to
				 * neighbouring ranks ('domainEdge' sites). In site id terms, this means we start at the
				 * end of the sites whose neighbours all lie on this rank ('midDomain'), then progress
				 * through the sites of each type in turn.
				 */

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

#ifdef HEMELB_USE_GPU	// If exporting computation on GPUs
				// Boolean variable for sending macroVariables to GPU global memory (avoids the if statement time%_Send_MacroVars_DtH==0 in the GPU kernels)
				// Consider using: a) propertyCache.densityCache.RequiresRefresh() || propertyCache.velocityCache.RequiresRefresh() || propertyCache.wallShearStressMagnitudeCache.RequiresRefresh()
				bool Write_GlobalMem = (mState->GetTimeStep()%frequency_WriteGlobalMem == 0) ? 1 : 0;

				// Get the HEMELB_KERNEL (ie LBGK, LBGKSL etc)
				const std::string hemeKernel = QUOTE_CONTENTS(HEMELB_KERNEL);

				// Before the collision starts make sure that the swap of distr. functions at the previous step has Completed
				//if (myPiD!=0) deviceStreamSynchronize(stream_SwapOldAndNew);
				if (myPiD!=0) deviceStreamSynchronize(stream_ReceivedDistr);
				//----------------------------------------------------------------------

				//----------------------------------------------------------------------
				// Check the stability of the Code// Insert a GPU kernel launch here that checks for NaN values
				// Just check the density, as any NaN values will eventually affect all variables
				site_t offset_test = 0;	// site_t is type int64_t
				site_t nFluid_nodes_test = mLatDat->GetLocalFluidSiteCount();
				site_t first_Index_test = offset_test;
				site_t site_Count_test = nFluid_nodes_test;
				//----------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_Check = 128;				//Number of threads per block for checking the stability of the simulation
				dim3 nThreads_Check(nThreadsPerBlock_Check);
				// Number of fluid nodes involved in the collision/streaming : mLatDat->GetDomainEdgeCollisionCount(0)
				int nBlocks_Check = (site_Count_test)/nThreadsPerBlock_Check			+ ((site_Count_test % nThreadsPerBlock_Check > 0)         ? 1 : 0);


				// TODO: Frequency of checking stability set to 200 / 1000 time-steps. Modify/Check again in the future!
				if(nBlocks_Check!=0 && mState->GetTimeStep()%1000 ==0){
						hemelb::GPU_Check_Stability <<< nBlocks_Check, nThreads_Check, 0, stability_check_stream >>> (	(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																									(int*)mLatDat->d_Stability_GPU_mLatDat,
																									 nFluid_nodes_test,
																									 first_Index_test, (first_Index_test + site_Count_test), mState->GetTimeStep()); // (int64_t*)GPUDataAddr_int64_Neigh_b
						/*
						// Debugging purposes - Remove later and move in PreReceive()
						// 	(get the value returned by the kernel GPU_Check_Stability if unstable simulation -see StabilityTester.h)
						deviceStreamSynchronize(stability_check_stream);
						// MemCopy from Device To Host the value for the Stability - TODO!!!
						status = deviceMemcpyAsync( &(mLatDat->h_Stability_GPU_mLatDat), &(((int*)mLatDat->d_Stability_GPU_mLatDat)[0]), sizeof(int), memcpyDeviceToHost, stability_check_stream);
						//printf("Rank = %d - Host Stability flag: %d \n\n", myPiD, mLatDat->h_Stability_GPU_mLatDat);
						*/
				}
			 //-----------------------------------------------------------------------



				//#####################################################################################################################################################
				// Merge the first 2 Types of collision-streaming
				// Collision Type 1:
				site_t offset = mLatDat->GetMidDomainSiteCount();	// site_t is type int64_t
				site_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount();

				site_t first_Index = offset;
				site_t site_Count_MidFluid = mLatDat->GetDomainEdgeCollisionCount(0);
				site_t site_Count_Wall = mLatDat->GetDomainEdgeCollisionCount(1);

				site_t site_Count = site_Count_MidFluid + site_Count_Wall;

				//if (myPiD!=0) printf("Rank: %d, Collision 1 & 2: First Index MidFluid: %lld, Upper Index MidFluid: %lld, First Index Wall: %lld, Upper Index Wall: %lld  \n\n",myPiD, first_Index, (first_Index+site_Count_MidFluid),
			 	//											(first_Index + site_Count_MidFluid), (first_Index + site_Count_MidFluid + site_Count_Wall));


				//----------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_Collide = 128;				//Number of threads per block for the Collision step
				dim3 nThreads_Collide(nThreadsPerBlock_Collide);
				// Number of fluid nodes involved in the collision/streaming : mLatDat->GetDomainEdgeCollisionCount(0)
				int nBlocks_Collide = (site_Count)/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
				//----------------------------------

				// To access the data in GPU global memory:
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)
				if(nBlocks_Collide!=0){
					if (hemeKernel == "LBGKSL"){
						hemelb::GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_WallShearStress<<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_1>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(site_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Wall,
								nFluid_nodes,
								first_Index, (first_Index + site_Count_MidFluid),
								(first_Index + site_Count_MidFluid), (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
								(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type2,
								(distribn_t*)GPUDataAddr_WallNormal_Edge_Type2,
								mState->GetTimeStep(), myPiD,
								(distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
							);
					}
					else if(hemeKernel == "LBGK"){
					hemelb::GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_WallShearStress<<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_1>>> (
							(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
							(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
							(distribn_t*)GPUDataAddr_dbl_MacroVars,
							(site_t*)GPUDataAddr_int64_Neigh_d,
							(uint32_t*)GPUDataAddr_uint32_Wall,
							nFluid_nodes,
							first_Index, (first_Index + site_Count_MidFluid),
							(first_Index + site_Count_MidFluid), (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
							(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type2,
							(distribn_t*)GPUDataAddr_WallNormal_Edge_Type2,
							mState->GetTimeStep(), myPiD);
					}
				}
				/*if(nBlocks_Collide!=0)
					hemelb::GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_1>>> (
							(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
							(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
							(distribn_t*)GPUDataAddr_dbl_MacroVars,
							(site_t*)GPUDataAddr_int64_Neigh_d,
							(uint32_t*)GPUDataAddr_uint32_Wall,
							nFluid_nodes,
							first_Index, (first_Index + site_Count_MidFluid),
							(first_Index + site_Count_MidFluid), (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem);
				*/																									//#####################################################################################################################################################



				/*
				// ====================================================================================================================================================
				// Collision Type 1:
				site_t offset = mLatDat->GetMidDomainSiteCount();
				// printf("Rank: %d: Collision Type 1: Starting = %lld, SiteCount = %lld, Ending = %lld \n\n", myPiD, offset, mLatDat->GetDomainEdgeCollisionCount(0), (offset + mLatDat->GetDomainEdgeCollisionCount(0)));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				int64_t first_Index = offset;	// Start Fluid Index
				int64_t site_Count = mLatDat->GetDomainEdgeCollisionCount(0);

				//----------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_Collide = 128;				//Number of threads per block for the Collision step
				dim3 nThreads_Collide(nThreadsPerBlock_Collide);
				// Number of fluid nodes involved in the collision/streaming : mLatDat->GetDomainEdgeCollisionCount(0)
				int nBlocks_Collide = (mLatDat->GetDomainEdgeCollisionCount(0))/nThreadsPerBlock_Collide			+ ((mLatDat->GetDomainEdgeCollisionCount(0) % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
				//----------------------------------

				// To access the data in GPU global memory nArr_dbl is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)
				// nArr_dbl = (mLatDat->GetLocalFluidSiteCount()) = nFluid_nodes
				if(nBlocks_Collide!=0)
					hemelb::GPU_CollideStream_1_PreReceive_SaveMacroVars<<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_1>>> ( (double*)GPUDataAddr_dbl_fOld_b,
																																																														(double*)GPUDataAddr_dbl_fNew_b,
																																																														(double*)GPUDataAddr_dbl_MacroVars,
																																																														(int64_t*)GPUDataAddr_int64_Neigh_d,
																																																														(mLatDat->GetLocalFluidSiteCount()),
																																																														first_Index,
																																																														(first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep()); //
				//--------------------------------------------------------------------------------------------------------------------------------------------------
				// ====================================================================================================================================================
				*/

				// ====================================================================================================================================================
				// Place this here so that it overlaps with the calculations on the GPU for the Collision-streaming type 1...
				// Actually control is returned back to the CPU just after the launch of the kernel... Check where it would be best to place the function...

				// Inlets:
				// Receive values for Inlet
				// 		see lb/iolets/BoundaryValues.cc : calls Wait if it needs to receive information for the Iolets: GetLocalIolet(i)->GetComms()->Wait();
				mInletValues->FinishReceive();

				// Outlets:
				// Receive values for Outlet
				mOutletValues->FinishReceive();

				//**********************************************************************
				/*// Get the type of Iolet BCs from the CMake file compiling options
				std::string hemeIoletBC_Inlet, hemeIoletBC_Outlet;
				get_Iolet_BCs(hemeIoletBC_Inlet, hemeIoletBC_Outlet);
				*/
				// Check If I can get the type of Iolet BCs from the CMake file
				#define QUOTE_RAW(x) #x
				#define QUOTE_CONTENTS(x) QUOTE_RAW(x)

				std::string hemeIoletBC_Inlet, hemeIoletBC_Outlet;
				hemeIoletBC_Inlet = QUOTE_CONTENTS(HEMELB_INLET_BOUNDARY);
				//printf("Function Call: Inlet Type of BCs: hemeIoletBC_Inlet: %s \n", hemeLB_IoletBC_Inlet.c_str()); //note the use of c_str

				hemeIoletBC_Outlet = QUOTE_CONTENTS(HEMELB_OUTLET_BOUNDARY);
				//printf("Function Call: Outlet Type of BCs: hemeIoletBC_Outlet: %s \n\n", hemeLB_IoletBC_Outlet.c_str()); //note the use of c_str

				//----------------------------------------------------------------------
				// Iolets - general details
				//	Total GLOBAL iolets: n_Inlets = mInletValues->GetTotalIoletCount();
				int n_Inlets = mInletValues->GetTotalIoletCount();

				//	Total GLOBAL iolets: n_Outlets = mOutletValues->GetTotalIoletCount();
				int n_Outlets = mOutletValues->GetTotalIoletCount();
				//----------------------------------------------------------------------

				lb::MacroscopicPropertyCache& propertyCache = GetPropertyCache();

				// Inlets BCs
				if(hemeIoletBC_Inlet == "LADDIOLET"){
					//printf("Entering the LaddIolet loop \n\n");
					//--------------------------------------------------------------------
					/** Apply Velocity Boundary Conditions
								At the moment for the Vel BCs case (LADDIOLET) - subtype File
									Implement evaluating the wall momentum correction term on the GPU

								Loop over the number of local iolets (not the unique iolets, but the local iolets -
									remember the irregular indexing that can appear in the iolets listing,
									see function identify_Range_iolets_ID) with their corresponding Iolet index
					 */

					// 25 April 2023
					//	New approach for evaluating the wall momentum correction terms using the
					//	geometric prefactor.
					// 	Loop through the fluid sites in each of the collision streaming types
					// Uses kernels: GPU_WallMom_correction_File_prefactor & GPU_WallMom_correction_File_prefactor_v2
					//if (myPiD!=0) apply_Vel_BCs_File_GetWallMom_correction_ApprPref();

					// 30 April 2023
					//	New approach for evaluating the wall momentum correction terms using the
					//	geometric prefactor - Avoid searching for the iolet ID - Pass as an argument to the GPU kernel (should be faster)
					// 	Loop through the iolets list using the information from the struct Iolets - No need to search for the iolet ID this way (pass as argument to the GPU kernel)
					//if (myPiD!=0) apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch();

					// Split in 2 stages - First PreSend and then PreReceive related wall momentum correction calculations to enable higher degree of overlap
					// Uses kernel GPU_WallMom_correction_File_prefactor_NoIoletIDSearch.
					if (myPiD!=0) apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch_PreSend();

					//--------------------------------------------------------------------

					//====================================================================
					// Domain Edge
					// Collision Type 3 (mInletCollision):
					site_t start_Index_Inlet_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
					site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
					if (site_Count_Inlet_Edge!=0){
						// Jan.2023 - If we use void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction()
						// Comment out the following
						//------------------------------------------------------------------
						/*
						// Debugging - 24 April 2023
						// Directly getting the correction term without passing through propertyCache
						GetWallMom_correction_Direct(mInletCollision, start_Index_Inlet_Edge, site_Count_Inlet_Edge, propertyCache, wallMom_correction_Inlet_Edge_Direct);
						wallMom_correction_Inlet_Edge_Direct.resize(site_Count_Inlet_Edge*LatticeType::NUMVECTORS);
						//printf("\n Entering section 1: wallMom_correction_Inlet_Edge \n\n");
						*/

						/*
						// Function to allocate memory on the GPU's global memory for the wallMom
						//memCpy_HtD_GPUmem_WallMom_correction(start_Index_Inlet_Edge, site_Count_Inlet_Edge, wallMom_correction_Inlet_Edge_Direct, GPUDataAddr_wallMom_correction_Inlet_Edge);
						memCpy_HtD_GPUmem_WallMom_correction_cudaStream(start_Index_Inlet_Edge, site_Count_Inlet_Edge, wallMom_correction_Inlet_Edge_Direct,
																														GPUDataAddr_wallMom_correction_Inlet_Edge, Collide_Stream_PreSend_3);
						//------------------------------------------------------------------
						*/
					}

					// Collision Type 5 (mInletWallCollision):
					site_t start_Index_InletWall_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
					                                    + mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);
					site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
					if (site_Count_InletWall_Edge!=0){
						// Jan.2023 - If we use void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction()
						// Comment out the following
						//------------------------------------------------------------------
						// Directly getting the correction term without passing through propertyCache
/*						GetWallMom_correction_Direct(mInletWallCollision, start_Index_InletWall_Edge, site_Count_InletWall_Edge, propertyCache, wallMom_correction_InletWall_Edge_Direct);
						wallMom_correction_InletWall_Edge_Direct.resize(site_Count_InletWall_Edge*LatticeType::NUMVECTORS);
						//printf("\n Entering section 2: wallMom_correction_InletWall_Edge \n\n");

						// Function to allocate memory on the GPU's global memory for the wallMom
						//memCpy_HtD_GPUmem_WallMom_correction(start_Index_InletWall_Edge, site_Count_InletWall_Edge, wallMom_correction_InletWall_Edge_Direct, GPUDataAddr_wallMom_correction_InletWall_Edge);
						memCpy_HtD_GPUmem_WallMom_correction_cudaStream(start_Index_InletWall_Edge, site_Count_InletWall_Edge, wallMom_correction_InletWall_Edge_Direct,
																														GPUDataAddr_wallMom_correction_InletWall_Edge, Collide_Stream_PreSend_5);
						*/
						//------------------------------------------------------------------
					}

				} // Ends the if(hemeIoletBC_Inlet == "LADDIOLET") loop
				else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
					//--------------
					// Approach: Use pinned memory (IZ - July 2024)
					int n_bytes = n_Inlets * sizeof(distribn_t);

					// Allocate memory only the first time
					if (h_ghostDensity_inlet == nullptr){
						bool status = deviceHostAlloc((void**)&h_ghostDensity_inlet, n_bytes);
						memset(h_ghostDensity_inlet, 0, n_bytes);
						if(!status){
							fprintf(stderr,"deviceHostAlloc (pinned mem) for h_ghostDensity_inlet failed... Rank = %d, Time = %d \n", myPiD, mState->GetTimeStep());
						}
					}
					//--------------

					// Proceed with the collision type if the number of fluid nodes involved is not ZERO - HtD memcopy
					// This (n_Inlets) refers to the total number of inlets globally. NOT on local RANK - SHOULD REPLACE THIS with the local number of inlets
					if (n_Inlets!=0){
						for (int i=0; i<n_Inlets; i++){
							h_ghostDensity_inlet[i] = mInletValues->GetBoundaryDensity(i);
							//std::cout << "Cout: GhostDensity : " << h_ghostDensity_inlet[i] << std::endl;
						}
						if (myPiD!=0){ // MemCopy cudaMemcpyHostToDevice only if rank!=0
							// Memory copy from host (h_ghostDensity_inlet) to Device (d_ghostDensity)
							//cudaStatus = cudaMemcpy(d_ghostDensity, h_ghostDensity_inlet, n_Inlets * sizeof(distribn_t), cudaMemcpyHostToDevice);
							bool status = deviceMemcpyAsync(d_ghostDensity, h_ghostDensity_inlet, n_Inlets * sizeof(distribn_t), memcpyHostToDevice, stream_ghost_dens_inlet);
							if(!status){ fprintf(stderr, "GPU memory transfer (ghostDensity inlet) Host To Device failed\n"); //return false;
						}
					}
					//if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function.
				} // Closes the if n_Inlets!=0
			} // Closes the if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET")
			//====================================================================

				//----------------------------------------------------------------------
				// Outlets BCs
				// Jan.2023 - The appropriate modifications were not applied in void LBM<LatticeType>::apply_Vel_BCs_File_GetWallMom_correction()
				// Hence, DO NOT Comment out the following
				if(hemeIoletBC_Outlet == "LADDIOLET"){

					// Domain Edge
					// Collision Type 4 (mOutletCollision):
					site_t start_Index_Outlet_Edge = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1) + mLatDat->GetDomainEdgeCollisionCount(2);
					site_t site_Count_Outlet_Edge = mLatDat->GetDomainEdgeCollisionCount(3);
					if (site_Count_Outlet_Edge!=0){
						// Directly getting the correction term without passing through propertyCache
						GetWallMom_correction_Direct(mOutletCollision, start_Index_Outlet_Edge, site_Count_Outlet_Edge, propertyCache, wallMom_correction_Outlet_Edge_Direct); // Fills the propertyCache.wallMom_Cache
						wallMom_correction_Outlet_Edge_Direct.resize(site_Count_Outlet_Edge * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						//memCpy_HtD_GPUmem_WallMom_correction(start_Index_Outlet_Edge, site_Count_Outlet_Edge, wallMom_correction_Outlet_Edge, GPUDataAddr_wallMom_correction_Outlet_Edge);
						memCpy_HtD_GPUmem_WallMom_correction_cudaStream(start_Index_Outlet_Edge, site_Count_Outlet_Edge, wallMom_correction_Outlet_Edge_Direct,
																														GPUDataAddr_wallMom_correction_Outlet_Edge, Collide_Stream_PreSend_4);

					}

					// Collision Type 6 (mOutletWallCollision):
					site_t start_Index_OutletWall_Edge = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
					                                    + mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3) + mLatDat->GetDomainEdgeCollisionCount(4);
					site_t site_Count_OutletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(5);
					if (site_Count_OutletWall_Edge!=0){
						// Directly getting the correction term without passing through propertyCache
						GetWallMom_correction_Direct(mOutletWallCollision, start_Index_OutletWall_Edge, site_Count_OutletWall_Edge, propertyCache, wallMom_correction_OutletWall_Edge_Direct); // Fills the propertyCache.wallMom_Cache
						wallMom_correction_OutletWall_Edge_Direct.resize(site_Count_OutletWall_Edge * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						//memCpy_HtD_GPUmem_WallMom_correction(start_Index_OutletWall_Edge, site_Count_OutletWall_Edge, wallMom_correction_OutletWall_Edge, GPUDataAddr_wallMom_correction_OutletWall_Edge);
						memCpy_HtD_GPUmem_WallMom_correction_cudaStream(start_Index_OutletWall_Edge, site_Count_OutletWall_Edge, wallMom_correction_OutletWall_Edge_Direct,
																														GPUDataAddr_wallMom_correction_OutletWall_Edge, Collide_Stream_PreSend_6);

					}

				}
				else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){

					//--------------
					// Approach: Use pinned memory (IZ - July 2024)
					int n_bytes = n_Outlets * sizeof(distribn_t);

					// Allocate memory only the first time
					if (h_ghostDensity_outlet == nullptr){
						bool status = deviceHostAlloc((void**)&h_ghostDensity_outlet, n_bytes);
						memset(h_ghostDensity_outlet, 0, n_bytes);
						if(!status){
							fprintf(stderr,"deviceHostAlloc (pinned mem) for h_ghostDensity_inlet outlet... Rank = %d, Time = %d \n", myPiD, mState->GetTimeStep());
						}
					}
					//--------------

					// Proceed with the collision type if the number of fluid nodes involved is not ZERO - HtD memcopy
					// This (n_Inlets) refers to the total number of inlets globally. NOT on local RANK - SHOULD REPLACE THIS with the local number of inlets
					if (n_Outlets!=0){
						for (int i=0; i<n_Outlets; i++){
							h_ghostDensity_outlet[i] = mOutletValues->GetBoundaryDensity(i);
							//std::cout << "Cout: GhostDensity : " << h_ghostDensity_inlet[i] << std::endl;
						}
						if (myPiD!=0){ // MemCopy cudaMemcpyHostToDevice only if rank!=0
							// Memory copy from host (h_ghostDensity_outlet) to Device (d_ghostDensity_out)
							bool status = deviceMemcpyAsync(d_ghostDensity_out, h_ghostDensity_outlet, n_Outlets * sizeof(distribn_t), memcpyHostToDevice, stream_ghost_dens_outlet);
							if(!status){ fprintf(stderr, "GPU memory transfer (ghostDensity outlet) Host To Device failed\n"); //return false;
						}
					}
					//if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function.
				} // Closes the if n_Outlets!=0
			} // Closes the if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET")
			//====================================================================

				//**********************************************************************
				// ====================================================================================================================================================

				/*
				// ====================================================================================================================================================
				// Collision Type 2:
				offset += mLatDat->GetDomainEdgeCollisionCount(0);
				// StreamAndCollide(mWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(1));

				// GPU COLLISION KERNEL:
				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetDomainEdgeCollisionCount(1);

				//-------------------------------------
				// Kernel set-up
				nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

				// To access the data in GPU global memory:
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)
				if(nBlocks_Collide!=0)
					hemelb::GPU_CollideStream_mWallCollision_sBB_PreRec <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_2>>> (	(double*)GPUDataAddr_dbl_fOld_b,
															(double*)GPUDataAddr_dbl_fNew_b,
															(double*)GPUDataAddr_dbl_MacroVars,
															(int64_t*)GPUDataAddr_int64_Neigh_d,
															(uint32_t*)GPUDataAddr_uint32_Wall,
															(mLatDat->GetLocalFluidSiteCount()),
															first_Index,
															(first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep()); // (int64_t*)GPUDataAddr_int64_Neigh_b
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				// ====================================================================================================================================================
				*/


				// ====================================================================================================================================================
				// Collision Type 3 (mInletCollision):
				offset = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1); // Write this explicitly because of the merged kernels above (mMidFluidCollision and mWallCollision)
				//offset += mLatDat->GetDomainEdgeCollisionCount(1);
				//StreamAndCollide(mInletCollision, offset, mLatDat->GetDomainEdgeCollisionCount(2));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetDomainEdgeCollisionCount(2);

				// Proceed with the collision type if the number of fluid nodes involved is not ZERO
				if (site_Count!=0){
					//-------------------------------------
					// GPU COLLISION KERNEL:
					// Kernel set-up
					nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

					//--------------------------------------------------------------------
					// TODO: Choose the appropriate kernel depending on BCs:
					// Inlets BCs
					if(hemeIoletBC_Inlet == "LADDIOLET"){
						//
						if (hemeKernel == "LBGKSL"){
							hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_3>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(int64_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Iolet,
								(mLatDat->GetLocalFluidSiteCount()),
								(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
								first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
								mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
							);
						}
						else if(hemeKernel == "LBGK"){
							hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_3>>> ( (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(distribn_t*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(mLatDat->GetLocalFluidSiteCount()),
																															(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem);
						}
						//
					}
					else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
						// Make sure it has received the values for ghost density on the GPU for the case of Pressure BCs
						if (myPiD!=0) deviceStreamSynchronize(stream_ghost_dens_inlet);	// Maybe transfer this within the loop for Press. BCs below

						if (n_LocalInlets_mInlet_Edge <=(local_iolets_MaxSIZE/3)){
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_3>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity,
									(float*)d_inletNormal,
									n_Inlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalInlets_mInlet_Edge, Inlet_Edge,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_3>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity,
									(float*)d_inletNormal,
									n_Inlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalInlets_mInlet_Edge, Inlet_Edge);
							}
							//
						}
						else{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2 <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_3>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity,
									(float*)d_inletNormal,
									n_Inlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalInlets_mInlet_Edge, (site_t*)GPUDataAddr_Inlet_Edge,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2 <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_3>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity,
									(float*)d_inletNormal,
									n_Inlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalInlets_mInlet_Edge, (site_t*)GPUDataAddr_Inlet_Edge);
							}
							//
						}
					}
					//--------------------------------------------------------------------
				}
				// ends if (site_Count!=0), Collision type 3 (mInletCollision)
				// ====================================================================================================================================================

				// ====================================================================================================================================================
				// Collision Type 4 (mOutletCollision):
				offset += mLatDat->GetDomainEdgeCollisionCount(2);
				// StreamAndCollide(mOutletCollision, offset, mLatDat->GetDomainEdgeCollisionCount(3));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetDomainEdgeCollisionCount(3);

				// Proceed with the collision type if the number of fluid nodes involved is not ZERO
				if (site_Count!=0){
					//printf("Rank: %d: Collision Type 4 (Outlet): Starting = %lld, SiteCount = %lld, Ending = %lld \n\n", myPiD, first_Index, mLatDat->GetDomainEdgeCollisionCount(3), (offset + mLatDat->GetDomainEdgeCollisionCount(3)));

					//-------------------------------------
					// GPU COLLISION KERNEL:
					// Kernel set-up
					nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

					//--------------------------------------------------------------------
					// TODO: Choose the appropriate kernel depending on BCs:
					// Inlets BCs
					if(hemeIoletBC_Outlet == "LADDIOLET"){
						//
						if (hemeKernel == "LBGKSL"){
							hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_4>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(int64_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Iolet,
								(mLatDat->GetLocalFluidSiteCount()),
								(distribn_t*)GPUDataAddr_wallMom_correction_Outlet_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
								first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
								mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
							);
						}
						else if(hemeKernel == "LBGK"){
							hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_4>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(int64_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Iolet,
								(mLatDat->GetLocalFluidSiteCount()),
								(distribn_t*)GPUDataAddr_wallMom_correction_Outlet_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
								first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem);
						}
						//
					}
					else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
						// Make sure it has received the values for ghost density on the GPU
						if (myPiD!=0) deviceStreamSynchronize(stream_ghost_dens_outlet);

						if(n_LocalOutlets_mOutlet_Edge<=(local_iolets_MaxSIZE/3))
						{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_4>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity_out,
									(float*)d_outletNormal,
									n_Outlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalOutlets_mOutlet_Edge, Outlet_Edge,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_4>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity_out,
									(float*)d_outletNormal,
									n_Outlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalOutlets_mOutlet_Edge, Outlet_Edge);
							}
							//
						}
						else{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2 <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_4>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity_out,
									(float*)d_outletNormal,
									n_Outlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalOutlets_mOutlet_Edge, (site_t*)GPUDataAddr_Outlet_Edge,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2 <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_4>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity_out,
									(float*)d_outletNormal,
									n_Outlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalOutlets_mOutlet_Edge, (site_t*)GPUDataAddr_Outlet_Edge);
							}
							//
						}

					}
					//---------------------------------------------------------------------------------------------------------------------------------------------------
				}
				// ends the if site_Count!=0, for Collision Type 4.
				// ====================================================================================================================================================

				// ====================================================================================================================================================
				// Collision Type 5:
				// Walls BCs:	Simple Bounce-Back
				// Inlet/Outlet BCs:	NashZerothOrderPressure - Specify the ghost density for each inlet/outlet
				offset += mLatDat->GetDomainEdgeCollisionCount(3);
				// printf("Rank: %d: Collision Type 5: Starting = %lld, Ending = %lld, site Count = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(4)), mLatDat->GetDomainEdgeCollisionCount(4));
				// StreamAndCollide(mInletWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(4));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetDomainEdgeCollisionCount(4);

				// Proceed with the collision type if the number of fluid nodes involved is not ZERO
				if (site_Count!=0){

					// GPU COLLISION KERNEL:
					// Kernel set-up
					nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

					//--------------------------------------------------------------------
					// TODO: Choose the appropriate kernel depending on BCs:
					// Inlets BCs
					if(hemeIoletBC_Inlet == "LADDIOLET"){
						//
						if (hemeKernel == "LBGKSL"){
							hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_5>>> (
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(distribn_t*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Wall,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(mLatDat->GetLocalFluidSiteCount()),
									(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type5,
									(distribn_t*)GPUDataAddr_WallNormal_Edge_Type5,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
						}
						else if(hemeKernel == "LBGK"){
							hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_5>>> (
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(distribn_t*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Wall,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(mLatDat->GetLocalFluidSiteCount()),
									(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type5,
									(distribn_t*)GPUDataAddr_WallNormal_Edge_Type5);
						}
						//
					}
					else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){

						if(n_LocalInlets_mInletWall_Edge <=(local_iolets_MaxSIZE/3))
						{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_5>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity,
										(float*)d_inletNormal,
										n_Inlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalInlets_mInletWall_Edge, InletWall_Edge,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type5,
										(distribn_t*)GPUDataAddr_WallNormal_Edge_Type5,
										mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
									);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_5>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity,
										(float*)d_inletNormal,
										n_Inlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalInlets_mInletWall_Edge, InletWall_Edge,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type5,
										(distribn_t*)GPUDataAddr_WallNormal_Edge_Type5);
							}
							//
						}
						else{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_5>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity,
										(float*)d_inletNormal,
										n_Inlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalInlets_mInletWall_Edge, (site_t*)GPUDataAddr_InletWall_Edge,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type5,
										(distribn_t*)GPUDataAddr_WallNormal_Edge_Type5,
										mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
									);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_5>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity,
										(float*)d_inletNormal,
										n_Inlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalInlets_mInletWall_Edge, (site_t*)GPUDataAddr_InletWall_Edge,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type5,
										(distribn_t*)GPUDataAddr_WallNormal_Edge_Type5);
							}
							//
						}
					}
					//--------------------------------------------------------------------
				}
				// ends the if site_Count!=0, for Collision Type 5.
				// ====================================================================================================================================================

				// ====================================================================================================================================================
				// Collision Type 6 (mOutletWallCollision):
				// Walls BCs:	Simple Bounce-Back
				// Inlet/Outlet BCs:	NashZerothOrderPressure - Specify the ghost density for each inlet/outlet
				offset += mLatDat->GetDomainEdgeCollisionCount(4);
				// printf("Rank: %d: Collision Type 6: Starting = %lld, Ending = %lld, site Count = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(5)), mLatDat->GetDomainEdgeCollisionCount(5));
				// StreamAndCollide(mOutletWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(5));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetDomainEdgeCollisionCount(5);

				// Proceed with the collision type if the number of fluid nodes involved is not ZERO
				if (site_Count!=0){

					//-------------------------------------
					// GPU COLLISION KERNEL:
					// Kernel set-up
					nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

					//--------------------------------------------------------------------
					// TODO: Choose the appropriate kernel depending on BCs:
					// Inlets BCs
					if(hemeIoletBC_Outlet == "LADDIOLET"){
						//
						if (hemeKernel == "LBGKSL"){
							hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_6>>> (
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(distribn_t*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Wall,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(mLatDat->GetLocalFluidSiteCount()),
									(distribn_t*)GPUDataAddr_wallMom_correction_OutletWall_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type6,
									(distribn_t*)GPUDataAddr_WallNormal_Edge_Type6,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
						}
						else if(hemeKernel == "LBGK"){
							hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_6>>> (
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(distribn_t*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Wall,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(mLatDat->GetLocalFluidSiteCount()),
									(distribn_t*)GPUDataAddr_wallMom_correction_OutletWall_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type6,
									(distribn_t*)GPUDataAddr_WallNormal_Edge_Type6);
						}
						//

					}
					else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalOutlets_mOutletWall_Edge<=(local_iolets_MaxSIZE/3))
						{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_6>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity_out,
										(float*)d_outletNormal,
										n_Outlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalOutlets_mOutletWall_Edge, OutletWall_Edge,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type6,
										(distribn_t*)GPUDataAddr_WallNormal_Edge_Type6,
										mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
									);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_6>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity_out,
										(float*)d_outletNormal,
										n_Outlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalOutlets_mOutletWall_Edge, OutletWall_Edge,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type6,
										(distribn_t*)GPUDataAddr_WallNormal_Edge_Type6);
							}
							//
						}
						else{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_6>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity_out,
										(float*)d_outletNormal,
										n_Outlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalOutlets_mOutletWall_Edge, (site_t*)GPUDataAddr_OutletWall_Edge,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type6,
										(distribn_t*)GPUDataAddr_WallNormal_Edge_Type6,
										mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
									);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreSend_6>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity_out,
										(float*)d_outletNormal,
										n_Outlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalOutlets_mOutletWall_Edge, (site_t*)GPUDataAddr_OutletWall_Edge,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Edge_Type6,
										(distribn_t*)GPUDataAddr_WallNormal_Edge_Type6);
							}
							//
						}
					}
					//--------------------------------------------------------------------
				}
				// ends the if site_Count!=0, for Collision Type 6.
				// ====================================================================================================================================================

				/*
				// If we follow the same steps as in the CPU version of hemeLB (Send step following PreSend) - INCLUDE this here!!!
				// Synchronisation barrier
				if(myPiD!=0){
					deviceStreamSynchronize(Collide_Stream_PreSend_1);
					deviceStreamSynchronize(Collide_Stream_PreSend_2);
					deviceStreamSynchronize(Collide_Stream_PreSend_3);
					deviceStreamSynchronize(Collide_Stream_PreSend_4);
					deviceStreamSynchronize(Collide_Stream_PreSend_5);
					deviceStreamSynchronize(Collide_Stream_PreSend_6);
				}

				// Once all collision-streaming types are completed then send the distr. functions fNew in totalSharedFs to the CPU
				// For the exchange of f's at domain edges
				// Uses Asynch. MemCopy - Stream: stream_memCpy_GPU_CPU_domainEdge
				if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU_totalSharedFs();
				*/

#else	// If computations on CPUs

				// printf("Calling CPU PART \n\n");
				// Collision Type 1 (mMidFluidCollision):
				site_t offset = mLatDat->GetMidDomainSiteCount();
				StreamAndCollide(mMidFluidCollision, offset, mLatDat->GetDomainEdgeCollisionCount(0));

				// Collision Type 2 (mWallCollision):
				offset += mLatDat->GetDomainEdgeCollisionCount(0);
				// printf("Rank: %d: Collision Type 2: Starting = %lld, Ending = %lld \n\n",myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(1)));
				StreamAndCollide(mWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(1));

				// Collision Type 3 (mInletCollision):
				offset += mLatDat->GetDomainEdgeCollisionCount(1);
				// Receive values for Inlet
				mInletValues->FinishReceive();
				// printf("Rank: %d: Collision Type 3: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(2)));
				StreamAndCollide(mInletCollision, offset, mLatDat->GetDomainEdgeCollisionCount(2));

				// Collision Type 4 (mOutletCollision):
				offset += mLatDat->GetDomainEdgeCollisionCount(2);
				// Receive values for Outlet
				mOutletValues->FinishReceive();
				// printf("Rank: %d: Collision Type 4: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(3)));
				StreamAndCollide(mOutletCollision, offset, mLatDat->GetDomainEdgeCollisionCount(3));

				// Collision Type 5 (mInletWallCollision):
				offset += mLatDat->GetDomainEdgeCollisionCount(3);
				// printf("Rank: %d: Collision Type 5: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(4)));
				StreamAndCollide(mInletWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(4));

				// Collision Type 6 (mOutletWallCollision):
				offset += mLatDat->GetDomainEdgeCollisionCount(4);
				// printf("Rank: %d: Collision Type 6: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(5)));
				StreamAndCollide(mOutletWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(5));

#endif

				timings[hemelb::reporting::Timers::lb_calc].Stop();
				timings[hemelb::reporting::Timers::lb].Stop();
			}


		template<class LatticeType>
			void LBM<LatticeType>::PreReceive()
			{
				timings[hemelb::reporting::Timers::lb].Start();
				timings[hemelb::reporting::Timers::lb_calc].Start();

				/**
				 * In the PreReceive phase, we perform LB for all the sites whose neighbours lie on this
				 * rank ('midDomain' rather than 'domainEdge' sites). Ideally this phase is the longest bit (maximising time for the asynchronous sends
				 * and receives to complete).
				 *
				 * In site id terms, this means starting at the first site and progressing through the
				 * midDomain sites, one type at a time.

				 *** GPU version ***
				 *		Change the enum Step: sequence:
				 * 		     BeginAll = -1, // Called only before first phase
				           BeginPhase = 0,
				           Receive = 1,
				           PreSend = 2,
				           PreWait = 3, 	// PreReceive - Stream synchronization point here for the PreSend streams and the Asynch. MemCopy - CUDA Stream: stream_memCpy_GPU_CPU_domainEdge, before Send !!!
				           Send = 4,
				           Wait = 5,
				           EndPhase = 6,
				           EndAll = 7, // Called only after final phase...
				 ***/

#ifdef HEMELB_USE_GPU	// If exporting computation on GPUs

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				// Boolean variable for sending macroVariables to GPU global memory (avoids the if statement time%_Send_MacroVars_DtH==0 in the GPU kernels)
				bool Write_GlobalMem = (mState->GetTimeStep()%frequency_WriteGlobalMem == 0) ? 1 : 0;

				// Consider whether to send boolean output_shearStressMagn (evaluate wall shear stress magnitude on the GPU) to GPU constant memory, or just pass as an argument to the kernels
				bool output_shearStressMagn = true;

				const std::string hemeKernel = QUOTE_CONTENTS(HEMELB_KERNEL);
				//######################################################################
				/*
				// Initial Location of Synch was at the end of PreReceive - Moved here just for testing
				// Overlap the calculations during PreReceive and the memory transfer at domain edges
				// Only if the steps sequence is modified.
				// 		a. PreSend
				//		b. PreReceive
				//		c. Send
				// Synchronisation barrier
				if(myPiD!=0){
					deviceStreamSynchronize(Collide_Stream_PreSend_1);
					deviceStreamSynchronize(Collide_Stream_PreSend_2);
					deviceStreamSynchronize(Collide_Stream_PreSend_3);
					deviceStreamSynchronize(Collide_Stream_PreSend_4);
					deviceStreamSynchronize(Collide_Stream_PreSend_5);
					deviceStreamSynchronize(Collide_Stream_PreSend_6);
				}
				*/
				//######################################################################


				//#####################################################################################################################################################
				// Merge the first 2 Types of collision-streaming
				// Collision Types 1 & 2:
				site_t offset = 0;	// site_t is type int64_t
				site_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount();

				site_t first_Index = offset;
				site_t site_Count_MidFluid = mLatDat->GetMidDomainCollisionCount(0);
				site_t site_Count_Wall = mLatDat->GetMidDomainCollisionCount(1);

				site_t site_Count = site_Count_MidFluid + site_Count_Wall;

				//if (myPiD!=0) printf("Rank: %d, Collision 1 & 2: First Index MidFluid: %lld, Upper Index MidFluid: %lld, First Index Wall: %lld, Upper Index Wall: %lld  \n\n",myPiD, first_Index, (first_Index+site_Count_MidFluid),
			 	//											(first_Index + site_Count_MidFluid), (first_Index + site_Count_MidFluid + site_Count_Wall));


				//----------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_Collide = 256;				//Number of threads per block for the Collision step
				dim3 nThreads_Collide(nThreadsPerBlock_Collide);
				// Number of fluid nodes involved in the collision/streaming : mLatDat->GetDomainEdgeCollisionCount(0)
				int nBlocks_Collide = (site_Count)/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
				//----------------------------------

				// To access the data in GPU global memory:
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)

				if(nBlocks_Collide!=0){
					if (hemeKernel == "LBGKSL"){
						hemelb::GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_1>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(site_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Wall,
								nFluid_nodes,
								first_Index, (first_Index + site_Count_MidFluid),
								(first_Index + site_Count_MidFluid), (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
								(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type2,
								(distribn_t*)GPUDataAddr_WallNormal_Inner_Type2,
								mState->GetTimeStep(), myPiD,
								(distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
							);
					}
					else if(hemeKernel == "LBGK"){
						hemelb::GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_1>>> (
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(distribn_t*)GPUDataAddr_dbl_MacroVars,
									(site_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Wall,
									nFluid_nodes,
									first_Index, (first_Index + site_Count_MidFluid),
									(first_Index + site_Count_MidFluid), (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type2,
									(distribn_t*)GPUDataAddr_WallNormal_Inner_Type2,
									mState->GetTimeStep(), myPiD);
					}
				}

				/*
				if(nBlocks_Collide!=0)
					hemelb::GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_1>>> (
							(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
							(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
							(distribn_t*)GPUDataAddr_dbl_MacroVars,
							(site_t*)GPUDataAddr_int64_Neigh_d,
							(uint32_t*)GPUDataAddr_uint32_Wall,
							nFluid_nodes,
							first_Index, (first_Index + site_Count_MidFluid),
							(first_Index + site_Count_MidFluid), (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem); // (int64_t*)GPUDataAddr_int64_Neigh_b
				*/
				//#####################################################################################################################################################


				/*
				// ====================================================================================================================================================
				// Collision Type 1:
				site_t offset = 0;	// site_t is type int64_t
				// StreamAndCollide(mMidFluidCollision, offset, mLatDat->GetMidDomainCollisionCount(0));

				int64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount();
				int64_t first_Index = offset;
				int64_t site_Count = mLatDat->GetMidDomainCollisionCount(0);
				//if (myPiD!=0) printf("Rank: %d, Collision 1: First Index: %lld, Upper Index: %lld \n\n",myPiD, first_Index, (first_Index+site_Count));

				//----------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_Collide = 128;				//Number of threads per block for the Collision step
				dim3 nThreads_Collide(nThreadsPerBlock_Collide);
				// Number of fluid nodes involved in the collision/streaming : mLatDat->GetDomainEdgeCollisionCount(0)
				int nBlocks_Collide = (site_Count)/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
				//----------------------------------

				// To access the data in GPU global memory:
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)
				if(nBlocks_Collide!=0)
					hemelb::GPU_CollideStream_1_PreReceive_SaveMacroVars <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_1>>> (	(double*)GPUDataAddr_dbl_fOld_b,
																									(double*)GPUDataAddr_dbl_fNew_b,
																									(double*)GPUDataAddr_dbl_MacroVars,
																									(int64_t*)GPUDataAddr_int64_Neigh_d,
																									(mLatDat->GetLocalFluidSiteCount()),
																									offset, (offset + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep()); //
				// ====================================================================================================================================================
				*/

				/*
				//-------------------------------------------------------------------------------------------------------------
				// To do:
				//	Think whether this should be here or at the end of PreReceive(), once all the collision-streaming kernels has been launched
				// 		Control is returned back to the host once the kernels are launched, hence putting this MemCopy at the end will overlap with these calculations
				// 		On the other hand it will delay slightly the begining of the Send step (Maybe ... ) Needs to be investigated
				// 	Ask Julich support on the above!!!

				// Overlap the calculations during PreReceive and the memory transfer at domain edges
				// Only if the steps sequence is modified.
				// 		a. PreSend
				//		b. PreReceive
				//		c. Send
				// Synchronisation barrier
				if(myPiD!=0){
					deviceStreamSynchronize(Collide_Stream_PreSend_1);
					deviceStreamSynchronize(Collide_Stream_PreSend_2);
					deviceStreamSynchronize(Collide_Stream_PreSend_3);
					deviceStreamSynchronize(Collide_Stream_PreSend_4);
					deviceStreamSynchronize(Collide_Stream_PreSend_5);
					deviceStreamSynchronize(Collide_Stream_PreSend_6);
				}

				// Once all collision-streaming types are completed then send the distr. functions fNew in totalSharedFs to the CPU
				// For the exchange of f's at domain edges
				// Uses Asynch. MemCopy - Stream: stream_memCpy_GPU_CPU_domainEdge
				if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU_totalSharedFs();
				//-------------------------------------------------------------------------------------------------------------
				*/

				/*
				// ====================================================================================================================================================
				// Collision Type 2 (Simple Bounce Back!!!):
				offset += mLatDat->GetMidDomainCollisionCount(0);
				// StreamAndCollide(mWallCollision, offset, mLatDat->GetMidDomainCollisionCount(1));

				// GPU COLLISION KERNEL:
				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetMidDomainCollisionCount(1);
				//if (myPiD!=0) printf("Rank: %d, Collision 2: First Index: %lld, Upper Index: %lld \n\n",myPiD, first_Index, (first_Index+site_Count));
				//-------------------------------------
				// Kernel set-up
				nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

				// To access the data in GPU global memory:
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)
				// Wall BCs: Remember that at the moment this is ONLY valid for Simple Bounce Back
				if(nBlocks_Collide!=0)
					hemelb::GPU_CollideStream_mWallCollision_sBB_PreRec <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_2>>> (	(double*)GPUDataAddr_dbl_fOld_b,
																						(double*)GPUDataAddr_dbl_fNew_b,
																						(double*)GPUDataAddr_dbl_MacroVars,
																						(int64_t*)GPUDataAddr_int64_Neigh_d,
																						(uint32_t*)GPUDataAddr_uint32_Wall,
																						(mLatDat->GetLocalFluidSiteCount()),
																						first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep()); // (int64_t*)GPUDataAddr_int64_Neigh_b

				//---------------------------------------------------------------------------------------------------------------------------------------------------
				// ====================================================================================================================================================
				*/

				//**********************************************************************
				/*// Get the type of Iolet BCs from the CMake file compiling options
				std::string hemeIoletBC_Inlet, hemeIoletBC_Outlet;
				get_Iolet_BCs(hemeIoletBC_Inlet, hemeIoletBC_Outlet);
				*/
				// Check If I can get the type of Iolet BCs from the CMake file
				#define QUOTE_RAW(x) #x
				#define QUOTE_CONTENTS(x) QUOTE_RAW(x)

				std::string hemeIoletBC_Inlet, hemeIoletBC_Outlet;
				hemeIoletBC_Inlet = QUOTE_CONTENTS(HEMELB_INLET_BOUNDARY);
				//printf("Function Call: Inlet Type of BCs: hemeIoletBC_Inlet: %s \n", hemeLB_IoletBC_Inlet.c_str()); //note the use of c_str

				hemeIoletBC_Outlet = QUOTE_CONTENTS(HEMELB_OUTLET_BOUNDARY);
				//printf("Function Call: Outlet Type of BCs: hemeIoletBC_Outlet: %s \n\n", hemeLB_IoletBC_Outlet.c_str()); //note the use of c_str

				//**********************************************************************


				// Added Sept 2022 - Moved the Inner domain part (Vel. BCs case) from the PreSend to here - Check if it makes a difference
				// ====================================================================================================================================================
				// Inlets BCs
				if(hemeIoletBC_Inlet == "LADDIOLET"){
					//printf("Entering the LaddIolet loop \n\n");
					if (myPiD!=0) apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch_PreReceive();
				} // Ends the if(hemeIoletBC_Inlet == "LADDIOLET") loop

				//----------------------------------------------------------------------
				// Outlets BCs - Old Implementation of BCs (need to switch to everything on the GPU as for the inlet)
				if(hemeIoletBC_Outlet == "LADDIOLET"){
					//====================================================================
					// Inner Domain
					// Collision Type 4 (mOutletCollision):
					site_t start_Index_Outlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2);
					site_t site_Count_Outlet_Inner = mLatDat->GetMidDomainCollisionCount(3);
					if (site_Count_Outlet_Inner!=0){
						/*
						GetWallMom(mOutletCollision, start_Index_Outlet_Inner, site_Count_Outlet_Inner, propertyCache); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_Outlet_Inner, site_Count_Outlet_Inner, propertyCache, wallMom_Outlet_Inner);
						wallMom_Outlet_Inner.resize(site_Count_Outlet_Inner * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_Outlet_Inner, site_Count_Outlet_Inner, wallMom_Outlet_Inner, GPUDataAddr_wallMom_Outlet_Inner);
						*/
						/*
						//-----------------
						// With the momentum correction approach
						GetWallMom_correction(mOutletCollision, start_Index_Outlet_Inner, site_Count_Outlet_Inner, propertyCache); // Fills the propertyCache.wallMom_Cache
						read_WallMom_correction_from_propertyCache(start_Index_Outlet_Inner, site_Count_Outlet_Inner, propertyCache, wallMom_correction_Outlet_Inner);
						wallMom_correction_Outlet_Inner.resize(site_Count_Outlet_Inner * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom_correction(start_Index_Outlet_Inner, site_Count_Outlet_Inner, wallMom_correction_Outlet_Inner, GPUDataAddr_wallMom_correction_Outlet_Inner);
						//-----------------
						*/
						//-----------------
						// Directly getting the correction term without passing through propertyCache
						GetWallMom_correction_Direct(mOutletCollision, start_Index_Outlet_Inner, site_Count_Outlet_Inner, propertyCache, wallMom_correction_Outlet_Inner_Direct); // Fills the propertyCache.wallMom_Cache
						wallMom_correction_Outlet_Inner_Direct.resize(site_Count_Outlet_Inner * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom_correction_cudaStream(start_Index_Outlet_Inner, site_Count_Outlet_Inner, wallMom_correction_Outlet_Inner_Direct,
																									GPUDataAddr_wallMom_correction_Outlet_Inner, Collide_Stream_PreRec_4);
						//-----------------

					}
					//====================================================================

					//====================================================================
					// Collision Type 6 (mOutletWallCollision):
					site_t start_Index_OutletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2)
					                                      + mLatDat->GetMidDomainCollisionCount(3) + mLatDat->GetMidDomainCollisionCount(4);
					site_t site_Count_OutletWall_Inner = mLatDat->GetMidDomainCollisionCount(5);
					if (site_Count_OutletWall_Inner!=0){
						/*
						GetWallMom(mOutletWallCollision, start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, propertyCache); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, propertyCache, wallMom_OutletWall_Inner);
						wallMom_OutletWall_Inner.resize(site_Count_OutletWall_Inner * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, wallMom_OutletWall_Inner, GPUDataAddr_wallMom_OutletWall_Inner);
						*/
						/*
						//-----------------
						// With the momentum correction approach
						GetWallMom_correction(mOutletWallCollision, start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, propertyCache); // Fills the propertyCache.wallMom_Cache
						read_WallMom_correction_from_propertyCache(start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, propertyCache, wallMom_correction_OutletWall_Inner);
						wallMom_correction_OutletWall_Inner.resize(site_Count_OutletWall_Inner * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom_correction(start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, wallMom_correction_OutletWall_Inner, GPUDataAddr_wallMom_correction_OutletWall_Inner);
						//-----------------
						*/

						//-----------------
						// Directly getting the correction term without passing through propertyCache
						GetWallMom_correction_Direct(mOutletWallCollision, start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, propertyCache, wallMom_correction_OutletWall_Inner_Direct); // Fills the propertyCache.wallMom_Cache
						wallMom_correction_OutletWall_Inner_Direct.resize(site_Count_OutletWall_Inner * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						//memCpy_HtD_GPUmem_WallMom_correction(start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, wallMom_correction_OutletWall_Inner, GPUDataAddr_wallMom_correction_OutletWall_Inner);
						memCpy_HtD_GPUmem_WallMom_correction_cudaStream(start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, wallMom_correction_OutletWall_Inner_Direct,
																														GPUDataAddr_wallMom_correction_OutletWall_Inner, Collide_Stream_PreRec_6);
						//-----------------


					}
					//====================================================================
				}
				//**********************************************************************
				// ====================================================================================================================================================



				// ====================================================================================================================================================
				// Collision Type 3:
				// Inlet BCs: NashZerothOrderPressure - Specify the ghost density for each inlet
				offset = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
				//offset += mLatDat->GetMidDomainCollisionCount(1);
				//StreamAndCollide(mInletCollision, offset, mLatDat->GetMidDomainCollisionCount(2));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetMidDomainCollisionCount(2);

				//	Total GLOBAL iolets (NOT ONLY ON LOCAL RANK): n_Inlets = mInletValues->GetTotalIoletCount();
				int n_Inlets = mInletValues->GetTotalIoletCount();

				// Proceed with the collision type if the number of fluid nodes involved is not ZERO
				if (site_Count!=0){
					//-------------------------------------
					// GPU COLLISION KERNEL:
					// Kernel set-up
					nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

					//--------------------------------------------------------------------
					// TODO: Choose the appropriate kernel depending on BCs:
					// Inlets BCs
					if(hemeIoletBC_Inlet == "LADDIOLET"){
						//
						if (hemeKernel == "LBGKSL"){
							hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_3>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(int64_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Iolet,
								(mLatDat->GetLocalFluidSiteCount()),
								(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
								first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
								mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
							);
						}
						else if(hemeKernel == "LBGK"){
							hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_3>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(int64_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Iolet,
								(mLatDat->GetLocalFluidSiteCount()),
								(distribn_t*)GPUDataAddr_wallMom_correction_Inlet_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
								first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem);
						}
						//

					}
					else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalInlets_mInlet<=(local_iolets_MaxSIZE/3))
						{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure<<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_3>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity,
									(float*)d_inletNormal,
									n_Inlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalInlets_mInlet, Inlet_Inner,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure<<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_3>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity,
									(float*)d_inletNormal,
									n_Inlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalInlets_mInlet, Inlet_Inner); // (int64_t*)GPUDataAddr_int64_Neigh_b
							}
							//
						}
						else{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2<<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_3>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity,
									(float*)d_inletNormal,
									n_Inlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalInlets_mInlet, (site_t*)GPUDataAddr_Inlet_Inner,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2<<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_3>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity,
									(float*)d_inletNormal,
									n_Inlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalInlets_mInlet, (site_t*)GPUDataAddr_Inlet_Inner);
							}
							//
						}
					}
					//--------------------------------------------------------------------
				}
				// ends the if site_Count!=0
				// ====================================================================================================================================================

				// ====================================================================================================================================================
				// Collision Type 4:
				// Outlet BCs: NashZerothOrderPressure - Specify the ghost density for each outlet
				offset += mLatDat->GetMidDomainCollisionCount(2);
				//StreamAndCollide(mOutletCollision, offset, mLatDat->GetMidDomainCollisionCount(3));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetMidDomainCollisionCount(3);

				//	Total GLOBAL iolets: n_Outlets = mOutletValues->GetTotalIoletCount();
				int n_Outlets = mOutletValues->GetTotalIoletCount();

				// Proceed with the collision type if the number of fluid nodes involved is not ZERO
				if (site_Count!=0){
					//printf("Rank: %d: Collision Type 4 (Outlet): Starting = %lld, SiteCount = %lld, Ending = %lld \n\n", myPiD, first_Index, mLatDat->GetMidDomainCollisionCount(3), (offset + mLatDat->GetMidDomainCollisionCount(3)));
					//-------------------------------------
					// GPU COLLISION KERNEL:
					// Kernel set-up
					nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

					//--------------------------------------------------------------------
					// TODO: Choose the appropriate kernel depending on BCs:
					// Inlets BCs
					if(hemeIoletBC_Outlet == "LADDIOLET"){
						//
						if (hemeKernel == "LBGKSL"){
							hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_4>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(int64_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Iolet,
								(mLatDat->GetLocalFluidSiteCount()),
								(distribn_t*)GPUDataAddr_wallMom_correction_Outlet_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
								first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
								mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
							);
						}
						else if(hemeKernel == "LBGK"){
							hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_4>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(int64_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Iolet,
								(mLatDat->GetLocalFluidSiteCount()),
								(distribn_t*)GPUDataAddr_wallMom_correction_Outlet_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
								first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem);
						}
						//
					}
					else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalOutlets_mOutlet<=(local_iolets_MaxSIZE/3))
						{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_4>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity_out,
									(float*)d_outletNormal,
									n_Outlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalOutlets_mOutlet, Outlet_Inner,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_4>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity_out,
									(float*)d_outletNormal,
									n_Outlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalOutlets_mOutlet, Outlet_Inner);
							}
							//
						}
						else{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2 <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_4>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity_out,
									(float*)d_outletNormal,
									n_Outlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalOutlets_mOutlet, (site_t*)GPUDataAddr_Outlet_Inner,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2 <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_4>>> (
									(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(double*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(distribn_t*)d_ghostDensity_out,
									(float*)d_outletNormal,
									n_Outlets,
									(mLatDat->GetLocalFluidSiteCount()),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									n_LocalOutlets_mOutlet, (site_t*)GPUDataAddr_Outlet_Inner);
							}
							//
						}

					}
					//--------------------------------------------------------------------
				}
				// ends the if site_Count!=0, for Collision Type 4.
				// ====================================================================================================================================================

				// ====================================================================================================================================================
				// Collision Type 5:
				// Walls BCs:	Simple Bounce-Back
				// Inlet/Outlet BCs:	NashZerothOrderPressure - Specify the ghost density for each inlet/outlet
				offset += mLatDat->GetMidDomainCollisionCount(3);
				//StreamAndCollide(mInletWallCollision, offset, mLatDat->GetMidDomainCollisionCount(4));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetMidDomainCollisionCount(4);

				n_Inlets = mInletValues->GetTotalIoletCount(); // Probably not necessary. Check and Remove!!! To do!!!

				// Proceed with the collision type if the number of fluid nodes involved is not ZERO
				if (site_Count!=0){

					//-------------------------------------
					// GPU COLLISION KERNEL:
					// Kernel set-up
					nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

					//--------------------------------------------------------------------
					// TODO: Choose the appropriate kernel depending on BCs:
					// Inlets BCs
					if(hemeIoletBC_Inlet == "LADDIOLET"){
						//
						if (hemeKernel == "LBGKSL"){
							hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_5>>> (
								 (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								 (distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								 (distribn_t*)GPUDataAddr_dbl_MacroVars,
								 (int64_t*)GPUDataAddr_int64_Neigh_d,
								 (uint32_t*)GPUDataAddr_uint32_Wall,
								 (uint32_t*)GPUDataAddr_uint32_Iolet,
								 (mLatDat->GetLocalFluidSiteCount()),
								 (distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
								 first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
								 (distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type5,
								 (distribn_t*)GPUDataAddr_WallNormal_Inner_Type5,
								 mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
							 );
						}
						else if(hemeKernel == "LBGK"){
							hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_5>>> (
 								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
 								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
 								(distribn_t*)GPUDataAddr_dbl_MacroVars,
 								(int64_t*)GPUDataAddr_int64_Neigh_d,
 								(uint32_t*)GPUDataAddr_uint32_Wall,
 								(uint32_t*)GPUDataAddr_uint32_Iolet,
 								(mLatDat->GetLocalFluidSiteCount()),
 								(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
 								first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
 								(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type5,
 								(distribn_t*)GPUDataAddr_WallNormal_Inner_Type5);
						}

						/*
						hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_5>>> (
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
								(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
								(distribn_t*)GPUDataAddr_dbl_MacroVars,
								(int64_t*)GPUDataAddr_int64_Neigh_d,
								(uint32_t*)GPUDataAddr_uint32_Wall,
								(uint32_t*)GPUDataAddr_uint32_Iolet,
								(mLatDat->GetLocalFluidSiteCount()),
								(distribn_t*)GPUDataAddr_wallMom_correction_InletWall_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
								first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem);
						*/
					}
					else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalInlets_mInletWall<=(local_iolets_MaxSIZE/3))
						{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_5>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity,
										(float*)d_inletNormal,
										n_Inlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalInlets_mInletWall, InletWall_Inner,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type5,
										(distribn_t*)GPUDataAddr_WallNormal_Inner_Type5,
										mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
									);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_5>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity,
										(float*)d_inletNormal,
										n_Inlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalInlets_mInletWall, InletWall_Inner,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type5,
										(distribn_t*)GPUDataAddr_WallNormal_Inner_Type5);
							}
							//
						}
						else{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_5>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity,
										(float*)d_inletNormal,
										n_Inlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalInlets_mInletWall, (site_t*)GPUDataAddr_InletWall_Inner,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type5,
										(distribn_t*)GPUDataAddr_WallNormal_Inner_Type5,
										mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
									);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_5>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity,
										(float*)d_inletNormal,
										n_Inlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalInlets_mInletWall, (site_t*)GPUDataAddr_InletWall_Inner,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type5,
										(distribn_t*)GPUDataAddr_WallNormal_Inner_Type5);
							}
							//
						}

					}
					//--------------------------------------------------------------------
				}
				// ends the if site_Count!=0, for Collision Type 5.
				// ====================================================================================================================================================

				// ====================================================================================================================================================
				// Collision Type 6:
				// Walls BCs:	Simple Bounce-Back
				// Inlet/Outlet BCs:	NashZerothOrderPressure - Specify the ghost density for each inlet/outlet

				offset += mLatDat->GetMidDomainCollisionCount(4);
				//StreamAndCollide(mOutletWallCollision, offset, mLatDat->GetMidDomainCollisionCount(5));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetMidDomainCollisionCount(5);

				n_Outlets = mOutletValues->GetTotalIoletCount(); // Probably not necessary. Check and Remove!!! To do!!!

				// Proceed with the collision type if the number of fluid nodes involved is not ZERO
				if (site_Count!=0){

					//-------------------------------------
					// GPU COLLISION KERNEL:
					// Kernel set-up
					nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

					//--------------------------------------------------------------------
					// TODO: Choose the appropriate kernel depending on BCs:
					// Inlets BCs
					if(hemeIoletBC_Outlet == "LADDIOLET"){
						//
						if (hemeKernel == "LBGKSL"){
							hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_6>>> (
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(distribn_t*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Wall,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(mLatDat->GetLocalFluidSiteCount()),
									(distribn_t*)GPUDataAddr_wallMom_correction_OutletWall_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type6,
									(distribn_t*)GPUDataAddr_WallNormal_Inner_Type6,
									mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
								);
						}
						else if(hemeKernel == "LBGK"){
							hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_6>>> (
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
									(distribn_t*)GPUDataAddr_dbl_MacroVars,
									(int64_t*)GPUDataAddr_int64_Neigh_d,
									(uint32_t*)GPUDataAddr_uint32_Wall,
									(uint32_t*)GPUDataAddr_uint32_Iolet,
									(mLatDat->GetLocalFluidSiteCount()),
									(distribn_t*)GPUDataAddr_wallMom_correction_OutletWall_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
									first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
									(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type6,
									(distribn_t*)GPUDataAddr_WallNormal_Inner_Type6);
						}
						//
					}
					else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalOutlets_mOutletWall<=(local_iolets_MaxSIZE/3))
						{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_6>>> (
												(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
												(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
												(double*)GPUDataAddr_dbl_MacroVars,
												(int64_t*)GPUDataAddr_int64_Neigh_d,
												(uint32_t*)GPUDataAddr_uint32_Wall,
												(uint32_t*)GPUDataAddr_uint32_Iolet,
												(distribn_t*)d_ghostDensity_out,
												(float*)d_outletNormal,
												n_Outlets,
												(mLatDat->GetLocalFluidSiteCount()),
												first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
												n_LocalOutlets_mOutletWall, OutletWall_Inner,
		 					 					(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type6,
		 					 					(distribn_t*)GPUDataAddr_WallNormal_Inner_Type6,
												mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
											);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_6>>> (
												(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
												(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
												(double*)GPUDataAddr_dbl_MacroVars,
												(int64_t*)GPUDataAddr_int64_Neigh_d,
												(uint32_t*)GPUDataAddr_uint32_Wall,
												(uint32_t*)GPUDataAddr_uint32_Iolet,
												(distribn_t*)d_ghostDensity_out,
												(float*)d_outletNormal,
												n_Outlets,
												(mLatDat->GetLocalFluidSiteCount()),
												first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
												n_LocalOutlets_mOutletWall, OutletWall_Inner,
		 					 					(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type6,
		 					 					(distribn_t*)GPUDataAddr_WallNormal_Inner_Type6);
							}
							//
						}
						else{
							//
							if (hemeKernel == "LBGKSL"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_6>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity_out,
										(float*)d_outletNormal,
										n_Outlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalOutlets_mOutletWall, (site_t*)GPUDataAddr_OutletWall_Inner,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type6,
										(distribn_t*)GPUDataAddr_WallNormal_Inner_Type6,
										mState->GetTimeStep(), (distribn_t*)GPUDataAddr_vTau, mSimConfig->GetSpongeLayerLifetime()
									);
							}
							else if(hemeKernel == "LBGK"){
								hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress <<<nBlocks_Collide, nThreads_Collide, 0, Collide_Stream_PreRec_6>>> (
										(double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
										(double*)GPUDataAddr_dbl_MacroVars,
										(int64_t*)GPUDataAddr_int64_Neigh_d,
										(uint32_t*)GPUDataAddr_uint32_Wall,
										(uint32_t*)GPUDataAddr_uint32_Iolet,
										(distribn_t*)d_ghostDensity_out,
										(float*)d_outletNormal,
										n_Outlets,
										(mLatDat->GetLocalFluidSiteCount()),
										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, Write_GlobalMem,
										n_LocalOutlets_mOutletWall, (site_t*)GPUDataAddr_OutletWall_Inner,
										(distribn_t*)GPUDataAddr_WallShearStressMagn_Inner_Type6,
										(distribn_t*)GPUDataAddr_WallNormal_Inner_Type6);
							}
							//
						}

					}
					//--------------------------------------------------------------------
				}
				// ends the if site_Count!=0, for Collision Type 6.
				// ====================================================================================================================================================

				//-------------------------------------------------------------------------------------------------------------

				// Initial Location of Synch (1) - Moved for testing at the beginning of PreReceive()
				// Overlap the calculations during PreReceive and the memory transfer at domain edges
				// Only if the steps sequence is modified.
				// 		a. PreSend
				//		b. PreReceive
				//		c. Send
				// Synchronisation barrier
				if(myPiD!=0){
					deviceStreamSynchronize(Collide_Stream_PreSend_1);
					deviceStreamSynchronize(Collide_Stream_PreSend_2);
					deviceStreamSynchronize(Collide_Stream_PreSend_3);
					deviceStreamSynchronize(Collide_Stream_PreSend_4);
					deviceStreamSynchronize(Collide_Stream_PreSend_5);
					deviceStreamSynchronize(Collide_Stream_PreSend_6);
				}

				// Comments:
				// CUDA-aware mpi enabled OR not???
				// 1. CUDA-aware mpi case: No need to send data D2H.
#ifndef HEMELB_CUDA_AWARE_MPI
				/**
				 	2. No CUDA-aware mpi
					Once all collision-streaming types are completed then send the distr. functions fNew in totalSharedFs to the CPU
				 	For the exchange of f's at domain edges
				 	Uses Asynch. MemCopy - Stream: stream_memCpy_GPU_CPU_domainEdge
					*/
				//std::cout << "No CUDA-aware mpi branch: D2H mem.copies... " << std::endl;
				if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU_totalSharedFs();
#endif
				//-------------------------------------------------------------------------------------------------------------

				// Stream for the asynchronous MemCopy DtH - f's at domain edges - after the collision-streaming kernels in PreSend().
				//if(myPiD!=0) deviceStreamSynchronize(stream_memCpy_GPU_CPU_domainEdge);


				// Synchronisation point for the kernel GPU_Check_Stability launched at the beginning of PreSend() step. Ensure the stability check has completed and the results are ready
				// memcopy D2H value of stability copied to mLatDat->h_Stability_GPU_mLatDat
				if(myPiD!=0 && mState->GetTimeStep()%1000 ==0){
						deviceStreamSynchronize(stability_check_stream);
						// MemCopy from Device To Host the value for the Stability - TODO!!!
						// status = deviceMemcpyAsync( &(mLatDat->h_Stability_GPU_mLatDat), &(((int*)mLatDat->d_Stability_GPU_mLatDat)[0]), sizeof(int), memcpyDeviceToHost, stability_check_stream);
						bool status = deviceMemcpy( &(mLatDat->h_Stability_GPU_mLatDat), &(((int*)mLatDat->d_Stability_GPU_mLatDat)[0]), sizeof(int), memcpyDeviceToHost);

						if(mLatDat->h_Stability_GPU_mLatDat==0)
							printf("Rank = %d - Unstable SImulation: Host Stability flag: %d \n\n", myPiD, mLatDat->h_Stability_GPU_mLatDat);
				}

/*
				// ====================================================================================================================================================
				// Send the MacroVariables (density and Velocity) to the CPU
				// THink where to place this!!! To do!!!
				if (mState->GetTimeStep() % 100 == 0)
				{
					if(myPiD!=0) {
						// Must ensure that writing the updated macroVariables from the above kernels has completed.
						deviceStreamSynchronize(Collide_Stream_PreRec_1);
						deviceStreamSynchronize(Collide_Stream_PreRec_2);
						deviceStreamSynchronize(Collide_Stream_PreRec_3);
						deviceStreamSynchronize(Collide_Stream_PreRec_4);
						deviceStreamSynchronize(Collide_Stream_PreRec_5);
						deviceStreamSynchronize(Collide_Stream_PreRec_6);
					}

					// Check whether the hemeLB picks up the macroVariables at the PostReceive step???
					lb::MacroscopicPropertyCache& propertyCache = GetPropertyCache();
					if(myPiD!=0) Read_Macrovariables_GPU_to_CPU(0, mLatDat->GetLocalFluidSiteCount(), propertyCache);
				}
				// ====================================================================================================================================================
*/

#else	// If computations on CPUs
				//=====================================================================================
				// Collision Type 1:
				site_t offset = 0;
				StreamAndCollide(mMidFluidCollision, offset, mLatDat->GetMidDomainCollisionCount(0));

				// Collision Type 2:
				offset += mLatDat->GetMidDomainCollisionCount(0);
				StreamAndCollide(mWallCollision, offset, mLatDat->GetMidDomainCollisionCount(1));

				// Collision Type 3:
				offset += mLatDat->GetMidDomainCollisionCount(1);
				StreamAndCollide(mInletCollision, offset, mLatDat->GetMidDomainCollisionCount(2));

				// Collision Type 4:
				offset += mLatDat->GetMidDomainCollisionCount(2);
				StreamAndCollide(mOutletCollision, offset, mLatDat->GetMidDomainCollisionCount(3));

				// Collision Type 5:
				offset += mLatDat->GetMidDomainCollisionCount(3);
				StreamAndCollide(mInletWallCollision, offset, mLatDat->GetMidDomainCollisionCount(4));

				// Collision Type 6:
				offset += mLatDat->GetMidDomainCollisionCount(4);
				StreamAndCollide(mOutletWallCollision, offset, mLatDat->GetMidDomainCollisionCount(5));

				//=====================================================================================
#endif

				timings[hemelb::reporting::Timers::lb_calc].Stop();
				timings[hemelb::reporting::Timers::lb].Stop();
			}

		template<class LatticeType>
			void LBM<LatticeType>::PostReceive()
			{
				timings[hemelb::reporting::Timers::lb].Start();

				// Copy the distribution functions received from the neighbouring
				// processors into the destination buffer "f_new".
				// This is done here, after receiving the sent distributions from neighbours.

#ifdef HEMELB_USE_GPU

				// 1*. host-to-device memcopy (NOT needed when CUDA-aware mpi is enabled!!!):
				// 		Send the totalSharedFs distr. functions in fOld to the GPU
				// 		( these have been already received - MPI exchange completed)

				// 2*. do the appropriate re-allocation into the destination buffer "f_new" using the  streamingIndicesForReceivedDistributions
				// 		see: *GetFNew(streamingIndicesForReceivedDistributions[i]) = *GetFOld(neighbouringProcs[0].FirstSharedDistribution + i);
				// 		from LatticeData::CopyReceived()

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

#ifndef HEMELB_CUDA_AWARE_MPI

				// NO CUDA-aware mpi branch -
				//std::cout << "NO CUDA-aware mpi branch: Current rank: " << myPiD << " Need to do H2D memcopy totalSharedFs distr. functions in fOld to the GPU " << std::endl;

				// Think how it could be possible to call this earlier. To do!!!
				// It requires the completion of the MPI exchange step... Step: Send

				// 1*. host-to-device memcopy: 1. Send the totalSharedFs distr. functions in fOld to the GPU
				// 			Previously Used the cuda stream: stream_memCpy_CPU_GPU_domainEdge
				//				Now Switched to stream: stream_ReceivedDistr
				if(myPiD!=0)
					Read_DistrFunctions_CPU_to_GPU_totalSharedFs();

				// Syncrhonisation Barrier for the above stream involved in the host-to-device memcopy (domain edges)
				/** 8-7-2020:
						Maybe remove the synch point: deviceStreamSynchronize(stream_memCpy_CPU_GPU_domainEdge);
						 	and just use the same cuda stream used in the HtD memcpy above in function Read_DistrFunctions_CPU_to_GPU_totalSharedFs (stream_memCpy_CPU_GPU_domainEdge)
						for launching the cuda kernel
							hemelb::GPU_StreamReceivedDistr
						OR THE REVERSE CASE: Use the stream: stream_ReceivedDistr. Follow this approach !!!
				*/
				/*
				// Not needed if using the stream: stream_ReceivedDistr in Read_DistrFunctions_CPU_to_GPU_totalSharedFs.
				if(myPiD!=0) {
					// deviceStreamSynchronize(stream_memCpy_CPU_GPU_domainEdge); // Needed if we switch to asynch memcopy and use this stream in Read_DistrFunctions_CPU_to_GPU_totalSharedFs

					// The following might be needed here for cases where the PostReceive Step is usefull, e.g. for interpolating types of BCs,
					// Otherwise could be moved before the GPU_SwapOldAndNew kernel
					deviceStreamSynchronize(Collide_Stream_PreRec_1);
					deviceStreamSynchronize(Collide_Stream_PreRec_2);
					deviceStreamSynchronize(Collide_Stream_PreRec_3);
					deviceStreamSynchronize(Collide_Stream_PreRec_4);
					deviceStreamSynchronize(Collide_Stream_PreRec_5);
					deviceStreamSynchronize(Collide_Stream_PreRec_6);
				}
				*/
#endif

				//----------------------------------
				// 2*. Cuda kernel to do the re-allocation into the destination buffer "f_new" using the  streamingIndicesForReceivedDistributions
				// Cuda kernel set-up
				site_t totSharedFs = mLatDat->totalSharedFs;
				int nThreadsPerBlock_StreamRecDistr = 128;				//Number of threads per block for the Collision step
				dim3 nThreads_StreamRecDistr(nThreadsPerBlock_StreamRecDistr);
				int nBlocks_StreamRecDistr = totSharedFs/nThreadsPerBlock_StreamRecDistr			+ ((totSharedFs % nThreadsPerBlock_StreamRecDistr > 0)         ? 1 : 0);

				if (nBlocks_StreamRecDistr!=0)
					hemelb::GPU_StreamReceivedDistr <<<nBlocks_StreamRecDistr, nThreads_StreamRecDistr, 0, stream_ReceivedDistr>>> ( (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																																																	(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																																																	(site_t*)GPUDataAddr_int64_streamInd, (mLatDat->GetLocalFluidSiteCount()), totSharedFs);
				//----------------------------------

#else		// Computations on CPU
				mLatDat->CopyReceived();


				// Do any cleanup steps necessary on boundary nodes
				site_t offset = mLatDat->GetMidDomainSiteCount();

				timings[hemelb::reporting::Timers::lb_calc].Start();

				//TODO yup, this is horrible. If you read this, please improve the following code.
				PostStep(mMidFluidCollision, offset, mLatDat->GetDomainEdgeCollisionCount(0));
				offset += mLatDat->GetDomainEdgeCollisionCount(0);

				PostStep(mWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(1));
				offset += mLatDat->GetDomainEdgeCollisionCount(1);

				PostStep(mInletCollision, offset, mLatDat->GetDomainEdgeCollisionCount(2));
				offset += mLatDat->GetDomainEdgeCollisionCount(2);

				PostStep(mOutletCollision, offset, mLatDat->GetDomainEdgeCollisionCount(3));
				offset += mLatDat->GetDomainEdgeCollisionCount(3);

				PostStep(mInletWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(4));
				offset += mLatDat->GetDomainEdgeCollisionCount(4);

				PostStep(mOutletWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(5));

				offset = 0;

				PostStep(mMidFluidCollision, offset, mLatDat->GetMidDomainCollisionCount(0));
				offset += mLatDat->GetMidDomainCollisionCount(0);

				PostStep(mWallCollision, offset, mLatDat->GetMidDomainCollisionCount(1));
				offset += mLatDat->GetMidDomainCollisionCount(1);

				PostStep(mInletCollision, offset, mLatDat->GetMidDomainCollisionCount(2));
				offset += mLatDat->GetMidDomainCollisionCount(2);

				PostStep(mOutletCollision, offset, mLatDat->GetMidDomainCollisionCount(3));
				offset += mLatDat->GetMidDomainCollisionCount(3);

				PostStep(mInletWallCollision, offset, mLatDat->GetMidDomainCollisionCount(4));
				offset += mLatDat->GetMidDomainCollisionCount(4);

				PostStep(mOutletWallCollision, offset, mLatDat->GetMidDomainCollisionCount(5));
#endif

				timings[hemelb::reporting::Timers::lb_calc].Stop();
				timings[hemelb::reporting::Timers::lb].Stop();
			}


		template<class LatticeType>
			void LBM<LatticeType>::EndIteration()
			{
				timings[hemelb::reporting::Timers::lb].Start();
				timings[hemelb::reporting::Timers::lb_calc].Start();

#ifdef HEMELB_USE_GPU
				// Sends macrovariables (density and velocity) from the GPU to the CPU at the requested frequency

				// Local rank
			  const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
			  int myPiD = rank_Com.Rank();

				// Synchronisation barrier for stream_ReceivedDistr
				// 		Ensure that the received distr. functions have been placed in fNew beforing swaping the populations (fNew -> fOld)
				//		if (myPiD!=0) deviceStreamSynchronize(stream_ReceivedDistr);
				// Or simply use the same cuda stream: stream_ReceivedDistr

				// 25-3-2021
				// Swap the f's (Place fNew in fOld).
				// fluid sites limits (just swap the distr. functions of the fluid sites (ignore the totalSharedFs):
				site_t offset = 0;
				site_t site_Count = mLatDat->GetLocalFluidSiteCount(); // Total number of fluid sites: GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)here

				// Syncrhonisation Barrier for the PreReceive collision cuda streams
				if(myPiD!=0) {
					// The following might be needed in PostReceive() for cases where the PostReceive Step is usefull, e.g. for interpolating types of BCs,
					// Otherwise could be moved here before the GPU_SwapOldAndNew kernel
					deviceStreamSynchronize(Collide_Stream_PreRec_1);
					deviceStreamSynchronize(Collide_Stream_PreRec_2);
					deviceStreamSynchronize(Collide_Stream_PreRec_3);
					deviceStreamSynchronize(Collide_Stream_PreRec_4);
					deviceStreamSynchronize(Collide_Stream_PreRec_5);
					deviceStreamSynchronize(Collide_Stream_PreRec_6);
				}

				/*
				// Approach 1: Using a GPU copy kernel
				// Cuda kernel set-up
				int nThreadsPerBlock_SwapOldAndNew = 128;				//Number of threads per block for the Collision step
				dim3 nThreads_Swap(nThreadsPerBlock_SwapOldAndNew);
				int nBlocks_Swap = site_Count/nThreadsPerBlock_SwapOldAndNew			+ ((site_Count % nThreadsPerBlock_SwapOldAndNew > 0)         ? 1 : 0);

				if(nBlocks_Swap!=0)
					hemelb::GPU_SwapOldAndNew <<<nBlocks_Swap, nThreads_Swap, 0, stream_ReceivedDistr>>> ( (double*)GPUDataAddr_dbl_fOld_b, (double*)GPUDataAddr_dbl_fNew_b, site_Count, offset, (offset + site_Count));
					//hemelb::GPU_SwapOldAndNew <<<nBlocks_Swap, nThreads_Swap, 0, stream_SwapOldAndNew>>> ( (double*)GPUDataAddr_dbl_fOld_b, (double*)GPUDataAddr_dbl_fNew_b, site_Count, offset, (offset + site_Count));
				// End of Approach 1
				*/

				/*
				// 25-3-2021
				// Consider whether to Comment out the following and transfer in Step PostReceive()
				//========================================================================================================
				// Approach 2: Using memcpyDeviceToDevice:
				// As this is a single large copy from device global memory to device global memory, then  memcpyDeviceToDevice should be ok.
				// See the discussion here: https://stackoverflow.com/questions/22345391/cuda-device-memory-copies-memcpyDevicetodevice-vs-copy-kernel
				if (myPiD!=0) {
					bool status;
					unsigned long long MemSz = site_Count * LatticeType::NUMVECTORS * sizeof(distribn_t); // Total memory size
					status = deviceMemcpyAsync(&(((distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat)[0]), &(((distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat)[0]), MemSz, memcpyDeviceToDevice, stream_ReceivedDistr);
					if (!status) fprintf(stderr, "GPU memory copy device-to-device failed ... \n");
				}
				// End of Approach 2
				//========================================================================================================
				*/

				//========================================================================================================
				// Approach 3: SWap the pointers to GPU global memory:
				swap_Pointers_GPU_glb_mem(&(mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat),&(mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat));
				// End of Approach 3
				//========================================================================================================

				//========================================================================================================
				// Think where to place this!!! To do!!!
				//kernels::HydroVarsBase<LatticeType> hydroVars(geometry::Site<geometry::LatticeData> _site);
				//kernels::HydroVarsBase<LatticeType> hydroVars;
				// TODO: Need to use the frequency as specified in the input file (.xml) - Need to add something there
				// Or use the variable frequency_WriteGlobalMem defined in cuda_params.h

				// Dec 2023
				// TODO: Another option would be to use (in Read_Macrovariables_GPU_to_CPU) the variables:
				// Actually the first 2 RequiresRefresh at every time-step... Not useful then..
				// 1. Density/Pressure: 	propertyCache.densityCache.RequiresRefresh()
				// 2. Velocity: 					propertyCache.velocityCache.RequiresRefresh()
				// 3. Wall Shear Stress: propertyCache.wallShearStressMagnitudeCache.RequiresRefresh()

				lb::MacroscopicPropertyCache& propertyCache = GetPropertyCache();
				/*bool requires_MacroVars = (propertyCache.densityCache.RequiresRefresh() ||
																		propertyCache.velocityCache.RequiresRefresh() ||
																		propertyCache.velocityCache.RequiresRefresh()
																	) ? 1 : 0;
				if(requires_MacroVars)*/
				if (mState->GetTimeStep() % frequency_WriteGlobalMem == 0)
				{
					// Check whether the hemeLB picks up the macroVariables at the EndIteration step???
					// Only the data in propertyCache, i.e. propertyCache.densityCache and propertyCache.velocityCache
					if(myPiD!=0){
						//Read_Macrovariables_GPU_to_CPU(0, mLatDat->GetLocalFluidSiteCount(), propertyCache, kernels::HydroVars<LB_KERNEL> hydroVars(const geometry::Site<geometry::LatticeData>& _site)); // Copy the whole array GPUDataAddr_dbl_fNew_b from the GPU to CPUDataAddr_dbl_fNew_b. Then just read just the elements needed.
						bool res_Read_MacroVars_FromGPU = Read_Macrovariables_GPU_to_CPU(0, mLatDat->GetLocalFluidSiteCount(), propertyCache); // Practicaly in a synchronous way... Check if it can be modified in the future.
						if (!res_Read_MacroVars_FromGPU) printf("Rank: %d - Time: %ld - Error getting macroVars from GPU ... \n", myPiD, mState->GetTimeStep());
					}
				}

				//----------------------------

				// If checkpointing  functionality is required
				if(mLatDat->checkpointing_Get_Distr_To_Host)
				{
					// printf("Time: %ld, Boolean Checkpointing_Get_Distr_To_Host: %d\n", mState->GetTimeStep(), mLatDat->checkpointing_Get_Distr_To_Host );
					// Send the distribution functions (fNew GPU global memory) to fOld in CPU host memory
					if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU_FluidSites();
				}
				//========================================================================================================

				// Make sure the swap of distr. functions is completed, before the next iteration

				/*// Testing - Remove later:
				if (myPiD!=0)
				{
					cudaDeviceSynchronize(); // Included a deviceStreamSynchronize at the beginning of PreSend(); Should do the same job

					//kernels::HydroVarsBase<LatticeType> hydroVars(geometry::Site<geometry::LatticeData> const &site);

					for (site_t site_Index=0; site_Index< mLatDat->GetLocalFluidSiteCount(); site_Index++)
					{
						geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_Index);

						// Need to make it more general - Pass the Collision Kernel Impl. typename - To do!!!
			    	//kernels::HydroVars<lb::kernels::LBGK<lb::lattices::D3Q19> > hydroVars(site);
						kernels::HydroVars<LB_KERNEL> hydroVars(site);
						//printf("Density from densityCache: %.5f \n\n", propertyCache.densityCache.Get(site_Index));
						printf("Density from hydroVars: %.5f \n\n", hydroVars.density);
					}
					// Testing fails... Density = 0 from hydroVars.density...
				}
				*/

#else // If computations on CPU

				// Swap f_old and f_new ready for the next timestep.
				mLatDat->SwapOldAndNew();

#endif

				timings[hemelb::reporting::Timers::lb_calc].Stop();
				timings[hemelb::reporting::Timers::lb].Stop();
			}

		template<class LatticeType>
			LBM<LatticeType>::~LBM()
			{
				// Delete the collision and stream objects we've been using
				delete mMidFluidCollision;
				delete mWallCollision;
				delete mInletCollision;
				delete mOutletCollision;
				delete mInletWallCollision;
				delete mOutletWallCollision;

#ifdef HEMELB_USE_GPU
				Cleanup_GPU_pinned_Memory();
#endif
			}

		template<class LatticeType>
			void LBM<LatticeType>::ReadParameters()
			{
				std::vector<lb::iolets::InOutLet*> inlets = mSimConfig->GetInlets();
				std::vector<lb::iolets::InOutLet*> outlets = mSimConfig->GetOutlets();
				inletCount = inlets.size();
				outletCount = outlets.size();
				mParams.StressType = mSimConfig->GetStressType();

				//mParams.SetRelaxationParameter(mSimConfig->GetRelaxationParameter());
				//mParams.ElasticWallStiffness = mSimConfig->GetElasticWallStiffness();
				//mParams.BoundaryVelocityRatio = mSimConfig->GetBoundaryVelocityRatio();
				mParams.ViscosityRatio = mSimConfig->GetViscosityRatio();
				mParams.SpongeLayerWidth = mSimConfig->GetSpongeLayerWidth();
				mParams.SpongeLayerLifetime = mSimConfig->GetSpongeLayerLifetime();
				mParams.Smagorinsky_const = mSimConfig->GetCSmagorinsky();

				//printf("Number of inlets: %d, outlets: %d \n\n", inletCount, outletCount);
				//printf("Smagorinsky const: %.3f  \n\n", mParams.Smagorinsky_const);

			}

	}
}

#endif /* HEMELB_LB_LB_HPP */
