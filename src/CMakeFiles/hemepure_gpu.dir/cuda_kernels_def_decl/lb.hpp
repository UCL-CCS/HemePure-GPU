// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_LB_HPP
#define HEMELB_LB_LB_HPP


#include "io/writers/xdr/XdrMemWriter.h"
#include "lb/lb.h"


// Add the following line when calling the function:
// hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // Check for last cuda error: Remember that it is in DEBUG flag
inline void hemelb::check_cuda_errors(const char *filename, const int line_number, int myProc)
{
#ifdef DEBUG
	//printf("Debug mode...\n\n");
  //cudaDeviceSynchronize();
	  hipError_t error = hipGetLastError();
	  if(error != hipSuccess)
	  {
		printf("CUDA error at %s:%i: \"%s\" at proc: %i\n", filename, line_number, hipGetErrorString(error), myProc);
		abort();
		exit(-1);
	  }
#endif
}



namespace hemelb
{
	namespace lb
	{

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

				// IZ
				// Remove later the following
				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();
				// std::printf("Local Rank = %i \n\n", myPiD);
				// IZ


				kernels::InitParams initParams = kernels::InitParams();
				initParams.latDat = mLatDat;
				initParams.lbmParams = &mParams;
				initParams.neighbouringDataManager = neighbouringDataManager;

				unsigned collId;
				InitInitParamsSiteRanges(initParams, collId);
				mMidFluidCollision = new tMidFluidCollision(initParams);

				/*
				//IZ
				if(myPiD==1){
					for(int i = 0; i < initParams.siteRanges.size(); i++)
					{
						std::cout << "Value of state = " << collId << " siteRanges[i = " << i << "].first " << initParams.siteRanges[i].first << ", " << " siteRanges[i = " << i << "].second " << initParams.siteRanges[i].second << std::endl;
					}
				}
				//IZ
				*/

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

				/*
				std::cout << "Value of state = " << collId << std::endl;
				std::cout << "Value of  initParams.siteRanges[0].first = " << initParams.siteRanges[0].first  << std::endl;
				std::cout << "Value of  initParams.siteRanges[1].first = " << initParams.siteRanges[1].first  << std::endl;
				std::cout << "Value of  initParams.siteRanges[0].second = " << initParams.siteRanges[0].second  << std::endl;
				std::cout << "Value of  initParams.siteRanges[1].second = " << initParams.siteRanges[1].second  << std::endl;
				*/
				/*
				for(int i = 0; i < initParams.siteRanges.size(); i++)
				{
					std::cout << "i = " << i << " .First = " << initParams.siteRanges[i].first << ", " << initParams.siteRanges[i].second << std::endl;
				}
				*/

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

				SetInitialConditions();

/*
// Transfered calling the function Initialise_GPU() in SimulationMaster.cu
#ifdef HEMELB_USE_GPU

				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				// std::printf("Local Rank for Initialise= %i\n\n", myPiD);
				if (myPiD!=0){
					// Initialise the GPU here - Memory allocations etc
					bool res_InitGPU = Initialise_GPU();

					//hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // Check for last cuda error: Remember that it is in DEBUG flag

					// Initialise the kernels' setup
					// Initialise_kernels_GPU();
				}
#endif
*/
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
				hipError_t cudaStatus;

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

				//cudaStatus = cudaMemcpyAsync(&(((distribn_t*)GPUDataAddr_dbl_fOld_b)[nFluid_nodes * LatticeType::NUMVECTORS +1]), &(Data_dbl_fOld_Tr[0]), MemSz, cudaMemcpyHostToDevice, stream_memCpy_CPU_GPU_domainEdge);
				// This works as well:
				// Sept 2020 - Switch to Unified Memory (from GPUDataAddr_dbl_fOld_b to mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat)
				cudaStatus = hipMemcpyAsync(&(((distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat)[nFluid_nodes * LatticeType::NUMVECTORS +1]), mLatDat->GetFOld(mLatDat->neighbouringProcs[0].FirstSharedDistribution), MemSz, hipMemcpyHostToDevice, stream_ReceivedDistr); // stream_memCpy_CPU_GPU_domainEdge);

				//cudaStatus = cudaMemcpy(&(((distribn_t*)GPUDataAddr_dbl_fOld_b)[nFluid_nodes * LatticeType::NUMVECTORS +1]), &(Data_dbl_fOld_Tr[0]), MemSz, cudaMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) fprintf(stderr, "GPU memory copy host-to-device failed ... \n");

				// Delete when the mem.copy is complete
				//delete[] Data_dbl_fOld_Tr; 				// Cannot delete as it is pointing to the following: mLatDat->GetFOld(mLatDat->neighbouringProcs[0].FirstSharedDistribution);
				//delete[] Data_dbl_fOld_Tr_test;		// Cannot delete for the same reason!!! points to : mLatDat->GetFOld(nFluid_nodes * LatticeType::NUMVECTORS +1)
				// delete[] Data_dbl_fOld_Tr_test2; 	// This one is possible

				return true;
			}



		template<class LatticeType>
			bool LBM<LatticeType>::Read_DistrFunctions_CPU_to_GPU(int64_t firstIndex, int64_t siteCount)
			{
				hipError_t cudaStatus;

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
				// cudaStatus = cudaMemcpy(GPUDataAddr_dbl_fOld, Data_dbl_fOld, nArray_oldDistr * sizeof(distribn_t), cudaMemcpyHostToDevice);
				// if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer Host To Device failed - \n"); return false; }

				// Send iteratively the f_0, f_1, f_2, ..., f_(q-1) to the corresponding GPU mem. address
				long long MemSz = siteCount * sizeof(distribn_t);	// Memory size for each of the fi's send - Carefull: This is not the total Memory Size!!!

				for (int LB_ind=0; LB_ind < LatticeType::NUMVECTORS; LB_ind++)
				{
					cudaStatus = hipMemcpy(&(((distribn_t*)GPUDataAddr_dbl_fOld_b)[(LB_ind*nFluid_nodes)+firstIndex]), &(Data_dbl_fOld_Tr[LB_ind * siteCount]), MemSz, hipMemcpyHostToDevice);
					if (cudaStatus != hipSuccess) fprintf(stderr, "GPU memory copy failed (%d)\n", LB_ind);
				}


				delete[] Data_dbl_fOld_Tr;

				return true;
			}


			// Function for reading:
			//	a. the Distribution Functions post-collision, fNew, in totalSharedFs
			// 				that will be send to the neighbouring ranks
			// 		from the GPU and copying to the CPU (device-to-host mem. copy - Asynchronous)

			//
			// If we use cudaMemcpy: Remember that from the host perspective the mem copy is synchronous, i.e. blocking
			// so the host will wait the data transfer to complete and then proceed to the next function call

			// Switched to cudaMemcpyAsync(): non-blocking on the host,
			//		so control returns to the host thread immediately after the transfer is issued.
			// 	cuda stream: mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2()
			//=================================================================================================
		template<class LatticeType>
			bool LBM<LatticeType>::Read_DistrFunctions_GPU_to_CPU_totalSharedFs()
			{
				hipError_t cudaStatus;

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
				cudaStatus = cudaMemcpyAsync(&(fNew_GPU_totalSharedFs[0]), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[nFluid_nodes * LatticeType::NUMVECTORS]), MemSz, cudaMemcpyDeviceToHost, mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2() );
				//cudaStatus = cudaMemcpyAsync(&(fNew_GPU_totalSharedFs[0]), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[nFluid_nodes * LatticeType::NUMVECTORS]), MemSz, cudaMemcpyDeviceToHost, stream_memCpy_GPU_CPU_domainEdge);

				//cudaStatus = cudaMemcpy(&(fNew_GPU_totalSharedFs[0]), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[nFluid_nodes * LatticeType::NUMVECTORS]), MemSz, cudaMemcpyDeviceToHost);

				if(cudaStatus != cudaSuccess){
					const char * eStr = cudaGetErrorString (cudaStatus);
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
				//cudaStatus = cudaMemcpyAsync( mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[nFluid_nodes * LatticeType::NUMVECTORS]), MemSz, cudaMemcpyDeviceToHost, mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2() );
				// Sept 2020 - Switching to cuda-aware mpi makes the D2H mem.copy not necessary. Also switching to Using Unified Memory
				// Does the following make sense though: case of NO cuda-aware mpi (in which case have to call D2H memcpy) and Unified Memory???
				cudaStatus = hipMemcpyAsync( mLatDat->GetFNew(nFluid_nodes * LatticeType::NUMVECTORS), &(((distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat)[nFluid_nodes * LatticeType::NUMVECTORS]), MemSz, hipMemcpyDeviceToHost, mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2() );

				if(cudaStatus != hipSuccess){
					const char * eStr = hipGetErrorString (cudaStatus);
					printf("GPU memory transfer for ReadGPU_distr totalSharedFs failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					return false;
				}
				//======================================================================


				return true;
			} // Ends the Read_DistrFunctions_GPU_to_CPU_totalSharedFs


			//=================================================================================================
			// Function to:
			// 	Perform a memory copy from Host to Device (to GPU global memory) for the wall momentum and the case of Velocity Inlet/Outlet BCs
			// 		Synchronous memcpy at the moment. TODO: change to Asynchronous memcpy.
			//=================================================================================================
		template<class LatticeType>
			bool LBM<LatticeType>::memCpy_HtD_GPUmem_WallMom(site_t firstIndex, site_t siteCount, std::vector<util::Vector3D<double> >& wallMom_Iolet, void *GPUDataAddr_wallMom)
			{
				hipError_t cudaStatus;

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
				cudaStatus = hipMemcpy(GPUDataAddr_wallMom, Data_dbl_WallMom, 3*nArr_wallMom * sizeof(distribn_t), hipMemcpyHostToDevice);

				//======================================================================
				if(cudaStatus != hipSuccess){
					const char * eStr = hipGetErrorString (cudaStatus);
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
			// 1. read the wall momentum for the case of Velocity Inlet/Outlet BCs
			// 2. fill the appropriate vector that will be used to send the data to the GPU global memory
			//=================================================================================================
		template<class LatticeType>
			void LBM<LatticeType>::read_WallMom_from_propertyCache(site_t firstIndex, site_t siteCount, lb::MacroscopicPropertyCache& propertyCache,
																															std::vector<util::Vector3D<double> >& wallMom_Iolet)
			{
				for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
				{
					for (unsigned int direction = 0; direction < LatticeType::NUMVECTORS; direction++)
					{
						LatticeVelocity site_WallMom = propertyCache.wallMom_Cache.Get((siteIndex-firstIndex)*LatticeType::NUMVECTORS+direction);
						/*
						if(site_WallMom.x !=0 || site_WallMom.y !=0 || site_WallMom.z !=0)
							printf("Received Wall Mom in LBM - Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction,
																			site_WallMom.x,
																			site_WallMom.y,
																			site_WallMom.z);
						*/
						wallMom_Iolet.push_back(site_WallMom);
					}
				}

				/*
				for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
				{
					for (unsigned int direction = 0; direction < LatticeType::NUMVECTORS; direction++)
					{
						printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction,
																		wallMom_Iolet[(siteIndex-firstIndex)*LatticeType::NUMVECTORS+direction].x,
																		wallMom_Iolet[(siteIndex-firstIndex)*LatticeType::NUMVECTORS+direction].y,
																		wallMom_Iolet[(siteIndex-firstIndex)*LatticeType::NUMVECTORS+direction].z);
					}
				}
				*/

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
				hipError_t cudaStatus;

			  // Total number of fluid sites
			  uint64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount(); // Actually GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)

				//--------------------------------------------------------------------------
			  //	a. Density

			  distribn_t* dens_GPU = new distribn_t[siteCount];

			  if(dens_GPU==0){printf("Density Memory allocation failure"); return false;}

			  unsigned long long MemSz = siteCount*sizeof(distribn_t);

			  //cudaStatus = cudaMemcpy(dens_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[firstIndex]), MemSz, cudaMemcpyDeviceToHost);
			  cudaStatus = hipMemcpyAsync(dens_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[firstIndex]), MemSz, hipMemcpyDeviceToHost, stream_Read_Data_GPU_Dens);

			  if(cudaStatus != hipSuccess){
			    printf("GPU memory transfer for density failed\n");
			    delete[] dens_GPU;
			    return false;
			  }

			  // b. Velocity
			  distribn_t* vx_GPU = new distribn_t[siteCount];
			  distribn_t* vy_GPU = new distribn_t[siteCount];
			  distribn_t* vz_GPU = new distribn_t[siteCount];

			  if(vx_GPU==0 || vy_GPU==0 || vz_GPU==0){ printf("Memory allocation failure"); return false;}

			  cudaStatus = hipMemcpyAsync(vx_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[1ULL*nFluid_nodes + firstIndex]), MemSz, hipMemcpyDeviceToHost, stream_Read_Data_GPU_Dens);
			  //cudaStatus = cudaMemcpyAsync(vx_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[1ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Vel);
			  if(cudaStatus != hipSuccess){
			    printf("GPU memory transfer Vel(1) failed\n");
			    delete[] vx_GPU;
			    return false;
			  }

			  cudaStatus = hipMemcpyAsync(vy_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[2ULL*nFluid_nodes + firstIndex]), MemSz, hipMemcpyDeviceToHost, stream_Read_Data_GPU_Dens);
			  //cudaStatus = cudaMemcpyAsync(vy_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[2ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Vel);
			  if(cudaStatus != hipSuccess){
			    printf("GPU memory transfer Vel(2) failed\n");
			    delete[] vy_GPU;
			    return false;
			  }

			  cudaStatus = hipMemcpyAsync(vz_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[3ULL*nFluid_nodes + firstIndex]), MemSz, hipMemcpyDeviceToHost, stream_Read_Data_GPU_Dens);
			  //cudaStatus = cudaMemcpyAsync(vz_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[3ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Vel);
			  if(cudaStatus != hipSuccess){
			    printf("GPU memory transfer Vel(2) failed\n");
			    delete[] vz_GPU;
			    return false;
			  }
			  //--------------------------------------------------------------------------
			  //hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // Check for last cuda error: Remember that it is in DEBUG flag

				hipStreamSynchronize(stream_Read_Data_GPU_Dens);
				//
			  // Read only the density, velocity and fNew[] that needs to be passed to the CPU at the updated sites: The ones that had been updated in the GPU collision kernel
			  for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
			  {
			    geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(siteIndex);
			    // printf("site.GetIndex() = %lld Vs siteIndex = %lld \n\n", site.GetIndex(), siteIndex); // Works fine - Access to the correct site

			    //
			    // Need to make it more general - Pass the Collision Kernel Impl. typename - To do!!!
					kernels::HydroVars<LB_KERNEL> hydroVars(site);
					//kernels::HydroVars<lb::kernels::LBGK<lb::lattices::D3Q19> > hydroVars(site);
					//kernels::HydroVarsBase<LatticeType> hydroVars(site);

			    // Pass the density and velocity to the hydroVars and the densityCache, velocityCache
			    hydroVars.density = dens_GPU[siteIndex-firstIndex];
			    hydroVars.velocity.x = vx_GPU[siteIndex-firstIndex];
			    hydroVars.velocity.y = vy_GPU[siteIndex-firstIndex];
			    hydroVars.velocity.z = vz_GPU[siteIndex-firstIndex];

					// TODO: I will need to change the following so that it gets updated only
					// if (propertyCache.densityCache.RequiresRefresh())
					// if (propertyCache.velocityCache.RequiresRefresh())
			    propertyCache.densityCache.Put(siteIndex, hydroVars.density);		//propertyCache.densityCache.Put(site.GetIndex(), hydroVars.density);
			    propertyCache.velocityCache.Put(siteIndex, hydroVars.velocity);	//propertyCache.velocityCache.Put(site.GetIndex(), hydroVars.velocity);

					// TODO: Check that the MacroVariables (density etc)  are actually written
					//printf("Reading Density: %.5f \n\n", dens_GPU[siteIndex-firstIndex]); // Successful !
					//printf("Reading Density from HydroVars: %.5f \n\n", hydroVars.density);
				}


				// Free memory once the mem.copies are Completed
				delete[] dens_GPU;
				delete[] vx_GPU, vy_GPU, vz_GPU;

				return true;
			}



			//=================================================================================================
			// Function for reading:
			//	a. the Distribution Functions post-collision, fNew,
			//	b. Density [nFluid nodes]
			//	c. Velocity[nFluid nodes*3]
			// from the GPU and copying to the CPU (device-to-host mem. copy - Synchronous)
			//
			// Development phase:
			//	Necessary at each time step as ALL data need to reside on the CPU
			// Final phase: (All collision/streaming types implemented)
			//	a.	to be called at the domain bundaries
			//		for the exchange of the fNew to be exchanged
			//	b.	When data needs to be saved to the disk on the CPU
			//
			// Remember that from the host perspective the mem copy is synchronous, i.e. blocking
			// so the host will wait the data transfer to complete and then proceed to the next function call
			//=================================================================================================
		template<class LatticeType>
			bool LBM<LatticeType>::Read_DistrFunctions_GPU_to_CPU(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache) // Is it necessary to use lb::MacroscopicPropertyCache& propertyCache or just propertyCache, as it is being initialised with the LBM constructor???
			{
				hipError_t cudaStatus;

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

				//cudaStatus = cudaMemcpyAsync(fNew_GPU_b, &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[0]), TotalMem_dbl_fNew_b, cudaMemcpyDeviceToHost, stream_Read_distr_Data_GPU);
				cudaStatus = hipMemcpy(&(fNew_GPU_b[0]), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[0]), TotalMem_dbl_fNew_b, hipMemcpyDeviceToHost);
				if(cudaStatus != hipSuccess){
					const char * eStr = hipGetErrorString (cudaStatus);
					printf("GPU memory transfer for ReadGPU_distr failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					delete[] fNew_GPU_b;
					return false;
				}

				//--------------------------------------------------------------------------
				//	b. Density

				distribn_t* dens_GPU = new distribn_t[nFluid_nodes];

				if(dens_GPU==0){printf("Density Memory allocation failure"); return false;}

				unsigned long long MemSz = nFluid_nodes*sizeof(distribn_t);

				cudaStatus = hipMemcpy(dens_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[0]), MemSz, hipMemcpyDeviceToHost);
				//cudaStatus = cudaMemcpyAsync(dens_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[0]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Dens);

				if(cudaStatus != hipSuccess){
					printf("GPU memory transfer for density failed\n");
					delete[] dens_GPU;
					return false;
				}

				// c. Velocity
				distribn_t* vx_GPU = new distribn_t[nFluid_nodes];
				distribn_t* vy_GPU = new distribn_t[nFluid_nodes];
				distribn_t* vz_GPU = new distribn_t[nFluid_nodes];

				if(vx_GPU==0 || vy_GPU==0 || vz_GPU==0){ printf("Memory allocation failure"); return false;}

				cudaStatus = hipMemcpy(vx_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[1ULL*nFluid_nodes]), MemSz, hipMemcpyDeviceToHost);
				//cudaStatus = cudaMemcpyAsync(vx_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[1ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Vel);
				if(cudaStatus != hipSuccess){
					printf("GPU memory transfer Vel(1) failed\n");
					delete[] vx_GPU;
					return false;
				}

				cudaStatus = hipMemcpy(vy_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[2ULL*nFluid_nodes]), MemSz, hipMemcpyDeviceToHost);
				//cudaStatus = cudaMemcpyAsync(vy_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[2ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Vel);
				if(cudaStatus != hipSuccess){
					printf("GPU memory transfer Vel(2) failed\n");
					delete[] vy_GPU;
					return false;
				}

				cudaStatus = hipMemcpy(vz_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[3ULL*nFluid_nodes]), MemSz, hipMemcpyDeviceToHost);
				//cudaStatus = cudaMemcpyAsync(vz_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[3ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Vel);
				if(cudaStatus != hipSuccess){
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
						// FNew index in hemeLB array (after streaming): site.GetStreamedIndex<LatticeType> (ii) = the element in the array neighbourIndices[iSiteIndex * LatticeType::NUMVECTORS + iDirectionIndex];
						//
						// int64_t streamedIndex = site.GetStreamedIndex<LatticeType> (ii); // ii: direction

						// given the streamed index value find the fluid ID index: iFluidIndex = (Array_Index - iDirectionIndex)/NumVectors,
						//	i.e. iFluidIndex = (site.GetStreamedIndex<LatticeType> (ii) - ii)/NumVectors;
						// Applies if streaming ends within the domain in the same rank.
						// If not then the postcollision fNew will stream in the neighbouring rank.
						// It will be placed then in location for the totalSharedFs

						// Need to include the case of inlet BCs - Unstreamed Unknown populations - To do!!!
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
			bool LBM<LatticeType>::Initialise_kernels_GPU()
			{
				// Maybe better to have these details outside of a class
				// See file cuda_params.cu
				// Think about this

				// Include the info for the kernels set-up
				// Kernel related parameters
				// int nThreadsPerBlock_Collide = 32;				//Number of threads per block for the Collision step

				return true;
			}

		template<class LatticeType>
			void LBM<LatticeType>::get_Iolet_BCs(std::string hemeLB_IoletBC_Inlet, std::string hemeLB_IoletBC_Outlet)
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



		template<class LatticeType>
			bool LBM<LatticeType>::FinaliseGPU()
			{
				hipError_t cudaStatus;
				std::string hemeIoletBC_Inlet, hemeIoletBC_Outlet;
				get_Iolet_BCs(hemeIoletBC_Inlet, hemeIoletBC_Outlet);

				// Cuda Streams
				hipStreamDestroy(Collide_Stream_PreSend_1);
				hipStreamDestroy(Collide_Stream_PreSend_2);
				hipStreamDestroy(Collide_Stream_PreSend_3);
				hipStreamDestroy(Collide_Stream_PreSend_4);
				hipStreamDestroy(Collide_Stream_PreSend_5);
				hipStreamDestroy(Collide_Stream_PreSend_6);

				hipStreamDestroy(Collide_Stream_PreRec_1);
				hipStreamDestroy(Collide_Stream_PreRec_2);
				hipStreamDestroy(Collide_Stream_PreRec_3);
				hipStreamDestroy(Collide_Stream_PreRec_4);
				hipStreamDestroy(Collide_Stream_PreRec_5);
				hipStreamDestroy(Collide_Stream_PreRec_6);
				//	cudaStreamDestroy(stream_Read_distr_Data_GPU);

				hipStreamDestroy(stream_Read_Data_GPU_Dens);

				hipStreamDestroy(stream_ghost_dens_inlet);
				hipStreamDestroy(stream_ghost_dens_outlet);
				hipStreamDestroy(stream_ReceivedDistr);
				hipStreamDestroy(stream_SwapOldAndNew);
				hipStreamDestroy(stream_memCpy_CPU_GPU_domainEdge);

				// Destroy the cuda stream created for the asynch. MemCopy DtH at the domain edges: created a stream in net::BaseNet object
				hemelb::net::Net& mNet_cuda_stream = *mNet;	// Access the mNet object
				mNet_cuda_stream.Destroy_stream_memCpy_GPU_CPU_domainEdge_new2(); // Which one is correct? Does it actually create the stream and then it imposes a barrier in net::BaseNet::Send

				//cudaStreamDestroy(stream_memCpy_GPU_CPU_domainEdge);



				// Free GPU memory
				/*
				cudaStatus = cudaFree(GPUDataAddr_dbl_fOld);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }

				cudaStatus = cudaFree(GPUDataAddr_dbl_fNew);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }
				*/

				cudaStatus = hipFree(GPUDataAddr_dbl_MacroVars);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }

				/*cudaStatus = cudaFree(GPUDataAddr_int64_Neigh);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }
				*/

				cudaStatus = hipFree(GPUDataAddr_uint32_Wall);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }

				cudaStatus = hipFree(GPUDataAddr_uint32_Iolet);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }

				if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
					cudaStatus = hipFree(d_ghostDensity);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree ghost Density inlet failed\n"); return false; }
				}

				cudaStatus = hipFree(d_inletNormal);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }

				if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
					cudaStatus = hipFree(d_ghostDensity_out);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree ghost density outlet failed\n"); return false; }
				}

				cudaStatus = hipFree(d_outletNormal);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }


				cudaStatus = hipFree(GPUDataAddr_dbl_fOld_b);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }

				cudaStatus = hipFree(GPUDataAddr_dbl_fNew_b);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }

				cudaStatus = hipFree(mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }

				cudaStatus = hipFree(mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }

				cudaStatus = hipFree(GPUDataAddr_int64_Neigh_d);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "hipFree failed\n"); return false; }

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


				//printf("CudaFree - Delete dynamically allocated memory on the GPU.\n\n");

				return true;
			}



template<class LatticeType>
			bool LBM<LatticeType>::Initialise_GPU(iolets::BoundaryValues* iInletValues,
					iolets::BoundaryValues* iOutletValues, const util::UnitConverter* iUnits)
			{

				mInletValues = iInletValues;
				mOutletValues = iOutletValues;
				mUnits = iUnits;

				hipError_t cudaStatus;

				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();
				// std::printf("Local Rank = %i and local fluid sites = %i \n\n", myPiD, mLatDat->GetLocalFluidSiteCount());

				// --------------------------------------------------------------------------------------------------------------------------------------------------------------
				// Distribution functions:
				// 	Option: Separately each element in memory
				//	f_old, f_new -	Comment: add the totalSharedFs values +1: Done!!!

				// Arrange the data in 2 ways - Done!!!
				//	a. Arrange by fluid index (as is oldDistributions), i.e f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]
				//	b. Arrange by index_LB, i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., fq[0 to (nFluid_nodes-1)]
				// KEEP ONLY Option (b) - Refer to earlier versions of the code for option (a).

				// Total number of fluid sites
				uint64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount(); // Actually GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)
				uint64_t totSharedFs = mLatDat->totalSharedFs;
				// std::printf("Proc# %i : Total Fluid nodes = %i, totalSharedFs = %i \n\n", myPiD, nFluid_nodes, totSharedFs);	// Test that I can access the value of totalSharedFs (protected member of class LatticeData (geometry/LatticeData.h) - declares class LBM as friend)

				unsigned long long TotalMem_dbl_fOld = ( nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs)  * sizeof(distribn_t); // Total memory size for fOld
				unsigned long long TotalMem_dbl_fNew = TotalMem_dbl_fOld;	// Total memory size for fNew
				unsigned long long TotalMem_dbl_MacroVars = (1+3) * nFluid_nodes  * sizeof(distribn_t); // Total memory size for macroVariables: density and Velocity n[nFluid_nodes], u[nFluid_nodes][3]

				//--------------------------------------------------------------------------------------------------
				// Alocate memory on the GPU for MacroVariables: density and Velocity
				// Number of elements (type double / distribn_t)
				// uint64_t nArray_MacroVars = nFluid_nodes; // uint64_t (unsigned long long int)

				cudaStatus = hipMalloc((void**)&GPUDataAddr_dbl_MacroVars, TotalMem_dbl_MacroVars);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation failed\n"); return false; }
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
					return false;
				}

				// 	f_old - Done!!!
				for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++)
				{
					for (site_t i = 0; i < mLatDat->GetLocalFluidSiteCount(); i++)
					{
						*(&Data_dbl_fOld_b[l * mLatDat->GetLocalFluidSiteCount() + i]) = *(mLatDat->GetFOld(i * LatticeType::NUMVECTORS + l)); // distribn_t (type double) - Data_dbl_fOld contains the oldDistributions re-arranged
					}
				}

				// 	f_new - Done!!!
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
				cudaStatus = hipMalloc((void**)&GPUDataAddr_dbl_fOld_b, nArray_Distr * sizeof(distribn_t));
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation failed\n"); return false; }

				cudaStatus = hipMalloc((void**)&GPUDataAddr_dbl_fNew_b, nArray_Distr * sizeof(distribn_t));
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation failed\n"); return false; }

				// Memory copy from host (Data_dbl_fOld_b) to Device (GPUDataAddr_dbl_fOld_b)
				cudaStatus = hipMemcpy(GPUDataAddr_dbl_fOld_b, Data_dbl_fOld_b, nArray_Distr * sizeof(distribn_t), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device failed\n"); return false; }

				// Memory copy from host (Data_dbl_fNew_b) to Device (GPUDataAddr_dbl_fNew_b)
				cudaStatus = hipMemcpy(GPUDataAddr_dbl_fNew_b, Data_dbl_fNew_b, nArray_Distr * sizeof(distribn_t), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device failed\n"); return false; }

				// For cuda-aware mpi we need to have access to
				// Test if I can access the pointer to global memory declared in class LatticeData (GPUDataAddr_dbl_fOld_b_mLatDat)
				// (geometry::LatticeData* mLatDat;)
				// Memory copy from host (Data_dbl_fOld_b) to Device (mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat)
				//cudaStatus = cudaMallocManaged((void**)&(mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat), nArray_Distr * sizeof(distribn_t));
				cudaStatus = hipMalloc((void**)&(mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat), nArray_Distr * sizeof(distribn_t));
				cudaStatus = hipMemcpy(mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																	Data_dbl_fOld_b, nArray_Distr * sizeof(distribn_t), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device failed\n"); return false; }

				//cudaStatus = cudaMallocManaged((void**)&(mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat), nArray_Distr * sizeof(distribn_t));
				cudaStatus = hipMalloc((void**)&(mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat), nArray_Distr * sizeof(distribn_t));
				cudaStatus = hipMemcpy(mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																	Data_dbl_fNew_b, nArray_Distr * sizeof(distribn_t), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device failed\n"); return false; }
				//=================================================================================================================================


				//=================================================================================================================================
				// Neighbouring indices - necessary for the STREAMING STEP

				// 	The total size of the neighbouring indices should be: neighbourIndices.resize(latticeInfo.GetNumVectors() * localFluidSites); (see geometry/LatticeData.cc:369)
				// 	Keep only method d: refer to the actual streaming index in f's array (arranged following method (b): Arrange by index_LB)

				//		Type site_t (units.h:28:		typedef int64_t site_t;)
				//		geometry/LatticeData.h:634:		std::vector<site_t> neighbourIndices; //! Data about neighbouring fluid sites.
				//	Memory requirements
				unsigned long long TotalMem_int64_Neigh = ( nFluid_nodes * LatticeType::NUMVECTORS)  * sizeof(site_t); // Total memory size for neighbouring Indices

				// -----------------------------------------------------------------------
				// d. Arrange by index_LB, i.e. neigh_0[0 to (nFluid_nodes-1)], neigh_1[0 to (nFluid_nodes-1)], ..., neigh_(q-1)[0 to (nFluid_nodes-1)]
				//		But instead of keeping the array index from HemeLB, convert to refer to the actual fluid ID and then to the actual address in global Memory
				site_t* Data_int64_Neigh_d = new site_t[TotalMem_int64_Neigh/sizeof(site_t)];	// site_t (type int64_t)
				if(!Data_int64_Neigh_d){
					std::cout << "Memory allocation error - Neigh. (d)" << std::endl;
					return false;
				}

				// Re-arrange the neighbouring data - organised by index LB
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
				cudaStatus = hipMalloc((void**)&GPUDataAddr_int64_Neigh_d, nArray_Neigh * sizeof(site_t));
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation for Neigh.(d) failed\n"); return false; }

				// Memory copy from host (Data_int64_Neigh_b) to Device (GPUDataAddr_int64_Neigh_b)
				cudaStatus = hipMemcpy(GPUDataAddr_int64_Neigh_d, Data_int64_Neigh_d, nArray_Neigh * sizeof(site_t), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Neigh.(d) failed\n"); return false; }
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

				unsigned long long TotalMem_uint32_WallIntersect = nFluid_nodes * sizeof(uint32_t); // Total memory size

				// Allocate memory on the host
				// Think about the following: Do I need to allocate nFluid_nodes or just the siteCount for this type of collision (check the limits for the mWallCollision). To do!!!
				uint32_t* Data_uint32_WallIntersect = new uint32_t[nFluid_nodes];	// distribn_t (type double)
				if(!Data_uint32_WallIntersect){ std::cout << "Memory allocation error - Neigh." << std::endl; return false;}

				// Fill the array Data_uint32_WallIntersect
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
				cudaStatus = hipMalloc((void**)&GPUDataAddr_uint32_Wall, nFluid_nodes * sizeof(uint32_t));
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation for Wall-Fluid Intersection failed\n"); return false; }

				// Memory copy from host (Data_uint32_WallIntersect) to Device (GPUDataAddr_uint32_Wall)
				cudaStatus = hipMemcpy(GPUDataAddr_uint32_Wall, Data_uint32_WallIntersect, nFluid_nodes * sizeof(uint32_t), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Neigh. failed\n"); return false; }
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

				unsigned long long TotalMem_uint32_IoletIntersect = nFluid_nodes * sizeof(uint32_t); // Total memory size

				// Allocate memory on the host
				uint32_t* Data_uint32_IoletIntersect = new uint32_t[nFluid_nodes];	// distribn_t (type double)
				if(!Data_uint32_IoletIntersect){ std::cout << "Memory allocation error - iolet" << std::endl; return false;}

				// Fill the array Data_uint32_IoletIntersect
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
				cudaStatus = hipMalloc((void**)&GPUDataAddr_uint32_Iolet, nFluid_nodes * sizeof(uint32_t));
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation for Iolet-Fluid Intersection failed\n"); return false; }

				// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
				cudaStatus = hipMemcpy(GPUDataAddr_uint32_Iolet, Data_uint32_IoletIntersect, nFluid_nodes * sizeof(uint32_t), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet failed\n"); return false; }
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
				//		Note that the value returned from GetLocalIoletCount() is the global iolet count!!! NOT the local iolet count on the current RANK.
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
				int n_Inlets = mInletValues->GetLocalIoletCount();
				int n_Outlets = mOutletValues->GetLocalIoletCount();
				// printf("Rank: %d, Number of inlets: %d, Outlets: %d \n\n", myPiD, n_Inlets, n_Outlets);

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
					Iolets_Inlet_Edge = identify_Range_iolets_ID(start_Index_Inlet_Edge, (start_Index_Inlet_Edge + site_Count_Inlet_Edge), &n_LocalInlets_mInlet_Edge, &n_unique_LocalInlets_mInlet_Edge);
					// TODO: Call function to prepare the struct object (with FAM for the array Iolets_ID_range):
					//				struct Iolets *createIolet(struct Iolets *iolet_member, int number_LocalIolets, int number_UniqueLocalIolets)
					Inlet_Edge.n_local_iolets = n_LocalInlets_mInlet_Edge;
					memcpy(&Inlet_Edge.Iolets_ID_range, &Iolets_Inlet_Edge[0], sizeof(Inlet_Edge.Iolets_ID_range));

					/*std::cout << "Rank: " << myPiD << " - Unique local Inlets: " << n_unique_LocalInlets_mInlet_Edge << " - Total local Inlets on current Rank (1st Round - mInlet_Edge): " << n_LocalInlets_mInlet_Edge << " with Inlet ID:";
					for (int index = 0; index < n_LocalInlets_mInlet_Edge; index++ )
						std::cout << ' ' << Iolets_Inlet_Edge[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalInlets_mInlet_Edge *  sizeof(site_t);
					cudaStatus = hipMalloc((void**)&GPUDataAddr_Inlet_Edge, MemSz);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation Iolet: Inlet Edge failed...\n"); return false; }

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					cudaStatus = hipMemcpy(GPUDataAddr_Inlet_Edge, &Iolets_Inlet_Edge[0], MemSz, hipMemcpyHostToDevice);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Inlet Edge failed... \n"); return false; }
				}

				// Case 1.1.b. Collision Type 5: mInletWallCollision
				//std::vector<site_t> Iolets_InletWall_Edge;
				n_LocalInlets_mInletWall_Edge = 0; // number of local Inlets involved during the mInletWallCollision collision- Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalInlets_mInletWall_Edge = 0;
				if(site_Count_InletWall_Edge != 0){
					Iolets_InletWall_Edge = identify_Range_iolets_ID(start_Index_InletWall_Edge, (start_Index_InletWall_Edge + site_Count_InletWall_Edge), &n_LocalInlets_mInletWall_Edge, &n_unique_LocalInlets_mInletWall_Edge);
					// Remove later the struct object InletWall_Edge
					InletWall_Edge.n_local_iolets = n_LocalInlets_mInletWall_Edge;
					memcpy(&InletWall_Edge.Iolets_ID_range, &Iolets_InletWall_Edge[0], sizeof(InletWall_Edge.Iolets_ID_range));
					//

					/*std::cout << "Rank: " << myPiD << " - Unique local Inlets: " << n_unique_LocalInlets_mInletWall_Edge << " - Total local Inlets on current Rank (1st Round - mInletWall_Edge): " << n_LocalInlets_mInletWall_Edge << " with Inlet ID:";
					for (int index = 0; index < n_LocalInlets_mInletWall_Edge; index++ )
						std::cout << ' ' << Iolets_InletWall_Edge[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalInlets_mInletWall_Edge *  sizeof(site_t);
					cudaStatus = hipMalloc((void**)&GPUDataAddr_InletWall_Edge, MemSz);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation Iolet: Inlet Wall Edge failed...\n"); return false; }

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					cudaStatus = hipMemcpy(GPUDataAddr_InletWall_Edge, &Iolets_InletWall_Edge[0], MemSz, hipMemcpyHostToDevice);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Inlet Wall Edge failed... \n"); return false; }
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
					// Remove later the object Inlet_Inner - Info in GPU global memory
					Inlet_Inner.n_local_iolets = n_LocalInlets_mInlet;
					memcpy(&Inlet_Inner.Iolets_ID_range, &Iolets_Inlet_Inner[0], sizeof(Inlet_Inner.Iolets_ID_range));
					//

					/*std::cout << "Rank: " << myPiD << " - Unique local Inlets: " << n_unique_LocalInlets_mInlet << " - Total local Inlets on current Rank (1st Round - mInlet): " << n_LocalInlets_mInlet << " with Inlet ID:";
					for (int index = 0; index < n_LocalInlets_mInlet; index++ )
						std::cout << ' ' << Iolets_Inlet_Inner[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalInlets_mInlet *  sizeof(site_t);
					cudaStatus = hipMalloc((void**)&GPUDataAddr_Inlet_Inner, MemSz);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation Iolet: Inlet Inner failed...\n"); return false; }

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					cudaStatus = hipMemcpy(GPUDataAddr_Inlet_Inner, &Iolets_Inlet_Inner[0], MemSz, hipMemcpyHostToDevice);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Inlet Inner failed\n"); return false; }
				}

				// Case 1.2.b. 	Collision Type 5: mInletWallCollision
				//std::vector<site_t> Iolets_InletWall_Inner;
				n_LocalInlets_mInletWall = 0; // number of local Inlets involved during the mInletWallCollision collision - Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalInlets_mInletWall = 0;
				if(site_Count_InletWall_Inner!=0)
				{
					Iolets_InletWall_Inner = identify_Range_iolets_ID(start_Index_InletWall_Inner, (start_Index_InletWall_Inner + site_Count_InletWall_Inner), &n_LocalInlets_mInletWall, &n_unique_LocalInlets_mInletWall);
					// Remove later the object InletWall_Inner - Replaced with array in GPU global memory
					InletWall_Inner.n_local_iolets = n_LocalInlets_mInletWall;
					memcpy(&InletWall_Inner.Iolets_ID_range, &Iolets_InletWall_Inner[0], sizeof(InletWall_Inner.Iolets_ID_range));
					//

					/*std::cout << "Rank: " << myPiD << " - Unique local Inlets: " << n_unique_LocalInlets_mInletWall << " - Total local Inlets on current Rank (1st Round - mInletWall): " << n_LocalInlets_mInletWall << " with Inlet ID:";
					for (int index = 0; index < n_LocalInlets_mInletWall; index++ )
						std::cout << ' ' << Iolets_InletWall_Inner[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalInlets_mInletWall *  sizeof(site_t);
					cudaStatus = hipMalloc((void**)&GPUDataAddr_InletWall_Inner, MemSz);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation Iolet: Inlet Wall Inner failed...\n"); return false; }

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					cudaStatus = hipMemcpy(GPUDataAddr_InletWall_Inner, &Iolets_InletWall_Inner[0], MemSz, hipMemcpyHostToDevice);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Inlet Wall Inner failed... \n"); return false; }
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
					// Remove later the struct object Outlet_Edge - Replaced with GPU global memory array
					Outlet_Edge.n_local_iolets = n_LocalOutlets_mOutlet_Edge;
					memcpy(&Outlet_Edge.Iolets_ID_range, &Iolets_Outlet_Edge[0], sizeof(Outlet_Edge.Iolets_ID_range));
					//

					/*std::cout << "Rank: " << myPiD << " - Unique local Outlets: " << n_unique_LocalOutlets_mOutlet_Edge << " - Total local Outlets on current Rank (1st Round - mOutlet_Edge): " << n_LocalOutlets_mOutlet_Edge << " with Inlet ID:";
					for (int index = 0; index < n_LocalOutlets_mOutlet_Edge; index++ )
						std::cout << ' ' << Iolets_Outlet_Edge[3*index];
					std::cout << "\n\n";*/
					site_t MemSz = 3 * n_LocalOutlets_mOutlet_Edge *  sizeof(site_t);
					cudaStatus = hipMalloc((void**)&GPUDataAddr_Outlet_Edge, MemSz);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation Iolet: Outlet Edge failed...\n"); return false; }

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					cudaStatus = hipMemcpy(GPUDataAddr_Outlet_Edge, &Iolets_Outlet_Edge[0], MemSz, hipMemcpyHostToDevice);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Outlet Edge failed... \n"); return false; }
				}

				// Case 2.1.b. Collision Type 6: mOutletWallCollision
				//std::vector<site_t> Iolets_OutletWall_Edge;
				n_LocalOutlets_mOutletWall_Edge = 0; // number of local Outlets involved during the mOutletCollision collision- Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalOutlets_mOutletWall_Edge = 0;
				if (site_Count_OutletWall_Edge!=0){
					Iolets_OutletWall_Edge = identify_Range_iolets_ID(start_Index_OutletWall_Edge, (start_Index_OutletWall_Edge + site_Count_OutletWall_Edge), &n_LocalOutlets_mOutletWall_Edge, &n_unique_LocalOutlets_mOutletWall_Edge);
					// Remove later the struct object OutletWall_Edge - Replaced by GPU Global memory  Array
					OutletWall_Edge.n_local_iolets = n_LocalOutlets_mOutletWall_Edge;
					memcpy(&OutletWall_Edge.Iolets_ID_range, &Iolets_OutletWall_Edge[0], sizeof(OutletWall_Edge.Iolets_ID_range));
					//
					/*std::cout << "Rank: " << myPiD << " - Unique local Outlets: " << n_unique_LocalOutlets_mOutletWall_Edge << " - Total local Outlets on current Rank (1st Round - mOutletWall_Edge): " << n_LocalOutlets_mOutletWall_Edge << " with Inlet ID:";
					for (int index = 0; index < n_LocalOutlets_mOutletWall_Edge; index++ )
						std::cout << ' ' << Iolets_OutletWall_Edge[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalOutlets_mOutletWall_Edge *  sizeof(site_t);
					cudaStatus = hipMalloc((void**)&GPUDataAddr_OutletWall_Edge, MemSz);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation Iolet: Outlet Wall Edge failed...\n"); return false; }

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					cudaStatus = hipMemcpy(GPUDataAddr_OutletWall_Edge, &Iolets_OutletWall_Edge[0], MemSz, hipMemcpyHostToDevice);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Outlet Wall Edge failed... \n"); return false; }
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
					// Remove later the struct object Outlet_Inner
					Outlet_Inner.n_local_iolets = n_LocalOutlets_mOutlet;
					memcpy(&Outlet_Inner.Iolets_ID_range, &Iolets_Outlet_Inner[0], sizeof(Outlet_Inner.Iolets_ID_range));
					//
					/*std::cout << "Rank: " << myPiD << " - Unique local Outlets: " << n_unique_LocalOutlets_mOutlet << " - Total local Outlets on current Rank (1st Round - mOutlet): " << n_LocalOutlets_mOutlet << " with Inlet ID:";
					for (int index = 0; index < n_LocalOutlets_mOutlet; index++ )
						std::cout << ' ' << Iolets_Outlet_Inner[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalOutlets_mOutlet *  sizeof(site_t);
					cudaStatus = hipMalloc((void**)&GPUDataAddr_Outlet_Inner, MemSz);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation Iolet: Outlet Inner failed...\n"); return false; }

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					cudaStatus = hipMemcpy(GPUDataAddr_Outlet_Inner, &Iolets_Outlet_Inner[0], MemSz, hipMemcpyHostToDevice);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Outlet Inner failed... \n"); return false; }
				}

				// Case 2.2.b. 	Collision Type 6: mOutletWallCollision
				//std::vector<site_t> Iolets_OutletWall_Inner;
				n_LocalOutlets_mOutletWall = 0; // number of local Outlets involved during the mOutletWallCollision collision - Move this declaration in the header file (lb.h - public or private member) later.
				n_unique_LocalOutlets_mOutletWall = 0;
				if (site_Count_OutletWall_Inner!=0){
					Iolets_OutletWall_Inner = identify_Range_iolets_ID(start_Index_OutletWall_Inner, (start_Index_OutletWall_Inner + site_Count_OutletWall_Inner), &n_LocalOutlets_mOutletWall, &n_unique_LocalOutlets_mOutletWall);
					// Remove later teh struct object OutletWall_Inner
					OutletWall_Inner.n_local_iolets = n_LocalOutlets_mOutletWall;
					memcpy(&OutletWall_Inner.Iolets_ID_range, &Iolets_OutletWall_Inner[0], sizeof(OutletWall_Inner.Iolets_ID_range));
					//
					/*std::cout << "Rank: " << myPiD << " - Unique local Outlets: " << n_unique_LocalOutlets_mOutletWall << " - Total local Outlets on current Rank (1st Round - mOutletWall): " << n_LocalOutlets_mOutletWall << " with Inlet ID:";
					for (int index = 0; index < n_LocalOutlets_mOutletWall; index++ )
						std::cout << ' ' << Iolets_OutletWall_Inner[3*index];
					std::cout << "\n\n";*/

					site_t MemSz = 3 * n_LocalOutlets_mOutletWall *  sizeof(site_t);
					cudaStatus = hipMalloc((void**)&GPUDataAddr_OutletWall_Inner, MemSz);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation Iolet: Outlet Wall Inner failed...\n"); return false; }

					// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet)
					cudaStatus = hipMemcpy(GPUDataAddr_OutletWall_Inner, &Iolets_OutletWall_Inner[0], MemSz, hipMemcpyHostToDevice);
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet: Outlet Wall Inner failed... \n"); return false; }
				}
				//-------------------------------------------------------------------------------------------------------------------------------------------------------------
				/*
				printf("Rank: %d, n_Inlets_Inner: %d, n_InletsWall_Inner: %d, n_Inlets_Edge: %d, n_InletsWall_Edge: %d \n", myPiD, n_LocalInlets_mInlet, n_LocalInlets_mInletWall, n_LocalInlets_mInlet_Edge, n_LocalInlets_mInletWall_Edge);
				printf("Rank: %d, n_Outlets_Inner: %d, n_OutletsWall_Inner: %d, n_Outlets_Edge: %d, n_OutletsWall_Edge: %d \n", myPiD, n_LocalOutlets_mOutlet, n_LocalOutlets_mOutletWall, n_LocalOutlets_mOutlet_Edge, n_LocalOutlets_mOutletWall_Edge);
				*/
				//=============================================================================================================================================================

				//=============================================================================================================================================================
				// Examine the type of Iolets BCs:
				//----------------------------------------------------------------------
				// ***** Velocity BCs ***** Option: "LADDIOLET" - CMake file
				//	Possible options: a. parabolic, b. file, c. womersley
				// See SimConfig.cc

				/**
				 Possible approach:
					Allocate the memory on the GPU for the desired/specified velocity (wallMom[3*NUMVECTORS]) at each of the iolets with Velocity BCs
					Parameters needed:
						a. Number of fluid sites for each iolet (known iolet ID, ACTUALLY this is not needed - treat as a whole)
						b.	Value for wallMom[3*NUMVECTORS] at each fluid site,
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


				//----------------------------------------------------------------------
				// Allocate memory on the GPU for each case:
				site_t MemSz; 		// Memory Size to be allocated

				// Inlets BCs
				if(hemeIoletBC_Inlet == "LADDIOLET"){
					printf("INDEED the Inlet Type of BCs is: %s \n", hemeIoletBC_Inlet.c_str()); //note the use of c_str

					// Allocate memory on the GPU for the case of Inlet Velocity BCs
					site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
					if(site_Count_Inlet_Edge!=0){
						MemSz = 3 * site_Count_Inlet_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						cudaStatus = hipMalloc((void**)&GPUDataAddr_wallMom_Inlet_Edge, MemSz);
						if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation wallMom - Inlet Edge failed\n"); return false; }
					}

					site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
					if(site_Count_InletWall_Edge!=0){
						MemSz = 3 * site_Count_InletWall_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						cudaStatus = hipMalloc((void**)&GPUDataAddr_wallMom_InletWall_Edge, MemSz);
						if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation wallMom - InletWall Edge failed\n"); return false; }
					}

					site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
					if(site_Count_Inlet_Inner!=0){
						MemSz = 3 * site_Count_Inlet_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						cudaStatus = hipMalloc((void**)&GPUDataAddr_wallMom_Inlet_Inner, MemSz);
						if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation wallMom - Inlet Inner failed\n"); return false; }
					}

					site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
					if(site_Count_InletWall_Inner!=0){
						MemSz = 3 * site_Count_InletWall_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						cudaStatus = hipMalloc((void**)&GPUDataAddr_wallMom_InletWall_Inner, MemSz);
						if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation wallMom - InletWall Inner failed\n"); return false; }
					}

				}
				else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
					printf("INDEED the Inlet Type of BCs is: %s \n", hemeIoletBC_Inlet.c_str()); //note the use of c_str

					// Allocate memory on the GPU for the case of Inlet Pressure BCs
					// Ghost Density Inlet
					cudaStatus = hipMalloc((void**)&d_ghostDensity, n_Inlets * sizeof(distribn_t));
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation ghostDensity failed\n"); return false; }
				}
				/*// TODO: must REMOVE the following later
				cudaStatus = cudaMalloc((void**)&d_ghostDensity, n_Inlets * sizeof(distribn_t));
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation ghostDensity failed\n"); return false; }
				*///
				//----------------------------------------------------------------------

				//----------------------------------------------------------------------
				// Outlets BCs
				if(hemeIoletBC_Outlet == "LADDIOLET"){
					printf("INDEED the Outlet Type of BCs is: %s \n", hemeIoletBC_Outlet.c_str()); //note the use of c_str

					// Allocate memory on the GPU for the case of Outlet Velocity BCs
					site_t site_Count_Outlet_Edge = mLatDat->GetDomainEdgeCollisionCount(3);
					if(site_Count_Outlet_Edge!=0){
						MemSz = 3 * site_Count_Outlet_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						cudaStatus = hipMalloc((void**)&GPUDataAddr_wallMom_Outlet_Edge, MemSz);
						if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation wallMom - Outlet Edge failed\n"); return false; }
					}

					site_t site_Count_OutletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(5);
					if(site_Count_OutletWall_Edge!=0){
						MemSz = 3 * site_Count_OutletWall_Edge * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						cudaStatus = hipMalloc((void**)&GPUDataAddr_wallMom_OutletWall_Edge, MemSz);
						if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation wallMom - OutletWall Edge failed\n"); return false; }
					}

					site_t site_Count_Outlet_Inner = mLatDat->GetMidDomainCollisionCount(3);
					if(site_Count_Outlet_Inner!=0){
						MemSz = 3 * site_Count_Outlet_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						cudaStatus = hipMalloc((void**)&GPUDataAddr_wallMom_Outlet_Inner, MemSz);
						if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation wallMom - Outlet Inner failed\n"); return false; }
					}

					site_t site_Count_OutletWall_Inner = mLatDat->GetMidDomainCollisionCount(5);
					if(site_Count_OutletWall_Inner!=0){
						MemSz = 3 * site_Count_OutletWall_Inner * (LatticeType::NUMVECTORS - 1) * sizeof(distribn_t);
						cudaStatus = hipMalloc((void**)&GPUDataAddr_wallMom_OutletWall_Inner, MemSz);
						if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation wallMom - OutletWall Inner failed\n"); return false; }
					}

				}
				else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
					printf("INDEED the Outlet Type of BCs is: %s \n", hemeIoletBC_Outlet.c_str()); //note the use of c_str

					// Allocate memory on the GPU for the case of Outlet Pressure BCs
					// Ghost Density Outlet
					cudaStatus = hipMalloc((void**)&d_ghostDensity_out, n_Outlets * sizeof(distribn_t));
					if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation ghostDensity failed\n"); return false; }
				}
				//----------------------------------------------------------------------



				//----------------------------------------------------------------------
				// ***** Pressure BCs ***** Option: "NASHZEROTHORDERPRESSUREIOLET" - CMake file
				// 	Set the Ghost Density if Inlet/Outlet BCs is set to NashZerothOrderPressure
				// 	Just allocate the memory as the ghostDensity can change as a function of time. MemCopies(host-to-device) before the gpu inlet/outlet collision kernels
				//----------------------------------------------------------------------

				// Ghost Density Inlet - Outlet
				//cudaStatus = cudaMalloc((void**)&d_ghostDensity, n_Inlets * sizeof(distribn_t));
				//if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation ghostDensity failed\n"); return false; }

				//cudaStatus = cudaMalloc((void**)&d_ghostDensity_out, n_Outlets * sizeof(distribn_t));
				//if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation ghostDensity failed\n"); return false; }
				//

				//----------------------------------------------------------------------
				// Normals to Iolets
				// Inlets:
				float* h_inletNormal = new float[3*n_Inlets]; 	// x,y,z components
				for (int i=0; i<n_Inlets; i++){
					util::Vector3D<float> ioletNormal = mInletValues->GetLocalIolet(i)->GetNormal();
					h_inletNormal[3*i] = ioletNormal.x;
					h_inletNormal[3*i+1] = ioletNormal.y;
					h_inletNormal[3*i+2] = ioletNormal.z;
					//std::cout << "Cout: ioletNormal.x : " <<  h_inletNormal[i] << " - ioletNormal.y : " <<  h_inletNormal[i+1] << " - ioletNormal.z : " <<  h_inletNormal[i+2] << std::endl;
				}

				cudaStatus = hipMalloc((void**)&d_inletNormal, 3*n_Inlets * sizeof(float));
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation inletNormal failed\n"); return false; }
				// Memory copy from host (h_inletNormal) to Device (d_inletNormal)
				cudaStatus = hipMemcpy(d_inletNormal, h_inletNormal, 3*n_Inlets * sizeof(float), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer (inletNormal) Host To Device failed\n"); return false; }

				// Outlets:
				float* h_outletNormal = new float[3*n_Outlets]; 	// x,y,z components
				for (int i=0; i<n_Outlets; i++){
					util::Vector3D<float> ioletNormal = mOutletValues->GetLocalIolet(i)->GetNormal();
					h_outletNormal[3*i] = ioletNormal.x;
					h_outletNormal[3*i+1] = ioletNormal.y;
					h_outletNormal[3*i+2] = ioletNormal.z;
					//std::cout << "Cout: ioletNormal.x : " <<  h_outletNormal[3*i] << " - ioletNormal.y : " <<  h_outletNormal[3*i+1] << " - ioletNormal.z : " <<  h_outletNormal[3*i+2] << std::endl;
				}

				cudaStatus = hipMalloc((void**)&d_outletNormal, 3*n_Outlets * sizeof(float));
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation outletNormal failed\n"); return false; }
				// Memory copy from host (h_outletNormal) to Device (d_outletNormal)
				cudaStatus = hipMemcpy(d_outletNormal, h_outletNormal, 3*n_Outlets * sizeof(float), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer (inletNormal) Host To Device failed\n"); return false; }
				//----------------------------------------------------------------------
				//***********************************************************************************************************************************


				//***********************************************************************************************************************************
				// Allocate memory for streamingIndicesForReceivedDistributions on the GPU constant Memory
				// From geometry/LatticeData.h:	std::vector<site_t> streamingIndicesForReceivedDistributions; //! The indices to stream to for distributions received from other processors.

				unsigned long long TotalMem_int64_streamInd = totSharedFs * sizeof(site_t); // Total memory size for streamingIndicesForReceivedDistributions
				site_t* Data_int64_streamInd = new site_t[totSharedFs];	// site_t (type int64_t)

				if(!Data_int64_streamInd){
					std::cout << "Memory allocation error - streamingIndicesForReceivedDistributions" << std::endl;
					return false;
				}

				Data_int64_streamInd = &(mLatDat->streamingIndicesForReceivedDistributions[0]);  // Data_int64_streamInd points to &(mLatDat->streamingIndicesForReceivedDistributions[0])

				// Debugging
				/* for (site_t i = 0; i < totSharedFs; i++){
					 site_t streamIndex = Data_int64_streamInd[i];
					 printf("Index = %lld, Streamed Index = %lld \n\n", i, streamIndex);
				}*/

				// Alocate memory on the GPU
				cudaStatus = hipMalloc((void**)&GPUDataAddr_int64_streamInd, totSharedFs * sizeof(site_t));
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory allocation for streamingIndicesForReceivedDistributions failed\n"); return false; }

				// Memory copy from host (Data_int64_streamInd) to Device (GPUDataAddr_int64_Neigh)
				cudaStatus = hipMemcpy(GPUDataAddr_int64_streamInd, Data_int64_streamInd, totSharedFs * sizeof(site_t), hipMemcpyHostToDevice);
				if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for streamingIndicesForReceivedDistributions failed\n"); return false; }
				//***********************************************************************************************************************************

				//***********************************************************************************************************************************
				// Check the total memory requirements
				// Change this in the future as roughly half of this (either memory arrangement (a) or (b)) will be needed. To do!!!
				// Add a check whether the memory on the GPU global memory is sufficient!!! Abort if not or split the domain into smaller subdomains and pass info gradually! To do!!!
				// unsigned long long TotalMem_req = (TotalMem_dbl_fOld * 4 +  TotalMem_dbl_MacroVars + TotalMem_int64_Neigh *4 + TotalMem_uint32_WallIntersect + TotalMem_uint32_IoletIntersect + TotalMem_int64_streamInd); //
				unsigned long long TotalMem_req = (TotalMem_dbl_fOld * 2 +  TotalMem_dbl_MacroVars + TotalMem_int64_Neigh + TotalMem_uint32_WallIntersect + TotalMem_uint32_IoletIntersect + TotalMem_int64_streamInd); //
				printf("Rank: %d - Total requested global memory %.2fGB \n\n", myPiD, ((double)TotalMem_req/1073741824.0));
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

				// 2.a. Weight coefficients for the equilibrium distr. functions
				cudaStatus = hipMemcpyToSymbol(hemelb::_EQMWEIGHTS_19, LatticeType::EQMWEIGHTS, LatticeType::NUMVECTORS*sizeof(double), 0, hipMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) { fprintf(stderr, "GPU constant memory copy failed (1)\n"); return false;
					//goto Error;
				}

				// 2.b. Number of vectors: LatticeType::NUMVECTORS
				static const unsigned int num_Vectors = LatticeType::NUMVECTORS;
				cudaStatus = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::_NUMVECTORS), &num_Vectors, sizeof(num_Vectors), 0, hipMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) { fprintf(stderr, "GPU constant memory copy failed (2)\n"); return false;
					//goto Error;
				}

				// 2.c. Inverse directions for the bounce back LatticeType::INVERSEDIRECTIONS[direction]
				cudaStatus = hipMemcpyToSymbol(hemelb::_InvDirections_19, LatticeType::INVERSEDIRECTIONS, LatticeType::NUMVECTORS*sizeof(int), 0, hipMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) { fprintf(stderr, "GPU constant memory copy failed (3)\n"); return false;
					//goto Error;
				}

				// 2.d. Lattice Velocity directions CX[DmQn::NUMVECTORS], CY[DmQn::NUMVECTORS], CZ[DmQn::NUMVECTORS]
				cudaStatus = hipMemcpyToSymbol(hemelb::_CX_19, LatticeType::CX, LatticeType::NUMVECTORS*sizeof(int), 0, hipMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) { fprintf(stderr, "GPU constant memory copy failed (4)\n"); return false; //goto Error;
				}
				cudaStatus = hipMemcpyToSymbol(hemelb::_CY_19, LatticeType::CY, LatticeType::NUMVECTORS*sizeof(int), 0, hipMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) { fprintf(stderr, "GPU constant memory copy failed (5)\n"); return false; //goto Error;
				}
				cudaStatus = hipMemcpyToSymbol(hemelb::_CZ_19, LatticeType::CZ, LatticeType::NUMVECTORS*sizeof(int), 0, hipMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) { fprintf(stderr, "GPU constant memory copy failed (6)\n"); return false; //goto Error;
				}

				// 2.e. Relaxation Time tau
				//static const int num_Vectors = LatticeType::NUMVECTORS;
				// mParams object of type hemelb::lb::LbmParameters (struct LbmParameters)
				double tau = mParams.GetTau();				// printf("Relaxation Time = %.5f\n\n", tau);
				double minus_inv_tau = mParams.GetOmega();	// printf("Minus Inv. Relaxation Time = %.5f\n\n", minus_inv_tau);

				cudaStatus = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::dev_tau), &tau, sizeof(tau), 0, hipMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) { fprintf(stderr, "GPU constant memory copy failed (7)\n"); return false; //goto Error;
				}

				cudaStatus = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::dev_minusInvTau), &minus_inv_tau, sizeof(minus_inv_tau), 0, hipMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) { fprintf(stderr, "GPU constant memory copy failed (8)\n"); return false; //goto Error;
				}

				cudaStatus = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::_Cs2), &Cs2, sizeof(Cs2), 0, hipMemcpyHostToDevice);
				if (cudaStatus != hipSuccess) { fprintf(stderr, "GPU constant memory copy failed (9)\n"); return false; //goto Error;
				}
				//=================================================================================================================================

				// Remove later...
				//if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function.


				// Create the Streams here
				hipStreamCreate(&Collide_Stream_PreSend_1);
				hipStreamCreate(&Collide_Stream_PreSend_2);
				hipStreamCreate(&Collide_Stream_PreSend_3);
				hipStreamCreate(&Collide_Stream_PreSend_4);
				hipStreamCreate(&Collide_Stream_PreSend_5);
				hipStreamCreate(&Collide_Stream_PreSend_6);

				hipStreamCreate(&Collide_Stream_PreRec_1);
				hipStreamCreate(&Collide_Stream_PreRec_2);
				hipStreamCreate(&Collide_Stream_PreRec_3);
				hipStreamCreate(&Collide_Stream_PreRec_4);
				hipStreamCreate(&Collide_Stream_PreRec_5);
				hipStreamCreate(&Collide_Stream_PreRec_6);

				hipStreamCreate(&stream_ghost_dens_inlet);
				hipStreamCreate(&stream_ghost_dens_outlet);

				hipStreamCreate(&stream_ReceivedDistr);
				hipStreamCreate(&stream_SwapOldAndNew);
				hipStreamCreate(&stream_memCpy_CPU_GPU_domainEdge);

				hipStreamCreate(&stream_Read_Data_GPU_Dens);

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
				delete[] Data_uint32_WallIntersect;
				delete[] Data_uint32_IoletIntersect;
				delete[] h_inletNormal, h_outletNormal;

				return true;
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
				//cudaError_t cudaStatus;


				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();

				//if (myPiD!=0) cudaStreamSynchronize(stream_ReceivedDistr);

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

				hipError_t cudaStatus;

				// Before the collision starts make sure that the swap of distr. functions at the previous step has Completed
				//if (myPiD!=0) cudaStreamSynchronize(stream_SwapOldAndNew);
				if (myPiD!=0) hipStreamSynchronize(stream_ReceivedDistr);

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
				if(nBlocks_Collide!=0)
					hipLaunchKernelGGL(hemelb::GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_1, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																									(distribn_t*)GPUDataAddr_dbl_MacroVars,
																									(site_t*)GPUDataAddr_int64_Neigh_d,
																									(uint32_t*)GPUDataAddr_uint32_Wall,
																									nFluid_nodes,
																									first_Index, (first_Index + site_Count_MidFluid),
																									(first_Index + site_Count_MidFluid), (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep()); // (int64_t*)GPUDataAddr_int64_Neigh_b
				//#####################################################################################################################################################



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
				//Receive values for Inlet
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
				//	Total GLOBAL iolets: n_Inlets = mInletValues->GetLocalIoletCount();
				int n_Inlets = mInletValues->GetLocalIoletCount();
				distribn_t* h_ghostDensity; // pointer to the ghost density for the inlets

				//	Total GLOBAL iolets: n_Outlets = mOutletValues->GetLocalIoletCount();
				int n_Outlets = mOutletValues->GetLocalIoletCount();
				distribn_t* h_ghostDensity_out;
				//----------------------------------------------------------------------

				// Inlets BCs
				if(hemeIoletBC_Inlet == "LADDIOLET"){
					//printf("Entering the LaddIolet loop \n\n");
					propertyCache.wallMom_Cache.SetRefreshFlag(); // Is this needed??? TODO: Check
					//====================================================================
					// Domain Edge
					// Collision Type 3 (mInletCollision):
					site_t start_Index_Inlet_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
					site_t site_Count_Inlet_Edge = mLatDat->GetDomainEdgeCollisionCount(2);
					if (site_Count_Inlet_Edge!=0){
						GetWallMom(mInletCollision, start_Index_Inlet_Edge, site_Count_Inlet_Edge); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_Inlet_Edge, site_Count_Inlet_Edge, propertyCache, wallMom_Inlet_Edge);
						wallMom_Inlet_Edge.resize(site_Count_Inlet_Edge*LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_Inlet_Edge, site_Count_Inlet_Edge, wallMom_Inlet_Edge, GPUDataAddr_wallMom_Inlet_Edge);
					}

					// Collision Type 5 (mInletWallCollision):
					site_t start_Index_InletWall_Edge =	mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
					                                    + mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3);
					site_t site_Count_InletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(4);
					if (site_Count_InletWall_Edge!=0){
						GetWallMom(mInletWallCollision, start_Index_InletWall_Edge, site_Count_InletWall_Edge); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_InletWall_Edge, site_Count_InletWall_Edge, propertyCache, wallMom_InletWall_Edge);
						wallMom_InletWall_Edge.resize(site_Count_InletWall_Edge*LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_InletWall_Edge, site_Count_InletWall_Edge, wallMom_InletWall_Edge, GPUDataAddr_wallMom_InletWall_Edge);
					}

					//====================================================================
					// Inner domain - TODO: Think whether to keep this here...
					// Collision Type 3 (mInletCollision):
					site_t start_Index_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
					site_t site_Count_Inlet_Inner = mLatDat->GetMidDomainCollisionCount(2);
					if (site_Count_Inlet_Inner!=0){
						GetWallMom(mInletCollision, start_Index_Inlet_Inner, site_Count_Inlet_Inner); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_Inlet_Inner, site_Count_Inlet_Inner, propertyCache, wallMom_Inlet_Inner);
						wallMom_Inlet_Inner.resize(site_Count_Inlet_Inner*LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_Inlet_Inner, site_Count_Inlet_Inner, wallMom_Inlet_Inner, GPUDataAddr_wallMom_Inlet_Inner);
					}

					// Collision Type 5 (mInletWallCollision):
					site_t start_Index_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2) + mLatDat->GetMidDomainCollisionCount(3);
					site_t site_Count_InletWall_Inner = mLatDat->GetMidDomainCollisionCount(4);
					if (site_Count_InletWall_Inner!=0){
						GetWallMom(mInletWallCollision, start_Index_InletWall_Inner, site_Count_InletWall_Inner); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_InletWall_Inner, site_Count_InletWall_Inner, propertyCache, wallMom_InletWall_Inner);
						wallMom_InletWall_Inner.resize(site_Count_InletWall_Inner*LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_InletWall_Inner, site_Count_InletWall_Inner, wallMom_InletWall_Inner, GPUDataAddr_wallMom_InletWall_Inner);
					}
					//====================================================================
				} // Ends the if(hemeIoletBC_Inlet == "LADDIOLET") loop
				else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){

					// Inlet BCs: NashZerothOrderPressure - Specify the ghost density for each inlet
					//	Pass the ghost density[nInlets] to the GPU kernel (cudaMemcpy):
					h_ghostDensity = new distribn_t[n_Inlets];

					// Proceed with the collision type if the number of fluid nodes involved is not ZERO - HtD memcopy
					// This (n_Inlets) refers to the total number of inlets globally. NOT on local RANK - SHOULD REPLACE THIS with the local number of inlets
					if (n_Inlets!=0){
						for (int i=0; i<n_Inlets; i++){
							h_ghostDensity[i] = mInletValues->GetBoundaryDensity(i);
							//std::cout << "Cout: GhostDensity : " << h_ghostDensity[i] << std::endl;
						}
						if (myPiD!=0){ // MemCopy cudaMemcpyHostToDevice only if rank!=0
							// Memory copy from host (h_ghostDensity) to Device (d_ghostDensity)
							//cudaStatus = cudaMemcpy(d_ghostDensity, h_ghostDensity, n_Inlets * sizeof(distribn_t), cudaMemcpyHostToDevice);
							cudaStatus = hipMemcpyAsync(d_ghostDensity, h_ghostDensity, n_Inlets * sizeof(distribn_t), hipMemcpyHostToDevice, stream_ghost_dens_inlet);
							if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer (ghostDensity) Host To Device failed\n"); //return false;
							}
						}
						//if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function.
					} // Closes the if n_Inlets!=0

				} // Closes the if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET")
				//----------------------------------------------------------------------
				// Outlets BCs
				if(hemeIoletBC_Outlet == "LADDIOLET"){

					// Domain Edge
					// Collision Type 4 (mOutletCollision):
					site_t start_Index_Outlet_Edge = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1) + mLatDat->GetDomainEdgeCollisionCount(2);
					site_t site_Count_Outlet_Edge = mLatDat->GetDomainEdgeCollisionCount(3);
					if (site_Count_Outlet_Edge!=0){
						GetWallMom(mOutletCollision, start_Index_Outlet_Edge, site_Count_Outlet_Edge); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_Outlet_Edge, site_Count_Outlet_Edge, propertyCache, wallMom_Outlet_Edge);
						wallMom_Outlet_Edge.resize(site_Count_Outlet_Edge * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_Outlet_Edge, site_Count_Outlet_Edge, wallMom_Outlet_Edge, GPUDataAddr_wallMom_Outlet_Edge);
					}

					// Collision Type 6 (mOutletWallCollision):
					site_t start_Index_OutletWall_Edge = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1)
					                                    + mLatDat->GetDomainEdgeCollisionCount(2) + mLatDat->GetDomainEdgeCollisionCount(3) + mLatDat->GetDomainEdgeCollisionCount(4);
					site_t site_Count_OutletWall_Edge = mLatDat->GetDomainEdgeCollisionCount(5);
					if (site_Count_OutletWall_Edge!=0){
						GetWallMom(mOutletWallCollision, start_Index_OutletWall_Edge, site_Count_OutletWall_Edge); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_OutletWall_Edge, site_Count_OutletWall_Edge, propertyCache, wallMom_OutletWall_Edge);
						wallMom_OutletWall_Edge.resize(site_Count_OutletWall_Edge * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_OutletWall_Edge, site_Count_OutletWall_Edge, wallMom_OutletWall_Edge, GPUDataAddr_wallMom_OutletWall_Edge);
					}

					// Inner Domain
					// Collision Type 4 (mOutletCollision):
					site_t start_Index_Outlet_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2);
					site_t site_Count_Outlet_Inner = mLatDat->GetMidDomainCollisionCount(3);
					if (site_Count_Outlet_Inner!=0){
						GetWallMom(mOutletCollision, start_Index_Outlet_Inner, site_Count_Outlet_Inner); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_Outlet_Inner, site_Count_Outlet_Inner, propertyCache, wallMom_Outlet_Inner);
						wallMom_Outlet_Inner.resize(site_Count_Outlet_Inner * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_Outlet_Inner, site_Count_Outlet_Inner, wallMom_Outlet_Inner, GPUDataAddr_wallMom_Outlet_Inner);
					}


					// Collision Type 6 (mOutletWallCollision):
					site_t start_Index_OutletWall_Inner = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1) + mLatDat->GetMidDomainCollisionCount(2)
					                                      + mLatDat->GetMidDomainCollisionCount(3) + mLatDat->GetMidDomainCollisionCount(4);
					site_t site_Count_OutletWall_Inner = mLatDat->GetMidDomainCollisionCount(5);
					if (site_Count_OutletWall_Inner!=0){
						GetWallMom(mOutletWallCollision, start_Index_OutletWall_Inner, site_Count_OutletWall_Inner); // Fills the propertyCache.wallMom_Cache
						read_WallMom_from_propertyCache(start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, propertyCache, wallMom_OutletWall_Inner);
						wallMom_OutletWall_Inner.resize(site_Count_OutletWall_Inner * LatticeType::NUMVECTORS);

						// Function to allocate memory on the GPU's global memory for the wallMom
						memCpy_HtD_GPUmem_WallMom(start_Index_OutletWall_Inner, site_Count_OutletWall_Inner, wallMom_OutletWall_Inner, GPUDataAddr_wallMom_OutletWall_Inner);
					}
				}
				else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){

					// Outlet BCs: NashZerothOrderPressure - Specify the ghost density for each outlet
					//	Pass the ghost density_out[nInlets] to the GPU kernel (cudaMemcpy):
					h_ghostDensity_out = new distribn_t[n_Outlets];

					// Proceed with the collision type if the number of fluid nodes involved is not ZERO
					if (n_Outlets!=0){ // even rank 0 can "see" this info

						for (int i=0; i<n_Outlets; i++){
							h_ghostDensity_out[i] = mOutletValues->GetBoundaryDensity(i);
							//std::cout << "Rank: " << myPiD <<  " Cout: GhostDensity Out: " << h_ghostDensity_out[i] << std::endl;
						}
						if (myPiD!=0){ // MemCopy cudaMemcpyHostToDevice only if rank!=0
							// Memory copy from host (h_ghostDensity) to Device (d_ghostDensity)
							//cudaStatus = cudaMemcpy(d_ghostDensity_out, h_ghostDensity_out, n_Outlets * sizeof(distribn_t), cudaMemcpyHostToDevice);
							cudaStatus = hipMemcpyAsync(d_ghostDensity_out, h_ghostDensity_out, n_Outlets * sizeof(distribn_t), hipMemcpyHostToDevice, stream_ghost_dens_outlet);
							if(cudaStatus != hipSuccess){ fprintf(stderr, "GPU memory transfer (ghostDensity_out) Host To Device failed\n"); //return false;
							}
						}
					} // Closes the if n_Oulets!=0
					//
				}
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
						hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_3, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																														(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																														(distribn_t*)GPUDataAddr_dbl_MacroVars,
																														(int64_t*)GPUDataAddr_int64_Neigh_d,
																														(uint32_t*)GPUDataAddr_uint32_Iolet,
																														(mLatDat->GetLocalFluidSiteCount()),
																														(distribn_t*)GPUDataAddr_wallMom_Inlet_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
																														first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep());
					}
					else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
						// Make sure it has received the values for ghost density on the GPU for the case of Pressure BCs
						if (myPiD!=0) hipStreamSynchronize(stream_ghost_dens_inlet);	// Maybe transfer this within the loop for Press. BCs below

						if (n_LocalInlets_mInlet_Edge <=(local_iolets_MaxSIZE/3)){
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_3, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																														(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																														(double*)GPUDataAddr_dbl_MacroVars,
																														(int64_t*)GPUDataAddr_int64_Neigh_d,
																														(uint32_t*)GPUDataAddr_uint32_Iolet,
																														(distribn_t*)d_ghostDensity,
																														(float*)d_inletNormal,
																														n_Inlets,
																														(mLatDat->GetLocalFluidSiteCount()),
																														first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																														n_LocalInlets_mInlet_Edge, Inlet_Edge);
						}
						else{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_3, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity,
																															(float*)d_inletNormal,
																															n_Inlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalInlets_mInlet_Edge, (site_t*)GPUDataAddr_Inlet_Edge);
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
						hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_4, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																														(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																														(distribn_t*)GPUDataAddr_dbl_MacroVars,
																														(int64_t*)GPUDataAddr_int64_Neigh_d,
																														(uint32_t*)GPUDataAddr_uint32_Iolet,
																														(mLatDat->GetLocalFluidSiteCount()),
																														(distribn_t*)GPUDataAddr_wallMom_Outlet_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
																														first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep());
					}
					else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
						// Make sure it has received the values for ghost density on the GPU
						if (myPiD!=0) hipStreamSynchronize(stream_ghost_dens_outlet);

						if(n_LocalOutlets_mOutlet_Edge<=(local_iolets_MaxSIZE/3))
						{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_4, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity_out,
																															(float*)d_outletNormal,
																															n_Outlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalOutlets_mOutlet_Edge, Outlet_Edge); //
						}
						else{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_4, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity_out,
																															(float*)d_outletNormal,
																															n_Outlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalOutlets_mOutlet_Edge, (site_t*)GPUDataAddr_Outlet_Edge);
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
						hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_5, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
	 																												 (distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
	 																												 (distribn_t*)GPUDataAddr_dbl_MacroVars,
	 																												 (int64_t*)GPUDataAddr_int64_Neigh_d,
	 																												 (uint32_t*)GPUDataAddr_uint32_Wall,
	 																												 (uint32_t*)GPUDataAddr_uint32_Iolet,
																													 (mLatDat->GetLocalFluidSiteCount()),
																													 (distribn_t*)GPUDataAddr_wallMom_InletWall_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
																													 first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep());
					}
					else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){

						if(n_LocalInlets_mInletWall_Edge <=(local_iolets_MaxSIZE/3))
						{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_iolet_Nash, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_5, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																														 (double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																														 (double*)GPUDataAddr_dbl_MacroVars,
																														 (int64_t*)GPUDataAddr_int64_Neigh_d,
																														 (uint32_t*)GPUDataAddr_uint32_Wall,
																														 (uint32_t*)GPUDataAddr_uint32_Iolet,
																														 (distribn_t*)d_ghostDensity,
																														 (float*)d_inletNormal,
																														 n_Inlets,
																														 (mLatDat->GetLocalFluidSiteCount()),
																														 first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																														 n_LocalInlets_mInletWall_Edge, InletWall_Edge); //
						}
						else{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_5, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																														 (double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																														 (double*)GPUDataAddr_dbl_MacroVars,
																														 (int64_t*)GPUDataAddr_int64_Neigh_d,
																														 (uint32_t*)GPUDataAddr_uint32_Wall,
																														 (uint32_t*)GPUDataAddr_uint32_Iolet,
																														 (distribn_t*)d_ghostDensity,
																														 (float*)d_inletNormal,
																														 n_Inlets,
																														 (mLatDat->GetLocalFluidSiteCount()),
																														 first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																														 n_LocalInlets_mInletWall_Edge, (site_t*)GPUDataAddr_InletWall_Edge); //
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
						hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_6, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
	 																												 (distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
	 																												 (distribn_t*)GPUDataAddr_dbl_MacroVars,
	 																												 (int64_t*)GPUDataAddr_int64_Neigh_d,
	 																												 (uint32_t*)GPUDataAddr_uint32_Wall,
	 																												 (uint32_t*)GPUDataAddr_uint32_Iolet,
																													 (mLatDat->GetLocalFluidSiteCount()),
																													 (distribn_t*)GPUDataAddr_wallMom_OutletWall_Edge, site_Count*(LatticeType::NUMVECTORS - 1),
																													 first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep());
					}
					else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalOutlets_mOutletWall_Edge<=(local_iolets_MaxSIZE/3))
						{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_iolet_Nash, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_6, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Wall,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity_out,
																															(float*)d_outletNormal,
																															n_Outlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalOutlets_mOutletWall_Edge, OutletWall_Edge);
						}
						else{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreSend_6, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Wall,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity_out,
																															(float*)d_outletNormal,
																															n_Outlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalOutlets_mOutletWall_Edge, (site_t*)GPUDataAddr_OutletWall_Edge);
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
					cudaStreamSynchronize(Collide_Stream_PreSend_1);
					cudaStreamSynchronize(Collide_Stream_PreSend_2);
					cudaStreamSynchronize(Collide_Stream_PreSend_3);
					cudaStreamSynchronize(Collide_Stream_PreSend_4);
					cudaStreamSynchronize(Collide_Stream_PreSend_5);
					cudaStreamSynchronize(Collide_Stream_PreSend_6);
				}

				// Once all collision-streaming types are completed then send the distr. functions fNew in totalSharedFs to the CPU
				// For the exchange of f's at domain edges
				// Uses Asynch. MemCopy - Stream: stream_memCpy_GPU_CPU_domainEdge
				if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU_totalSharedFs();
				*/

					// Delete the variables used for cudaMemcpy
					if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET") delete[] h_ghostDensity_out;
					if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET") delete[] h_ghostDensity;

#else	// If computations on CPUs

				// printf("Calling CPU PART \n\n");
				// Collision Type 1 (mMidFluidCollision):
				site_t offset1 = mLatDat->GetMidDomainSiteCount();
				StreamAndCollide(mMidFluidCollision, offset1, mLatDat->GetDomainEdgeCollisionCount(0));

				// Collision Type 2 (mWallCollision):
				offset1 += mLatDat->GetDomainEdgeCollisionCount(0);
				// printf("Rank: %d: Collision Type 2: Starting = %lld, Ending = %lld \n\n",myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(1)));
				StreamAndCollide(mWallCollision, offset1, mLatDat->GetDomainEdgeCollisionCount(1));

				// Collision Type 3 (mInletCollision):
				offset1 += mLatDat->GetDomainEdgeCollisionCount(1);
				// Receive values for Inlet
				mInletValues->FinishReceive();
				// printf("Rank: %d: Collision Type 3: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(2)));
				StreamAndCollide(mInletCollision, offset1, mLatDat->GetDomainEdgeCollisionCount(2));

				// Collision Type 4 (mOutletCollision):
				offset1 += mLatDat->GetDomainEdgeCollisionCount(2);
				// Receive values for Outlet
				mOutletValues->FinishReceive();
				// printf("Rank: %d: Collision Type 4: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(3)));
				StreamAndCollide(mOutletCollision, offset1, mLatDat->GetDomainEdgeCollisionCount(3));

				// Collision Type 5 (mInletWallCollision):
				offset1 += mLatDat->GetDomainEdgeCollisionCount(3);
				// printf("Rank: %d: Collision Type 5: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(4)));
				StreamAndCollide(mInletWallCollision, offset1, mLatDat->GetDomainEdgeCollisionCount(4));

				// Collision Type 6 (mOutletWallCollision):
				offset1 += mLatDat->GetDomainEdgeCollisionCount(4);
				// printf("Rank: %d: Collision Type 6: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(5)));
				StreamAndCollide(mOutletWallCollision, offset1, mLatDat->GetDomainEdgeCollisionCount(5));

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

				hipError_t cudaStatus;

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
				int nThreadsPerBlock_Collide = 128;				//Number of threads per block for the Collision step
				dim3 nThreads_Collide(nThreadsPerBlock_Collide);
				// Number of fluid nodes involved in the collision/streaming : mLatDat->GetDomainEdgeCollisionCount(0)
				int nBlocks_Collide = (site_Count)/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
				//----------------------------------

				// To access the data in GPU global memory:
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)
				if(nBlocks_Collide!=0)
					hipLaunchKernelGGL(hemelb::GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_1, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																									(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																									(distribn_t*)GPUDataAddr_dbl_MacroVars,
																									(site_t*)GPUDataAddr_int64_Neigh_d,
																									(uint32_t*)GPUDataAddr_uint32_Wall,
																									nFluid_nodes,
																									first_Index, (first_Index + site_Count_MidFluid),
																									(first_Index + site_Count_MidFluid), (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep()); // (int64_t*)GPUDataAddr_int64_Neigh_b
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
				//	Think whether this should be here or at the end of PreReceive(), once all the colliion-streaming kernels has been launched
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
					cudaStreamSynchronize(Collide_Stream_PreSend_1);
					cudaStreamSynchronize(Collide_Stream_PreSend_2);
					cudaStreamSynchronize(Collide_Stream_PreSend_3);
					cudaStreamSynchronize(Collide_Stream_PreSend_4);
					cudaStreamSynchronize(Collide_Stream_PreSend_5);
					cudaStreamSynchronize(Collide_Stream_PreSend_6);
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


				// ====================================================================================================================================================
				// Collision Type 3:
				// Inlet BCs: NashZerothOrderPressure - Specify the ghost density for each inlet
				offset = mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
				//offset += mLatDat->GetMidDomainCollisionCount(1);
				//StreamAndCollide(mInletCollision, offset, mLatDat->GetMidDomainCollisionCount(2));

				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetMidDomainCollisionCount(2);

				//	Total GLOBAL iolets (NOT ONLY ON LOCAL RANK): n_Inlets = mInletValues->GetLocalIoletCount();
				int n_Inlets = mInletValues->GetLocalIoletCount();

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
						hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_3, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																														(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																														(distribn_t*)GPUDataAddr_dbl_MacroVars,
																														(int64_t*)GPUDataAddr_int64_Neigh_d,
																														(uint32_t*)GPUDataAddr_uint32_Iolet,
																														(mLatDat->GetLocalFluidSiteCount()),
																														(distribn_t*)GPUDataAddr_wallMom_Inlet_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
																														first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep());
					}
					else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalInlets_mInlet<=(local_iolets_MaxSIZE/3))
						{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_3, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity,
																															(float*)d_inletNormal,
																															n_Inlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalInlets_mInlet, Inlet_Inner); // (int64_t*)GPUDataAddr_int64_Neigh_b
						}
						else{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_3, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																										(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																										(double*)GPUDataAddr_dbl_MacroVars,
																										(int64_t*)GPUDataAddr_int64_Neigh_d,
																										(uint32_t*)GPUDataAddr_uint32_Iolet,
																										(distribn_t*)d_ghostDensity,
																										(float*)d_inletNormal,
																										n_Inlets,
																										(mLatDat->GetLocalFluidSiteCount()),
																										first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																										n_LocalInlets_mInlet, (site_t*)GPUDataAddr_Inlet_Inner);
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

				//	Total GLOBAL iolets: n_Outlets = mOutletValues->GetLocalIoletCount();
				int n_Outlets = mOutletValues->GetLocalIoletCount();

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
						hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_Ladd_VelBCs, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_4, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																														(distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																														(distribn_t*)GPUDataAddr_dbl_MacroVars,
																														(int64_t*)GPUDataAddr_int64_Neigh_d,
																														(uint32_t*)GPUDataAddr_uint32_Iolet,
																														(mLatDat->GetLocalFluidSiteCount()),
																														(distribn_t*)GPUDataAddr_wallMom_Outlet_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
																														first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep());
					}
					else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalOutlets_mOutlet<=(local_iolets_MaxSIZE/3))
						{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_4, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity_out,
																															(float*)d_outletNormal,
																															n_Outlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalOutlets_mOutlet, Outlet_Inner); //
						}
						else{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_v2, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_4, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity_out,
																															(float*)d_outletNormal,
																															n_Outlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalOutlets_mOutlet, (site_t*)GPUDataAddr_Outlet_Inner);
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

				n_Inlets = mInletValues->GetLocalIoletCount(); // Probably not necessary. Check and Remove!!! To do!!!

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
						hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_5, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
	 																												 (distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
	 																												 (distribn_t*)GPUDataAddr_dbl_MacroVars,
	 																												 (int64_t*)GPUDataAddr_int64_Neigh_d,
	 																												 (uint32_t*)GPUDataAddr_uint32_Wall,
	 																												 (uint32_t*)GPUDataAddr_uint32_Iolet,
																													 (mLatDat->GetLocalFluidSiteCount()),
																													 (distribn_t*)GPUDataAddr_wallMom_InletWall_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
																													 first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep());
					}
					else if (hemeIoletBC_Inlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalInlets_mInletWall<=(local_iolets_MaxSIZE/3))
						{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_iolet_Nash, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_5, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Wall,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity,
																															(float*)d_inletNormal,
																															n_Inlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalInlets_mInletWall, InletWall_Inner);
						}
						else{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_5, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Wall,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity,
																															(float*)d_inletNormal,
																															n_Inlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalInlets_mInletWall, (site_t*)GPUDataAddr_InletWall_Inner);
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

				n_Outlets = mOutletValues->GetLocalIoletCount(); // Probably not necessary. Check and Remove!!! To do!!!

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
						hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_6, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
	 																												 (distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
	 																												 (distribn_t*)GPUDataAddr_dbl_MacroVars,
	 																												 (int64_t*)GPUDataAddr_int64_Neigh_d,
	 																												 (uint32_t*)GPUDataAddr_uint32_Wall,
	 																												 (uint32_t*)GPUDataAddr_uint32_Iolet,
																													 (mLatDat->GetLocalFluidSiteCount()),
																													 (distribn_t*)GPUDataAddr_wallMom_OutletWall_Inner, site_Count*(LatticeType::NUMVECTORS - 1),
																													 first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep());
					}
					else if (hemeIoletBC_Outlet == "NASHZEROTHORDERPRESSUREIOLET"){
						if(n_LocalOutlets_mOutletWall<=(local_iolets_MaxSIZE/3))
						{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_iolet_Nash, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_6, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Wall,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity_out,
																															(float*)d_outletNormal,
																															n_Outlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalOutlets_mOutletWall, OutletWall_Inner); //
						}
						else{
							hipLaunchKernelGGL(hemelb::GPU_CollideStream_wall_sBB_iolet_Nash_v2, dim3(nBlocks_Collide), dim3(nThreads_Collide), 0, Collide_Stream_PreRec_6, (double*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
																															(double*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat,
																															(double*)GPUDataAddr_dbl_MacroVars,
																															(int64_t*)GPUDataAddr_int64_Neigh_d,
																															(uint32_t*)GPUDataAddr_uint32_Wall,
																															(uint32_t*)GPUDataAddr_uint32_Iolet,
																															(distribn_t*)d_ghostDensity_out,
																															(float*)d_outletNormal,
																															n_Outlets,
																															(mLatDat->GetLocalFluidSiteCount()),
																															first_Index, (first_Index + site_Count), mLatDat->totalSharedFs, mState->GetTimeStep(),
																															n_LocalOutlets_mOutletWall, (site_t*)GPUDataAddr_OutletWall_Inner);
						}

					}
					//--------------------------------------------------------------------
				}
				// ends the if site_Count!=0, for Collision Type 6.
				// ====================================================================================================================================================

				//-------------------------------------------------------------------------------------------------------------
				// Overlap the calculations during PreReceive and the memory transfer at domain edges
				// Only if the steps sequence is modified.
				// 		a. PreSend
				//		b. PreReceive
				//		c. Send
				// Synchronisation barrier
				if(myPiD!=0){
					hipStreamSynchronize(Collide_Stream_PreSend_1);
					hipStreamSynchronize(Collide_Stream_PreSend_2);
					hipStreamSynchronize(Collide_Stream_PreSend_3);
					hipStreamSynchronize(Collide_Stream_PreSend_4);
					hipStreamSynchronize(Collide_Stream_PreSend_5);
					hipStreamSynchronize(Collide_Stream_PreSend_6);
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
				//if(myPiD!=0) cudaStreamSynchronize(stream_memCpy_GPU_CPU_domainEdge);

/*
				// ====================================================================================================================================================
				// Send the MacroVariables (density and Velocity) to the CPU
				// THink where to place this!!! To do!!!
				if (mState->GetTimeStep() % 100 == 0)
				{
					if(myPiD!=0) {
						// Must ensure that writing the updated macroVariables from the above kernels has completed.
						cudaStreamSynchronize(Collide_Stream_PreRec_1);
						cudaStreamSynchronize(Collide_Stream_PreRec_2);
						cudaStreamSynchronize(Collide_Stream_PreRec_3);
						cudaStreamSynchronize(Collide_Stream_PreRec_4);
						cudaStreamSynchronize(Collide_Stream_PreRec_5);
						cudaStreamSynchronize(Collide_Stream_PreRec_6);
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
				site_t offset1 = 0;
				StreamAndCollide(mMidFluidCollision, offset1, mLatDat->GetMidDomainCollisionCount(0));

				// Collision Type 2:
				offset1 += mLatDat->GetMidDomainCollisionCount(0);
				StreamAndCollide(mWallCollision, offset1, mLatDat->GetMidDomainCollisionCount(1));

				// Collision Type 3:
				offset1 += mLatDat->GetMidDomainCollisionCount(1);
				StreamAndCollide(mInletCollision, offset1, mLatDat->GetMidDomainCollisionCount(2));

				// Collision Type 4:
				offset1 += mLatDat->GetMidDomainCollisionCount(2);
				StreamAndCollide(mOutletCollision, offset1, mLatDat->GetMidDomainCollisionCount(3));

				// Collision Type 5:
				offset1 += mLatDat->GetMidDomainCollisionCount(3);
				StreamAndCollide(mInletWallCollision, offset1, mLatDat->GetMidDomainCollisionCount(4));

				// Collision Type 6:
				offset1 += mLatDat->GetMidDomainCollisionCount(4);
				StreamAndCollide(mOutletWallCollision, offset1, mLatDat->GetMidDomainCollisionCount(5));

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

#ifndef HEMELB_CUDA_AWARE_MPI
				// Local rank
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();
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
						Maybe remove the synch point: cudaStreamSynchronize(stream_memCpy_CPU_GPU_domainEdge);
						 	and just use the same cuda stream used in the HtD memcpy above in function Read_DistrFunctions_CPU_to_GPU_totalSharedFs (stream_memCpy_CPU_GPU_domainEdge)
						for launching the cuda kernel
							hemelb::GPU_StreamReceivedDistr
						OR THE REVERSE CASE: Use the stream: stream_ReceivedDistr. Follow this approach !!!
				*/
				if(myPiD!=0) {
					// Not needed if using the stream: stream_ReceivedDistr in Read_DistrFunctions_CPU_to_GPU_totalSharedFs.
					// cudaStreamSynchronize(stream_memCpy_CPU_GPU_domainEdge); // Needed if we switch to asynch memcopy and use this stream in Read_DistrFunctions_CPU_to_GPU_totalSharedFs
					/*
					// The following might be needed here for cases where the PostReceive Step is usefull, e.g. for interpolating types of BCs,
					// Otherwise could be moved before the GPU_SwapOldAndNew kernel
					cudaStreamSynchronize(Collide_Stream_PreRec_1);
					cudaStreamSynchronize(Collide_Stream_PreRec_2);
					cudaStreamSynchronize(Collide_Stream_PreRec_3);
					cudaStreamSynchronize(Collide_Stream_PreRec_4);
					cudaStreamSynchronize(Collide_Stream_PreRec_5);
					cudaStreamSynchronize(Collide_Stream_PreRec_6);
					*/
				}
#endif

				//----------------------------------
				// 2*. Cuda kernel to do the re-allocation into the destination buffer "f_new" using the  streamingIndicesForReceivedDistributions
				// Cuda kernel set-up
				site_t totSharedFs = mLatDat->totalSharedFs;
				int nThreadsPerBlock_StreamRecDistr = 128;				//Number of threads per block for the Collision step
				dim3 nThreads_StreamRecDistr(nThreadsPerBlock_StreamRecDistr);
				int nBlocks_StreamRecDistr = totSharedFs/nThreadsPerBlock_StreamRecDistr			+ ((totSharedFs % nThreadsPerBlock_StreamRecDistr > 0)         ? 1 : 0);

				if (nBlocks_StreamRecDistr!=0)
					hipLaunchKernelGGL(hemelb::GPU_StreamReceivedDistr, dim3(nBlocks_StreamRecDistr), dim3(nThreads_StreamRecDistr), 0, stream_ReceivedDistr, (distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat,
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
				// Ensure that the received distr. functions have been placed in fNew beforing swaping the populations (fNew -> fOld)
				//if (myPiD!=0) cudaStreamSynchronize(stream_ReceivedDistr);

				// Swap the f's (Place fNew in fOld).
				// fluid sites limits (just swap the distr. functions of the fluid sites (ignore the totalSharedFs):
				site_t offset = 0;
				site_t site_Count = mLatDat->GetLocalFluidSiteCount(); // Total number of fluid sites: GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t)here

				// Syncrhonisation Barrier for the PreReceive collision cuda streams
				if(myPiD!=0) {
					// The following might be needed in PostReceive() for cases where the PostReceive Step is usefull, e.g. for interpolating types of BCs,
					// Otherwise could be moved here before the GPU_SwapOldAndNew kernel
					hipStreamSynchronize(Collide_Stream_PreRec_1);
					hipStreamSynchronize(Collide_Stream_PreRec_2);
					hipStreamSynchronize(Collide_Stream_PreRec_3);
					hipStreamSynchronize(Collide_Stream_PreRec_4);
					hipStreamSynchronize(Collide_Stream_PreRec_5);
					hipStreamSynchronize(Collide_Stream_PreRec_6);
				}

				/*
				// Approach 1: Using a GPU copy kernel
				// Cuda kernel set-up
				int nThreadsPerBlock_SwapOldAndNew = 64;				//Number of threads per block for the Collision step
				dim3 nThreads_Swap(nThreadsPerBlock_SwapOldAndNew);
				int nBlocks_Swap = site_Count/nThreadsPerBlock_SwapOldAndNew			+ ((site_Count % nThreadsPerBlock_SwapOldAndNew > 0)         ? 1 : 0);

				if(nBlocks_Swap!=0)
					hemelb::GPU_SwapOldAndNew <<<nBlocks_Swap, nThreads_Swap, 0, stream_ReceivedDistr>>> ( (double*)GPUDataAddr_dbl_fOld_b, (double*)GPUDataAddr_dbl_fNew_b, site_Count, offset, (offset + site_Count));
					//hemelb::GPU_SwapOldAndNew <<<nBlocks_Swap, nThreads_Swap, 0, stream_SwapOldAndNew>>> ( (double*)GPUDataAddr_dbl_fOld_b, (double*)GPUDataAddr_dbl_fNew_b, site_Count, offset, (offset + site_Count));
				// End of Approach 1
				*/

				//========================================================================================================
				// Approach 2: Using cudaMemcpyDeviceToDevice:
				// As this is a single large copy from device global memory to device global memory, then  cudaMemcpyDeviceToDevice should be ok.
				// See the discussion here: https://stackoverflow.com/questions/22345391/cuda-device-memory-copies-cudamemcpydevicetodevice-vs-copy-kernel
				if (myPiD!=0) {
				hipError_t cudaStatus;
				unsigned long long MemSz = site_Count * LatticeType::NUMVECTORS * sizeof(distribn_t); // Total memory size
				cudaStatus = hipMemcpyAsync(&(((distribn_t*)mLatDat->GPUDataAddr_dbl_fOld_b_mLatDat)[0]), &(((distribn_t*)mLatDat->GPUDataAddr_dbl_fNew_b_mLatDat)[0]), MemSz, hipMemcpyDeviceToDevice, stream_ReceivedDistr);
				if (cudaStatus != hipSuccess) fprintf(stderr, "GPU memory copy device-to-device failed ... \n");
				}
				// End of Approach 2
				//========================================================================================================


				//========================================================================================================
				// Think where to place this!!! To do!!!
				//kernels::HydroVarsBase<LatticeType> hydroVars(geometry::Site<geometry::LatticeData> _site);
				//kernels::HydroVarsBase<LatticeType> hydroVars;
				if (mState->GetTimeStep() % 1000 == 0)
				{
					// Check whether the hemeLB picks up the macroVariables at the EndIteration step???
					// Only the data in propertyCache, i.e. propertyCache.densityCache and propertyCache.velocityCache
					lb::MacroscopicPropertyCache& propertyCache = GetPropertyCache();
					if(myPiD!=0)
						//Read_Macrovariables_GPU_to_CPU(0, mLatDat->GetLocalFluidSiteCount(), propertyCache, kernels::HydroVars<LB_KERNEL> hydroVars(const geometry::Site<geometry::LatticeData>& _site)); // Copy the whole array GPUDataAddr_dbl_fNew_b from the GPU to CPUDataAddr_dbl_fNew_b. Then just read just the elements needed.
						Read_Macrovariables_GPU_to_CPU(0, mLatDat->GetLocalFluidSiteCount(), propertyCache);
				}
				//========================================================================================================

				// Make sure the swap of distr. functions is completed, before the next iteration

				/*// Testing - Remove later:
				if (myPiD!=0)
				{
					cudaDeviceSynchronize(); // Included a cudaStreamSynchronize at the beginning of PreSend(); Should do the same job

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
			}

		template<class LatticeType>
			void LBM<LatticeType>::ReadParameters()
			{
				std::vector<lb::iolets::InOutLet*> inlets = mSimConfig->GetInlets();
				std::vector<lb::iolets::InOutLet*> outlets = mSimConfig->GetOutlets();
				inletCount = inlets.size();
				outletCount = outlets.size();
				mParams.StressType = mSimConfig->GetStressType();

				//printf("Number of inlets: %d, outlets: %d \n\n", inletCount, outletCount);
			}

	}
}

#endif /* HEMELB_LB_LB_HPP */
