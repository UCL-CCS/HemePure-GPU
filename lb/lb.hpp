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
	  cudaDeviceSynchronize();
	  cudaError_t error = cudaGetLastError();
	  if(error != cudaSuccess)
	  {
		printf("CUDA error at %s:%i: \"%s\" at proc: %i\n", filename, line_number, cudaGetErrorString(error), myProc);
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

#ifdef HEMELB_USE_GPU

				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();				
				
				// std::printf("Local Rank for Initialise= %i\n\n", myPiD);
				if (myPiD!=0){
					// Initialise the GPU here - Memory allocations etc
					Initialise_GPU();
					hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // Check for last cuda error: Remember that it is in DEBUG flag

					// Initialise the kernels' setup	
					// Initialise_kernels_GPU();
				}
#endif

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

		template<class LatticeType>	
			bool LBM<LatticeType>::Read_DistrFunctions_CPU_to_GPU(int64_t firstIndex, int64_t siteCount)
			{
				cudaError_t cudaStatus;

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
					cudaStatus = cudaMemcpy(&(((distribn_t*)GPUDataAddr_dbl_fOld_b)[(LB_ind*nFluid_nodes)+firstIndex]), &(Data_dbl_fOld_Tr[LB_ind * siteCount]), MemSz, cudaMemcpyHostToDevice);
					if (cudaStatus != cudaSuccess) fprintf(stderr, "GPU memory copy failed (%d)\n", LB_ind);		
				}


				delete[] Data_dbl_fOld_Tr;

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
				cudaError_t cudaStatus;

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
				cudaStatus = cudaMemcpy(&(fNew_GPU_b[0]), &(((distribn_t*)GPUDataAddr_dbl_fNew_b)[0]), TotalMem_dbl_fNew_b, cudaMemcpyDeviceToHost);	    	
				if(cudaStatus != cudaSuccess){
					const char * eStr = cudaGetErrorString (cudaStatus);
					printf("GPU memory transfer for ReadGPU_distr failed with error: \"%s\" at proc# %i\n", eStr, myPiD);
					delete[] fNew_GPU_b; 	
					return false;
				}  									

				//--------------------------------------------------------------------------
				//	b. Density 
							
				distribn_t* dens_GPU = new distribn_t[nFluid_nodes];
				
				if(dens_GPU==0){printf("Density Memory allocation failure"); return false;}
				
				unsigned long long MemSz = nFluid_nodes*sizeof(distribn_t);

				cudaStatus = cudaMemcpy(dens_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[0]), MemSz, cudaMemcpyDeviceToHost);	    					
				//cudaStatus = cudaMemcpyAsync(dens_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[0]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Dens);	    	
								
				if(cudaStatus != cudaSuccess){	
					printf("GPU memory transfer for density failed\n");				
					delete[] dens_GPU; 	
					return false;
				}

				// c. Velocity
				distribn_t* vx_GPU = new distribn_t[nFluid_nodes];
				distribn_t* vy_GPU = new distribn_t[nFluid_nodes];
				distribn_t* vz_GPU = new distribn_t[nFluid_nodes];
  
				if(vx_GPU==0 || vy_GPU==0 || vz_GPU==0){ printf("Memory allocation failure"); return false;}
  
				cudaStatus = cudaMemcpy(vx_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[1ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost);	    	
				//cudaStatus = cudaMemcpyAsync(vx_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[1ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Vel);	    	
				if(cudaStatus != cudaSuccess){				
					printf("GPU memory transfer Vel(1) failed\n");
					delete[] vx_GPU; 	
					return false;				
				}
  
				cudaStatus = cudaMemcpy(vy_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[2ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost);	    	
				//cudaStatus = cudaMemcpyAsync(vy_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[2ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Vel);	    	
				if(cudaStatus != cudaSuccess){				
					printf("GPU memory transfer Vel(2) failed\n");
					delete[] vy_GPU; 	
					return false;				
				}
  
				cudaStatus = cudaMemcpy(vz_GPU, &(((distribn_t*)GPUDataAddr_dbl_MacroVars)[3ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost);	    	
				//cudaStatus = cudaMemcpyAsync(vz_GPU, &(((distribn_t*)GMem_dbl_MacroVars)[3ULL*nFluid_nodes]), MemSz, cudaMemcpyDeviceToHost, stream_Read_Data_GPU_Vel);	    	
				if(cudaStatus != cudaSuccess){				
					printf("GPU memory transfer Vel(2) failed\n");
					delete[] vz_GPU; 	
					return false;				
				}							  
				//--------------------------------------------------------------------------
				hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // Check for last cuda error: Remember that it is in DEBUG flag
				

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
										
					for (unsigned int ii = 0; ii < LatticeType::NUMVECTORS; ii++)
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
						//******************************************************************************

						if (site.HasWall(ii)){
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
			bool LBM<LatticeType>::FinaliseGPU()
			{			
				cudaError_t cudaStatus;
				
				/*
					cudaStreamDestroy(stream1);
					cudaStreamDestroy(stream_Read_distr_Data_GPU);	
				*/				
											
				cudaStatus = cudaFree(GPUDataAddr_dbl_fOld);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }
				
				cudaStatus = cudaFree(GPUDataAddr_dbl_fNew);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }
								
				cudaStatus = cudaFree(GPUDataAddr_dbl_MacroVars);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }
								
				cudaStatus = cudaFree(GPUDataAddr_int64_Neigh);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }
	  
				cudaStatus = cudaFree(GPUDataAddr_uint32_Wall);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }

				cudaStatus = cudaFree(GPUDataAddr_uint32_Iolet);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }

				cudaStatus = cudaFree(d_ghostDensity);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }

				cudaStatus = cudaFree(d_inletNormal);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }

				

				
				cudaStatus = cudaFree(GPUDataAddr_dbl_fOld_b);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }
				
				cudaStatus = cudaFree(GPUDataAddr_dbl_fNew_b);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }
				
				cudaStatus = cudaFree(GPUDataAddr_int64_Neigh_b);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "cudaFree failed\n"); return false; }
				
				printf("CudaFree - Delete dynamically allocated memory on the GPU.\n\n");	
				
				return true;
			}


template<class LatticeType>
			bool LBM<LatticeType>::Initialise_GPU()
			{			
				
				// Local rank									
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();				

				cudaError_t cudaStatus;
								
				// std::printf("Local Rank = %i and local fluid sites = %i \n\n", myPiD, mLatDat->GetLocalFluidSiteCount());
				
				// --------------------------------------------------------------------------------------------------------------------------------------------------------------
				// Option: Separately each element in memory: 
				// f_old, f_new															- Comment: need to add the totalSharedFs values +1: Done!!!
				// density ???, velocity???, limits for the various collision types		- Comment: To do!!! 

				// Arrange the data in 2 ways - To do!!! 
				//	a. Arrange by fluid index (as is oldDistributions), i.e f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind] 
				//	b. Arrange by index_LB, i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., fq[0 to (nFluid_nodes-1)]

				// Total number of fluid sites					
				uint64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount(); // Actually GetLocalFluidSiteCount returns localFluidSites of type int64_t (site_t) 				
				uint64_t totSharedFs = mLatDat->totalSharedFs;

				// unsigned long long nArr_dbl = nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs;	//Number of doubles in array									
				// std::printf("Proc# %i : Total Fluid nodes = %i, totalSharedFs = %i \n\n", myPiD, nFluid_nodes, totSharedFs);	// Test that I can access the value of totalSharedFs (protected member of class LatticeData (geometry/LatticeData.h) - declares class LBM as friend)   							 
				
				unsigned long long TotalMem_dbl_fOld = ( nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs)  * sizeof(distribn_t); // Total memory size for fOld 					
				unsigned long long TotalMem_dbl_fNew = TotalMem_dbl_fOld;	// Total memory size for fNew
				unsigned long long TotalMem_dbl_MacroVars = (1+3) * nFluid_nodes  * sizeof(distribn_t); // Total memory size for macroVariables: density and Velocity n[nFluid_nodes], u[nFluid_nodes][3]					

				distribn_t* Data_dbl_fOld = new distribn_t[TotalMem_dbl_fOld/sizeof(distribn_t)];	// distribn_t (type double)
				distribn_t* Data_dbl_fNew = new distribn_t[TotalMem_dbl_fNew/sizeof(distribn_t)];	// distribn_t (type double)
				// distribn_t* Data_dbl_MacroVars = new distribn_t[TotalMem_dbl_MacroVars/sizeof(distribn_t)];	// distribn_t (type double)
								  
				//if(!Data_dbl_fOld || !Data_dbl_fNew || !Data_dbl_MacroVars){									
				if(!Data_dbl_fOld || !Data_dbl_fNew){									
					std::cout << "Memory allocation error" << std::endl;
					return false;														  				 
				}			 
				//else{ std::printf("Memory allocation Data_dbl successful from Proc# %i \n\n", myPiD);}				  
				  

				//--------------------------------------------------------------------------------------------------
				// Alocate memory on the GPU for MacroVariables: density and Velocity
				// Number of elements (type double/distribn_t) 
				// uint64_t nArray_MacroVars = nFluid_nodes; // uint64_t (unsigned long long int)			 				
								
				cudaStatus = cudaMalloc((void**)&GPUDataAddr_dbl_MacroVars, TotalMem_dbl_MacroVars);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation failed\n"); return false; }									
				//--------------------------------------------------------------------------------------------------
				

				//--------------------------------------------------------------------------------------------------
				// std::vector<distribn_t> oldDistributions; //! The distribution function fi's values for the previous time step.
				// oldDistributions.resize(localFluidSites * latticeInfo.GetNumVectors() + 1 + totalSharedFs);  -  see src/geometry/LatticeData.h (line 422 in function void PopulateWithReadData)				
				//--------------------------------------------------------------------------------------------------
				//	a. Arrange by fluid index (as is oldDistributions), i.e f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind] 			
				Data_dbl_fOld = mLatDat->GetFOld(0);  // distribn_t (type double) - Data_dbl points to &oldDistributions[0]
				Data_dbl_fNew = mLatDat->GetFNew(0);  // distribn_t (type double) - Data_dbl points to &newDistributions[0]
								
				for (site_t i = 0; i < mLatDat->GetLocalFluidSiteCount(); i++)								 
				{
					distribn_t ff[LatticeType::NUMVECTORS];
					for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++)					
					{						
						ff[l] = *(mLatDat->GetFOld(i * LatticeType::NUMVECTORS + l));
						/*
						std::printf("Distribution Functions: ff[%d] = %.5f \n\n", l, ff[l]);
						std::printf("Distribution Functions Data_dbl: ff[%d] = %.5f \n\n", l, *(&Data_dbl_fOld[i * LatticeType::NUMVECTORS + l]));						
						*/
						if (ff[l] != *(&Data_dbl_fOld[i * LatticeType::NUMVECTORS + l]))
						{
							std::printf("Distribution Functions: ff[%d] = %.5f \n\n", l, ff[l]);
							std::printf("Distribution Functions Data_dbl: ff[%d] = %.5f \n\n", l, *(&Data_dbl_fOld[i * LatticeType::NUMVECTORS + l]));						
						}				
					}				
				} // Passed the test
				// End of allocation by fluid index
				//--------------------------------------------------------------------------------------------------

				//--------------------------------------------------------------------------------------------------
				//	b. Arrange by index_LB, i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., f_(q-1)[0 to (nFluid_nodes-1)]
				distribn_t* Data_dbl_fOld_b = new distribn_t[TotalMem_dbl_fOld/sizeof(distribn_t)];	// distribn_t (type double)
				distribn_t* Data_dbl_fNew_b = new distribn_t[TotalMem_dbl_fNew/sizeof(distribn_t)];	// distribn_t (type double)
								  
				if(!Data_dbl_fOld_b || !Data_dbl_fNew_b){									
					std::cout << "Memory allocation error" << std::endl;
					return false;														  				 
				}			 
				//else{ std::printf("Memory allocation Data_dbl_b (by index LB) successful from Proc# %i \n\n", myPiD);}
								
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

				
				for (site_t i = 0; i < mLatDat->GetLocalFluidSiteCount(); i++)								 
				{
					distribn_t ff[LatticeType::NUMVECTORS];
					for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++)					
					{						
						ff[l] = *(mLatDat->GetFOld(i * LatticeType::NUMVECTORS + l));
						
						// std::printf("Distribution Functions: ff[%d] = %.5f \n\n", l, ff[l]);
						// std::printf("Distribution Functions Data_dbl: ff[%d] = %.5f \n\n", l, *(&Data_dbl_fOld[l * mLatDat->GetLocalFluidSiteCount() + i]));						
						
						if (ff[l] != *(&Data_dbl_fOld_b[l * mLatDat->GetLocalFluidSiteCount() + i]))
						{
							std::printf("Distribution Functions: ff[%d] = %.5f \n\n", l, ff[l]);
							std::printf("Distribution Functions Data_dbl_b: ff[%d] = %.5f \n\n", l, *(&Data_dbl_fOld_b[l * mLatDat->GetLocalFluidSiteCount() + i]));						
						}				
					}				
				} // Check the second method - Arrange by index LBM		
				
				//--------------------------------------------------------------------------------------------------

				//
				// Alocate memory on the GPU
				// Number of elements (type double/distribn_t) in oldDistributions and newDistributions 
				// including the extra part (+1 + totalSharedFs) - Done!!! 
				uint64_t nArray_oldDistr = nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs; // uint64_t (unsigned long long int)			 
				uint64_t nArray_newDistr = nFluid_nodes * LatticeType::NUMVECTORS + 1 + totSharedFs; // Remove this in the future - keep only nArray_oldDistr = nArray_newDistr = nArray_Distr- To do!!!


				//--------------------------------------------------------------------------------------------------
				//	a. Arrange by fluid index (as is oldDistributions), i.e f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind] 						
				cudaStatus = cudaMalloc((void**)&GPUDataAddr_dbl_fOld, nArray_oldDistr * sizeof(distribn_t));
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation failed\n"); return false; }

				cudaStatus = cudaMalloc((void**)&GPUDataAddr_dbl_fNew, nArray_newDistr * sizeof(distribn_t));  
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation failed\n"); return false; }
				
				// Memory copy from host (Data_dbl_fOld) to Device (GPUDataAddr_dbl_fOld) 				
				cudaStatus = cudaMemcpy(GPUDataAddr_dbl_fOld, Data_dbl_fOld, nArray_oldDistr * sizeof(distribn_t), cudaMemcpyHostToDevice);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer Host To Device failed\n"); return false; }

				// Memory copy from host (Data_dbl_fNew) to Device (GPUDataAddr_dbl_fNew) 
				cudaStatus = cudaMemcpy(GPUDataAddr_dbl_fNew, Data_dbl_fNew, nArray_newDistr * sizeof(distribn_t), cudaMemcpyHostToDevice);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer Host To Device failed\n"); return false; }
				//

				//--------------------------------------------------------------------------------------------------	
				//	b. Arrange by index_LB
				//		i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., f_(q-1)[0 to (nFluid_nodes-1)]
				// 	Include here for future testing!!! - To do!!!
				
				cudaStatus = cudaMalloc((void**)&GPUDataAddr_dbl_fOld_b, nArray_oldDistr * sizeof(distribn_t));
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation failed\n"); return false; }

				cudaStatus = cudaMalloc((void**)&GPUDataAddr_dbl_fNew_b, nArray_newDistr * sizeof(distribn_t));  
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation failed\n"); return false; }
				
				// Memory copy from host (Data_dbl_fOld_b) to Device (GPUDataAddr_dbl_fOld_b) 
				cudaStatus = cudaMemcpy(GPUDataAddr_dbl_fOld_b, Data_dbl_fOld_b, nArray_oldDistr * sizeof(distribn_t), cudaMemcpyHostToDevice);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer Host To Device failed\n"); return false; }

				// Memory copy from host (Data_dbl_fNew_b) to Device (GPUDataAddr_dbl_fNew_b) 
				cudaStatus = cudaMemcpy(GPUDataAddr_dbl_fNew_b, Data_dbl_fNew_b, nArray_newDistr * sizeof(distribn_t), cudaMemcpyHostToDevice);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer Host To Device failed\n"); return false; }
								
				//=================================================================================================================================
								
				
				//=================================================================================================================================
				// Neighbouring indices - necessary for the STREAMING STEP

				// 	The total size of the neighbouring indices should be: neighbourIndices.resize(latticeInfo.GetNumVectors() * localFluidSites); (see geometry/LatticeData.cc:369)
				//		Organised in the same way as the distribution functions in the HemeLB code, i.e. based on the fluid index. (What is refered below as method (a))
				
				//		Type site_t (units.h:28:		typedef int64_t site_t;)
				//		geometry/LatticeData.h:634:		std::vector<site_t> neighbourIndices; //! Data about neighbouring fluid sites.
				//	Memory requirements - Just in case that we check whether data fit in Device Global memory - Need to add a statement that will check this, and exit sim if resources are not sufficient. To do!!!
				unsigned long long TotalMem_int64_Neigh = ( nFluid_nodes * LatticeType::NUMVECTORS)  * sizeof(site_t); // Total memory size for neighbouring Indices 					
				
				
				// -----------------------------------------------------------------------			
				//	a. Arrange by fluid index (as is neighbourIndices[]), i.e neigh_0[0], neigh_1[0], neigh_2[0], ..., neigh_(q-1)[0] and for the general case with Fluid Index Ind : neigh_0[Ind], neigh_1[Ind], neigh_2[Ind], ..., neigh_(q-1)[Ind] 							
				site_t* Data_int64_Neigh = new site_t[TotalMem_int64_Neigh/sizeof(site_t)];	// site_t (type int64_t)
				
				if(!Data_int64_Neigh){									
					std::cout << "Memory allocation error - Neigh." << std::endl;
					return false;														  				 
				}			 
				//else{ std::printf("Memory allocation Data_int64_Neigh successful from Proc# %i \n\n", myPiD); }				  
				
				Data_int64_Neigh = &(mLatDat->neighbourIndices[0]);  // Data_int64_Neigh points to &(mLatDat->neighbourIndices[0])
				// -----------------------------------------------------------------------
				
				// -----------------------------------------------------------------------
				// b. Arrange by index_LB, i.e. neigh_0[0 to (nFluid_nodes-1)], neigh_1[0 to (nFluid_nodes-1)], ..., neigh_(q-1)[0 to (nFluid_nodes-1)]
				site_t* Data_int64_Neigh_b = new site_t[TotalMem_int64_Neigh/sizeof(site_t)];	// site_t (type int64_t)
				if(!Data_int64_Neigh_b){									
					std::cout << "Memory allocation error - Neigh." << std::endl;
					return false;														  				 
				}			 
				//else{ std::printf("Memory allocation Data_int64_Neigh(b) successful from Proc# %i \n\n", myPiD);}				  
			
				// Re-arrange the neighbouring data - organised by index LB 	
				for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++)																	 
				{					
					for (site_t i = 0; i < mLatDat->GetLocalFluidSiteCount(); i++)
					{							
						Data_int64_Neigh_b[(int64_t)l * mLatDat->GetLocalFluidSiteCount() + i] = mLatDat->neighbourIndices[(int64_t)(LatticeType::NUMVECTORS)*i  + l];
						
						//std::printf("Memory allocation Data_int64_Neigh(b) successful from Proc# %i \n\n", myPiD);								 				  

					}				
				}				
				// ------------------------------------------------------------------------
				
				/*
				// Testing:								
				for (int64_t site_Index = 0; site_Index < 10; site_Index++){										
					geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_Index);					
					for (Direction direction=0; direction< LatticeType::NUMVECTORS; direction++){

						int64_t streamedIndex_Data_Neigh_b = Data_int64_Neigh_b[(unsigned long long)direction * mLatDat->GetLocalFluidSiteCount() + site_Index];
						
						int64_t streamedIndex_Data_Neigh = Data_int64_Neigh[site_Index * LatticeType::NUMVECTORS + direction];

						int64_t streamedIndex = site.GetStreamedIndex<LatticeType> (direction);
						int64_t streamedIndex_2 = *(&(mLatDat->neighbourIndices[site_Index * LatticeType::NUMVECTORS + direction]));
						printf("Site Index = %lld, Stream. Dir.[%d] = %lld - Vs Stream. Dir._2 = %lld - Vs Data_Neigh_b[%lld] = %lld - Vs Data_Neigh[%lld] = %lld \n\n", site_Index, direction, streamedIndex, streamedIndex_2, ((unsigned long long)direction * mLatDat->GetLocalFluidSiteCount() + site_Index),streamedIndex_Data_Neigh_b, (site_Index * LatticeType::NUMVECTORS + direction), streamedIndex_Data_Neigh);
					}
				}	
				*/

				/*
				for (int64_t site_Index = 0; site_Index < 39; site_Index++){	
					printf("Index Array= %lld, Data_Neigh_b[%lld] = %lld - Vs Data_Neigh[%lld] = %lld - Vs neighbourIndices[%lld]= %lld \n\n", site_Index, site_Index, Data_int64_Neigh_b[site_Index], site_Index, Data_int64_Neigh[site_Index], site_Index,mLatDat->neighbourIndices[site_Index]);
				}
				*/


				// Number of elements (type long long int/site_t) in neighbourIndices  - To do!!!								 
				uint64_t nArray_Neigh = nFluid_nodes * LatticeType::NUMVECTORS; // uint64_t (unsigned long long int)			 				
				
				// ------------------------------------------------------------------------
				// a. Arrange by fluid index (as is neighbourIndices)
				// Alocate memory on the GPU				
				cudaStatus = cudaMalloc((void**)&GPUDataAddr_int64_Neigh, nArray_Neigh * sizeof(site_t));
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation for Neigh. failed\n"); return false; }
								
				// Memory copy from host (Data_int64_Neigh) to Device (GPUDataAddr_int64_Neigh) 				
				cudaStatus = cudaMemcpy(GPUDataAddr_int64_Neigh, Data_int64_Neigh, nArray_Neigh * sizeof(site_t), cudaMemcpyHostToDevice);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Neigh. failed\n"); return false; }
				//

				// ------------------------------------------------------------------------
				//	b. Arrange by index_LB, i.e. neigh_0[0 to (nFluid_nodes-1)], neigh_1[0 to (nFluid_nodes-1)], ..., neigh_(q-1)[0 to (nFluid_nodes-1)]
				cudaStatus = cudaMalloc((void**)&GPUDataAddr_int64_Neigh_b, nArray_Neigh * sizeof(site_t));
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation for Neigh.(b) failed\n"); return false; }
				
				// Memory copy from host (Data_int64_Neigh_b) to Device (GPUDataAddr_int64_Neigh_b) 				
				cudaStatus = cudaMemcpy(GPUDataAddr_int64_Neigh_b, Data_int64_Neigh_b, nArray_Neigh * sizeof(site_t), cudaMemcpyHostToDevice);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Neigh.(b) failed\n"); return false; }
				
				//=================================================================================================================================
				
				

				//***********************************************************************************************************************************
				// Fluid-Wall links
				// Access the information for the fluid-wall links:
				//	function GetWallIntersectionData returns wallIntersection variable that we want... 
				
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
					}				

				}		
				// Ends the loop for Filling the array Data_uint32_WallIntersect  


				// Alocate memory on the GPU				
				cudaStatus = cudaMalloc((void**)&GPUDataAddr_uint32_Wall, nFluid_nodes * sizeof(uint32_t));
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation for Wall-Fluid Intersection failed\n"); return false; }
								
				// Memory copy from host (Data_uint32_WallIntersect) to Device (GPUDataAddr_uint32_Wall) 				
				cudaStatus = cudaMemcpy(GPUDataAddr_uint32_Wall, Data_uint32_WallIntersect, nFluid_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Neigh. failed\n"); return false; }
				//***********************************************************************************************************************************

				//***********************************************************************************************************************************
				// Fluid-Inlet links
				// Access the information for the fluid-inlet links:
				//	function GetIoletIntersectionData() returns ioletIntersection variable that we want... 
				
				// Do we need nFluid_nodes elements of type uint32_t??? Think... 
				// In PreSend() the site limits for mInletCollision:
				// offset = mLatDat->GetMidDomainSiteCount() + mLatDat->GetDomainEdgeCollisionCount(0) + mLatDat->GetDomainEdgeCollisionCount(1);
				// siteCount_iolet_PreSend = mLatDat->GetDomainEdgeCollisionCount(2);
				
				// In PreReceive() the site limits for mInletCollision:
				// offset = 0 + mLatDat->GetMidDomainCollisionCount(0) + mLatDat->GetMidDomainCollisionCount(1);
				// siteCount_iolet_PreReceive = mLatDat->GetMidDomainCollisionCount(2);
				
				// Maybe consider allocating just (siteCount_iolet_PreSend + siteCount_iolet_PreReceive) elements. To do!!!
				
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
					test_Iolet_Intersect = site.GetSiteData().GetIoletIntersectionData(); // Returns the value of wallIntersection (type uint32_t) 

					Data_uint32_IoletIntersect[site_Index] = test_Iolet_Intersect;
					
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

				}		
				// Ends the loop for Filling the array Data_uint32_IoletIntersect  


				// Alocate memory on the GPU				
				cudaStatus = cudaMalloc((void**)&GPUDataAddr_uint32_Iolet, nFluid_nodes * sizeof(uint32_t));
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation for Iolet-Fluid Intersection failed\n"); return false; }
								
				// Memory copy from host (Data_uint32_IoletIntersect) to Device (GPUDataAddr_uint32_Iolet) 				
				cudaStatus = cudaMemcpy(GPUDataAddr_uint32_Iolet, Data_uint32_IoletIntersect, nFluid_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer Host To Device for Iolet failed\n"); return false; }
				//***********************************************************************************************************************************

				//***********************************************************************************************************************************
				// Ghost Density if Inlet/Outlet BCs is set to NashZerothOrderPressure
				// Just allocate the memory as the ghostDensity can change as a function of time. MemCopies(host-to-device) before the gpu inlet/outlet collision kernels
				int n_Inlets = mInletValues->GetLocalIoletCount();
				//printf("Number of inlets: %d \n\n", n_Inlets);

				cudaStatus = cudaMalloc((void**)&d_ghostDensity, n_Inlets * sizeof(distribn_t));
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation ghostDensity failed\n"); return false; }																
				//
				
				// Normals to Iolets					
				float* h_inletNormal = new float[3*n_Inlets]; 	// x,y,z components		
				for (int i=0; i<n_Inlets; i++){
					util::Vector3D<float> ioletNormal = mInletValues->GetLocalIolet(i)->GetNormal();					
					h_inletNormal[3*i] = ioletNormal.x;
					h_inletNormal[3*i+1] = ioletNormal.y;
					h_inletNormal[3*i+2] = ioletNormal.z;					
					//std::cout << "Cout: ioletNormal.x : " <<  h_inletNormal[i] << " - ioletNormal.y : " <<  h_inletNormal[i+1] << " - ioletNormal.z : " <<  h_inletNormal[i+2] << std::endl;
				}
				
				cudaStatus = cudaMalloc((void**)&d_inletNormal, 3*n_Inlets * sizeof(float));
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory allocation inletNormal failed\n"); return false; }												
				// Memory copy from host (h_inletNormal) to Device (d_inletNormal) 				
				cudaStatus = cudaMemcpy(d_inletNormal, h_inletNormal, 3*n_Inlets * sizeof(float), cudaMemcpyHostToDevice);
				if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer (inletNormal) Host To Device failed\n"); return false; }
				//***********************************************************************************************************************************





				// Check the total memory requirements
				// Change this in the future as roughly half of this (either memory arrangement (a) or (b)) will be needed. To do!!!
				// Add a check whether the memory on the GPU global memory is sufficient!!! Abort if not or split the domain into smaller subdomains and pass info gradually! To do!!!
				unsigned long long TotalMem_req = (TotalMem_dbl_fOld * 4 +  TotalMem_dbl_MacroVars + TotalMem_int64_Neigh *2 + TotalMem_uint32_WallIntersect + TotalMem_uint32_IoletIntersect); // 
				printf("Rank: %d - Total requested global memory %.2fGB \n\n", myPiD, ((double)TotalMem_req/1073741824.0));

				

				//=================================================================================================================================
				// Copy constants to the GPU memory - Limit is 64 kB
				//	2. Constants:
				//		a. weights for the equilibrium distr. functions
				//		b. Number of vectors: LatticeType::NUMVECTORS				
				//		c. INVERSEDIRECTIONS for the bounce simple back simple (Wall BCs): LatticeType::INVERSEDIRECTIONS
				//		d. Lattice Velocity directions CX[DmQn::NUMVECTORS], CY[DmQn::NUMVECTORS], CZ[DmQn::NUMVECTORS]
				//		e. Relaxation Time tau
				
				// 2.a. Weight coefficients for the equilibrium distr. functions				
				cudaStatus = cudaMemcpyToSymbol(hemelb::_EQMWEIGHTS_19, LatticeType::EQMWEIGHTS, LatticeType::NUMVECTORS*sizeof(double), 0, cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "GPU constant memory copy failed\n"); return false; 
					//goto Error;
				}					

				// 2.b. Number of vectors: LatticeType::NUMVECTORS
				static const unsigned int num_Vectors = LatticeType::NUMVECTORS;
				cudaStatus = cudaMemcpyToSymbol(hemelb::_NUMVECTORS, &num_Vectors, sizeof(num_Vectors), 0, cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "GPU constant memory copy failed\n"); return false; 
					//goto Error;
				}
															
				// 2.c. Inverse directions for the bounce back LatticeType::INVERSEDIRECTIONS[direction]				
				cudaStatus = cudaMemcpyToSymbol(hemelb::_InvDirections_19, LatticeType::INVERSEDIRECTIONS, LatticeType::NUMVECTORS*sizeof(int), 0, cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "GPU constant memory copy failed\n"); return false; 
					//goto Error;
				}

				// 2.d. Lattice Velocity directions CX[DmQn::NUMVECTORS], CY[DmQn::NUMVECTORS], CZ[DmQn::NUMVECTORS]
				cudaStatus = cudaMemcpyToSymbol(hemelb::_CX_19, LatticeType::CX, LatticeType::NUMVECTORS*sizeof(int), 0, cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "GPU constant memory copy failed\n"); return false; //goto Error;
				}				
				cudaStatus = cudaMemcpyToSymbol(hemelb::_CY_19, LatticeType::CY, LatticeType::NUMVECTORS*sizeof(int), 0, cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "GPU constant memory copy failed\n"); return false; //goto Error;			
				}
				cudaStatus = cudaMemcpyToSymbol(hemelb::_CZ_19, LatticeType::CZ, LatticeType::NUMVECTORS*sizeof(int), 0, cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "GPU constant memory copy failed\n"); return false; //goto Error;
				}			

				// 2.e. Relaxation Time tau
				//static const int num_Vectors = LatticeType::NUMVECTORS;
				// mParams object of type hemelb::lb::LbmParameters (struct LbmParameters)
				double tau = mParams.GetTau();				// printf("Relaxation Time = %.5f\n\n", tau);
				double minus_inv_tau = mParams.GetOmega();	// printf("Minus Inv. Relaxation Time = %.5f\n\n", minus_inv_tau);

				cudaStatus = cudaMemcpyToSymbol(hemelb::dev_tau, &tau, sizeof(tau), 0, cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "GPU constant memory copy failed\n"); return false; //goto Error;
				}

				cudaStatus = cudaMemcpyToSymbol(hemelb::dev_minusInvTau, &minus_inv_tau, sizeof(minus_inv_tau), 0, cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) { fprintf(stderr, "GPU constant memory copy failed\n"); return false; //goto Error;
				}							
				//=================================================================================================================================

				// Remove later... 
				if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function. 

				/*
				// Check the following - To do!!!
				// This is causing a problem - Why???
				delete[] Data_dbl_fOld;
				delete[] Data_dbl_fNew;				
				delete[] Data_int64_Neigh;
				*/

				return true;
  								
			}
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
				
				//IZ
				// Local rank									
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();
				//IZ

				// IZ
#ifdef HEMELB_USE_GPU	// If exporting computation on GPUs:	Create Cuda streams in the future - before the LBM time loop - To do!!!
				
				// ====================================================================================================================================================
				// Collision Type 1:
				site_t offset = mLatDat->GetMidDomainSiteCount();
				// printf("Rank: %d: Collision Type 1: Starting = %lld, SiteCount = %lld, Ending = %lld \n\n", myPiD, offset, mLatDat->GetDomainEdgeCollisionCount(0), (offset + mLatDat->GetDomainEdgeCollisionCount(0)));
								
				// To do!!!
				// For the development phase - one collision type at a time, we need to pass the info for the fOld from the CPU to the GPU so that all information is in the GPU global memory
				if(myPiD!=0) Read_DistrFunctions_CPU_to_GPU(offset, mLatDat->GetDomainEdgeCollisionCount(0)); 
				// cudaDeviceSynchronize(); // Probably not needed - Check. To do!!!				
												
				//----------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_Collide = 64;				//Number of threads per block for the Collision step
				dim3 nThreads_Collide(nThreadsPerBlock_Collide);								
				// Number of fluid nodes involved in the collision/streaming : mLatDat->GetDomainEdgeCollisionCount(0)
				int nBlocks_Collide = (mLatDat->GetDomainEdgeCollisionCount(0))/nThreadsPerBlock_Collide			+ ((mLatDat->GetDomainEdgeCollisionCount(0) % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
				//----------------------------------
								
				/* printf("Kernels' details: Simulation is being executed ... \n");
				std::cout << "Using " << nBlocks_Collide << " blocks of " << nThreadsPerBlock_Collide << " threads for the Collision (type 1) step of the Lattice Boltzmann Method - Domain Edges" << '\n'; */							
								
				// To access the data in GPU global memory nArr_dbl is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)
				// nArr_dbl = (mLatDat->GetLocalFluidSiteCount()) = nFluid_nodes
				hemelb::GPU_CollideStream_1_PreReceive <<<nBlocks_Collide, nThreads_Collide>>> ( (double*)GPUDataAddr_dbl_fOld_b, (double*)GPUDataAddr_dbl_fNew_b, (double*)GPUDataAddr_dbl_MacroVars, (int64_t*)GPUDataAddr_int64_Neigh_b, (mLatDat->GetLocalFluidSiteCount()), offset, (offset + mLatDat->GetDomainEdgeCollisionCount(0)), mLatDat->totalSharedFs); // 
				
				// Synchronisation point - Must include this at the end of all the GPU kernels launched in PreReceive, as the cuda kernels are launched and then control is returned to the CPU - Must wait to complete
				cudaDeviceSynchronize();				
				if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function. 
				//--------------------------------------------------------------------------------------------------------------------------------------------------
				//
				// Memory transfer from the GPU to the CPU
				// Has to be placed at the end of PreSend so that all the info is passed from the GPU to the CPU and then exchanged between neighbouring ranks. To do!!!
				lb::MacroscopicPropertyCache& propertyCache = GetPropertyCache();
				if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU(offset, mLatDat->GetDomainEdgeCollisionCount(0), propertyCache); // Copy the whole array GPUDataAddr_dbl_fNew_b from the GPU to CPUDataAddr_dbl_fNew_b. Then just read just the elements needed.  
				//
				
				
				// ====================================================================================================================================================
				// Collision Type 2:
				offset += mLatDat->GetDomainEdgeCollisionCount(0);								
				// StreamAndCollide(mWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(1));

				// GPU COLLISION KERNEL: 
				// Fluid ID range: [first_Index, first_Index + site_Count)
				int64_t first_Index = offset;	// Start Fluid Index
				int64_t site_Count = mLatDat->GetDomainEdgeCollisionCount(1);

				if(myPiD!=0) Read_DistrFunctions_CPU_to_GPU(first_Index, site_Count); // Mem. Copy Host to Device - Synchronous!!!
				
				//-------------------------------------
				// Kernel set-up
				nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
								
				// To access the data in GPU global memory: 
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)				
				hemelb::GPU_CollideStream_2 <<<nBlocks_Collide, nThreads_Collide>>> (	(double*)GPUDataAddr_dbl_fOld_b, 
																						(double*)GPUDataAddr_dbl_fNew_b, 
																						(double*)GPUDataAddr_dbl_MacroVars, 
																						(int64_t*)GPUDataAddr_int64_Neigh_b, 
																						(uint32_t*)GPUDataAddr_uint32_Wall,
																						(mLatDat->GetLocalFluidSiteCount()), 
																						first_Index, (first_Index + site_Count), mLatDat->totalSharedFs); // 
				
				// Synchronisation point - Must include this at the end of all the GPU kernels launched in PreReceive, as the cuda kernels are launched and then control is returned to the CPU - Must wait to complete
				cudaDeviceSynchronize();				
				if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function. 
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				// Memory transfer from the GPU to the CPU
				// Has to be placed at the end of PreSend (or at the end of each collision Type - Think about this!!! To do!!!) so that all the info is passed from the GPU to the CPU and then exchanged between neighbouring ranks. To do!!!
				// Already defined above: lb::MacroscopicPropertyCache& propertyCache = GetPropertyCache();
				if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU(first_Index, site_Count, propertyCache); // Copy the whole array GPUDataAddr_dbl_fNew_b from the GPU to CPUDataAddr_dbl_fNew_b. Then just read just the elements needed.  
				//
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				// ====================================================================================================================================================


#else	// If computations on CPUs

				// printf("Calling CPU PART \n\n");
				// Collision Type 1:
				site_t offset = mLatDat->GetMidDomainSiteCount();
				StreamAndCollide(mMidFluidCollision, offset, mLatDat->GetDomainEdgeCollisionCount(0));

				// Collision Type 2:
				offset += mLatDat->GetDomainEdgeCollisionCount(0);				
				// printf("Rank: %d: Collision Type 2: Starting = %lld, Ending = %lld \n\n",myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(1)));				
				StreamAndCollide(mWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(1));

#endif
							
				// Collision Type 3:
				offset += mLatDat->GetDomainEdgeCollisionCount(1);
				
				// Receive values for Inlet
				mInletValues->FinishReceive();

				// printf("Rank: %d: Collision Type 3: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(2)));
				StreamAndCollide(mInletCollision, offset, mLatDat->GetDomainEdgeCollisionCount(2));

				offset += mLatDat->GetDomainEdgeCollisionCount(2);


				// Receive values for Outlet
				mOutletValues->FinishReceive();
				
				// Collision Type 4:
				// printf("Rank: %d: Collision Type 4: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(3)));
				StreamAndCollide(mOutletCollision, offset, mLatDat->GetDomainEdgeCollisionCount(3));
				offset += mLatDat->GetDomainEdgeCollisionCount(3);

				// Collision Type 5:
				// printf("Rank: %d: Collision Type 5: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(4)));
				StreamAndCollide(mInletWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(4));
				offset += mLatDat->GetDomainEdgeCollisionCount(4);

				// Collision Type 6:
				// printf("Rank: %d: Collision Type 6: Starting = %lld, Ending = %lld \n\n", myPiD, offset, (offset + mLatDat->GetDomainEdgeCollisionCount(5)));
				StreamAndCollide(mOutletWallCollision, offset, mLatDat->GetDomainEdgeCollisionCount(5));

				
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
				 */

				//IZ
				// Local rank									
				const hemelb::net::Net& rank_Com = *mNet;	// Needs the constructor and be initialised
				int myPiD = rank_Com.Rank();
				//IZ

				// IZ
#ifdef HEMELB_USE_GPU	// If exporting computation on GPUs:	Create Cuda streams in the future - before the LBM time loop - To do!!!

				// ====================================================================================================================================================				
				// Collision Type 1:				
				site_t offset = 0;	// site_t is type int64_t

				// To do!!!
				// For the development phase - one collision type at a time, we need to pass the info for the fOld from the CPU to the GPU so that all information is in the GPU global memory
				int64_t nFluid_nodes = mLatDat->GetLocalFluidSiteCount();
				int64_t first_Index = offset;
				int64_t site_Count = mLatDat->GetMidDomainCollisionCount(0);

				if(myPiD!=0) Read_DistrFunctions_CPU_to_GPU(first_Index, site_Count); // Mem. Copy Host to Device - Synchronous!!!
				
				// Remove the following in the future:
				// StreamAndCollide(mMidFluidCollision, offset, mLatDat->GetMidDomainCollisionCount(0));

				//----------------------------------
				// Cuda kernel set-up
				int nThreadsPerBlock_Collide = 64;				//Number of threads per block for the Collision step
				dim3 nThreads_Collide(nThreadsPerBlock_Collide);								
				// Number of fluid nodes involved in the collision/streaming : mLatDat->GetDomainEdgeCollisionCount(0)
				int nBlocks_Collide = (mLatDat->GetMidDomainCollisionCount(0))/nThreadsPerBlock_Collide			+ ((mLatDat->GetMidDomainCollisionCount(0) % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
				//----------------------------------
				
				// To access the data in GPU global memory: 
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)				
				hemelb::GPU_CollideStream_1_PreReceive <<<nBlocks_Collide, nThreads_Collide>>> (	(double*)GPUDataAddr_dbl_fOld_b, 
																									(double*)GPUDataAddr_dbl_fNew_b, 
																									(double*)GPUDataAddr_dbl_MacroVars, 
																									(int64_t*)GPUDataAddr_int64_Neigh_b, 
																									(mLatDat->GetLocalFluidSiteCount()), 
																									offset, (offset + mLatDat->GetMidDomainCollisionCount(0)), mLatDat->totalSharedFs); // 
				
				// Synchronisation point - Must include this at the end of all the GPU kernels launched in PreReceive, as the cuda kernels are launched and then control is returned to the CPU - Must wait to complete
				cudaDeviceSynchronize();				
				if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function. 
				//--------------------------------------------------------------------------------------------------------------------------------------------------
				
				// For the development phase - Add at the end of all the GPU collision kernels in PreReceive() with a synchronization barrier before it! To do!!!
				// Memory transfer from the GPU to the CPU - so that all info for the fNew is on the CPU rather than the GPU				

				lb::MacroscopicPropertyCache& propertyCache = GetPropertyCache();
				if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU(offset, mLatDat->GetMidDomainCollisionCount(0), propertyCache); // Copy the whole array GPUDataAddr_dbl_fNew_b from the GPU to CPUDataAddr_dbl_fNew_b. Then just read just the elements needed.  
				//
				// ====================================================================================================================================================

				// ====================================================================================================================================================
				// Collision Type 2 (Simple Bounce Back!!!):
				offset += mLatDat->GetMidDomainCollisionCount(0);				
				// StreamAndCollide(mWallCollision, offset, mLatDat->GetMidDomainCollisionCount(1));

				// GPU COLLISION KERNEL: 
				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetMidDomainCollisionCount(1);

				// Read the fOld distr. from host-to-device - Remove later: Necessary at Development only phase!!!
				if(myPiD!=0) Read_DistrFunctions_CPU_to_GPU(first_Index, site_Count); // Mem. Copy Host to Device - Synchronous!!!
				
				//-------------------------------------
				// Kernel set-up
				nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
								
				// To access the data in GPU global memory: 
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)			
				// Wall BCs: Remember that at the moment this is ONLY valid for Simple Bounce Back 
				hemelb::GPU_CollideStream_2 <<<nBlocks_Collide, nThreads_Collide>>> (	(double*)GPUDataAddr_dbl_fOld_b, 
																						(double*)GPUDataAddr_dbl_fNew_b, 
																						(double*)GPUDataAddr_dbl_MacroVars, 
																						(int64_t*)GPUDataAddr_int64_Neigh_b, 
																						(uint32_t*)GPUDataAddr_uint32_Wall,
																						(mLatDat->GetLocalFluidSiteCount()), 
																						first_Index, (first_Index + site_Count), mLatDat->totalSharedFs); // 
				
				// Synchronisation point - Must include this at the end of all the GPU kernels launched in PreReceive, as the cuda kernels are launched and then control is returned to the CPU - Must wait to complete
				cudaDeviceSynchronize();				
				if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function. 
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				// For the development phase - Add at the end of all the GPU collision kernels in PreReceive() with a synchronization barrier before it! To do!!!
				// Memory transfer from the GPU to the CPU - so that all info for the fNew is on the CPU rather than the GPU				
				// Already defined above: lb::MacroscopicPropertyCache& propertyCache = GetPropertyCache();
				if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU(first_Index, site_Count, propertyCache); // Copy the whole array GPUDataAddr_dbl_fNew_b from the GPU to CPUDataAddr_dbl_fNew_b. Then just read just the elements needed.  
				//
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				// ====================================================================================================================================================



				// Collision Type 3:
				offset += mLatDat->GetMidDomainCollisionCount(1);
				StreamAndCollide(mInletCollision, offset, mLatDat->GetMidDomainCollisionCount(2));

				
				// Inlet BCs: NashZerothOrderPressure - Specify the ghost density for each inlet
				// Pass the ghost density[nInlets] to the GPU kernel (cudaMemcpy):
				cudaError_t cudaStatus;
				int n_Inlets = mInletValues->GetLocalIoletCount();
				// printf("Before the Kernel: Number of inlets: %d \n\n", n_Inlets);				

				distribn_t* h_ghostDensity = new distribn_t[n_Inlets];

				for (int i=0; i<n_Inlets; i++){
					h_ghostDensity[i] = mInletValues->GetBoundaryDensity(i);										
					// std::cout << "Cout: GhostDensity : " << h_ghostDensity[i] << std::endl;
				}				
				if (myPiD!=0){ // MemCopy cudaMemcpyHostToDevice only if rank!=0													
					// Memory copy from host (h_ghostDensity) to Device (d_ghostDensity) 				
					cudaStatus = cudaMemcpy(d_ghostDensity, h_ghostDensity, n_Inlets * sizeof(distribn_t), cudaMemcpyHostToDevice);
					if(cudaStatus != cudaSuccess){ fprintf(stderr, "GPU memory transfer (ghostDensity) Host To Device failed\n"); //return false; 
					}					
				}
				if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function. 

				// GPU COLLISION KERNEL: 
				// Fluid ID range: [first_Index, first_Index + site_Count)
				first_Index = offset;	// Start Fluid Index
				site_Count = mLatDat->GetMidDomainCollisionCount(2);

				// Read the fOld distr. from host-to-device - Remove later: Necessary at Development only phase!!!
				if(myPiD!=0) Read_DistrFunctions_CPU_to_GPU(first_Index, site_Count); // Mem. Copy Host to Device - Synchronous!!!

				if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function. 
				//-------------------------------------
				// Kernel set-up
				nBlocks_Collide = site_Count/nThreadsPerBlock_Collide			+ ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);
								
				// To access the data in GPU global memory: 
				// nArr_dbl =  (mLatDat->GetLocalFluidSiteCount()) is the number of fluid elements that sets how these are organised in memory; see Initialise_GPU (method b - by index LB)			
				//Inlet BCs: Remember that at the moment this is ONLY valid for NashZerothOrderPressure
				
				hemelb::GPU_CollideStream_3_NashZerothOrderPressure <<<nBlocks_Collide, nThreads_Collide>>> (	(double*)GPUDataAddr_dbl_fOld_b, 
																												(double*)GPUDataAddr_dbl_fNew_b, 
																												(double*)GPUDataAddr_dbl_MacroVars, 
																												(int64_t*)GPUDataAddr_int64_Neigh_b,
																												(uint32_t*)GPUDataAddr_uint32_Iolet,
																												(distribn_t*)d_ghostDensity,
																												(float*)d_inletNormal,
																												n_Inlets,
																												(mLatDat->GetLocalFluidSiteCount()), 
																												first_Index, (first_Index + site_Count), mLatDat->totalSharedFs); // 
				
				// Synchronisation point - Must include this at the end of all the GPU kernels launched in PreReceive, as the cuda kernels are launched and then control is returned to the CPU - Must wait to complete
				cudaDeviceSynchronize();				
				if (myPiD!=0) hemelb::check_cuda_errors(__FILE__, __LINE__, myPiD); // In the future remove the DEBUG from this function. 
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				// For the development phase - Add at the end of all the GPU collision kernels in PreReceive() with a synchronization barrier before it! To do!!!
				// Memory transfer from the GPU to the CPU - so that all info for the fNew is on the CPU rather than the GPU				
				// Already defined above: lb::MacroscopicPropertyCache& propertyCache = GetPropertyCache();
				if(myPiD!=0) Read_DistrFunctions_GPU_to_CPU(first_Index, site_Count, propertyCache); // Copy the whole array GPUDataAddr_dbl_fNew_b from the GPU to CPUDataAddr_dbl_fNew_b. Then just read just the elements needed.  




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

				//=====================================================================================
#endif
				

				//=====================================================================================
				// The following is only valid for NashZerothOrderPressureDelegate Inlet BCs!!!
				// I need the following to do the Collision Type 3:
				//	1.	ghost density: 
				//			from NashZerothOrderPressureDelegate.h:
				//			distribn_t ghostDensity = iolet.GetBoundaryDensity(boundaryId);
				//		Needs to be passed to the GPU_CollideStream kernel at each time-step:
				//					ONE value for each boundaryId = site.GetIoletId()
				//					Hence, I need the number of local Iolets for Inlet:  mInletValues->GetLocalIoletCount() - One value of density passed (Remember this is valid ONLY for the NashZerothOrderPressure case)

				//	2.	Normal to the iolet:
				//			util::Vector3D<float> ioletNormal = iolet.GetLocalIolet(boundaryId)->GetNormal();
				//		Needs to be passed to the constant GPU memory at initialisation, since it doesn't change for the duration of the simulations
				//		number of elements:  mInletValues->GetLocalIoletCount()
				//		ioletNormal_x, ioletNormal_y, ioletNormal_z = vector<float> or maybe just an array of floats

				//	3.	Velocity at the ghost site, as the component normal to the iolet.
				//			distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);
				//			Note that the division by density compensates for the fact that v_x etc have momentum
				//			not velocity.


				

				/*
				for (site_t site_Index = offset; site_Index < (offset + mLatDat->GetMidDomainCollisionCount(2)); site_Index++)
				{
					
					geometry::Site<geometry::LatticeData> site = mLatDat->GetSite(site_Index);

					int boundaryId = site.GetIoletId();
					// This can be in general a function of time. Hence, probably best to pass at every time-step to the GPU by using a cuda kernel - GPU_Iolets...
					distribn_t ghostDensity = mInletValues->GetBoundaryDensity(boundaryId);

					// Calculate the velocity at the ghost site, as the component normal to the iolet.
					util::Vector3D<float> ioletNormal = mInletValues->GetLocalIolet(boundaryId)->GetNormal();
					
					// Seems to be ok: 
					// Comments: 
					// a. localIoletCount returns the value of 1 : mInletValues->GetLocalIoletCount() = 1
					//		Hence Iolet considers the inlet and the outlet as labelled the same??? 
					// b. 
					
					printf("Printf 1: Site Index: %lld , zCoord = %d - BoundaryId: %d - Value for ghostDensity = %.5e, iolet.z = %.5f, localIoletCount = %lld \n", site.GetIndex(), site.GetGlobalSiteCoords().z, boundaryId, ghostDensity, ioletNormal.z, mInletValues->GetLocalIoletCount());
					std::cout << "Cout 2: ioletNormal.x : " <<  ioletNormal.x << " - ioletNormal.y : " <<  ioletNormal.y << " - ioletNormal.z : " <<  ioletNormal.z << std::endl;
					printf("Printf 3: Site Index: %lld , zCoord = %d - BoundaryId: %d - Value for ghostDensity = %.5e, iolet.z = %.5f \n\n", site.GetIndex(), site.GetGlobalSiteCoords().z, boundaryId, ghostDensity, ioletNormal.z);
					
					//distribn_t ghostDensity = iolet.GetBoundaryDensity(boundaryId);
				}*/

				//=====================================================================================

				
				
				// Collision Type 4:
				offset += mLatDat->GetMidDomainCollisionCount(2);
				StreamAndCollide(mOutletCollision, offset, mLatDat->GetMidDomainCollisionCount(3));

				// Collision Type 5:
				offset += mLatDat->GetMidDomainCollisionCount(3);
				StreamAndCollide(mInletWallCollision, offset, mLatDat->GetMidDomainCollisionCount(4));
				
				// Collision Type 6:
				offset += mLatDat->GetMidDomainCollisionCount(4);
				StreamAndCollide(mOutletWallCollision, offset, mLatDat->GetMidDomainCollisionCount(5));


				//----------------------------------------------------------
				// Delete the variables use for cudaMemcpy 				
				delete[] h_ghostDensity;
				
				//----------------------------------------------------------

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

				timings[hemelb::reporting::Timers::lb_calc].Stop();
				timings[hemelb::reporting::Timers::lb].Stop();
			}

		template<class LatticeType>
			void LBM<LatticeType>::EndIteration()
			{
				timings[hemelb::reporting::Timers::lb].Start();
				timings[hemelb::reporting::Timers::lb_calc].Start();

				// Swap f_old and f_new ready for the next timestep.
				mLatDat->SwapOldAndNew();

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
			}
					
	}
}

#endif /* HEMELB_LB_LB_HPP */
