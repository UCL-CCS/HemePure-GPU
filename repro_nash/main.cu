#include "GPU_Collide_Stream_Iolets.hpp"
#include <vector>

using namespace hemelb;
using namespace ParamData;


bool testNash()
{
	bool success=true;

	printf("Running testNashZerothOrderPressure\n");

	// Device pointers that hold read only values
	// these are dumped into the file: nash_data_1.h from the original program
	// So we need to alloc these device pointers and copy data to the device
	distribn_t *MacroVars_d;
	int64_t    *Neigh_d;
	uint32_t   *Iolet_Link_d;
	distribn_t *ghostDensity_out_d;
	distribn_t *outletNormal_d;


	// *** ----- Allocate and copy -------
	// MacroVars	
	printf("Copying Auxiliary fields to device\n");
	bool status = GPU::deviceMalloc((void **)&MacroVars_d, nElem_MacroVars*sizeof(distribn_t));
	if( ! status ) {
	   	fprintf(stderr, "Couldnt alloc device MacroVars array\n");
		abort();
	}
	status = GPU::deviceMemcpy((void *)MacroVars_d, (const void *)MacroVars, nElem_MacroVars*sizeof(distribn_t), GPU::memcpyHostToDevice);
	if (! status )  {
		fprintf(stderr, "Couldnt copy Macro Vars array to Device\n"); 
		abort();
	}

	// Neighbor table
	status = GPU::deviceMalloc((void **)&Neigh_d, nElem_Neigh*sizeof(int64_t));
	if( ! status ) {
	   	fprintf(stderr, "Couldnt alloc device Neigh_d array\n");
		abort();
	}
	status = GPU::deviceMemcpy((void *)Neigh_d, (const void *)Neigh, nElem_Neigh*sizeof(int64_t), GPU::memcpyHostToDevice);
	if (! status )  {
		fprintf(stderr, "Couldnt copy Neigh Vars array to Device\n"); 
		abort();
	}

	// Iolet table
	status = GPU::deviceMalloc((void **)&Iolet_Link_d, nElem_Iolet*sizeof(uint32_t) );
	if( ! status ) {
	   	fprintf(stderr, "Couldnt alloc device Iolet_Link_d array\n");
		abort();
	}
	status = GPU::deviceMemcpy((void *)Iolet_Link_d, (const void *)Iolet, nElem_Iolet*sizeof(int32_t), GPU::memcpyHostToDevice);
	if (! status )  {
		fprintf(stderr, "Couldnt copy Iolet array to Device\n"); 
		abort();
	}

	// Ghost Zone?
	status = GPU::deviceMalloc((void **)&ghostDensity_out_d, nElem_ghostDensity_out*sizeof(distribn_t));
	if( ! status ) {
	   	fprintf(stderr, "Couldnt alloc device wallMom_correction array\n");
		abort();
	}
	status = GPU::deviceMemcpy((void *)ghostDensity_out_d, (const void *)ghostDensity_out, nElem_ghostDensity_out*sizeof(distribn_t), GPU::memcpyHostToDevice);
	if (! status )  {
		fprintf(stderr, "Couldnt copy wallMomCorrection array to Device\n"); 
		abort();
	}

	// Normals to the outlets
	status = GPU::deviceMalloc((void **)&outletNormal_d, nElem_outletNormal*sizeof(float));
	if( ! status ) {
	   	fprintf(stderr, "Couldnt alloc device wallMom_correction array\n");
		abort();
	}
	status = GPU::deviceMemcpy((void *)outletNormal_d, (const void *)outletNormal, nElem_outletNormal*sizeof(float), GPU::memcpyHostToDevice);
	if (! status )  {
		fprintf(stderr, "Couldnt copy wallMomCorrection array to Device\n"); 
		abort();
	}
	
	// These are the main arrays the LBM method works with. Again -- dumped into the .h file
	// Fold and FNew
	distribn_t* fOld_d;
	distribn_t* fNew_d;

	// Alloc and copy
	status = GPU::deviceMalloc((void **)&fOld_d, nElem_fOld*sizeof(distribn_t));
	if( ! status ) {
	   	fprintf(stderr, "Couldnt alloc device fOld array\n");
		abort();
	}
	status = GPU::deviceMemcpy((void *)fOld_d, (const void *)fOld, nElem_fOld*sizeof(distribn_t), GPU::memcpyHostToDevice);
	if (! status )  {
		fprintf(stderr, "Couldnt copy fOld array to Device\n"); 
		abort();
	}

	status = GPU::deviceMalloc((void **)&fNew_d, nElem_fNew*sizeof(distribn_t));
	if( ! status ) {
	   	fprintf(stderr, "Couldnt alloc fNew_d array\n");
		abort();
	}
	status = GPU::deviceMemcpy((void *)fNew_d, (const void *)fNew, nElem_fNew*sizeof(distribn_t), GPU::memcpyHostToDevice);
	if (! status )  {
		fprintf(stderr, "Couldnt copy fNew array to Device\n"); 
		abort();
	}


	// Setup the stream -- to mimic the kernel calls
	GPU::Stream_t Collide_Stream_PreRec_4;
	status = GPU::deviceStreamCreate(&Collide_Stream_PreRec_4);
	if ( !status ) {
		fprintf(stderr, "Couldnt create Collide_Stream_PreRec_5\n");
		abort();	
	}

	// Setup a struct -- this function was dumped in the program. 
	// The struct is a fixed length array and a size.
	// The routine just fills in the array
	setupIoletsOutletInner();

	// We can use these on the host and bring back data after the kernel run
    std::vector<distribn_t> host_fNew(nElem_fNew);
    std::vector<distribn_t> host_MacroVars(nElem_MacroVars);

    // Call the Functor version of the routine
	hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure_Functor<lattices::D3Q19> collide_kern(
						(double *)fOld_d,
						(double*)fNew_d,
						(double*)MacroVars_d,
						(int64_t *)Neigh_d, (uint32_t *)Iolet_Link_d, 	
						(distribn_t *) ghostDensity_out_d, (float *)outletNormal_d,
						n_Outlets, nArr_dbl, lower_limit,upper_limit, totalSharedFs, Write_GlobalMem,
						n_LocalOutlets_mOutlet, Outlet_Inner, minusInvTau, 1, __LINE__);

	// Launch Params: blocksize and nblocks
	size_t site_Count = upper_limit - lower_limit;
    size_t nThreadsPerBlock_Collide = 256;	
    size_t nBlocks_Collide = site_Count/nThreadsPerBlock_Collide  
	         + ((site_Count % nThreadsPerBlock_Collide > 0) ? 1 : 0);

	// Launch the kernel -- see the api.h file for the definition of the kernel launcher	
	GPU::kernelLaunch(collide_kern, nBlocks_Collide, nThreadsPerBlock_Collide, 0, Collide_Stream_PreRec_4);	

	// Sync
    GPU::deviceStreamSynchronize(Collide_Stream_PreRec_4);

	// Retreive fNew and Macro vars and check against expectation
	//
	GPU::deviceMemcpy( (void *)host_fNew.data(),
					   (const void *)fNew_d,
					   nElem_fNew*sizeof(distribn_t), GPU::memcpyDeviceToHost);

	GPU::deviceMemcpy( (void *)host_MacroVars.data(),
				   	   (const void *)MacroVars_d,
						nElem_MacroVars*sizeof(distribn_t), GPU::memcpyDeviceToHost);

	// Check: fNewResult[] and fMacroVars[] result are in nash_data_1.h 
	// 
	{
		printf("Checking Functor fNew: ...");
		// Now diff them 
		size_t differences_fNew=0; 
		for(int i=0; i < host_fNew.size(); ++i) {
			distribn_t diff = fabs( host_fNew[i] - fNewResult[i] ) ;
			if ( diff != 0 ) diff /= fabs( fNewResult[i] ); // Turn it into a relative error
			if ( diff > 1.0e-13 ) {
				differences_fNew++;
			}
		}
		if( differences_fNew > 0 ) {
			success = false;
			printf("FAILED!!!!\n");
		}
		else {
			printf("OK!\n");
		}

		printf("Checking Functor MacroVars: ...");
		size_t differences_Mv=0;	
		for(int i=0; i < host_MacroVars.size(); ++i) {
			distribn_t diff = fabs( host_MacroVars[i] - MacroVarsResult[i] ) ;
			if ( diff != 0 ) diff /= fabs( MacroVarsResult[i] ); // Turn it into a relative error
			if ( diff > 1.0e-13 ) {
				 differences_Mv++;
			}
		}
				
		if (differences_Mv >  0 )  {
		  success=false;
		  printf("FAILED !!!\n");
		}
		else {
		  printf("OK!\n");
		}
	}

	// Now restore the orignal backed up stuff prior to calling the regular kernel version
    GPU::deviceMemcpy( (void *)fNew_d,
					   (const void *)fNew,
						nElem_fNew*sizeof(distribn_t), GPU::memcpyHostToDevice);


	GPU::deviceMemcpy( (void *)MacroVars_d,
					   (const void *)MacroVars,
						nElem_MacroVars*sizeof(distribn_t), GPU::memcpyHostToDevice);
				
	// Call the Good Kernel
    hemelb::GPU_CollideStream_Iolets_NashZerothOrderPressure <<<nBlocks_Collide, nThreadsPerBlock_Collide, 0, Collide_Stream_PreRec_4>>> (	(double*)fOld_d,
						(double *)fNew_d,
						(double *)MacroVars_d,
						(int64_t*)Neigh_d,
						(uint32_t*)Iolet_Link_d,
						(distribn_t*)ghostDensity_out_d,
						(float*)outletNormal_d,
						n_Outlets,
						nArr_dbl,
						lower_limit,upper_limit, totalSharedFs, Write_GlobalMem,
						n_LocalOutlets_mOutlet, Outlet_Inner); //

	// Sync and dump the results to disk.	 
    GPU::deviceStreamSynchronize(Collide_Stream_PreRec_4);
	
	// Save the fNew and the Macrovars back to host (over write previous stuff) and check against expectation
	GPU::deviceMemcpy( (void *)host_fNew.data(),
					   (const void *)fNew_d, 
					    nElem_fNew*sizeof(distribn_t), GPU::memcpyDeviceToHost);

	GPU::deviceMemcpy( (void *)host_MacroVars.data(),
				   	   (const void *)MacroVars_d,
						nElem_MacroVars*sizeof(distribn_t), GPU::memcpyDeviceToHost);
	// Check
	{
		printf("Checking Kernel fNew: ...");
		// Now diff them 
		size_t differences_fNew=0; 
		for(int i=0; i < host_fNew.size(); ++i) {
			distribn_t diff = fabs( host_fNew[i] - fNewResult[i] ) ;
			if ( diff != 0 ) diff /= fabs( fNewResult[i] ); // Turn it into a relative error
			if ( diff > 1.0e-13 ) {
				differences_fNew++;
			}
		}
		if( differences_fNew > 0 ) {
			success = false;
			printf("FAILED!!!!\n");
		}
		else {
			printf("OK!\n");
		}

		printf("Checking Kernel MacroVars: ...");
		size_t differences_Mv=0;	
		for(int i=0; i < host_MacroVars.size(); ++i) {
			distribn_t diff = fabs( host_MacroVars[i] - MacroVarsResult[i] ) ;
			if ( diff != 0 ) diff /= fabs( MacroVarsResult[i] ); // Turn it into a relative error
			if ( diff > 1.0e-13 ) {
				 differences_Mv++;
			}
		}
				
		if (differences_Mv >  0 )  {
		  success=false;
		  printf("FAILED !!!\n");
		}
		else {
		  printf("OK!\n");
		}
	}

	// Cleanup: destroy stream	
	GPU::deviceStreamDestroy(Collide_Stream_PreRec_4);

	// Cleanup: free device arrays
	GPU::deviceFree(fNew_d);
	GPU::deviceFree(fOld_d);
	GPU::deviceFree(MacroVars_d);
	GPU::deviceFree(Neigh_d);
	GPU::deviceFree(Iolet_Link_d);
	GPU::deviceFree(ghostDensity_out_d);
	GPU::deviceFree(outletNormal_d);

	return success;

}


int main()
{
  GPU::deviceAttach(0);
  GPU::deviceGetProperties(0);
  if(  !setup<hemelb::lattices::D3Q19>() ) {
    fprintf(stderr, "failed to initialize\n");
    abort();
  }
  else {
    printf("Initialization completed. Symbols copied\n");
  }

  bool succ = testNash();
  if ( succ ) {
	printf("Test Succeeded\n");
  }
  else {
	printf("Test Failed\n");
  }
}

