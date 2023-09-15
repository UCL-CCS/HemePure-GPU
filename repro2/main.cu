#include "GPU_Collide_Stream_wall_sBB_Iolets.hpp"
#include <vector>
using namespace std;
using namespace hemelb;
using namespace ParamData;

bool testSBBIoletsLaddVelBC()
{
	bool success=true;

	printf("Running testSBBIoletsLaddVelBC\n");

	distribn_t *MacroVars_d;
	int64_t    *Neigh_d;
	uint32_t   *Wall_Link_d;
	uint32_t   *Iolet_Link_d;
	distribn_t *WallMom_d;


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


	status = GPU::deviceMalloc((void **)&Wall_Link_d, nElem_Wall*sizeof(uint32_t));
	if( ! status ) {
	   	fprintf(stderr, "Couldnt alloc device Wall_Link_d array\n");
		abort();
	}
	status = GPU::deviceMemcpy((void *)Wall_Link_d, (const void *)Wall, nElem_Wall*sizeof(int32_t), GPU::memcpyHostToDevice);
	if (! status )  {
		fprintf(stderr, "Couldnt copy Wall_Link array to Device\n"); 
		abort();
	}

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

	status = GPU::deviceMalloc((void **)&WallMom_d, nElem_wallMom_correction*sizeof(distribn_t));
	if( ! status ) {
	   	fprintf(stderr, "Couldnt alloc device wallMom_correction array\n");
		abort();
	}
	status = GPU::deviceMemcpy((void *)WallMom_d, (const void *)wallMom_correction, nElem_wallMom_correction*sizeof(distribn_t), GPU::memcpyHostToDevice);
	if (! status )  {
		fprintf(stderr, "Couldnt copy wallMomCorrection array to Device\n"); 
		abort();
	}


	// Fold and FNew
	distribn_t* fOld_d;
	distribn_t* fNew_d;

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


	// Setup the stream
	GPU::Stream_t Collide_Stream_PreRec_5;
	status = GPU::deviceStreamCreate(&Collide_Stream_PreRec_5);
	if ( !status ) {
		fprintf(stderr, "Couldnt create Collide_Stream_PreRec_5\n");
		abort();	}


	// Storage to save post-call darta
	std::vector<distribn_t> host_fnew(nElem_fNew);
	std::vector<distribn_t> host_mvars(nElem_MacroVars);

	size_t site_Count = upper_limit - lower_limit;
	size_t nThreadsPerBlock_Collide=256;
	size_t nBlocks_Collide = site_Count/nThreadsPerBlock_Collide + ((site_Count % nThreadsPerBlock_Collide > 0)         ? 1 : 0);

	// Launch Kernel into Stream
	printf("Calling Functor\n");

	GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_Functor<lattices::D3Q19> velbc_kernel(
		(distribn_t *)fOld_d,
		(distribn_t*)fNew_d,
		(distribn_t*)MacroVars_d,
		(int64_t *)Neigh_d, (uint32_t *)Wall_Link_d,
		(uint32_t *)Iolet_Link_d, nArr_dbl, (distribn_t *)WallMom_d,
		nArr_wallMom, lower_limit, upper_limit, totalSharedFs, Write_GlobalMem, minusInvTau, Cs2);

	GPU::kernelLaunch(velbc_kernel, nBlocks_Collide, nThreadsPerBlock_Collide, 0, Collide_Stream_PreRec_5);
	GPU::deviceStreamSynchronize(Collide_Stream_PreRec_5);

	printf("Copying Back Results\n");
	GPU::deviceMemcpy((void *)host_fnew.data(), (const void *)fNew_d, nElem_fNew*sizeof(distribn_t), GPU::memcpyDeviceToHost);

	printf("Checking Resulting fNew....");
	size_t diffcount = 0;
	for(int i=0; i < nElem_fNew; ++i) {
		double absdiff = fabs( host_fnew[i] - fNewResult[i] );
	    double rel_err = absdiff;
	    if( host_fnew[i] != 0 ) rel_err  /= fabs( fNewResult[i] );
		if ( rel_err > 1.0e-13  ) {
			diffcount++;	
	    }
	}

	if ( diffcount > 0 ) {
			printf(" FAILED\n");
			success = false;
	}
	else { 
		printf(" OK!\n");
	}

	GPU::deviceMemcpy((void *)host_mvars.data(), (const void *)MacroVars_d, nElem_MacroVars*sizeof(distribn_t), GPU::memcpyDeviceToHost);
	printf("Checking Resulting Macrovars....");
	diffcount = 0;
	for(int i=0; i < nElem_MacroVars; ++i) {
		double absdiff = fabs( host_mvars[i] - MacroVarsResult[i] );
	    double rel_err = absdiff;
	    if( host_mvars[i] != 0 ) rel_err  /= fabs( MacroVarsResult[i] );
	    if ( rel_err > 1.0e-13  ) {
			diffcount++;	
	    }
	}
	if( diffcount > 0 ){
		 printf(" FAILED\n");
		success=false;
	}
	else {
		printf(" OK!\n");
	}

	GPU::deviceStreamDestroy(Collide_Stream_PreRec_5);
	GPU::deviceFree(fNew_d);
	GPU::deviceFree(fOld_d);
	GPU::deviceFree(MacroVars_d);
	GPU::deviceFree(Neigh_d);
	GPU::deviceFree(Wall_Link_d);
	GPU::deviceFree(Iolet_Link_d);
	GPU::deviceFree(WallMom_d);
	return success;

}



int main()
{
  GPU::deviceAttach(0);
  GPU::deviceGetProperties(0);
#if 0
  if(  !setup<hemelb::lattices::D3Q19>() ) {
    fprintf(stderr, "failed to initialize\n");
    abort();
  }
  else {
    printf("Initialization completed. Symbols copied\n");
  }
#endif
  bool succ = testSBBIoletsLaddVelBC();
  if ( succ ) {
	printf("Test Succeeded\n");
  }
  else {
	printf("Test Failed\n");
  }
}

