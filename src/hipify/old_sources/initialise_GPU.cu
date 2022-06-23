#include "cuda_kernels_def_decl/initialise_GPU.h"


namespace hemelb
{

	//==================================================
	// Initialise the GPU
	//==================================================
	bool InitGPU()
	{
		int dev_count=0;
		
		cudaGetDeviceCount( &dev_count);
		/*if(myPE==0){ 
			std::cout << "===============================================" << std::endl;
			std::printf("Details of the GPUs installed: \n\n");
			std::printf("Device Count: %i\n", dev_count); 
			fflush(stdout);
		}*/

		cudaDeviceProp dev_prop;
		/*
		for (int i = 0; i < 1; i++) {
			cudaGetDeviceProperties( &dev_prop, i);
			check_cuda_errors(__FILE__, __LINE__, myPE);
	
			if(myPE==0){
				// decide if device has sufficient resources and capabilities
		
				cout << "Device properties: " << endl;
				printf("Device name:       %s\n", dev_prop.name);
				printf("Total Global Mem:    %.1fGB\n", ((double)dev_prop.totalGlobalMem/1073741824.0));    
				cout << "Number of Streaming Multiprocessors:  "<< dev_prop.multiProcessorCount<< endl;
				printf("Shared Mem Per SM:   %.0fKB\n", ((double)dev_prop.sharedMemPerBlock/1024));
				//cout << "Clock Rate:  "<< dev_prop.clockRate<< endl;
				cout << "Max Number of Threads per Block:  "<< dev_prop.maxThreadsPerBlock << endl;
				cout << "Max Number of Blocks allowed in x-dir:  "<< dev_prop.maxGridSize[0]<< endl;
				cout << "Max Number of Blocks allowed in y-dir:  "<< dev_prop.maxGridSize[1]<< endl;
				cout << "Warp Size:  "<< dev_prop.warpSize<< endl;
				cout << "===============================================" << "\n\n";
				fflush(stdout);
			}

		}
		*/
	}// closes the bool InitGPU function
  
}

