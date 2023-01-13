#include <hip/hip_runtime.h>
#include <mpi.h>

__global__ void tmpKernel(float *x, const float *y, int N)
{
   auto tid = threadIdx.x + blockIdx.x*blockDim.x;

	 if( tid < N ) {
	 	x[tid] += y[tid];
	}	
}

void dummyLauncher()
{
	const size_t N=16*1024;
	float x_d[N];
	float y_d[N];
	float *x=nullptr;
	float *y=nullptr;

	if(  hipMalloc(&x, N*sizeof(float)) != hipSuccess ) {
		MPI_Abort(MPI_COMM_WORLD, 20);
   }

	if(  hipMalloc(&y, N*sizeof(float)) != hipSuccess ) {
		MPI_Abort(MPI_COMM_WORLD, 20);
   }

	float val=0.5;
	for(int i=0; i < N; ++i) {
		x_d[i] = val;
		val += 1.0;
		y_d[i] = val;
		val += 0.5;
  }

  if( hipMemcpy(x, x_d, N*sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
		 MPI_Abort(MPI_COMM_WORLD, 21);
  }


  if( hipMemcpy(y, y_d, N*sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
		 MPI_Abort(MPI_COMM_WORLD, 21);
  }

	tmpKernel<<<256, N/256>>>(x,y,N);

	hipFree(x);
	hipFree(y);
}

