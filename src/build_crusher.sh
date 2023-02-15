cmake -DHEMELB_GPU_BACKEND=HIP_ROCM \
		 -DCMAKE_CXX_COMPILER=CC \
		 -DCMAKE_C_COMPILER=cc \
		 -DHEMELB_COMPUTE_ARCHITECTURE=NEUTRAL \
		 -DCMAKE_CXX_EXTENSIONS=OFF \
		 -DHEMELB_CUDA_AWARE_MPI=ON \
		 -DCMAKE_HIP_FLAGS="-O3 -I/opt/cray/pe/mpich/8.1.17/ofi/amd/5.0/include -I/opt/cray/pe/dsmml/0.2.2/dsmml//include -I/opt/cray/pe/pmi/6.1.3/include -I/opt/cray/xpmem/2.4.4-2.3_11.2__gff0e1d9.shasta/include --offload-arch=gfx90a " \
			..

#  -DAMDGPU_TARGETS="gfx90a"\
#	 -DGPU_TARGETS="gfx90a" \
