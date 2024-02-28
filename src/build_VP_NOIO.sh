source ../env_crusher.sh
export HIP_PLATFORM="amd"

cmake -DHEMELB_GPU_BACKEND=HIP_ROCM \
		 -DCMAKE_CXX_COMPILER=/opt/rocm-5.7.0/llvm/bin/amdclang++ \
		 -DCMAKE_C_COMPILER=/opt/rocm-5.7.0/llvm/bin/amdclang \
		 -DCMAKE_HIP_COMPILER=/opt/rocm-5.7.0/llvm/bin/amdclang++ \
		 -DHIP_PLATFORM="amd" \
		 -DHEMELB_COMPUTE_ARCHITECTURE=NEUTRAL \
		 -DCMAKE_CXX_EXTENSIONS=OFF \
		 -DHEMELB_CUDA_AWARE_MPI=ON \
		 -DCMAKE_CXX_FLAGS="-O3 -fopenmp --gcc-toolchain=/opt/cray/pe/gcc/12.2.0/snos -g  ${MPI_CFLAGS}" \
         -DCMAKE_HIP_FLAGS="-O3 -fopenmp --gcc-toolchain=/opt/cray/pe/gcc/12.2.0/snos -g  ${MPI_CFLAGS} --offload-arch=gfx90a" \
         -DCMAKE_SHARED_LINKER_FLAGS="--gcc-toolchain=/opt/cray/pe/gcc/12.2.0/snos -g -O3 -fopenmp ${MPI_LDFLAGS} --offload-arch=gfx90a"\
         -DCMAKE_EXE_LINKER_FLAGS="--gcc-toolchain=/opt/cray/pe/gcc/12.2.0/snos -g -O3 -fopenmp ${MPI_LDFLAGS} --offload-arch=gfx90a" \
		-DHEMELB_LOG_LEVEL="Info" \
		-DHEMELB_USE_MPI_PARALLEL_IO=OFF \
        -DHEMELB_USE_VELOCITY_WEIGHTS_FILE="ON" \
     -DHEMELB_INLET_BOUNDARY="LADDIOLET" \
     -DHEMELB_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET" \
     -DHEMELB_WALL_INLET_BOUNDARY="LADDIOLETSBB" \
     -DHEMELB_WALL_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB" \
     -DHEMELB_WALL_BOUNDARY="SIMPLEBOUNCEBACK" \
		..

cmake --build . -j 16  -v
