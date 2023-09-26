source ../env_crusher_afar.sh

#
# This works with AFAR 2146, Do not increase the Optimization beyond -O1

OPT="-O3"


# -fgpu-rdc is needed for some reason when working with the AFAR compiler
cmake -DHEMELB_GPU_BACKEND=HIP_ROCM \
		 -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/amdclang++ \
		 -DCMAKE_CXX_FLAGS="${OPT} ${MPI_CFLAGS} --offload-arch=gfx90a" \
		 -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/amdclang  \
		 -DCMAKE_C_FLAGS="${OPT} ${MPI_CFLAGS} --offload-arch=gfx90a" \
		 -DHEMELB_COMPUTE_ARCHITECTURE=NEUTRAL \
		 -DCMAKE_CXX_EXTENSIONS=OFF \
		 -DHEMELB_CUDA_AWARE_MPI=ON \
		 -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/amdclang++ \
		 -DCMAKE_HIP_FLAGS="${OPT} ${MPI_CFLAGS} --offload-arch=gfx90a" \
		 -DCMAKE_EXE_LINKER_FLAGS="${OPT} ${MPI_LDFLAGS} -fgpu-rdc --offload-arch=gfx90a" \
		 -DHIP_CLANG_PATH=${ROCM_PATH}/llvm/bin \
		-DHEMELB_LOG_LEVEL="Info" \
		-DHEMELB_USE_MPI_PARALLEL_IO=ON \
		-DHEMELB_USE_VELOCITY_WEIGHTS_FILE="ON" \
     -DHEMELB_INLET_BOUNDARY="LADDIOLET" \
     -DHEMELB_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET" \
     -DHEMELB_WALL_INLET_BOUNDARY="LADDIOLETSBB" \
     -DHEMELB_WALL_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB" \
	 -DHIP_CLANG_PATH=${HIP_CLANG_PATH} \
		..

cmake --build . -j 16  -v
