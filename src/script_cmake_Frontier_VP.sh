#source ../env.sh

cmake -DHEMELB_GPU_BACKEND=HIP_ROCM \
                 -DCMAKE_CXX_COMPILER=CC \
                 -DCMAKE_C_COMPILER=cc \
                 -DHEMELB_COMPUTE_ARCHITECTURE=NEUTRAL \
                 -DCMAKE_CXX_EXTENSIONS=OFF \
                 -DHEMELB_CUDA_AWARE_MPI=ON \
                 -DCMAKE_HIP_FLAGS="-O3 -I/opt/cray/pe/mpich/8.1.23/ofi/amd/5.0/include ${CRAY_XPMEM_INCLUDE_OPTS} --offload-arch=gfx90a" \
                 -DHEMELB_LOG_LEVEL="Info" \
                 -DHEMELB_USE_MPI_PARALLEL_IO=ON \
                 -DHEMELB_USE_VELOCITY_WEIGHTS_FILE=ON \
                 -DHEMELB_INLET_BOUNDARY=LADDIOLET \
                 -DHEMELB_WALL_INLET_BOUNDARY=LADDIOLETSBB \
                 -DHEMELB_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSUREIOLET \
                 -DHEMELB_WALL_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSURESBB \
                ..
