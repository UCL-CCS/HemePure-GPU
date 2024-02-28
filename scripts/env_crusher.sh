module load cpe/22.12
module load PrgEnv-amd
module load craype-accel-amd-gfx90a   # For GPU Aware MPI
module load perftools-base
module load perftools
module load amd/5.1.0
module load cmake                     # Cmake
module unload cray-libsci             # Nuisance we don't need but PrgEnv loads
export PATH=${ROCM_PATH}/llvm/bin:${PATH}


# C/C++ Compilers through Cray Wrappers (why....?)
export CC=cc
export CXX=CC
export MPI_C_COMPILER=cc
export MPI_CXX_COMPILER=CC
export CUDA_CMAKE_FLAGS=""

# Load python
module load cray-python/3.9.13.1
