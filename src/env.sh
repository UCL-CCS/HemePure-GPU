module load cpe/22.12
module load PrgEnv-amd
module load craype-accel-amd-gfx90a   # For GPU Aware MPI
module load perftools-base
module load perftools
module load amd/5.1.0
module load cmake                     # Cmake
module unload cray-libsci             # Nuisance we don't need but PrgEnv loads
export PATH=${ROCM_PATH}/llvm/bin:${PATH}
export LD_LIBRARY_PATH=${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH}
