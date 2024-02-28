module load cpe/23.09
module load PrgEnv-amd
module load craype-accel-amd-gfx90a   # For GPU Aware MPI
module load perftools-base
module load perftools
module load amd/5.7.0
module load gcc-mixed/12.2.0
module load cmake                     # Cmake
module unload cray-libsci             # Nuisance we don't need but PrgEnv loads
module load cray-python/3.9.13.1
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export PATH=${ROCM_PATH}/llvm/bin:${PATH}
export LD_LIBRARY_PATH=${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH}
export HIP_PATH=${ROCM_PATH}
export MPICH_GPU_SUPPORT_ENABLED=1

export MPICH_ROOT=/opt/cray/pe/mpich/8.1.27
export GTL_ROOT=/opt/cray/pe/mpich/8.1.27/gtl/lib
export MPICH_DIR=${MPICH_ROOT}/ofi/amd/5.0

MPI_CFLAGS="${CRAY_XPMEM_INCLUDE_OPTS} -I${MPICH_DIR}/include "
MPI_LDFLAGS=" ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem  -Wl,-rpath=${MPICH_DIR}/lib -L${MPICH_DIR}/lib -lmpi -Wl,-rpath=${GTL_ROOT} -L${GTL_ROOT} -lmpi_gtl_hsa -L${ROCM_PATH}/llvm/lib -Wl,-rpath=${ROCM_PATH}/llvm/lib"
