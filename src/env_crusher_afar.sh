module load cpe/23.03
module load PrgEnv-amd/8.4.0
module load craype-accel-amd-gfx90a   # For GPU Aware MPI
module load cray-mpich/8.1.26
module load cray-python
module load amd/5.4.3
module load cmake                     # Cmake
module unload cray-libsci             # Nuisance we don't need but PrgEnv loads

export ROCM_PATH=/lustre/orion/proj-shared/stf006/bjoo/rocm-afar-2146
#export ROCM_PATH=/opt/rocm-5.4.3
export PATH=${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:$PATH
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$LD_LIBRARY_PATH
export HIP_PATH=${ROCM_PATH}/hip
export HIP_CLANG_PATH=${ROCM_PATH}/llvm/bin
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

export MPICH_ROOT=/opt/cray/pe/mpich/8.1.26
export GTL_ROOT=/opt/cray/pe/mpich/8.1.26/gtl/lib
export MPICH_DIR=${MPICH_ROOT}/ofi/amd/5.0
export CMAKE_PREFIX_PATH=${ROCM_PATH}/lib/cmake:${CMAKE_PREFIX_PATH}

MPI_CFLAGS="${CRAY_XPMEM_INCLUDE_OPTS} -I${MPICH_DIR}/include "
MPI_LDFLAGS=" ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem  -Wl,-rpath=${MPICH_DIR}/lib -L${MPICH_DIR}/lib -lmpi -Wl,-rpath=${GTL_ROOT} -L${GTL_ROOT} -lmpi_gtl_hsa -L${ROCM_PATH}/llvm/lib -Wl,-rpath=${ROCM_PATH}/llvm/lib"

export MPICH_GPU_SUPPORT_ENABLED=1
export PK_BUILD_TYPE="Release"
export PATH=${ROCM_PATH}/llvm/bin:$PATH

export LD_LIBRARY_PATH=${ROCM_PATH}/llvm/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH}
