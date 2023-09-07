module load cpe/23.05
module load PrgEnv-amd
module load craype-accel-amd-gfx90a   # For GPU Aware MPI
module load cray-mpich
module load cray-python
module load amd/5.4.0
module load cmake                     # Cmake
module use ~bjoo/modules
module load dpcpp/2023-08-28
module unload cray-libsci             # Nuisance we don't need but PrgEnv loads

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

export MPICH_ROOT=/opt/cray/pe/mpich/8.1.26
export GTL_ROOT=/opt/cray/pe/mpich/8.1.26/gtl/lib
export MPICH_DIR=${MPICH_ROOT}/ofi/amd/5.0

MPI_CFLAGS="${CRAY_XPMEM_INCLUDE_OPTS} -I${MPICH_DIR}/include "
MPI_LDFLAGS=" ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem  -Wl,-rpath=${MPICH_DIR}/lib -L${MPICH_DIR}/lib -lmpi -Wl,-rpath=${GTL_ROOT} -L${GTL_ROOT} -lmpi_gtl_hsa"

export MPICH_GPU_SUPPORT_ENABLED=1
export PK_BUILD_TYPE="Release"
