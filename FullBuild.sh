#!/bin/bash
## Compilation/build script for HEMELB
## Run from found location

HOST_DOMAIN=`/bin/hostname -d | cut -f1 -d.`

if [ "X${HOST_DOMAIN}" == "Xcrusher" ];
then
	echo "Building On Crusher."
	module load craype-accel-amd-gfx90a   # For GPU Aware MPI
	module load PrgEnv-amd							  # To use hipcc under CC wrapper
	module unload rocm										# If rocm is loaded loading amd will complain
	module load amd												# Load default version of hipcc
	module load rocm											# Load default version of ROCm
	module load cmake											# Cmake
	module unload cray-libsci							# Nuisance we don't need but PrgEnv loads it

  # The Cray/HPE wrappers cause MPI fields to remain unpopulated
  # They don't get passed to the HIP compiler.
  # Also if one doesn't want to use the Cray compiler wrappers one will need these
  #
  export MPI_CFLAGS="${CRAY_XPMEM_INCLUDE_OPTS} -I${CRAY_MPICH_DIR}/include"
  export MPI_LDFLAGS="${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem  -Wl,-rpath=${CRAY_MPICH_DIR}/lib -L${CRAY_MPICH_DIR}/lib -lmpi -Wl,-rpath=${GTL_ROOT} -L${GTL_ROOT} -lmpi_gtl_hsa"

	# C/C++ Compilers through Cray Wrappers (why....?)
	export CC=cc
	export CXX=CC
	export MPI_C_COMPILER=cc
	export MPI_CXX_COMPILER=CC
	export CUDA_CMAKE_FLAGS="" 
else
## MODULE loads
  export CC=$(which gcc)
  export CXX=$(which g++)
  export MPI_C_COMPILER=$(which mpicc)
  export MPI_CXX_COMPILER=$(which mpicxx)
  export CUDA_CMAKE_FLAGS="-DCMAKE_CUDA_FLAGS=\"-ccbin g++ -gencode arch=compute_70,code=sm_70 -lineinfo --ptxas-options=-v --disable-warnings\"" 

  # Any extra flags for MPI (especially for HIP sources which include <mpi.h>)
	export MPI_CFLAGS=""
	export MPI_LDFLAGS=""
fi

# Set as appropriate
export SOURCE_DIR=/Path_To_GPU_code

## HEMELB build
# 1) Dependencies
BuildDep(){
cd $SOURCE_DIR/dep
rm -rf build
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..
make -j  && echo "Done HemeLB Dependencies"
cd ../..
}

# 2) Source code
BuildSource(){
cd $SOURCE_DIR/src
rm -rf build
mkdir build


cd build


cmake -G "Unix Makefiles"  \
  -DCMAKE_CXX_FLAGS="-std=c++11 -g -Wno-narrowing" \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DHEMELB_INLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET"\
  -DHEMELB_WALL_INLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB"\
  -DHEMELB_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET"\
  -DHEMELB_WALL_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB"\
  -DHEMELB_CUDA_AWARE_MPI=ON \
	-DCMAKE_HIP_FLAGS="${MPI_CFLAGS}" \
 	$SOURCE_DIR/src
cmake --build . -j && echo "Done HemeLB Source"
cd ../..
}

BuildDep
BuildSource
echo "Done build all"
