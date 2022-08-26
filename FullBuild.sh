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
fi

export SOURCE_DIR=${PWD}

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
cmake  \
  -DCMAKE_CXX_FLAGS="-std=c++11 -g -Wno-narrowing" \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DHEMELB_INLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET"\
  -DHEMELB_WALL_INLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB"\
  -DHEMELB_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET"\
  -DHEMELB_WALL_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB"\
	${CUDA_CMAKE_FLAGS} \
 	$SOURCE_DIR/src
make -j && echo "Done HemeLB Source"
cd ../..
}

BuildDep
BuildSource
echo "Done build all"
