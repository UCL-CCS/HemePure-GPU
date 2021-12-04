#!/bin/bash
## Compilation/build script for HEMELB
## Run from found location

## MODULE loads
export CC=$(which gcc)
export CXX=$(which g++)
export MPI_C_COMPILER=$(which mpicc)
export MPI_CXX_COMPILER=$(which mpicxx)

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
cmake  \
  -DCMAKE_CXX_FLAGS="-std=c++11 -g -Wno-narrowing" \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DHEMELB_INLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET"\
  -DHEMELB_WALL_INLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB"\
  -DHEMELB_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSUREIOLET"\
  -DHEMELB_WALL_OUTLET_BOUNDARY="NASHZEROTHORDERPRESSURESBB"\
  -DCMAKE_CUDA_FLAGS="-ccbin g++ -gencode arch=compute_70, code=sm_70 -lineinfo --ptxas-options=-v --disable-warnings" \
  	$SOURCE_DIR/src
make -j && echo "Done HemeLB Source"
cd ../..
}

BuildDep
BuildSource
echo "Done build all"
