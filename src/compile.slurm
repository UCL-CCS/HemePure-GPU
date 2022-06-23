#!/bin/bash
#SBATCH --job-name=heme_lb
#SBATCH --mail-user=paul.karlshoefer@atos.net	
#SBATCH --ntasks=10
#SBATCH --time=00:30:00
#SBATCH --partition=MI100
pwd; hostname; date

source ~bguibertd/nix/var/nix/profiles/system/nixpack/setup.sh
module load hip/4.3.1

module load aocc openmpi

which mpicc
mpicc --version
mpirun --version

export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr

hipconfig --full

echo "Compiling..."

rm -rf build_A100_new
mkdir build_A100_new

GPU_TARGET=MI50
GPU_TARGET=MI100

case $GPU_TARGET in
 "MI50")
   amd_target=gfx906
 ;;
 "MI100")
   amd_target=gfx908
 ;;
esac

if test -f "build_${GPU_TARGET}"
then
  echo "build_${GPU_TARGET} directory already exists. Prefer to not destroy previous work. Exiting."
  exit -1
else
  mkdir -p build_${GPU_TARGET}
  cd build_${GPU_TARGET}
fi

export CC=mpicc
export CXX=mpicxx

unset HIPCC_COMPILE_FLAGS_APPEND

cmake ..

for file in $(grep -r "std=" | awk -F":" '{print $1}' | grep -v "\.log"); do sed -i -e "s/-march=native -ffast-math /-march=native -ffast-math --amdgpu-target=$amd_target/" $file ; done

for file in $(grep -r "std=" | awk -F":" '{print $1}' | grep -v "\.log"); do sed -i -e "s:CXX_INCLUDES = :CXX_INCLUDES = -I$OPENMPI_ROOT/include :" $file ; done

head -n 1 CMakeFiles/hemepure_gpu.dir/link.txt > CMakeFiles/hemepure_gpu.dir/link.txt.new
mv CMakeFiles/hemepure_gpu.dir/link.txt.new CMakeFiles/hemepure_gpu.dir/link.txt

sed -i -e "s/-march=native -ffast-math /-march=native -ffast-math --amdgpu-target=$amd_target/" CMakeFiles/hemepure_gpu.dir/link.txt
sed -i "s:$: -L$OPENMPI_ROOT/lib -l mpi -ltirpc :" CMakeFiles/hemepure_gpu.dir/link.txt 


make VERBOSE=1

cd ..

echo "Done"
