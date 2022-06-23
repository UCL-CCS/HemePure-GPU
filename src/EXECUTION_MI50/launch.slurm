#!/bin/bash

#SBATCH --time=02:30:00
#SBATCH --partition=Intel_8260_2s_24c_2t_GPU_edr_192GB_2933
#SBATCH -N 1 
#SBATCH --exclusive
##SBATCH --gres=gpu:4
#SBATCH -L scratch_lustre_na*1
#cmake
export BIN_PATH=/home_nfs/hablotl/2021_CompBioMed2/HemePure_GPU-master_copy/src/build_MI50_new/
export binary=hemepure_gpu

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home_nfs/hablotl/2021_CompBioMed/HemeLB/HemePure_GPU-master_copy/dep/install/lib/


#module load cuda/11.2
#module load openmpi/4.0.5.2
#module load amd/3.1.0/aocc
#module load openmpi-hpcx/4.1.2a1/aocc3.1_cuda_rhel8

source ~bguibertd/nix/var/nix/profiles/system/nixpack/setup.sh
module load aocc
#module load openmpi
module load openmpi/4.1.1/aocc-3.1.0

ldd $BIN_PATH/$binary

#nvidia-smi
rocm-smi

#export CFLAGS=-I/home_nfs_robin_ib/bkarlshoeferp/work_2021/install/include/tirpc/
export CXXFLAGS=-I/home_nfs/bkarlshoeferp/work/HemeLB/install/include/tirpc/
#export LIBS=-L/home_nfs/bkarlshoeferp/work/HemeLB/install/lib
export LDFLAGS="-L/home_nfs/bkarlshoeferp/work/HemeLB/install/lib -ltirpc"
#export LIBS=-ltirpc
FS=scratch_lustre_na
WORKDIR=/$FS/users/$USER/WORKDIR/HemeLB/

mkdir -p $WORKDIR

#mkdir build && cd build
#cmake -DCMAKE_CXX_FLAGS="-I/home_nfs/bkarlshoeferp/work/HemeLB/install/include/tirpc/" -DCMAKE_CUDA_FLAGS="-I/home_nfs/bkarlshoeferp/work/HemeLB/install/include/tirpc/" ..
#make -j VERBOSE=1

testcase=/home_nfs/hablotl/2021_CompBioMed2/Orig/HemePure_GPU-master_copy/cases/FileSharingIZPK/inputArtery.xml

#cd build
#mpirun -np 5 ./hemepure_gpu -in /home_nfs/bkarlshoeferp/work/HemeLB/HemePure_GPU-master/cases/bifurcation_hires/input.xml -out results
time mpirun -np 5 $BIN_PATH/$binary -in $testcase -out $WORKDIR/DIR_$SLURM_JOB_ID

#mpirun -np 5 nvprof -o simpleMPI.%q{OMPI_COMM_WORLD_RANK}.nvprof ./hemepure_gpu -in /home_nfs/bkarlshoeferp/work/HemeLB/HemePure_GPU-master/cases/FileSharingIZPK/inputArtery_prof.xml -out results
cd -
