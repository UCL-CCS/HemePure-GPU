#!/bin/bash

#SBATCH --time=02:30:00
#SBATCH --partition=AMD_7413_2s_24c_1t_GPU_edr_256GB_2933
#SBATCH -N 1 
#SBATCH --exclusive
##SBATCH --gres=gpu:4
#SBATCH -L scratch_lustre_na*1
#cmake
export BIN_PATH=/home_nfs/hablotl/2021_CompBioMed2/HemePure_GPU-master_copy/src/build_MI100_new/
export binary=hemepure_gpu

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home_nfs/hablotl/2021_CompBioMed/HemeLB/HemePure_GPU-master_copy/dep/install/lib/

source ~bguibertd/nix/var/nix/profiles/system/nixpack/setup.sh
module load aocc
module load openmpi/4.1.1/aocc-3.1.0

ldd $BIN_PATH/$binary

rocm-smi

FS=scratch_lustre_na
WORKDIR=/$FS/users/$USER/WORKDIR/HemeLB

mkdir -p $WORKDIR

testcase=/home_nfs/hablotl/2021_CompBioMed2/Orig/HemePure_GPU-master_copy/cases/FileSharingIZPK/inputArtery.xml

time mpirun -np 5 $BIN_PATH/$binary -in $testcase -out $WORKDIR/DIR_$SLURM_JOB_ID

cd -
