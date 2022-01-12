# HemePure-GPU
**GPU version of HemePure**.  <br />
The user can either:
1. Run the Full build script (FullBuild.sh), which first builds the dependencies and then compiles the source code or 
2. Do these steps as listed here: 

Build dependencies before attempting to build `hemepure_gpu`.
## DEPENDENCIES #
1) Create `dep/build/` .
2) In `dep/build/` run `ccmake -B. -H../` or `ccmake ..` .
3) Configure using CMake.
4) Run `make` in `dep/build/`.

## SOURCE #
1) Create `src/build/`.
2) In `src/build/` run `ccmake -B. -H../` or `ccmake ..` .
3) Configure using CMake.
4) Run `make` in `src/build/`.

**IMPORTANT NOTE** <br />
Remember to modify the following (in the script `FullBuild.sh`, or in `src/CMakeLists.txt`) depending on the GPU compute capability (`-gencode arch=compute_70,code=sm_70`) of the NVIDIA GPUs available on the system <br />
`-DCMAKE_CUDA_FLAGS="-ccbin g++ -gencode arch=compute_70,code=sm_70 -lineinfo --ptxas-options=-v --disable-warnings" `

## EXECUTION OF THE PROGRAM #
Running the executable can be done in the same way as the CPU version. <br />
**Launch a HemeLB simulation with the instruction:** <br />
mpirun -n N <hemelb executable address> -in <input file *.xml address> -out <output directory address> <br />
  e.g. mpirun -n 4 ./hemepure_gpu -in ./input.xml -out results <br />

A detailed description of the input file and how to run a simulation is provided in the official HemeLB website <br />
http://hemelb.org/tutorials/simulation/
  

## Cases
The folder `cases/bifurcation_hires/` contains:
  1) the geometry input file (`bifurcation.gmy`)
  2) the input file for the simulation (`input.xml`)
  
## Analysis of results
The analysis of results can be done following the instructions provided in the official HemeLB website <br />
  http://hemelb.org/tutorials/simulation/sim_section3/
