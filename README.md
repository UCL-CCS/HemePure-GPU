# HemePure_GPU
**GPU version of HemePure**  <br />
This repository represents a porting of the CPU-only [HemePure](https://github.com/UCL-CCS/HemePure) version of the HemeLB code to enable execution of GPUs. This version is able to run on both Nvidia and AMD hardware and has demonstrated strong scaling to tens of thousands of GPU cards on both vendors' hardware. It is available under the BSD-3-Clause license. Development of this version from the HemePure code began in 2019, the major developer of this implementation has been Ioannis Zacharoudiou (UCL). 

Pre- and post-processing of simulation domains for HemePure-GPU follow the same steps as utilised by the CPU version.

## Compilation #
As per the CPU version of the code, the dependencies need to be built before attempting to compile the source code and `hemepure_gpu` executable.

### DEPENDENCIES #
1) Create `dep/build/` .
2) In `dep/build/` run `ccmake -B. -H../` or `ccmake ..` .
3) Configure using CMake.
4) Run `make` in `dep/build/`.

### SOURCE #
1) Create `src/build/`.
2) In `src/build/` run `ccmake -B. -H../` or `ccmake ..` .
3) Configure using CMake.
4) Run `make` in `src/build/`.

### Compilation for different GPU backend

You can select different GPU backends (CUDA, HIP-CUDA, HIP-ROCM) with the
`HEMELB_GPU_BACKEND` option.

#### CUDA

```sh
mkdir src/build && cd src/build
cmake -DHEMELB_GPU_BACKEND=CUDA ..
make -j
```

#### HIP-CUDA

```sh
export CUDA_PATH=/path/to/cuda/installation
export HIP_PATH=/path/to/hip/installation
export HIP_PLATFORM=nvidia HIP_COMPILER=nvcc HIP_RUNTIME=cuda
mkdir src/build && cd src/build
cmake -DHEMELB_GPU_BACKEND=HIP_CUDA ..
make -j
```

#### HIP-ROCM

```sh
export HIP_PATH=/path/to/hip/installation
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr
mkdir src/build && cd src/build
cmake -DHEMELB_GPU_BACKEND=HIP_ROCM -DCMAKE_CXX_COMPILER=hipcc ..
make -j
```

You may also want to set `CMAKE_PREFIX_PATH` to directories containing cmake
modules for `AMDDeviceLibs`, `amd_comgr` and `hsa-runtime64` if they are not in
standard HIP location.

When cross-compiling you may want to set `HCC_AMDGPU_TARGET` to specify which
architecture you want to compile for:

```sh
export HCC_AMDGPU_TARGET="gfx906,gfx908,gfx90a"
```
