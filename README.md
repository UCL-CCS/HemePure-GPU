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
