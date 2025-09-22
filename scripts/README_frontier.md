# Frontier Compilation Guide

This repository provides instructions to build and compile **Frontier**.

---

## Prerequisites

Before building, ensure you have:

- A working **C++ compiler** (e.g., `CC`, `cc`).
- **CMake** installed.
- Access to the environment setup script:
  - `scripts/env_crusher_20May2024.sh`

---

## Build Dependencies

1. Enter the dependency build directory:
   ```bash
   cd dep/build
   ```

2. Load the build environment:
   ```bash
   source ../../scripts/env_crusher_20May2024.sh
   ```

3. Configure the build with CMake:
   ```bash
   cmake ../ -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc
   ```

4. Build with `make`:
   ```bash
   make -j32
   ```

---

## Build Source Code

1. Navigate to the source directory:
   ```bash
   cd ../../src/
   ```

2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Link the build script:
   ```bash
   ln -s ../build_VP_LBGKSL_Frontier_200525.sh
   ```

4. Run the build script:
   ```bash
   ./build_VP_LBGKSL_Frontier_200525.sb
   ```

5. Compile with `make`:
   ```bash
   make -j32
   ```

---

## Notes

- Adjust `-j32` to match the number of GPU cores available on your system.
- If compilation fails, ensure that:
  - You are using the correct environment script.
  - Required modules and libraries are loaded.

---

## License

This project follows the license provided in the repository.  
Please check the `LICENSE` file for more details.
