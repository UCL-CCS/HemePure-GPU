# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.3.0/cmake-3.21.3-5ufcufnkyw5xxspnuotfgqsdx7r4babk/bin/cmake

# The command to remove a file.
RM = /autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-9.3.0/cmake-3.21.3-5ufcufnkyw5xxspnuotfgqsdx7r4babk/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build

# Include any dependencies generated for this target.
include configuration/CMakeFiles/hemelb_configuration.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include configuration/CMakeFiles/hemelb_configuration.dir/compiler_depend.make

# Include the progress variables for this target.
include configuration/CMakeFiles/hemelb_configuration.dir/progress.make

# Include the compile flags for this target's objects.
include configuration/CMakeFiles/hemelb_configuration.dir/flags.make

configuration/CMakeFiles/hemelb_configuration.dir/CommandLine.cc.o: configuration/CMakeFiles/hemelb_configuration.dir/flags.make
configuration/CMakeFiles/hemelb_configuration.dir/CommandLine.cc.o: ../configuration/CommandLine.cc
configuration/CMakeFiles/hemelb_configuration.dir/CommandLine.cc.o: configuration/CMakeFiles/hemelb_configuration.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object configuration/CMakeFiles/hemelb_configuration.dir/CommandLine.cc.o"
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration && /sw/summit/hip-cuda/5.1.0/hip/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT configuration/CMakeFiles/hemelb_configuration.dir/CommandLine.cc.o -MF CMakeFiles/hemelb_configuration.dir/CommandLine.cc.o.d -o CMakeFiles/hemelb_configuration.dir/CommandLine.cc.o -c /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/configuration/CommandLine.cc

configuration/CMakeFiles/hemelb_configuration.dir/CommandLine.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hemelb_configuration.dir/CommandLine.cc.i"
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration && /sw/summit/hip-cuda/5.1.0/hip/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/configuration/CommandLine.cc > CMakeFiles/hemelb_configuration.dir/CommandLine.cc.i

configuration/CMakeFiles/hemelb_configuration.dir/CommandLine.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hemelb_configuration.dir/CommandLine.cc.s"
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration && /sw/summit/hip-cuda/5.1.0/hip/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/configuration/CommandLine.cc -o CMakeFiles/hemelb_configuration.dir/CommandLine.cc.s

configuration/CMakeFiles/hemelb_configuration.dir/SimConfig.cc.o: configuration/CMakeFiles/hemelb_configuration.dir/flags.make
configuration/CMakeFiles/hemelb_configuration.dir/SimConfig.cc.o: ../configuration/SimConfig.cc
configuration/CMakeFiles/hemelb_configuration.dir/SimConfig.cc.o: configuration/CMakeFiles/hemelb_configuration.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object configuration/CMakeFiles/hemelb_configuration.dir/SimConfig.cc.o"
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration && /sw/summit/hip-cuda/5.1.0/hip/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT configuration/CMakeFiles/hemelb_configuration.dir/SimConfig.cc.o -MF CMakeFiles/hemelb_configuration.dir/SimConfig.cc.o.d -o CMakeFiles/hemelb_configuration.dir/SimConfig.cc.o -c /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/configuration/SimConfig.cc

configuration/CMakeFiles/hemelb_configuration.dir/SimConfig.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hemelb_configuration.dir/SimConfig.cc.i"
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration && /sw/summit/hip-cuda/5.1.0/hip/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/configuration/SimConfig.cc > CMakeFiles/hemelb_configuration.dir/SimConfig.cc.i

configuration/CMakeFiles/hemelb_configuration.dir/SimConfig.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hemelb_configuration.dir/SimConfig.cc.s"
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration && /sw/summit/hip-cuda/5.1.0/hip/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/configuration/SimConfig.cc -o CMakeFiles/hemelb_configuration.dir/SimConfig.cc.s

# Object files for target hemelb_configuration
hemelb_configuration_OBJECTS = \
"CMakeFiles/hemelb_configuration.dir/CommandLine.cc.o" \
"CMakeFiles/hemelb_configuration.dir/SimConfig.cc.o"

# External object files for target hemelb_configuration
hemelb_configuration_EXTERNAL_OBJECTS =

configuration/libhemelb_configuration.a: configuration/CMakeFiles/hemelb_configuration.dir/CommandLine.cc.o
configuration/libhemelb_configuration.a: configuration/CMakeFiles/hemelb_configuration.dir/SimConfig.cc.o
configuration/libhemelb_configuration.a: configuration/CMakeFiles/hemelb_configuration.dir/build.make
configuration/libhemelb_configuration.a: configuration/CMakeFiles/hemelb_configuration.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libhemelb_configuration.a"
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration && $(CMAKE_COMMAND) -P CMakeFiles/hemelb_configuration.dir/cmake_clean_target.cmake
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hemelb_configuration.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
configuration/CMakeFiles/hemelb_configuration.dir/build: configuration/libhemelb_configuration.a
.PHONY : configuration/CMakeFiles/hemelb_configuration.dir/build

configuration/CMakeFiles/hemelb_configuration.dir/clean:
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration && $(CMAKE_COMMAND) -P CMakeFiles/hemelb_configuration.dir/cmake_clean.cmake
.PHONY : configuration/CMakeFiles/hemelb_configuration.dir/clean

configuration/CMakeFiles/hemelb_configuration.dir/depend:
	cd /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022 /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/configuration /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration /gpfs/alpine/proj-shared/chm155/izacharo/hemeLB_HIP/HemePure_GPU-master_hipified_June2022/src_v27_swapPointers_post_hack2022/build/configuration/CMakeFiles/hemelb_configuration.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : configuration/CMakeFiles/hemelb_configuration.dir/depend
