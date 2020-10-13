# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "CXX"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_CXX
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/Block.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/Block.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/BlockTraverser.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/BlockTraverser.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/BlockTraverserWithVisitedBlockTracker.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/BlockTraverserWithVisitedBlockTracker.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/GeometryReader.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/GeometryReader.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/LatticeData.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/LatticeData.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/SiteData.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/SiteData.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/SiteDataBare.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/SiteDataBare.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/SiteTraverser.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/SiteTraverser.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/VolumeTraverser.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/VolumeTraverser.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/decomposition/BasicDecomposition.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/decomposition/BasicDecomposition.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/decomposition/OptimisedDecomposition.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/decomposition/OptimisedDecomposition.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/needs/Needs.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/needs/Needs.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/neighbouring/NeighbouringDataManager.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/neighbouring/NeighbouringDataManager.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/neighbouring/NeighbouringLatticeData.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/neighbouring/NeighbouringLatticeData.cc.o"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/geometry/neighbouring/RequiredSiteInformation.cc" "/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v1_24a/build/geometry/CMakeFiles/hemelb_geometry.dir/neighbouring/RequiredSiteInformation.cc.o"
  )
set(CMAKE_CXX_COMPILER_ID "GNU")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_CXX
  "DEBUG"
  "HAVE_RUSAGE"
  "HAVE_STD_ISNAN"
  "HEMELB_CODE"
  "HEMELB_COMPUTE_ARCHITECTURE=NEUTRAL"
  "HEMELB_IMAGES_TO_NULL"
  "HEMELB_INLET_BOUNDARY=NASHZEROTHORDERPRESSUREIOLET"
  "HEMELB_KERNEL=LBGK"
  "HEMELB_LATTICE=D3Q19"
  "HEMELB_LOG_LEVEL=Info"
  "HEMELB_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSUREIOLET"
  "HEMELB_READING_GROUP_SIZE=2"
  "HEMELB_READING_GROUP_SPACING=1"
  "HEMELB_TRACER_PARTICLES"
  "HEMELB_USE_GPU"
  "HEMELB_USE_SSE3"
  "HEMELB_WALL_BOUNDARY=SIMPLEBOUNCEBACK"
  "HEMELB_WALL_INLET_BOUNDARY=NASHZEROTHORDERPRESSURESBB"
  "HEMELB_WALL_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSURESBB"
  "LINUX_SCANDIR"
  "TIXML_USE_STL"
  )

# The include file search paths:
set(CMAKE_CXX_TARGET_INCLUDE_PATH
  "/usr/local/cuda"
  "/usr/local/cuda/include"
  "/opt/openmpi-3.0.0/include"
  "/home/ioannis/Documents/UCL_project/code_development/GPU_code/dep/install/include"
  "."
  "../../dep/install/include"
  "../"
  )

# Targets to which this target links.
set(CMAKE_TARGET_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
