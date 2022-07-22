cmake_minimum_required(VERSION 3.10)

project(HemeLB_GPU)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_CXX_EXTENSIONS OFF)

# conditional compile options
# ---------------------------
option(HEMELB_BUILD_TESTS_UNIT "build the unit tests" OFF)
option(HEMELB_DEPENDENCIES_SET_RPATH "set runtime RPATH" ON)
option(HEMELB_IMAGES_TO_NULL "write images to null" ON)
option(HEMELB_STATIC_ASSERT "use simple compile-time assertions" ON)
option(HEMELB_TRACER_PARTICLES "use particles as tracers" ON)
option(HEMELB_USE_ALL_WARNINGS_GNU "show all compiler warnings on development builds (gnu-style-compilers)" OFF)
option(HEMELB_USE_SSE3 "use SSE3 intrinsics" ON)
option(HEMELB_USE_VELOCITY_WEIGHTS_FILE "use velocity weights file" ON)
option(HEMELB_USE_GMYPLUS "use GMY+ format" OFF)
option(HEMELB_USE_MPI_CALL "use standard MPI functions when reading blocks" OFF)
option(HEMELB_USE_MPI_WIN "" OFF)
option(HEMELB_USE_BIGMPI "use BigMPI to read data from file" OFF)
option(HEMELB_USE_PARMETIS "use ParMETIS" OFF)
option(HEMELB_USE_ZOLTAN "use Zoltan" OFF)
#option(HEMELB_VALIDATE_GEOMETRY "validate geometry" OFF)

option(HEMELB_USE_GPU "use GPU accel" ON)
option(HEMELB_CUDA_AWARE_MPI "use CUDA-AWARE MPI" OFF)
set(HEMELB_GPU_BACKEND CUDA CACHE STRING "GPU backend")
set(HEMELB_GPU_BACKENDS CUDA HIP_CUDA HIP_ROCM)
set_property(CACHE HEMELB_GPU_BACKEND PROPERTY STRINGS ${HEMELB_GPU_BACKENDS})

if(HEMELB_GPU_BACKEND STREQUAL "CUDA")
  set(USE_CUDA "ON")
elseif(HEMELB_GPU_BACKEND STREQUAL "HIP_ROCM")
  set(USE_HIP_ROCM "ON")
  set(GPU_COMPILER "HIP-clang" )
elseif(HEMELB_GPU_BACKEND STREQUAL "HIP_CUDA")
  set(USE_HIP_CUDA "ON")
  set(GPU_COMPILER "HIP-nvcc")
endif()
if (NOT CMAKE_CXX_COMPILER MATCHES ".*hipcc$" )
  set(GPU_COMPILER "HOST-default")
endif()

set(HEMELB_EXECUTABLE "hemepure_gpu"
	CACHE STRING "name of executable to produce")
set(HEMELB_READING_GROUP_SIZE 2
	CACHE INTEGER "number of cores to use to read geometry file")
set(HEMELB_READING_GROUP_SPACING 1
	CACHE INTEGER "spacing between cores to use to read geometry file")
set(HEMELB_LOG_LEVEL Info
	CACHE STRING "log level: 'Critical','Error','Warning','Info','Debug','Trace'")
set(HEMELB_DEPENDENCIES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../dep/"
	CACHE FILEPATH "path to find dependency find modules")
set(HEMELB_DEPENDENCIES_INSTALL_PATH "${HEMELB_DEPENDENCIES_PATH}/install"
	CACHE FILEPATH "path to find dependency includes and libraries")
set(HEMELB_OPTIMISATION "-O3"
	CACHE STRING "optimisation level: <blank> or -O1 to -O4)")
set(HEMELB_LATTICE "D3Q19"
	CACHE STRING "lattice type: 'D3Q15','D3Q19','D3Q27','D3Q15i'")
set(HEMELB_KERNEL "LBGK"
	CACHE STRING "kernel: 'LBGK','EntropicAnsumali','EntropicChik','MRT','TRT','NNCY','NNCYMOUSE','NNC','NNTPL'")
set(HEMELB_WALL_BOUNDARY "SIMPLEBOUNCEBACK"
	CACHE STRING "boundary conditions to be used at the walls: 'BFL','GZS','SIMPLEBOUNCEBACK','JUNKYANG'")
set(HEMELB_INLET_BOUNDARY "NASHZEROTHORDERPRESSUREIOLET"
	CACHE STRING "boundary conditions to be used at the inlet: 'NASHZEROTHORDERPRESSUREIOLET','LADDIOLET'")
set(HEMELB_OUTLET_BOUNDARY "NASHZEROTHORDERPRESSUREIOLET"
	CACHE STRING "boundary conditions to be used at the outlets: 'NASHZEROTHORDERPRESSUREIOLET','LADDIOLET'")
set(HEMELB_COMPUTE_ARCHITECTURE "NEUTRAL"
	CACHE STRING "architecture of the machine being used 'INTELSANDYBRIDGE','AMDBULLDOZER','NEUTRAL','ISBFILEVELOCITYINLET'")
set(HEMELB_WALL_INLET_BOUNDARY "NASHZEROTHORDERPRESSURESBB"
	CACHE STRING "boundary conditions to be used at corners between walls and inlets: 'NASHZEROTHORDERPRESSURESBB','NASHZEROTHORDERPRESSUREBFL','LADDIOLETSBB','LADDIOLETBFL'")
set(HEMELB_WALL_OUTLET_BOUNDARY "NASHZEROTHORDERPRESSURESBB"
	CACHE STRING "boundary conditions to be used at corners between walls and outlets: 'NASHZEROTHORDERPRESSURESBB','NASHZEROTHORDERPRESSUREBFL','LADDIOLETSBB','LADDIOLETBFL'")
set(HEMELB_POINTPOINT_IMPLEMENTATION Coalesce
	CACHE STRING "point to point comms implementation: 'Coalesce','Separated','Immediate'" )
set(HEMELB_GATHERS_IMPLEMENTATION Separated
	CACHE STRING "gather comms implementation: 'Separated','ViaPointPoint'" )
set(HEMELB_ALLTOALL_IMPLEMENTATION Separated
	CACHE STRING "alltoall comms implementation: 'Separated','ViaPointPoint'" )
option(HEMELB_SEPARATE_CONCERNS "communicate for each concern separately" OFF)


# stop warnings on ARCHER
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")


# add warnings flags to development build types
if(HEMELB_USE_ALL_WARNINGS_GNU)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
endif()


if(HEMELB_OPTIMISATION)
	set(CMAKE_CXX_FLAGS_RELEASE ${HEMELB_OPTIMISATION})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HEMELB_OPTIMISATION}")
endif()


add_definitions(-DHEMELB_CODE)
add_definitions(-DHEMELB_READING_GROUP_SIZE=${HEMELB_READING_GROUP_SIZE})
add_definitions(-DHEMELB_READING_GROUP_SPACING=${HEMELB_READING_GROUP_SPACING})
add_definitions(-DHEMELB_LATTICE=${HEMELB_LATTICE})
add_definitions(-DHEMELB_KERNEL=${HEMELB_KERNEL})
add_definitions(-DHEMELB_WALL_BOUNDARY=${HEMELB_WALL_BOUNDARY})
add_definitions(-DHEMELB_INLET_BOUNDARY=${HEMELB_INLET_BOUNDARY})
add_definitions(-DHEMELB_OUTLET_BOUNDARY=${HEMELB_OUTLET_BOUNDARY})
add_definitions(-DHEMELB_WALL_INLET_BOUNDARY=${HEMELB_WALL_INLET_BOUNDARY})
add_definitions(-DHEMELB_WALL_OUTLET_BOUNDARY=${HEMELB_WALL_OUTLET_BOUNDARY})
add_definitions(-DHEMELB_COMPUTE_ARCHITECTURE=${HEMELB_COMPUTE_ARCHITECTURE})
add_definitions(-DHEMELB_LOG_LEVEL=${HEMELB_LOG_LEVEL})


if(HEMELB_IMAGES_TO_NULL)
	add_definitions(-DHEMELB_IMAGES_TO_NULL)
endif()

if(NOT HEMELB_STATIC_ASSERT)
	add_definitions(-DHEMELB_NO_STATIC_ASSERT)
endif()

if(HEMELB_TRACER_PARTICLES)
	add_definitions(-DHEMELB_TRACER_PARTICLES)
endif()

if(HEMELB_USE_SSE3)
	add_definitions(-DHEMELB_USE_SSE3)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -msse3")
	set(CMAKE_CXX_FLAGS_RELEASE "${HEMELB_OPTIMISATION} -msse3")
endif()


## ==============================================================================================================================
# Print the CXX_FLAGS
message('CXX Flags = ' ${CMAKE_CXX_FLAGS})


## ==============================================================================================================================

if(HEMELB_USE_GPU)
  add_definitions(-DHEMELB_USE_GPU)
  add_definitions(-DDEBUG)
endif()

if (USE_CUDA)

  # enable_language(CUDA)

  # if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  #   set(CMAKE_CUDA_ARCHITECTURES 75)
  # endif()
  # find_package(CUDAToolkit REQUIRED)

  # set(NVCC_OPTIONS "-gencode=arch=compute_80,code=\"sm_80,compute_80\"")
  # message(STATUS "NVCC_OPTIONS: ${NVCC_OPTIONS}")

  ############################### OLD ###################################

  ##------------------------------------
  ## If CUDA-aware mpi is enabled
  if(HEMELB_CUDA_AWARE_MPI)
	add_definitions(-DHEMELB_CUDA_AWARE_MPI)
  endif()
  ##------------------------------------

  enable_language(CUDA)

  find_package(CUDA REQUIRED)
  if(NOT CUDA_FOUND)
     message(FATAL_ERROR "CUDA not found.")
  else()
     message("CUDA found.")
  endif()

  #ADDED 25-3-2020
  include_directories("${CUDA_TOOLKIT_ROOT_DIR}")
  include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include/")

  # set(CUDA_PROPAGATE_HOST_FLAGS ON)

  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
  # if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
  #   message("gnu COMPILER")
  #   set(CMAKE_CUDA_FLAGS "-ccbin g++ --compiler-options '-msse3' -gencode arch=compute_70,code=sm_70 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -lineinfo --ptx\
  # as-options=-v --disable-warnings")
  #   #set(CMAKE_CUDA_FLAGS "-ccbin g++ --compiler-options '-msse3' -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -lineinfo")
  #   set(CMAKE_CUDA_FLAGS_DEBUG " -ccbin g++ --compiler-options '-msse3' -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -lineinfo --ptxas-options=-v --disable-warnings")
  #   #set(CMAKE_CUDA_FLAGS_DEBUG " -ccbin g++ --compiler-options '-msse3' -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -lineinfo --disable-warnings")
  # elseif( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  #   # using Intel C++
  #   message("intel COMPILER")
  #   #	    set(CMAKE_CUDA_FLAGS "-D__ICC=1600 -ccbin icpc -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -lineinfo")
  #   #	    set(CMAKE_CUDA_FLAGS_DEBUG "-D__ICC=1600 -ccbin icpc -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -lineinfo")

  #   #  set(CMAKE_CUDA_FLAGS "-ccbin g++ --compiler-options '-msse3' -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -lineinfo")
  #   #  set(CMAKE_CUDA_FLAGS_DEBUG " -ccbin g++ --compiler-options '-msse3' -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -lineinfo")
  #   set(CMAKE_CUDA_FLAGS "-ccbin g++ --compiler-options '-msse3' -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -lineinfo --ptxas-options=-v --disable-warnings")
  #   set(CMAKE_CUDA_FLAGS_DEBUG " -ccbin g++ --compiler-options '-msse3' -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -lineinfo --ptxas-options=-v --disable-warnings")

  # endif()
  # message('Cuda Flags = ' ${CMAKE_CUDA_FLAGS})

endif()

if (USE_HIP_ROCM OR USE_HIP_CUDA)

  enable_language(CXX)
  if(NOT DEFINED HIP_PATH)
    if(DEFINED ENV{HIP_PATH})
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
  endif()
  set(CMAKE_MODULE_PATH
    "${HIP_PATH}/cmake"
    "${HIP_PATH}/share/cmake/hip"
    "${HIP_PATH}/hip/cmake"
    ${CMAKE_MODULE_PATH})

  if(GPU_COMPILER STREQUAL "HIP-nvcc")
    set( HIP_PLATFORM "nvidia" )
    set( HIP_COMPILER "nvcc" )
    set( HIP_RUNTIME "cuda" )

    set( CMAKE_C_COMPILE_OPTIONS_PIC -Xcompiler ${CMAKE_C_COMPILE_OPTIONS_PIC} )
    set( CMAKE_CXX_COMPILE_OPTIONS_PIC -Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_PIC} )
    set( CMAKE_SHARED_LIBRARY_C_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_C_FLAGS}" )
    set( CMAKE_SHARED_LIBRARY_CXX_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}" )
    set( CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Xlinker -soname," )
    set( CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "-Xlinker -soname," )

    set( CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Xlinker -rpath," )
    set( CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
    set( CMAKE_EXECUTABLE_RUNTIME_C_FLAG "-Xlinker -rpath," )
    set( CMAKE_EXECUTABLE_RUNTIME_CXX_FLAG "-Xlinker -rpath," )

    set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY "-Xcompiler='${CMAKE_C_COMPILE_OPTIONS_VISIBILITY}'" )
    set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY "-Xcompiler='${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY}'" )
    set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler='${CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}'" )
    set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler='${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}'" )
  endif()

  if(GPU_COMPILER STREQUAL "HIP-clang")
    set( HIP_PLATFORM "hcc" )
    set( HIP_COMPILER "clang" )
    set( HIP_RUNTIME "rocclr" )
    set( AMDGPU_TARGETS gfx803;gfx900;gfx906;gfx908 CACHE STRING "List of specific machine types for library to target" )
  endif()

  if (USE_HIP_ROCM)
    find_package(hip REQUIRED)
    find_package(HIP REQUIRED) # hip_add_library
  endif()

  if (USE_HIP_CUDA)
    find_package(CUDA REQUIRED)
    find_package(HIP REQUIRED)
    #list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
    list( APPEND HIP_INCLUDE_DIRS "${HIP_PATH}/include" )

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    find_package(CUDAToolkit REQUIRED)

    set(NVCC_OPTIONS "-rdc true -gencode=arch=compute_80,code=\"sm_80,compute_80\"")
    message(STATUS "NVCC_OPTIONS: ${NVCC_OPTIONS}")
  endif()

  add_library(hemelb_options INTERFACE)
  if(GPU_COMPILER STREQUAL "HOST-default")
    if (USE_HIP_ROCM)
      target_compile_definitions(hemelb_options INTERFACE __HIP_PLATFORM_HCC__)
    elseif(USE_HIP_CUDA)
      target_compile_definitions(hemelb_options INTERFACE __HIP_PLATFORM_NVCC__)
    endif()
  endif()

  set(CMAKE_HIP_ARCHITECTURES OFF)
  message("HIP found")
  message("   HIP version:      ${HIP_VERSION_STRING}")
  message("   HIP platform:     ${HIP_PLATFORM}")
  message("   HIP runtime:      ${HIP_RUNTIME}")
  message("   HIP compiler:     ${HIP_COMPILER}")
  message("   HIP Include Path: ${HIP_INCLUDE_DIRS}")
  message("   HIP Libraries:    ${HIP_LIBRARIES}")
  message("   GPU compiler:     ${CMAKE_CXX_COMPILER}")
endif()
## ==============================================================================================================================




if(HEMELB_USE_VELOCITY_WEIGHTS_FILE)
	add_definitions(-DHEMELB_USE_VELOCITY_WEIGHTS_FILE)
endif()

if(HEMELB_USE_PARMETIS)
	add_definitions(-DHEMELB_USE_PARMETIS)
endif()

if(HEMELB_USE_GMYPLUS)
	add_definitions(-DHEMELB_USE_GMYPLUS)
endif()

if(HEMELB_USE_MPI_CALL)
	add_definitions(-DHEMELB_USE_MPI_CALL)
endif()

if(HEMELB_USE_MPI_WIN)
	add_definitions(-DHEMELB_USE_MPI_WIN)
endif()

if(HEMELB_USE_BIGMPI)
	add_definitions(-DHEMELB_USE_BIGMPI)
endif()

#if(HEMELB_VALIDATE_GEOMETRY)
#	add_definitions(-DHEMELB_VALIDATE_GEOMETRY)
#endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" "${HEMELB_DEPENDENCIES_PATH}/modules/")
list(APPEND CMAKE_INCLUDE_PATH ${HEMELB_DEPENDENCIES_INSTALL_PATH}/include)
list(APPEND CMAKE_LIBRARY_PATH ${HEMELB_DEPENDENCIES_INSTALL_PATH}/lib)




# set up runtime search path for dlls
#------------------------------------
if(HEMELB_DEPENDENCIES_SET_RPATH)
	set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
	set(CMAKE_INSTALL_RPATH "${HEMELB_DEPENDENCIES_INSTALL_PATH}/lib")
endif()

include(platform_checks)

add_definitions(-DHAVE_STD_ISNAN)	# add_definitions(-DHAVE_ISNAN) is the alternative
add_definitions(-DLINUX_SCANDIR)	# definitely using Linux!!

if(HAVE_STD_ISNAN)
	add_definitions(-DHAVE_STD_ISNAN)
endif()

if(HAVE_RUSAGE)
	add_definitions(-DHAVE_RUSAGE)
endif()

# without the standard names for the xdr functions, create aliases for the existing ones
if(NOT HAVE_XDRUINTXX_T)
	add_definitions(-Dxdr_uint16_t=xdr_u_int16_t)
	add_definitions(-Dxdr_uint32_t=xdr_u_int32_t)
	add_definitions(-Dxdr_uint64_t=xdr_u_int64_t)
endif()

include(mpi)
include(dependencies)

#if(HEMELB_USE_ZOLTAN)
#
#	set(HEMELB_DECOMPOSITION "PARMETIS"
#		CACHE STRING "decomposition method: 'PARMETIS','PTSCOTCH','BLOCK'")
#
#	if(
#		HEMELB_DECOMPOSITION STREQUAL "PARMETIS")
#		add_definitions(-DHEMELB_USE_PARMETIS)
#	elseif(
#		HEMELB_DECOMPOSITION STREQUAL "PTSCOTCH")
#		add_definitions(-DHEMELB_USE_PTSCOTCH)
#	elseif(
#		HEMELB_DECOMPOSITION STREQUAL "BLOCK")
#		add_definitions(-DHEMELB_USE_BLOCK)
#	endif()
#
#	set(DECOMPOSE_LIBRARIES ${ZOLTAN_LIBRARY} ${PTSCOTCH_LIBRARIES} ${PARMETIS_LIBRARIES})
#	add_definitions(-DHEMELB_USE_ZOLTAN)
#
#elseif(NOT HEMELB_USE_ZOLTAN)
set(DECOMPOSE_LIBRARIES ${PARMETIS_LIBRARIES})
#endif()


## ==============================================================================================================================
# resources
# ---------
set(BUILD_RESOURCE_PATH ${PROJECT_BINARY_DIR}/resources)
file(MAKE_DIRECTORY ${BUILD_RESOURCE_PATH})
set(INSTALL_RESOURCE_PATH ${CMAKE_INSTALL_PREFIX}/share/hemelb/resources)

configure_file(
	"${PROJECT_SOURCE_DIR}/resources/path_parameters.h.in"
	"${PROJECT_BINARY_DIR}/resources/path_parameters.h")

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${HEMELB_DEPENDENCIES_INSTALL_PATH}/include")


## ==============================================================================================================================
# HemeLB
# ------

if(USE_CUDA)
  set(cuda_sources SimulationMaster.cu
    cuda_kernels_def_decl/cuda_params.cu
    cuda_kernels_def_decl/GPU_Collide_Stream_Iolets.cu
    cuda_kernels_def_decl/GPU_Collide_Stream_wall_sBB_Iolets.cu)
  add_library(gpu_objs OBJECT ${cuda_sources})
  set_property(TARGET gpu_objs PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  # set_source_files_properties(main.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT 1)
elseif (USE_HIP_ROCM OR USE_HIP_CUDA)
  set(hip_sources SimulationMaster_hip.cc
	cuda_kernels_def_decl/cuda_params_hip.cc
	cuda_kernels_def_decl/GPU_Collide_Stream_Iolets_hip.cc
	cuda_kernels_def_decl/GPU_Collide_Stream_wall_sBB_Iolets_hip.cc)
  set_source_files_properties(${hip_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

  message("HIPCC_OPTIONS ${HIPCC_OPTIONS}")
  message("NVCC_OPTIONS ${NVCC_OPTIONS}")
  message("CLANG_OPTIONS ${CLANG_OPTIONS}")

  hip_add_library(gpu_objs ${hip_sources}
    HIPCC_OPTIONS "${HIPCC_OPTIONS}"
    NVCC_OPTIONS "${NVCC_OPTIONS}"
    CLANG_OPTIONS "${CLANG_OPTIONS}")

  target_link_libraries(gpu_objs PRIVATE $<$<BOOL:${USE_HIP_ROCM}>:hip::device>)
  #specfem_options)

  # hipcc(with nvcc backend) build has problem for share library visibility,
  # need to figure out the reason and enable visibility "hidden" for nvcc eventually.
  if(NOT GPU_COMPILER STREQUAL "HIP-nvcc")
    set_target_properties(gpu_objs PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
  endif()
endif()

target_link_libraries(gpu_objs PRIVATE hemelb_options)

add_executable(${HEMELB_EXECUTABLE} main.cc $<TARGET_OBJECTS:gpu_objs>)

if(USE_HIP_ROCM OR USE_HIP_CUDA)
  include_directories(${HIP_INCLUDE_DIRS})
  if (USE_HIP_ROCM)
    add_definitions(-D__HIP_PLATFORM_HCC__)
  elseif(USE_HIP_CUDA)
    add_definitions(-D__HIP_PLATFORM_NVCC__)
    include_directories("${CUDA_TOOLKIT_ROOT_DIR}")
    include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include/")
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR})
set(package_subdirs
	configuration
	extraction
	reporting
	geometry
	lb
	net
	util
	io
	log
	colloids)

foreach(subdir ${package_subdirs})
	set(lib "hemelb_${subdir}")
	list(APPEND HEMELB_LIBRARIES ${lib})
	add_subdirectory(${subdir})
endforeach()

add_subdirectory(resources)

if (USE_CUDA)
  set_property(TARGET ${HEMELB_EXECUTABLE} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ${HEMELB_EXECUTABLE} PROPERTY CUDA_STANDARD 11)
endif()

if (USE_HIP_ROCM)
  set_target_properties(${HEMELB_EXECUTABLE} PROPERTIES LINKER_LANGUAGE HIP)
endif()

if (USE_HIP_CUDA)
  target_link_libraries(${HEMELB_EXECUTABLE} PUBLIC gpu_objs CUDA::cudart -lstdc++)
endif()

target_link_libraries(${HEMELB_EXECUTABLE} PUBLIC
  ${HEMELB_LIBRARIES}
  ${MPI_LIBRARIES}
  ${DECOMPOSE_LIBRARIES}
  ${TINYXML_LIBRARY}
  ${Boost_LIBRARIES}
  ${CTEMPLATE_LIBRARY}
  ${ZLIB_LIBRARIES}
  ${CUDA_LINK_LIBRARIES_KEYWORD})

# add_executable(${HEMELB_EXECUTABLE} main.cu ${cuda_sources})
# include_directories(${PROJECT_SOURCE_DIR})
# set(package_subdirs configuration extraction reporting geometry lb net util io log colloids)
# foreach(subdir ${package_subdirs})
# 	set(lib "hemelb_${subdir}")
# 	list(APPEND HEMELB_LIBRARIES ${lib})
# 	add_subdirectory(${subdir})
# endforeach()
# add_subdirectory(resources)
# set_property(TARGET ${HEMELB_EXECUTABLE} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# set_property(TARGET ${HEMELB_EXECUTABLE} PROPERTY CUDA_STANDARD 11)
# target_link_libraries(${HEMELB_EXECUTABLE} ${HEMELB_LIBRARIES} ${MPI_LIBRARIES} ${DECOMPOSE_LIBRARIES} ${TINYXML_LIBRARY} ${Boost_LIBRARIES} ${CTEMPLATE_LIBRARY} ${ZLIB_LIBRARIES} ${CUDA_LINK_LIBRARIES_KEYWORD})

install(TARGETS ${HEMELB_EXECUTABLE} RUNTIME DESTINATION bin)
list(APPEND RESOURCES resources/report.txt.ctp resources/report.xml.ctp)


# HemeLB unittests
# ----------------
if(HEMELB_BUILD_TESTS_UNIT)
	find_package(CPPUNIT REQUIRED)
	include_directories(${CPPUNIT_INCLUDE_DIR})
	add_executable(unittests_hemelb ${root_sources})
	add_subdirectory(unittests)
	target_link_libraries(unittests_hemelb
		hemelb_unittests
		${HEMELB_LIBRARIES}
		${MPI_LIBRARIES}
		${DECOMPOSE_LIBRARIES}
		${TINYXML_LIBRARY}
		${CPPUNIT_LIBRARY}
		${Boost_LIBRARIES}
		${CTEMPLATE_LIBRARY}
		${ZLIB_LIBRARIES}
		${CMAKE_DL_LIBS})
	install(TARGETS unittests_hemelb RUNTIME DESTINATION bin)
	list(APPEND RESOURCES
		unittests/resources/four_cube.gmy
		unittests/resources/four_cube.xml
		unittests/resources/config.xml unittests/resources/config0_2_0.xml
		unittests/resources/config_file_inlet.xml unittests/resources/iolet.txt
		unittests/resources/config-velocity-iolet.xml unittests/resources/config_new_velocity_inlets.xml
		unittests/resources/velocity_inlet.txt.weights.txt
		unittests/resources/xmltest.xml unittests/resources/config_file_velocity_inlet.xml unittests/resources/velocity_inlet.txt)
endif()

# copy and install resources
# --------------------------
foreach(resource ${RESOURCES})
	configure_file(${PROJECT_SOURCE_DIR}/${resource} ${BUILD_RESOURCE_PATH} COPYONLY)
	get_filename_component(resource_leaf ${resource} NAME)
	install(FILES ${BUILD_RESOURCE_PATH}/${resource_leaf} DESTINATION share/hemelb/resources)
endforeach()
