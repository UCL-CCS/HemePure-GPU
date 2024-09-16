# Install script for directory: /home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/hemepure_gpu" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/hemepure_gpu")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/hemepure_gpu"
         RPATH "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/../dep/install/lib:/opt/openmpi-3.0.0/lib:/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/dep/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/hemepure_gpu")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/hemepure_gpu" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/hemepure_gpu")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/hemepure_gpu"
         OLD_RPATH "/opt/openmpi-3.0.0/lib:/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/dep/install/lib:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
         NEW_RPATH "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/../dep/install/lib:/opt/openmpi-3.0.0/lib:/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/dep/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/hemepure_gpu")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/hemelb/resources" TYPE FILE FILES "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/resources/report.txt.ctp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/hemelb/resources" TYPE FILE FILES "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/resources/report.xml.ctp")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/configuration/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/extraction/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/reporting/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/lb/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/net/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/util/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/io/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/log/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/colloids/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/geometry/cmake_install.cmake")
  include("/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/resources/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ioannis/Documents/UCL_project/code_development/GPU_code/github_GPU_Hemepure_HIP_CUDA_ROCM_SL/HemePure_GPU/src/build_VP_LBGK/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
