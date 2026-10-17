cmake_minimum_required(VERSION 3.13.1)

# ===============================================================================
# PROJECT DEFINITION
# ===============================================================================
include(cmake/version.cmake)
compute_version()

cmake_policy(SET CMP0048 NEW)
set(CMAKE_BUILD_TYPE "Release")



project(Forvia-OSM-Library VERSION ${APP_VERSION}) # Also sets CPACK version
option(STATIC_BUILD "Whether to build static executables" OFF)
option(BUILD_TESTS "whether to build tests" ON)

set(USE_SYSTEM_BOOST ON)
message(STATUS "Project version is ${PROJECT_VERSION}")




if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR STREQUAL "")
  message(WARNING "CMAKE_SYSTEM_PROCESSOR is not defined. Perhaps CMake toolchain is broken")
endif()

message(STATUS "Detected processor: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(HOST_CPU x86)
  if(NOT DEFINED ENV{SOC} OR ENV{SOC} STREQUAL "" )
    message(FATAL_ERROR "SOC not specicfied, please export SOC variable")
  else()
    set(TARGET_DEVICE $ENV{SOC})
    message(STATUS "TARGET_DEVICE setting to: ${TARGET_DEVICE}")
  endif()
  if(NOT DEFINED TARGET_DEVICE)
    message(FATAL_ERROR "SOC not specicfied, please export SOC variable")
  endif()
  if(NOT DEFINED TARGET_CPU)
    set(TARGET_CPU x86)
    message(STATUS "TARGET_CPU not specicfied using ${TARGET_CPU} ")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
  set(HOST_CPU arm)
  if(NOT DEFINED ENV{SOC} OR ENV{SOC} STREQUAL "" )
    message(FATAL_ERROR "SOC not specicfied, please export SOC variable")
  else()
    set(TARGET_DEVICE $ENV{SOC})
    message(STATUS "TARGET_DEVICE setting to: ${TARGET_DEVICE}")
  endif()  
  if(NOT DEFINED TARGET_DEVICE)
    message(FATAL_ERROR "SOC not specicfied, please export SOC variable")
    endif()
    if(NOT DEFINED TARGET_CPU)
      set(TARGET_CPU arm)
      message(STATUS "TARGET_CPU not specicfied using ${TARGET_CPU} ")
    endif()
endif()
# ===============================================================================
# INCLUDES
# ===============================================================================
include(cmake/lib.cmake)
include(cmake/options.cmake)
include(cmake/test.cmake)
include(cmake/features.cmake)
include(cmake/common_ti.cmake)
# ===============================================================================
# COMPILER OPTIONS
# ===============================================================================
define_compiler_options(STATIC_BUILD)
add_compile_options("-DAPP_VERSION=\"${APP_VERSION}\"")
print_compiler_options()
#===============================================================================
# DOCUMENTATION
#===============================================================================

option(BUILD_DOC "Build documentation" OFF)
if (BUILD_DOC)
  # check if Doxygen is installed
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
      # set input and output files
      set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
      set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

      # request to configure the file
      configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
      message("Doxygen build started")

      # note the option ALL which allows to build the docs together with the application
      add_custom_target( doc_doxygen ALL
          COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          COMMENT "Generating API documentation with Doxygen"
          VERBATIM )
  else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
  endif (DOXYGEN_FOUND)
endif(BUILD_DOC)
# ===============================================================================
# LIBRARIES
# ===============================================================================




if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
    set(AARCH64 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
    set(ARM ON)
endif()

if (TARGET_CPU MATCHES "arm")
  add_subdirectory(foms_common)
  add_subdirectory(foms_age_gender)
  add_subdirectory(foms_positions)
  add_subdirectory(foms_samples)
else()
  find_package(Threads REQUIRED)
  find_package(Boost REQUIRED COMPONENTS date_time unit_test_framework)
  add_subdirectory(foms_common)
  add_subdirectory(foms_age_gender)
  add_subdirectory(foms_positions)
  add_subdirectory(foms_common/test)
  add_subdirectory(foms_age_gender/test)
  add_subdirectory(foms_positions/test)
  add_subdirectory(foms_samples)
endif()
message("USE_SYSTEM_BOOST: ${USE_SYSTEM_BOOST}")
message("BUILD_TESTS: ${BUILD_TESTS}")
