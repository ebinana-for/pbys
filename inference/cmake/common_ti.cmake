include(GNUInstallDirs)

# add_compile_options(-std=c++11)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

if (EXISTS /external/edgeai-tidl-tools)
  set(TIDLPATH /external/edgeai-tidl-tools)
else()
  set(TIDLPATH ${CMAKE_SOURCE_DIR}/external/edgeai-tidl-tool)
endif()

message(STATUS "TIDLPATH: ${TIDLPATH} ")



SET(CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib" ".so")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})


if(NOT EXISTS $ENV{TIDL_TOOLS_PATH}/)
  message (FATAL_ERROR  "TIDL_TOOLS_PATH: $ENV{TIDL_TOOLS_PATH} is not found")
endif()


if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.12/) # check for container
  set(TENSORFLOW_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.12/)
  message (STATUS  "setting TENSORFLOW_INSTALL_DIR path:${TENSORFLOW_INSTALL_DIR}") # check for PC
elseif (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.12_x86_u22/)
  set(TENSORFLOW_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.12_x86_u22/)
  message (STATUS  "setting TENSORFLOW_INSTALL_DIR path:${TENSORFLOW_INSTALL_DIR}")
else()
  # avoid warning in case of hw accelerated device which have this in filesystem
  if( NOT ( (NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
    message (WARNING  "TENSORFLOW_INSTALL_DIR is not set")
  endif()
endif()


if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/onnx_1.14.0_x86_u22/)
  set(ONNXRT_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/onnx_1.14.0_x86_u22/)# check for PC
else()
  # avoid warning in case of  hw accelerated device which have this in filesystem
  if( NOT ((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
    message (WARNING  "ONNXRT_INSTALL_DIR is not set")
  endif()
endif()

if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/dlr_1.10.0_x86_u22/)
  set(DLR_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/dlr_1.10.0_x86_u22/)
  message (STATUS  "setting DLR_INSTALL_DIR path:${DLR_INSTALL_DIR}")
else()
  # avoid warning in case of hw accelerated device which have this in filesystem
  if( NOT ((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
    message (WARNING  "DLR_INSTALL_DIR is not set")
  endif()
endif()


if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv_4.2.0_x86_u22/)
  set(OPENCV_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv_4.2.0_x86_u22/)
  message (STATUS  "setting OPENCV_INSTALL_DIR path:${OPENCV_INSTALL_DIR}")
elseif (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv-4.1.0/)
  set(OPENCV_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv-4.1.0/)
  message (STATUS  "setting opencv path:${OPENCV_INSTALL_DIR}")
else ()
  # avoid warning in case of hw accelerated device which have this in filesystem
  if( NOT ((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
    message (WARNING  "OPENCV_INSTALL_DIR is not set")
  endif()
endif()


if(ARMNN_ENABLE)
  if( ${TARGET_CPU} STREQUAL  "x86" OR (NOT ${TARGET_DEVICE} STREQUAL "am62") )
    message(WARNING "ARMNN NOT supported on X86 and hw accelerated device")
    set(ARMNN_ENABLE 0)
    add_compile_options(-DARMNN_ENABLE=0)    
  else()
    if(NOT ARMNN_PATH)
      if (EXISTS ${TIDLPATH}/armnn)
        set(ARMNN_PATH ${TIDLPATH}/armnn)
      else()
        message(WARNING "ARMNN_PATH is not set")
      endif()
    endif()
      add_compile_options(-DARMNN_ENABLE=1)
    endif()
  endif()

if(ENABLE_SDK_9_2_COMPATIBILITY)
  set(TARGET_DEVICE_PYTHON python3.10)
else()
  set(TARGET_DEVICE_PYTHON python3.12)
endif()

set(TFLITE_2_12_LIBS 
  # Enable these when migrating to tflite 2.12
  absl_base
  absl_log_severity
  absl_malloc_internal
  absl_raw_logging_internal
  absl_spinlock_wait
  absl_strerror
  absl_throw_delegate
  absl_hashtablez_sampler
  absl_raw_hash_set
  absl_debugging_internal
  absl_demangle_internal
  absl_stacktrace
  absl_symbolize
  absl_flags
  absl_flags_commandlineflag
  absl_flags_commandlineflag_internal
  absl_flags_config
  absl_flags_internal
  absl_flags_marshalling
  absl_flags_private_handle_accessor
  absl_flags_program_name
  absl_flags_reflection
  absl_city
  absl_hash
  absl_low_level_hash
  absl_int128
  absl_exponential_biased
  absl_status
  absl_cord
  absl_cord_internal
  absl_cordz_functions
  absl_cordz_handle
  absl_cordz_info
  absl_str_format_internal
  absl_strings
  absl_strings_internal
  absl_graphcycles_internal
  absl_synchronization
  absl_civil_time
  absl_time
  absl_time_zone
  absl_bad_optional_access
  absl_bad_variant_access
  flatbuffers
  fft2d_fftsg2d
  fft2d_fftsg
  cpuinfo
  farmhash
  ruy_allocator
  ruy_apply_multiplier
  ruy_blocking_counter
  ruy_block_map
  ruy_context
  ruy_context_get_ctx
  ruy_cpuinfo
  ruy_ctx
  ruy_denormal
  ruy_frontend
  ruy_have_built_path_for_avx2_fma
  ruy_have_built_path_for_avx512
  ruy_have_built_path_for_avx
  ruy_kernel_arm
  ruy_kernel_avx2_fma
  ruy_kernel_avx512
  ruy_kernel_avx
  ruy_pack_arm
  ruy_pack_avx2_fma
  ruy_pack_avx512
  ruy_pack_avx
  ruy_prepacked_cache
  ruy_prepare_packed_matrices
  ruy_system_aligned_alloc
  ruy_thread_pool
  ruy_trmul
  ruy_tune
  ruy_wait
  ruy_profiler_instrumentation
  pthreadpool
  #xnn lib
  XNNPACK
  # Enable these when migrating to tflite 2.12
)

set(TARGET_COMPILE_INCLUDE_DIR 
  
  /usr/local/include
  /usr/local/dlr
  
  /usr/include/opencv4/
  

 
)

set(PC_INCLUDE_DIR 
 

  /usr/local/include
  /usr/local/dlr
  # opencv libraries
  /usr/include/opencv4
  ${DLR_INSTALL_DIR}/neo-ai-dlr/include
  ${DLR_INSTALL_DIR}/neo-ai-dlr/3rdparty/tvm/3rdparty/dlpack/include

  $ENV{TIDL_TOOLS_PATH}
)
 
message("setting PC_INCLUDE_DIR path:${PC_INCLUDE_DIR}")

if(${TARGET_DEVICE} STREQUAL  "am62" AND  (${TARGET_CPU} STREQUAL  "x86" AND ${HOST_CPU} STREQUAL  "x86"))
  message(STATUS "Compiling for x86 with am62 config")
  add_compile_options(-DDEVICE_AM62=1)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  add_compile_options(-DDEVICE_AM62=1)
  add_compile_options(-DXNN_ENABLE=1)

  include_directories(
             
              /usr/local/include
              /usr/local/dlr
             
              
            
              
              # opencv libraries
              ${OPENCV_INSTALL_DIR}/cmake              
              ${OPENCV_INSTALL_DIR}/modules/highgui/include
              ${OPENCV_INSTALL_DIR}/modules/core/include
              ${OPENCV_INSTALL_DIR}/modules/imgproc/include/opencv2
              ${OPENCV_INSTALL_DIR}/modules/imgproc/include/
              ${OPENCV_INSTALL_DIR}/modules/imgcodecs/include

              
                       
              ${DLR_INSTALL_DIR}/include
              ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
              

              
            )

  set(SYSTEM_LINK_LIBS
                      opencv_imgproc
                      opencv_videoio
                      opencv_imgcodecs
                      opencv_highgui
                      opencv_core
                     
                      dlr
                      vx_tidl_rt
                      pthread
                      dl
                      yaml-cpp
                      stdc++fs
                      pthreadpool
                      #xnn lib
                     
    )

  link_directories(
                    /usr/lib 
                    /usr/local/dlr
                    /usr/lib/aarch64-linux-gnu
                    /usr/lib/python3.10/site-packages/dlr/
                    ${TIDLPATH}/.local/dlr/ 
                    # opencv libraries
                    ${OPENCV_INSTALL_DIR}/cmake/lib
                    ${OPENCV_INSTALL_DIR}/cmake/3rdparty/lib
                    ${OPENCV_INSTALL_DIR}/modules/core/include 

                   
                    
                    #for onnx  lib
                    $ENV{TIDL_TOOLS_PATH}
                  )
endif()

if(NOT ${TARGET_DEVICE} STREQUAL  "am62" AND  (${TARGET_CPU} STREQUAL  "x86" AND ${HOST_CPU} STREQUAL  "x86"))
  message(STATUS "Compiling for x86 with ${TARGET_DEVICE} config")
  add_compile_options(-DDEVICE_AM62A=1)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  #enbale xnn since tflite 2.12
  add_compile_options(-DXNN_ENABLE=1)

  include_directories(
    SYSTEM ${PC_INCLUDE_DIR}
  )

  set(SYSTEM_LINK_LIBS
    opencv_imgproc
    opencv_core
    opencv_videoio
    opencv_imgcodecs
    opencv_highgui
  
    dlr
    vx_tidl_rt
    pthread
    dl
    yaml-cpp
    
    )

  link_directories(
    /usr/lib 
    /usr/local/dlr
    /usr/lib/aarch64-linux-gnu
    /usr/lib/python3.10/site-packages/dlr/
    ${TIDLPATH}/.local/dlr/ 
    
    $ENV{TIDL_TOOLS_PATH}/osrt_deps/
    $ENV{TIDL_TOOLS_PATH}/
  )
endif()


if((${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "x86") )
  message(STATUS "cross compiling for AM62")
  add_compile_options(-DDEVICE_AM62=1)
  add_compile_options(-DXNN_ENABLE=1)

  
  if(NOT ARM64_GCC_PATH)
    if (EXISTS $ENV{ARM64_GCC_PATH})
      set(ARM64_GCC_PATH $ENV{ARM64_GCC_PATH})
      message (STATUS  "setting ARM64_GCC_PATH path:${ARM64_GCC_PATH}")
    else()
      message(WARNING "ARM64_GCC_PATH is not set")
    endif()
  endif()


  if(NOT TARGET_FS_PATH)
    if (EXISTS ${TIDLPATH}/targetfs)
      set(TARGET_FS_PATH ${TIDLPATH}/targetfs)
    else()
      message(WARNING "TARGET_FS_PATH is not set")
    endif()
  endif()

  set(ARMCC_PREFIX ${ARM64_GCC_PATH}/x86_64-arago-linux/usr/bin/aarch64-oe-linux/aarch64-oe-linux-)
  set(CMAKE_C_COMPILER ${ARMCC_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${ARMCC_PREFIX}g++)
  set(CMAKE_SYSROOT ${ARM64_GCC_PATH}/aarch64-oe-linux)

  link_directories(
                  #AM62 targetfs
                  ${TARGET_FS_PATH}/usr/lib
                  ${TARGET_FS_PATH}/usr/lib/glib-2.0
                  ${TARGET_FS_PATH}/usr/lib/python3.10/site-packages  

  )
  set(SYSTEM_LINK_LIBS
                  
                 
                  opencv_imgproc
                  opencv_core
                  opencv_highgui
                  opencv_imgcodecs
                  opencv_videoio
             
                  dl
                  dlr
                  yaml-cpp
                  pthread
                  tbb
  )
  include_directories(
                 

                  ${TARGET_FS_PATH}/usr/include
               
                  ${TARGET_FS_PATH}/usr/include/opencv4/
               
                  
                          
                  ${DLR_INSTALL_DIR}/include
                  ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/core/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/highgui/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/imgcodecs/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/videoio/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/imgproc/include
                  #${TARGET_FS_PATH}/usr/include/opencv4/opencv2/cmake
                  
                 
  )
endif()

if((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "x86") )
  message(STATUS "cross compiling for TARGET_DEVICE:${TARGET_DEVICE}")
  add_compile_options(-DXNN_ENABLE=1)
  
  if(NOT ARM64_GCC_PATH)
    if (EXISTS $ENV{ARM64_GCC_PATH})
      set(ARM64_GCC_PATH $ENV{ARM64_GCC_PATH})
      message (STATUS  "setting ARM64_GCC_PATH path:${ARM64_GCC_PATH}")
    else()
      message(WARNING "ARM64_GCC_PATH is not set")
    endif()
  endif()

  #User need to set this
  if(EXISTS $ENV{TARGET_FS_PATH})
    set(TARGET_FS_PATH $ENV{TARGET_FS_PATH})
    message (STATUS  "setting TARGET_FS_PATH path:$ENV{TARGET_FS_PATH}")
  else()
    message(FATAL_ERROR "TARGET_FS_PATH is not set")
  endif()

  set(ARMCC_PREFIX ${ARM64_GCC_PATH}/x86_64-arago-linux/usr/bin/aarch64-oe-linux/aarch64-oe-linux-)
  set(CMAKE_C_COMPILER ${ARMCC_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${ARMCC_PREFIX}g++)
  set(CMAKE_SYSROOT ${ARM64_GCC_PATH}/aarch64-oe-linux)

  link_directories(                
    #Device targetfs
    ${TARGET_FS_PATH}/usr/lib
    ${TARGET_FS_PATH}/usr/lib/glib-2.0
    ${TARGET_FS_PATH}/usr/lib/python3.10/site-packages  
  
  )
  set(SYSTEM_LINK_LIBS
    
    opencv_imgproc
    opencv_imgcodecs
    opencv_core
    opencv_highgui
    opencv_videoio
    opencv_video
 
    tbb
   
    dl
    dlr
    yaml-cpp
   
    
  )
  include_directories(
    

    ${TARGET_FS_PATH}/usr/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/core/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/build
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/highgui/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/imgcodecs/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/videoio/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/imgproc/include/
    

    #armnn
    ${ARMNN_PATH}/delegate/include
    ${ARMNN_PATH}/include

    
  )
endif()


if((${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm") )
  message(STATUS "native compiling for AM62")
  if(NOT FLATBUFFERS_DIR)
    if (EXISTS ${TIDLPATH}/flatbuffers)
      set(FLATBUFFERS_DIR ${TIDLPATH}/flatbuffers)
    else()
      message(WARNING "FLATBUFFERS_DIR is not set")
    endif()
  endif()
  add_compile_options(-DDEVICE_AM62=1)
  add_compile_options(-DXNN_ENABLE=1)

  include_directories(
              ${TARGET_FS_INCLUDE_DIR}
            )

  set(SYSTEM_LINK_LIBS
                opencv_imgproc
                opencv_imgcodecs
                opencv_core
                opencv_videoio
                opencv_video
                opencv_highgui
           
                # dlr
               
                pthread
                dl
                yaml-cpp
                pthreadpool
               
    )



    link_directories(
                    /usr/lib 
                    /usr/local/dlr
                    /usr/lib/aarch64-linux-gnu
                    /usr/lib/${TARGET_DEVICE_PYTHON}/site-packages/dlr/
                    ${TIDLPATH}/.local/dlr/                 
    )
endif()

if( ((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
  message(NOTICE "native compiling for TARGET_DEVICE:${TARGET_DEVICE} ")
  add_compile_options(-DXNN_ENABLE=1)

  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)

  include_directories(
    SYSTEM ${TARGET_FS_INCLUDE_DIR}
  )
  link_directories(  
                  /usr/lib/opencv/ #for container
                  /usr/lib
                 
                  /usr/local/dlr
                  /usr/lib/aarch64-linux-gnu
                  /usr/lib/${TARGET_DEVICE_PYTHON}/site-packages/dlr/
                  /usr/local/lib/${TARGET_DEVICE_PYTHON}/dist-packages/dlr/
                  ${TIDLPATH}/.local/dlr/                  
  )
  set(SYSTEM_LINK_LIBS
                  opencv_imgproc
                  opencv_imgcodecs
                  opencv_core
                  opencv_highgui
                  opencv_videoio
                  dlr
                  
                  vx_tidl_rt
                  pthread
                  dl
                  yaml-cpp
                  
                 
  )
  include_directories(
                  /usr/include
                  /usr/local/include
                  /usr/local/dlr
                  /usr/lib/${TARGET_DEVICE_PYTHON}/site-packages/dlr/include #for am68pa evm
                  
                  /usr/include/opencv4/
                  /usr/include/opencv-4.2.0/# for container
                  /usr/include/opencv-4.2.0/build# for container
                  /usr/include/opencv-4.2.0/modules/core/include/# for container
                  /usr/include/opencv-4.2.0/modules/highgui/include/                  # for container
                  /usr/include/opencv-4.2.0/modules/imgcodecs/include/# for container
                  /usr/include/opencv-4.2.0/modules/videoio/include/# for container
                  /usr/include/opencv-4.2.0/modules/imgproc/include/# for container
                  /usr/include/opencv-4.2.0/cmake/# for container
                 
                  
                  
                  /usr/include/neo-ai-dlr/include
                  /usr/include/neo-ai-dlr/3rdparty/tvm/3rdparty/dlpack/include
               
                  
  )
endif()


if(ARMNN_ENABLE)
  include_directories(
    ${ARMNN_PATH}/delegate/include
    ${ARMNN_PATH}/armnn/include
    ${ARMNN_PATH}/include
  )
  set(SYSTEM_LINK_LIBS
    ${SYSTEM_LINK_LIBS}
    armnn
    armnnDelegate
  )
  link_directories(
    #armnn lib need to remove once added to filesystem
    ${ARMNN_PATH}/build
    ${ARMNN_PATH}/build/delegate
    ${TIDL_TARGET_LIBS}
  )
endif()        

macro(add_library_target_ti LIB_NAME)

  # LIB SOURCE FILES
  file(GLOB_RECURSE "${LIB_NAME}_SRC" CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

  if(STATIC_BUILD)
    add_library(${LIB_NAME} STATIC ${${LIB_NAME}_SRC})
  else()
    add_library(${LIB_NAME} SHARED ${${LIB_NAME}_SRC})
  endif()

  # LIB HEADER FILES
  set("${LIB_NAME}_INCLUDE_DIRS" "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set("${LIB_NAME}_INCLUDE_DIRS" "${${LIB_NAME}_INCLUDE_DIRS}" PARENT_SCOPE)
  target_include_directories(${LIB_NAME} PUBLIC ${${LIB_NAME}_INCLUDE_DIRS})

  # LIB DEPENDENCIES
  foreach(dep_target ${ARGN})

    # include dependency include files, if they exist
    if(NOT "${${dep_target}_INCLUDE_DIRS}" STREQUAL "")
      target_include_directories(${LIB_NAME} PUBLIC ${${dep_target}_INCLUDE_DIRS})
    endif()

    # include dependency target
    target_link_libraries(${LIB_NAME} ${dep_target})

  endforeach(dep_target)

  
  target_link_libraries(${LIB_NAME}
                          -Wl,--unresolved-symbols=ignore-in-shared-libs,--start-group
                          
                          ${SYSTEM_LINK_LIBS}                        
                         
                          -Wl,--end-group)

  install(TARGETS ${LIB_NAME} DESTINATION "lib")

endmacro(add_library_target_ti LIB_NAME)

message("BUILD_TESTS ${BUILD_TESTS}")
enable_testing()
# ===============================================================================
# Macro used to add a new test target: unit tests or integration tests,
# only using targets for dependencies
# ===============================================================================
# name: name of the module
# type: type of the test (unit|integ|functional)
# All other arguments are used as libraries targets
macro(add_test_target_ti name type)
    if(BUILD_TESTS)
        set(_BINARY ${name}_${type}_test)
        file(GLOB_RECURSE "${_BINARY}_SRC" 
            LIST_DIRECTORIES false 
            CONFIGURE_DEPENDS "main_${type}.cpp"  "common/*.cpp" "${type}/*.h" 
                              "${type}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
        
        add_executable(${_BINARY} "${${_BINARY}_SRC}")
        
        add_definitions(-DFAST_LOG_ACTIVATE_DEBUG)

        # activates dynamic linking with list boost test framework
        if(NOT STATIC_BUILD)
            target_compile_definitions(${_BINARY} PRIVATE BOOST_TEST_DYN_LINK=1)
        endif()

        # Add default includes, "src" path is included to enable unit tests on implementation files only functions
        target_include_directories(${_BINARY} PRIVATE include ${${name}_INCLUDE_DIRS} ${${name}_INCLUDE_DIRS}/../src)

        foreach(dep_target ${ARGN})
            if(NOT "${${dep_target}_INCLUDE_DIRS}" STREQUAL "")
                target_include_directories(${_BINARY} PRIVATE ${${dep_target}_INCLUDE_DIRS})
            endif()
        endforeach(dep_target)

        target_link_libraries(${_BINARY} ${name} Boost::unit_test_framework ${ARGN})

    target_link_libraries(${_BINARY}
                          -Wl,--unresolved-symbols=ignore-in-shared-libs,--start-group
                          ${COMMON_LINK_LIBS}
                          ${TARGET_LINK_LIBS}
                          ${SYSTEM_LINK_LIBS}
                          ${lobs}
                          -Wl,--end-group)
        
        add_test(NAME ${_BINARY} COMMAND ${_BINARY} --log_level=message WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_property(TEST ${_BINARY} PROPERTY LABELS "${type}")
        unset(_BINARY)
    endif()
endmacro(add_test_target_ti name type)


# ===============================================================================
# Macro Used to build an application, only using targets for dependencies
# ===============================================================================
macro(add_target_application_ti APP_NAME)

  # LIB SOURCE FILES
  file(GLOB_RECURSE ${APP_NAME}_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${APP_NAME}.cpp)

  include_directories(${${APP_NAME}_INCLUDE_DIRS})
  add_executable(${APP_NAME} ${${APP_NAME}_SRC})

  # APP DEPENDENCIES
  foreach(dep_target ${ARGN})

    # include dependency include files, if they exist
    if(NOT "${${dep_target}_INCLUDE_DIRS}" STREQUAL "")
      target_include_directories(${APP_NAME} PUBLIC ${${dep_target}_INCLUDE_DIRS})
    endif()
  endforeach(dep_target)

  target_link_libraries(${APP_NAME} ${ARGN})

  if(${APP_NAME}_DEST)
    install(TARGETS ${${APP_NAME}} DESTINATION "bin")
  endif(${APP_NAME}_DEST)

  target_link_libraries(${APP_NAME}
                          -Wl,--unresolved-symbols=ignore-in-shared-libs,--start-group
                          ${COMMON_LINK_LIBS}
                          ${TARGET_LINK_LIBS}
                          ${SYSTEM_LINK_LIBS}
                          ${lobs}
                          -Wl,--end-group)

endmacro(add_target_application_ti APP_NAME)



# Function for building a node:
# ARG0: app name
# ARG1: source list
function(build_app)
    set(app ${ARGV0})
    set(src ${ARGV1})
    add_executable(${app} ${${src}})
    
    link_directories(${app} 
        PUBLIC ${CMAKE_SOURCE_DIR}/../lib/${CMAKE_BUILD_TYPE}        
        )
    target_link_libraries(${app}
                          -Wl,--unresolved-symbols=ignore-in-shared-libs,--start-group
                          ${COMMON_LINK_LIBS}
                          ${TARGET_LINK_LIBS}
                          ${SYSTEM_LINK_LIBS}
                          ${lobs}
                          -Wl,--end-group)
endfunction()
