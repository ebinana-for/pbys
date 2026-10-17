# ===============================================================================
# Functions used to import supported libraries into a project.
# The version of each supported library is static and depends on the version of
# this project.
# ===============================================================================

include(${CMAKE_CURRENT_LIST_DIR}/lib_fetcher.cmake)

set(ICU_VERSION 73.2)
set(BOOST_VERSION 1.83.0)
set(AWSSDK_VERSION 1.11.168)
set(OPENTELEMETRY_VERSION 1.11.0)
set(ORTOOLS_VERSION 9.10)
set(TBB_VERSION 2021.9.0)
set(OSRM_VERSION 5.27.1)
set(OSMIUM_VERSION 2.19.0)

# ===============================================================================
# Macro used to import ICU
# ===============================================================================
macro(find_icu)
    cmake_policy(SET CMP0074 NEW)

    fetch_external_lib_file("icu" ${ICU_VERSION})
    set(ICU_ROOT ${icu_LIB_DIR})
    list(APPEND CMAKE_PREFIX_PATH "${icu_LIB_DIR}") # needed sometimes
    set(var_lib_icu_INITIAL_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")

    if(STATIC_BUILD)
        # force the retrieval of static libraries, as there is no cmake native support for it
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    endif()

    find_package(ICU "${ICU_VERSION}" EXACT REQUIRED COMPONENTS data i18n uc)

    if(STATIC_BUILD)
        # restore the original variable value
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${var_lib_icu_INITIAL_CMAKE_FIND_LIBRARY_SUFFIXES}")
    endif()

    if(ICU_FOUND)
        include_directories(SYSTEM ${ICU_INCLUDE_DIRS})
        target_link_libraries(ICU::uc INTERFACE ICU::data)
        target_link_libraries(ICU::i18n INTERFACE ICU::uc)
    endif()
endmacro(find_icu)

# ===============================================================================
# Macro used to import Boost
# Expects the list of required BOOST components
# Notes:
# - Reference the Boost components via their target: Boost::boost, Boost::locale
# Example:
# find_icu(72.1)
# find_boost(1.81.0 locale)
# ===============================================================================
macro(find_boost)
    cmake_policy(SET CMP0074 NEW)

    # handling Boost dependencies
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)

    # handling Boost
    fetch_external_lib_file("boost" ${BOOST_VERSION})

    set(BOOST_ROOT ${boost_LIB_DIR})
    set(Boost_USE_DEBUG_LIBS OFF)
    set(Boost_USE_RELEASE_LIBS ON)
    set(Boost_DEBUG OFF)
    set(Boost_NO_SYSTEM_PATHS ON)
    set(Boost_NO_WARN_NEW_VERSIONS ON)
    set(Boost_USE_STATIC_RUNTIME OFF)

    if(STATIC_BUILD)
        set(Boost_USE_STATIC_LIBS ON)
    else()
        add_definitions(-DBOOST_ALL_DYN_LINK)
    endif()

    add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)

    # Cmake reports warnings, that should be ignored:
    # "New Boost version may have incorrect or missing dependencies and imported targets"
    # These warnings just mean that the Boost version is more recent than the cmake version,
    # which happens most of the time.
    find_package(Boost ${BOOST_VERSION} EXACT REQUIRED COMPONENTS ${ARGN})

    if(Boost_FOUND)
        include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    endif()

    if(Boost_LOCALE_FOUND)
        target_link_libraries(Boost::locale INTERFACE Boost::thread ICU::i18n ICU::uc)
    endif()

    if(Boost_REGEX_FOUND)
        target_link_libraries(Boost::regex INTERFACE ICU::i18n ICU::uc)
    endif()

    if(Boost_LOG_FOUND)
        target_link_libraries(Boost::log INTERFACE Boost::thread ICU::i18n ICU::uc)
    endif()

    if(Boost_TIMER_FOUND)
        target_link_libraries(Boost::timer INTERFACE Boost::chrono)
    endif()
endmacro(find_boost)

# ===============================================================================
# Macro used to import the AWS C++ SDK
# Expects the list of required AWSSDK components
# Currently supports the following components: s3-crt
# Exports the variable OPTIONAL_AWSSDK_TARGETS that contains all imported targets
# Runtime dependencies for shared AND static builds:
# - yum: libcurl, zlib, openssl
# - apt: libcurl4-openssl, zlib1g, libcrypto++, libssl
# ===============================================================================
macro(find_awssdk)
    cmake_policy(SET CMP0074 NEW)

    fetch_external_lib_file("awssdk" ${AWSSDK_VERSION})

    # static openssl won't work for all OS
    set(OPENSSL_USE_STATIC_LIBS OFF)
    list(APPEND CMAKE_PREFIX_PATH "${awssdk_LIB_DIR}")

    if(STATIC_BUILD)
        list(APPEND CMAKE_PREFIX_PATH "${awssdk_LIB_DIR}/static")
    else()
        add_definitions(-DUSE_IMPORT_EXPORT)
        list(APPEND CMAKE_PREFIX_PATH "${awssdk_LIB_DIR}/shared")
    endif()

    set(AWSSDK_NO_SYSTEM_PATHS ON)
    find_package(AWSSDK REQUIRED COMPONENTS s3-crt)
    target_link_libraries(aws-cpp-sdk-s3-crt INTERFACE aws-cpp-sdk-core)
    set(OPTIONAL_AWSSDK_TARGETS "aws-cpp-sdk-s3-crt")

    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    find_package(ZLIB REQUIRED)
endmacro(find_awssdk)

# ===============================================================================
# Macro used to import OpenSSL. Always in shared mode (static mode must not be used).
# ===============================================================================
macro(find_openssl)
    # static openssl won't work for all OS
    set(OPENSSL_USE_STATIC_LIBS OFF)
    find_package(OpenSSL REQUIRED)
endmacro(find_openssl)

# ===============================================================================
# Macro used to import the TBB dependency
# ===============================================================================
macro(find_tbb)
    fetch_external_lib_file("tbb" ${TBB_VERSION})
    list(APPEND CMAKE_PREFIX_PATH "${tbb_LIB_DIR}/lib/cmake")
    list(APPEND CMAKE_PREFIX_PATH "${tbb_LIB_DIR}/lib64/cmake")
    find_package(TBB REQUIRED)
endmacro(find_tbb)

# ===============================================================================
# Macro used to import the OSRM-backend dependency
# Runtime dependencies for shared AND static builds:
# - yum: lua53 luajit expat libxml2 bzip2 libzip
# - apt: lua5.3 luajit libbz2 libxml2 libzip
# ===============================================================================
macro(find_osrm_backend)
    set(_var_lib_extension ".so")
    set(_var_lib_mode SHARED)
    set(_var_lib_mode_dir shared)

    if(STATIC_BUILD)
        set(_var_lib_extension ".a")
        set(_var_lib_mode STATIC)
        set(_var_lib_mode_dir static)
    endif()

    find_library(LIBRT rt REQUIRED)

    fetch_external_lib_file("osrm-backend" ${OSRM_VERSION})

    add_library(OSRM::OSRM ${_var_lib_mode} IMPORTED)
    target_include_directories(OSRM::OSRM INTERFACE "${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/include/;${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/include/osrm")
    set_target_properties(OSRM::OSRM PROPERTIES
        IMPORTED_LOCATION "${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/lib/libosrm${_var_lib_extension}")
    target_link_libraries(OSRM::OSRM INTERFACE TBB::tbb Boost::iostreams Boost::thread Boost::filesystem Boost::system ${LIBRT})

    add_library(OSRM::extract ${_var_lib_mode} IMPORTED)
    set_target_properties(OSRM::extract PROPERTIES
        IMPORTED_LOCATION "${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/lib/libosrm_extract${_var_lib_extension}")
    target_link_libraries(OSRM::extract INTERFACE OSRM::OSRM TBB::tbb)

    add_library(OSRM::contract ${_var_lib_mode} IMPORTED)
    set_target_properties(OSRM::contract PROPERTIES
        IMPORTED_LOCATION "${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/lib/libosrm_contract${_var_lib_extension}")
    target_link_libraries(OSRM::contract INTERFACE OSRM::OSRM TBB::tbb)

    add_library(OSRM::customize ${_var_lib_mode} IMPORTED)
    set_target_properties(OSRM::customize PROPERTIES
        IMPORTED_LOCATION "${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/lib/libosrm_customize${_var_lib_extension}")
    target_link_libraries(OSRM::customize INTERFACE OSRM::OSRM TBB::tbb)

    add_library(OSRM::guidance ${_var_lib_mode} IMPORTED)
    set_target_properties(OSRM::guidance PROPERTIES
        IMPORTED_LOCATION "${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/lib/libosrm_guidance${_var_lib_extension}")
    target_link_libraries(OSRM::guidance INTERFACE OSRM::OSRM TBB::tbb)

    add_library(OSRM::store ${_var_lib_mode} IMPORTED)
    set_target_properties(OSRM::store PROPERTIES
        IMPORTED_LOCATION "${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/lib/libosrm_store${_var_lib_extension}")
    target_link_libraries(OSRM::store INTERFACE OSRM::OSRM TBB::tbb)

    add_library(OSRM::update ${_var_lib_mode} IMPORTED)
    set_target_properties(OSRM::update PROPERTIES
        IMPORTED_LOCATION "${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/lib/libosrm_update${_var_lib_extension}")
    target_link_libraries(OSRM::update INTERFACE OSRM::OSRM TBB::tbb)

    add_library(OSRM::partition ${_var_lib_mode} IMPORTED)
    set_target_properties(OSRM::partition PROPERTIES
        IMPORTED_LOCATION "${osrm-backend_LIB_DIR}/${_var_lib_mode_dir}/lib/libosrm_partition${_var_lib_extension}")
    target_link_libraries(OSRM::partition INTERFACE OSRM::OSRM TBB::tbb)
endmacro(find_osrm_backend)

# ===============================================================================
# Macro used to import the osmium dependency
# Runtime dependencies for shared AND static builds:
# - yum: TODO
# - apt: TODO
# ===============================================================================
macro(find_osmium)
    fetch_external_lib_file("osmium" ${OSMIUM_VERSION})

    add_library(osmium::osmium INTERFACE IMPORTED)
    target_include_directories(osmium::osmium INTERFACE "${osmium_LIB_DIR}/include")
endmacro(find_osmium)

# ===============================================================================
# Macro used to import the OpenTelemetry client.
# This library can only be referenced as a static library, even on shared builds.
# Runtime dependencies:
# - yum: libcurl, zlib, openssl
# - apt: libcurl4-openssl, zlib1g, libcrypto++, libssl
# ===============================================================================
macro(find_opentelemetry)
    fetch_external_lib_file("opentelemetry" ${OPENTELEMETRY_VERSION})
    list(APPEND CMAKE_PREFIX_PATH "${opentelemetry_LIB_DIR}")
    find_package(opentelemetry-cpp CONFIG REQUIRED)
    find_package(protobuf REQUIRED)
    find_package(ZLIB REQUIRED)
    find_package(CURL REQUIRED)

    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    target_include_directories(nlohmann_json::nlohmann_json INTERFACE "${opentelemetry_LIB_DIR}/include")
    target_link_libraries(opentelemetry-cpp::common INTERFACE nlohmann_json::nlohmann_json)
endmacro(find_opentelemetry)

# ===============================================================================
# Macro used to import the OR tools library.
# Exported target: ortools::ortools
# ===============================================================================
macro(find_ortools)
    fetch_external_lib_file("or-tools" ${ORTOOLS_VERSION})
    list(APPEND CMAKE_PREFIX_PATH "${or-tools_LIB_DIR}")
    find_package(ortools ${ORTOOLS_VERSION}.9999 EXACT REQUIRED)
endmacro(find_ortools)

# ===============================================================================
# Macro used to import the pybind library.
# ===============================================================================
macro(find_pybind)
    set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
    execute_process(
        COMMAND ${Python_EXECUTABLE} --version
        OUTPUT_VARIABLE PYTHON_VERSION_OUTPUT
        ERROR_VARIABLE PYTHON_VERSION_ERROR
        RESULT_VARIABLE PYTHON_VERSION_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(PYTHON_VERSION_RESULT EQUAL 0)
        string(REGEX REPLACE "Python ([0-9]+\\.[0-9]+).*" "\\1" PYTHON_VERSION ${PYTHON_VERSION_OUTPUT})
    else()
        message(FATAL_ERROR "Failed to get Python version: ${PYTHON_VERSION_ERROR}")
    endif()

    set(PYTHON_VERSION "${PYTHON_VERSION}")
    set(CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/pybind11/share/cmake/pybind11")
    set(CMAKE_INSTALL_PREFIX "$ENV{VIRTUAL_ENV}" PARENT_SCOPE)
    message("PYTHON_VERSION : ${PYTHON_VERSION}")
    message("CMAKE_PREFIX_PATH : ${CMAKE_PREFIX_PATH}")
    message("CMAKE_INSTALL_PREFIX : ${CMAKE_INSTALL_PREFIX}")

    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)

    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE PYTHON_VENV_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Python library directory: ${PYTHON_VENV_LIB_DIR}")

    if(PYTHON_VENV_LIB_DIR MATCHES "^/usr/lib")
        message(WARNING "Python virtual environment not activated.")
    endif()
endmacro(find_pybind)
