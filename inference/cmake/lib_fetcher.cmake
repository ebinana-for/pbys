# ===============================================================================
# Functions used to import a library binary package from a package registry
# ===============================================================================

set(DEPENDS_DIR "${CMAKE_SOURCE_DIR}/depends")

# Checks if the package already exists in the external libraries directory (defined in the variable DEPENDS_DIR)
# If not, downloads it from the libraries registry.
# If the library build script is available, but not the archive library, the library is built locally.
# Globally sets the path to the library through a variable named: ${LIB_NAME}_LIB_DIR
function(fetch_external_lib_file _LIB_NAME _LIB_VERSION)
    get_os_release_info(OS_NAME OS_VERSION)
    set(_var_package_extension "tar.gz")
    set(_var_os_id "${OS_NAME}${OS_VERSION}")

    string(REGEX REPLACE "[\.]" "" _var_os_id ${_var_os_id})

    set(_var_lib_charact "${_LIB_NAME}:${_LIB_VERSION}/${_var_os_id}")
    set(_var_lib_dir "${DEPENDS_DIR}/${_var_os_id}/${_LIB_NAME}")

    set(${_LIB_NAME}_LIB_DIR "${_var_lib_dir}" PARENT_SCOPE)

    set(_var_lib_archive_path "${_var_lib_dir}/${_LIB_NAME}_${_LIB_VERSION}.${_var_package_extension}")
    set(_var_lib_script_path "${_var_lib_dir}/${_LIB_NAME}.sh")

    file(GLOB _var_lib_existing_archives ${_var_lib_dir}/*.${_var_package_extension})

    if(NOT EXISTS ${_var_lib_archive_path} AND _var_lib_existing_archives)
        message(STATUS "Library package ${_LIB_NAME} already exists for another version, cleaning...")
        file(REMOVE_RECURSE ${_var_lib_dir})
    endif()

    if(NOT EXISTS ${_var_lib_archive_path} AND EXISTS ${_var_lib_script_path})
        message(STATUS "Library package ${_var_lib_charact} build script exists, building...")
        execute_process(
            COMMAND chmod +x "${_var_lib_script_path}"
            COMMAND bash -c "LIB_VERSION=${_LIB_VERSION} ${_var_lib_script_path}"
            WORKING_DIRECTORY "${_var_lib_dir}"
            RESULT_VARIABLE _LIB_BUILD_RESULT
        )

        if(NOT _LIB_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "Unable to build the library ${_var_lib_charact} package")
        endif()

        # extract the downloaded archive
        execute_process(COMMAND tar -xf ${_var_lib_archive_path} WORKING_DIRECTORY ${_var_lib_dir})
    endif()

    if(NOT EXISTS ${_var_lib_archive_path})
        message(STATUS "External library ${_var_lib_charact} package file not in cache, downloading...")

        set(_var_lib_ci_job_token "$ENV{CI_JOB_TOKEN}")
        set(_var_lib_gitlab_personal_token "$ENV{GITLAB_PERSONAL_TOKEN}")

        if(NOT "${_var_lib_ci_job_token}${_var_lib_gitlab_personal_token}" STREQUAL "")
            message(STATUS "Download from GitLab")

            # Either the job token or the private token can work at the same time
            file(DOWNLOAD
                "https://code.probayes.net/api/v4/projects/1338/packages/generic/${_LIB_NAME}/${_LIB_VERSION}/${_LIB_NAME}_${_var_os_id}.${_var_package_extension}"
                ${_var_lib_archive_path}
                HTTPHEADER "JOB-TOKEN:${_var_lib_ci_job_token}"
                HTTPHEADER "PRIVATE-TOKEN:${_var_lib_gitlab_personal_token}"
                STATUS _var_package_downloaded_status)
        else()
            message(STATUS "Download from Nexus")
            file(DOWNLOAD
                "https://nexus.probayes.net/repository/raw-noauth/devops/cpp-devops/${_LIB_NAME}/${_LIB_VERSION}/${_LIB_NAME}_${_var_os_id}.${_var_package_extension}"
                ${_var_lib_archive_path}
                STATUS _var_package_downloaded_status)
        endif()

        if("${_var_package_downloaded_status}" MATCHES "^0;")
            message(STATUS "External library ${_var_lib_charact} package file downloaded")
        else()
            message(FATAL_ERROR "External library ${_var_lib_charact} package file cannot be downloaded: ${_var_package_downloaded_status}")
            file(REMOVE_RECURSE ${_var_lib_archive_path}) # clean any malformed lib archive
        endif()

        # extract the downloaded archive
        execute_process(COMMAND tar -xf ${_var_lib_archive_path} WORKING_DIRECTORY ${_var_lib_dir})
    else()
        message(STATUS "External library ${_var_lib_charact} package file exists in cache")
    endif()
endfunction()

# ===============================================================================
# VENDOR TOOLS
# ===============================================================================

# Copyright (c) 2018, Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
if(NOT DEFINED _os_release_info)
    set(_os_release_info TRUE)

    # Function get_os_release_info - Determine and return OS name and version
    #
    # Args:
    # 1.  the name of a variable to receive os_name
    # 2.  the name of a variable to receive os_version
    #
    # Return values: (Quotation marks are always stripped).
    # Upon failure, return values are null strings.
    function(get_os_release_info _vn_id _vn_version_id)
        set(_var_id "")
        set(_var_version_id "")

        if("${_var_id}" STREQUAL "")
            set(file_path "/etc/centos-release")

            if(EXISTS "${file_path}")
                # Example: CentOS release 6.9 (Final)
                file(STRINGS "${file_path}" file_list LIMIT_COUNT 1)
                list(GET file_list 0 file_line)

                # Remove all parenthesized items.
                string(REGEX REPLACE "\\([^)]+\\)" "" file_line "${file_line}")

                # Extract start and end, discard optional "version" or "release"
                string(REGEX MATCH "^([A-Za-z0-9_]+)( +(version|release))? +(.*)$" _dummy "${file_line}")

                # 1              2  3                    4
                set(_var_id "${CMAKE_MATCH_1}")
                set(_var_version_id "${CMAKE_MATCH_4}")
            endif()
        endif()

        if("${_var_id}" STREQUAL "")
            if(EXISTS "/usr/lib/os-release")
                set(file_path "/usr/lib/os-release")
            elseif(EXISTS "/etc/os-release")
                set(file_path "/etc/os-release")
            else()
                set(file_path "")
            endif()

            if(NOT "${file_path}" STREQUAL "")
                file(STRINGS "${file_path}" data_list REGEX "^(ID|VERSION_ID)=")

                # Look for lines like "ID="..." and VERSION_ID="..."
                foreach(_var ${data_list})
                    if("${_var}" MATCHES "^(ID)=(.*)$")
                        set(_var_id "${CMAKE_MATCH_2}")
                    elseif("${_var}" MATCHES "^(VERSION_ID)=(.*)$")
                        set(_var_version_id "${CMAKE_MATCH_2}")
                    endif()
                endforeach()
            endif()
        endif()

        if("${_var_id}" STREQUAL "")
            set(file_path "/etc/lsb-release")

            if(EXISTS "${file_path}")
                file(STRINGS "${file_path}" data_list REGEX "^(DISTRIB_ID|DISTRIB_RELEASE)=")

                # Look for lines like "DISTRIB_ID="..." and DISTRIB_RELEASE="..."
                foreach(_var ${data_list})
                    if("${_var}" MATCHES "^(DISTRIB_ID)=(.*)$")
                        set(_var_id "${CMAKE_MATCH_2}")
                    elseif("${_var}" MATCHES "^(DISTRIB_RELEASE)=(.*)$")
                        set(_var_version_id "${CMAKE_MATCH_2}")
                    endif()
                endforeach()
            endif()
        endif()

        string(TOLOWER "${_var_id}" "_var_id")

        string(STRIP "${_var_id}" _var_id)
        string(STRIP "${_var_version_id}" _var_version_id)

        # Remove any enclosing quotation marks
        string(REGEX REPLACE "^\"(.*)\"$" "\\1" _var_id "${_var_id}")
        string(REGEX REPLACE "^\"(.*)\"$" "\\1" _var_version_id "${_var_version_id}")

        if(NOT "${_vn_id}" STREQUAL "")
            set(${_vn_id} "${_var_id}" PARENT_SCOPE)
        endif()

        if(NOT "${_vn_version_id}" STREQUAL "")
            set(${_vn_version_id} "${_var_version_id}" PARENT_SCOPE)
        endif()
    endfunction()
endif(NOT DEFINED _os_release_info)