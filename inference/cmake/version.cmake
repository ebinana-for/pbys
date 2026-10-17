# ===============================================================================
# Macro used to compute the application version.
# The version is taken either from the APP_VERSION variable, or if present, 
# from a .version file in the root directory. In this case, the version file 
# has precedence.
# ===============================================================================
macro(compute_version)

    ### Compute the application version

    # Only here to support building from the source package
    set("VERSION_FILE_PATH" "${CMAKE_SOURCE_DIR}/.version")
    if(EXISTS "${VERSION_FILE_PATH}")
        file(STRINGS "${VERSION_FILE_PATH}" APP_VERSION)
    endif()

    if(NOT APP_VERSION)
        set(APP_VERSION "2.1.0")
    endif()

    message(STATUS "Application version is ${APP_VERSION}")

endmacro(compute_version)
