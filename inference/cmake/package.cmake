# ===============================================================================
# Macro Used to add a template file
# ===============================================================================
macro(add_file_package files dest)
    foreach(src ${files})
        get_filename_component(filename ${src} NAME)
        configure_file(${src} "${CMAKE_CURRENT_BINARY_DIR}/temp/${filename}" @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/temp/${filename} DESTINATION "${dest}")
    endforeach(src)
endmacro(add_file_package file dest)

# ===============================================================================
# Macro Used to add a directory in package
# ===============================================================================
macro(add_directory_package directory dest)
    install(DIRECTORY ${directory} DESTINATION "${dest}" PATTERN "CVS" EXCLUDE)
endmacro(add_directory_package directory dest)

# ===============================================================================
# Macro Used to initialize some variables for template files
# ===============================================================================
macro(add_configuration_build name)
    set(${name}_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES ".*64.*")
        set(${name}_NUM_BITS "64")
    else(${CMAKE_SYSTEM_PROCESSOR} MATCHES ".*64.*")
        set(${name}_NUM_BITS "32")
    endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES ".*64.*")

    set(${name}_GENERIC_OS "${CMAKE_SYSTEM}")

    # COMPILATEUR
    if(WIN32)
        if(WIN32)
            set(${name}_COMPILER "Microsoft (R) 32-bit C/C++ Optimizing Compiler Version 14.00.50727.762 for 80x86")
        endif(WIN32)

        if(MSVC90)
            set(${name}_MSVC "9,00")
        endif(MSVC90)

        if(MSVC80)
            set(${name}_MSVC "8,00")
        endif(MSVC80)
    else(WIN32)
        execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --version COMMAND head -n 1 OUTPUT_VARIABLE ${name}_COMPILER)
    endif(WIN32)

    # DATE
    if(WIN32)
        execute_process(COMMAND "C:/cygwin/bin/date" --rfc-822 OUTPUT_VARIABLE ${name}_DATE)
        execute_process(COMMAND "C:/cygwin/bin/date" "+%Y-%m" OUTPUT_VARIABLE ${name}_DATE_LATEX)
    else(WIN32)
        execute_process(COMMAND date "+%Y/%m/%d-%H:%M" OUTPUT_VARIABLE ${name}_DATE)
        execute_process(COMMAND date "+%Y-%m" OUTPUT_VARIABLE ${name}_DATE_LATEX)
    endif(WIN32)

    string(REPLACE "\n" "" ${name}_DATE ${${name}_DATE})
endmacro(add_configuration_build name)
