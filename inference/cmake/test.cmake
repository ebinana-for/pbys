message("BUILD_TESTS ${BUILD_TESTS}")
enable_testing()


# ===============================================================================
# Macro used to add a new test target: unit tests or integration tests,
# only using targets for dependencies
# ===============================================================================
# name: name of the module
# type: type of the test (unit|integ|functional)
# All other arguments are used as libraries targets
macro(add_test_target name type)
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
        
        add_test(NAME ${_BINARY} COMMAND ${_BINARY} --log_level=message WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_property(TEST ${_BINARY} PROPERTY LABELS "${type}")
        unset(_BINARY)
    endif()
endmacro(add_test_target name type)