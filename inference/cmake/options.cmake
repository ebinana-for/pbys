# ===============================================================================
# Macro used to detect if the current compiler is compatible with GCC or Clang
# ===============================================================================
macro(detect_gnu)
    set(CXX_COMPILER_IS_GNU_COMPATIBLE OFF)
    set(CXX_COMPILER_IS_GCC OFF)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CXX_COMPILER_IS_GNU_COMPATIBLE ON)
        set(CXX_COMPILER_IS_GCC ON)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CXX_COMPILER_IS_GNU_COMPATIBLE ON)
    endif()

    message("CXX_COMPILER_IS_GNU_COMPATIBLE ${CXX_COMPILER_IS_GNU_COMPATIBLE}")
    message("CXX_COMPILER_IS_GCC ${CXX_COMPILER_IS_GCC}")
endmacro(detect_gnu)

# ===============================================================================
# Macro used to add standard compiler options: warnings, debug and release
# options, static build.
# Options are:
# - ACTIVATE_THREAD_SANITIZER: ON to activate the thread sanitizer
# - ACTIVATE_SANITIZERS: ON to activate the sanitizers but the thread sanitizer
# ===============================================================================
macro(define_compiler_options STATIC_BUILD)
    detect_gnu()
    set(CMAKE_CXX_STANDARD 17)

    option(ACTIVATE_TSAN "Activate all the sanitizers" OFF)
    option(ACTIVATE_SANITIZERS "Activate all the sanitizers but the thread sanitizer" OFF)

    if(STATIC_BUILD)
        add_definitions(-DSTATIC_BUILD)
    endif()

    if(CXX_COMPILER_IS_GNU_COMPATIBLE)

        if(ACTIVATE_THREAD_SANITIZER)
            message("ACTIVATE_SANITIZERS ON")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer -O1 -fno-optimize-sibling-calls")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
        elseif(ACTIVATE_SANITIZERS)
            message("ACTIVATE_SANITIZERS ON")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -O1 -fno-optimize-sibling-calls")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined")
        else()
            # standard compilation options
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -finline-functions")
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fdiagnostics-color=always")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wformat=2 ")
    
            if(CXX_COMPILER_IS_GCC)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wctor-dtor-privacy -Wnon-virtual-dtor -Wcast-qual -Wcast-align -fno-strict-aliasing -Wold-style-cast -Wno-unknown-pragmas -Wshadow=local -Wduplicated-cond -Wduplicated-branches -Wuninitialized -Wnull-dereference -ftemplate-backtrace-limit=10 -Wzero-as-null-pointer-constant -Wmisleading-indentation -Wuseless-cast")
                # -Wsuggest-final-methods -Wsuggest-attribute=noreturn -Wsuggest-override -Wsign-promo -Wfloat-equal -Wswitch-enum
            endif()
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC ")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")

        if(STATIC_BUILD)
            # Beware! The flag "-static" should never be set, as it is incompatible with some third-party libraries
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++ ")
        endif()
    endif()
endmacro(define_compiler_options STATIC_BUILD)

# ===============================================================================
# Macro used to print compiler options
# ===============================================================================
macro(print_compiler_options)
    message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
    message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
    message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}")
    message(STATUS "CMAKE_CXX_LINK_EXECUTABLE: ${CMAKE_CXX_LINK_EXECUTABLE}")
    message(STATUS "LDFLAGS: ${LDFLAGS}")
endmacro(print_compiler_options)