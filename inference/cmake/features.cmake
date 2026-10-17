# ===============================================================================
# Macro used to configure fast log
# ===============================================================================
macro(configure_fast_logs)
    if(NOT DEFINED FAST_LOGS_SCORE)
        message("FAST_LOGS_SCORE not defined, defining it")
        option(FAST_LOGS_SCORE "Whether to activate score level logs." OFF)
    endif()
    if(NOT DEFINED FAST_LOGS_DEBUG)
        message("FAST_LOGS_DEBUG not defined, defining it")
        option(FAST_LOGS_DEBUG "Whether to activate debug logs." OFF)
    endif()
    if(NOT DEFINED FAST_LOGS_ALL)
        message("FAST_LOGS_ALL not defined, defining it")
        option(FAST_LOGS_ALL "Whether to activate all levels of logs." OFF)
    endif()
    message("FAST_LOGS_DEBUG ${FAST_LOGS_DEBUG}")

    if(FAST_LOGS_SCORE OR FAST_LOGS_DEBUG OR FAST_LOGS_ALL)
        add_definitions(-DFAST_LOG_ACTIVATE_SCORE)
    endif()
    if(FAST_LOGS_DEBUG OR FAST_LOGS_ALL)
        add_definitions(-DFAST_LOG_ACTIVATE_DEBUG)
    endif()
    if(FAST_LOGS_ALL)
        add_definitions(-DFAST_LOG_ACTIVATE_TRACE)
    endif()
    message("FAST_LOGS_SCORE: ${FAST_LOGS_SCORE}")
    message("FAST_LOGS_DEBUG: ${FAST_LOGS_DEBUG}")
    message("FAST_LOGS_ALL: ${FAST_LOGS_ALL}")
endmacro(configure_fast_logs)

# ===============================================================================
# Macro used to configure Boost available features
# ===============================================================================
macro(configure_boost)
    message("BOOST VERSION IS: ${Boost_VERSION_MACRO}")
    if(Boost_VERSION_MACRO GREATER_EQUAL 107500)
        add_definitions(-DHAS_FEATURE_BOOST_JSON)
        set(OPTIONAL_BOOST_TARGETS "Boost::json")
    else()
        set(OPTIONAL_BOOST_TARGETS "")
    endif()
    if(Boost_VERSION_MACRO GREATER_EQUAL 108200)
        add_definitions(-DHAS_RECENT_BOOST_BEAST)
    endif()
endmacro(configure_boost)

# ===============================================================================
# Macro used to configure lib install path
# ===============================================================================
macro(configure_pybind)
    set(CMAKE_INSTALL_PREFIX "$ENV{VIRTUAL_ENV}")
    message("CMAKE_INSTALL_PREFIX : ${CMAKE_INSTALL_PREFIX}")
endmacro(configure_pybind)

