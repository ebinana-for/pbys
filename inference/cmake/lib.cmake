# ===============================================================================
# Macro used to add a library, only using targets for dependencies
# ===============================================================================
macro(add_library_target LIB_NAME)

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

  install(TARGETS ${LIB_NAME} DESTINATION "lib")

endmacro(add_library_target LIB_NAME)

# ===============================================================================
# Macro Used to build an application, only using targets for dependencies
# ===============================================================================
macro(add_target_application APP_NAME)

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

endmacro(add_target_application APP_NAME)
