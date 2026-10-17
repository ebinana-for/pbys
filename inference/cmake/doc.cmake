# ===============================================================================
# Macro used to generate doxygen doc.
# takes the relative path to the doxygen configuration (usually Doxygen.in) as argument
# ===============================================================================
macro(add_doc_target)
    if (CONF_PATH)
        find_package(Doxygen)
        if (DOXYGEN_FOUND)
            # set input and output files
            set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/${CONF_PATH})
            set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
            set(DOC_DOXYGEN_TARGET_NAME doc_doxygen)

            # request to configure the file
            configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

            # note the option ALL which allows to build the docs together with the application
            add_custom_target(${DOC_DOXYGEN_TARGET_NAME}
                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                    COMMAND echo "creating ${DOC_DOXYGEN_TARGET_NAME}.tar.gz archive"
                    COMMAND tar -czf ${DOC_DOXYGEN_TARGET_NAME}.tar.gz ${DOC_DOXYGEN_TARGET_NAME}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMENT "Generating API documentation with Doxygen"
                    VERBATIM)
            unset(DOC_DOXYGEN_TARGET_NAME)
            message("Doxygen configured")
        else (DOXYGEN_FOUND)
            message(WARNING "Doxygen needs to be installed to generate the doxygen documentation")
        endif (DOXYGEN_FOUND)
    else (CONF_PATH)
        message(WARNING "You have to pass -DCONF_PATH=path/to/Doxyfile.in at cmake invocation")
    endif (CONF_PATH)
endmacro(add_doc_target)
# ===============================================================================
# Latex
# ===============================================================================
macro(initLatex)
    find_package(Latex REQUIRED)

    if(LATEX_FOUND)
        message(STATUS "Latex found. documentation targets will be generated")
    else(LATEX_FOUND)
        message(STATUS "Latex not found. documentation targets will not be generated")
    endif(LATEX_FOUND)
endmacro(initLatex)

# ===============================================================================
# Macro Used to generate documentation with latex
# ===============================================================================
macro(add_latex name)

    initLatex()

    if(NOT WIN32 AND PDFLATEX_COMPILER)
        foreach(src ${${name}_LATEX_SRC})
            get_filename_component(filename ${src} NAME)
            configure_file("${src}" "${CMAKE_CURRENT_BINARY_DIR}/temp/${filename}" @ONLY)
        endforeach(src)

        foreach(src ${${name}_LATEX_IMG})
            get_filename_component(filename ${src} NAME)
            execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${src}" "${CMAKE_CURRENT_BINARY_DIR}/${filename}")
        endforeach(src)

        ADD_CUSTOM_TARGET(${name}_tex
            COMMAND ${PDFLATEX_COMPILER} "${CMAKE_CURRENT_BINARY_DIR}/temp/${${name}_LATEX_MAIN}.tex"
            COMMAND ${PDFLATEX_COMPILER} "${CMAKE_CURRENT_BINARY_DIR}/temp/${${name}_LATEX_MAIN}.tex")
    else(NOT WIN32 AND PDFLATEX_COMPILER)
        ADD_CUSTOM_TARGET(${name}_tex COMMAND ${CMAKE_COMMAND} -E copy "${${name}_LATEX_ALT_PDF}" "${CMAKE_CURRENT_BINARY_DIR}/user_manual.pdf")
    endif(NOT WIN32 AND PDFLATEX_COMPILER)

    ADD_DEPENDENCIES(${name}_doc ${name}_tex)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${${name}_LATEX_MAIN}.pdf DESTINATION "${${name}_LATEX_DEST}" RENAME ${name}_${${name}_LATEX_MAIN}.pdf)

endmacro(add_latex name)
