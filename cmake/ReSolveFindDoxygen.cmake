#[[

Looks for Doxygen tool and, if found, created `doxygen` target.

Command `make doxygen` will create Doxygen documentation in the
`doxygen` subdirectory within ReSolve build directory.

@author Slaven Peles (peless@ornl.gov)

]]
find_package(Doxygen)

if ( DOXYGEN_FOUND )
  set( DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/doxygen )
  # set( DOXYGEN_CREATE_SUBDIRS YES )
  set( DOXYGEN_COLLABORATION_GRAPH YES )
  set( DOXYGEN_EXTRACT_ALL YES )
  set( DOXYGEN_CLASS_DIAGRAMS YES )
  set( DOXYGEN_HIDE_UNDOC_RELATIONS NO )
  set( DOXYGEN_HAVE_DOT YES )
  set( DOXYGEN_CLASS_GRAPH YES )
  set( DOXYGEN_CALL_GRAPH YES )
  set( DOXYGEN_CALLER_GRAPH YES )
  set( DOXYGEN_COLLABORATION_GRAPH YES )
  set( DOXYGEN_BUILTIN_STL_SUPPORT YES )
  set( DOXYGEN_EXTRACT_PRIVATE YES )
  set( DOXYGEN_EXTRACT_PACKAGE YES )
  set( DOXYGEN_EXTRACT_STATIC YES )
  set( DOXYGEN_EXTRACT_LOCALMETHODS YES )
  set( DOXYGEN_UML_LOOK YES )
  set( DOXYGEN_UML_LIMIT_NUM_FIELDS 50 )
  set( DOXYGEN_TEMPLATE_RELATIONS YES )
  set( DOXYGEN_DOT_GRAPH_MAX_NODES 100 )
  set( DOXYGEN_MAX_DOT_GRAPH_DEPTH 0 )
  set( DOXYGEN_DOT_TRANSPARENT YES )
  set( DOXYGEN_DOT_IMAGE_FORMAT svg )
  set( DOXYGEN_INTERACTIVE_SVG YES )
  set(DOXYGEN_DISABLE_INDEX NO)
  set(DOXYGEN_FULL_SIDEBAR NO)
  set(DOXYGEN_GENERATE_TREEVIEW YES)
  set(DOXYGEN_HTML_EXTRA_STYLESHEET 
      "${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-awesome.css"
      # "${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-custom/custom.css"
      #"${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-awesome-sidebar-only.css"
      #"${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-awesome-sidebar-only-darkmode-toggle.css"
      # "${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-custom/custom-alternative.css"
     )
set(DOXYGEN_HTML_EXTRA_FILES 
      "${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-awesome-darkmode-toggle.js"
      "${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-awesome-fragment-copy-button.js"
      "${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-awesome-paragraph-link.js"
      "${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-custom/toggle-alternative-theme.js"
      "${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-awesome-interactive-toc.js"
      "${CMAKE_SOURCE_DIR}/docs/doxygen-awesome-css/doxygen-awesome-tabs.js")
  set(DOXYGEN_HTML_COLORSTYLE LIGHT)
  doxygen_add_docs( doxygen ${CMAKE_SOURCE_DIR}/resolve )

else()
  message( "Doxygen need to be installed to generate the doxygen documentation" )
endif()
