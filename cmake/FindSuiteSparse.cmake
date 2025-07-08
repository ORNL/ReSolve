#[[

Finds Sutiesparse include directory and libraries and exports target `Suitesparse`

User may set:
- SUITESPARSE_ROOT_DIR

Author(s):
- Cameron Rutherford <cameron.rutherford@pnnl.gov>

]]
set(SUITESPARSE_MODULES
  amd
  colamd
  klu
  suitesparseconfig)

find_library(SUITESPARSE_LIBRARY
  NAMES
  ${SUITESPARSE_MODULES}
  PATHS
  ${SUITESPARSE_DIR} $ENV{SUITESPARSE_DIR} ${SUITESPARSE_ROOT_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(SUITESPARSE_LIBRARY)
  set(SUITESPARSE_LIBRARY CACHE FILEPATH "Path to Suitesparse library")
  get_filename_component(SUITESPARSE_LIBRARY_DIR ${SUITESPARSE_LIBRARY} DIRECTORY CACHE "Suitesparse library directory")
  message(STATUS "Found Suitesparse libraries in: " ${SUITESPARSE_LIBRARY_DIR})
  mark_as_advanced(SUITESPARSE_LIBRARY SUITESPARSE_LIBRARY_DIR)
  if(NOT SUITESPARSE_DIR)
    get_filename_component(SUITESPARSE_DIR ${SUITESPARSE_LIBRARY_DIR} DIRECTORY CACHE)
  endif()
endif()

# Find SUITESPARSE header path and ensure all needed files are there
find_path(SUITESPARSE_INCLUDE_DIR
  NAMES
  amd.h
  colamd.h
  klu.h
  SuiteSparse_config.h
  PATHS
  ${SUITESPARSE_DIR} $ENV{SUITESPARSE_DIR} ${SUITESPARSE_ROOT_DIR} ${SUITESPARSE_LIBRARY_DIR}/..
  PATH_SUFFIXES
  include
  include/suitesparse)

if(SUITESPARSE_LIBRARY)
  message(STATUS "Found Suitesparse include: ${SUITESPARSE_INCLUDE_DIR}")
  mark_as_advanced(SUITESPARSE_INCLUDE_DIR)
  unset(SUITESPARSE_LIBRARY)
  add_library(SUITESPARSE INTERFACE IMPORTED)
  target_include_directories(SUITESPARSE INTERFACE ${SUITESPARSE_INCLUDE_DIR})
  foreach(mod ${SUITESPARSE_MODULES})
    find_library(suitesparse_${mod}
      NAMES ${mod}
      HINTS ${SUITESPARSE_LIBRARY_DIR})
    if(suitesparse_${mod})
      message(STATUS "Found suitesparse internal library " ${mod})
      target_link_libraries(SUITESPARSE INTERFACE ${suitesparse_${mod}})
      mark_as_advanced(suitesparse_${mod})
    else()
      message(SEND_ERROR "Suitesparse internal library " ${mod} " not found")
    endif()
  endforeach(mod)
else()
  if(NOT SUITESPARSE_ROOT_DIR)
    message(STATUS "Suitesparse dir not found! Please provide correct filepath.")
    set(SUITESPARSE_DIR ${SUITESPARSE_DIR} CACHE PATH "Path to Suitesparse installation root.")
    unset(SUITESPARSE_LIBRARY CACHE)
    unset(SUITESPARSE_INCLUDE_DIR CACHE)
    unset(SUITESPARSE_LIBRARY_DIR CACHE)
  elseif(NOT SUITESPARSE_LIBRARY)
    message(STATUS "Suitesparse library not found! Please provide correct filepath.")
  endif()
  if(SUITESPARSE_ROOT_DIR AND NOT SUITESPARSE_INCLUDE_DIR)
    message(STATUS "Suitesparse include dir not found! Please provide correct filepath.")
  endif()
endif()
