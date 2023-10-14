#[[

@brief Add functionality and unit tests for Re::Solve

@author Slaven Peles <peless@ornl.gov>

]]

if(RESOLVE_USE_CUDA)
  add_subdirectory(functionality) 
endif(RESOLVE_USE_CUDA)

add_subdirectory(unit)

