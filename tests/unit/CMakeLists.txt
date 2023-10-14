#[[

@brief Add unit tests for Re::Solve

@author Slaven Peles <peless@ornl.gov>

]]

if(RESOLVE_USE_CUDA)
  add_subdirectory(matrix)
  add_subdirectory(vector)
  add_subdirectory(utilities)
endif(RESOLVE_USE_CUDA)
