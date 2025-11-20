#[[

@brief Add unit tests for Re::Solve

@author Slaven Peles <peless@ornl.gov>

]]

add_subdirectory(matrix)
add_subdirectory(vector)
add_subdirectory(utilities)
add_subdirectory(memory)
add_subdirectory(params)
if(RESOLVE_USE_KLU)
  add_subdirectory(hykkt)
endif()
