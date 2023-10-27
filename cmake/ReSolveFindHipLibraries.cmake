# Exports target `resolve_hip` which finds all hip libraries needed by resolve.


add_library(resolve_hip INTERFACE)

find_package(hip REQUIRED)
find_package(hipblas REQUIRED)

target_link_libraries(resolve_hip INTERFACE
  #hip::host 
  hip::device
  rocblas
  rocsparse
  #roc::hipblas
)

# get_target_property(hip_includes hip::device INTERFACE_INCLUDE_DIRECTORIES)
# message(STATUS "HIP include directories: ${hip_includes}")

# get_target_property(resolve_hip_includes resolve_hip INTERFACE_INCLUDE_DIRECTORIES)
# message(STATUS "ReSolve HIP include directories: ${resolve_hip_includes}")

install(TARGETS resolve_hip EXPORT ReSolveTargets)
