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

install(TARGETS resolve_hip EXPORT ReSolveTargets)
