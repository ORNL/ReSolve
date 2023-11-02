# Exports target `resolve_hip` which finds all hip libraries needed by resolve.


add_library(resolve_hip INTERFACE)

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsparse REQUIRED)
find_package(rocsolver REQUIRED)

target_link_libraries(resolve_hip INTERFACE
  hip::host 
  hip::device
  roc::rocblas
  roc::rocsparse
  roc::rocsolver
)

install(TARGETS resolve_hip EXPORT ReSolveTargets)

