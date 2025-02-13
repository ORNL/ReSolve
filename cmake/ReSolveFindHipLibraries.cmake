# Exports target `resolve_hip` which finds all hip libraries needed by resolve.


add_library(resolve_hip INTERFACE)

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsparse REQUIRED)
find_package(rocsolver REQUIRED)
find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

target_link_libraries(resolve_hip INTERFACE 
  hip::host 
  hip::device
  roc::rocblas
  roc::rocsparse
  roc::rocsolver
)

# HIP/ROCm targets still don't have include directories set correctly
# We need this little hack for now :/
get_target_property(hip_includes hip::device INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "HIP include directories found at: ${hip_includes}")

target_include_directories(resolve_hip INTERFACE 
  $<BUILD_INTERFACE:${hip_includes}>)

install(TARGETS resolve_hip EXPORT ReSolveTargets)

