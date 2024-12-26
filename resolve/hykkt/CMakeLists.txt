#[[

@brief Build HyKKT 

@author Shaked Regev <regevs@ornl.gov>

]]

set (HyKKT_SRC
    PermClass.cpp
    cpuHykktPermutationKernels.cpp
)

set (HyKKT_HEADER_INSTALL
    PermClass.hpp
    cpuHykktPermutationKernels.hpp
)

add_library(hyktt SHARED ${HyKKT_SRC})
target_link_libraries(hyktt PUBLIC ReSolve)
# install include headers
install(FILES ${HyKKT_HEADER_INSTALL} DESTINATION include/hykkt)


