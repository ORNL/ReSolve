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

add_library(resolve_hyktt SHARED ${HyKKT_SRC})
target_link_libraries(resolve_hyktt PRIVATE resolve_tpl)
#target_include_directories(resolve_hykkt PRIVATE ${CMAKE_BINARY_DIR})

# install include headers
install(FILES ${HyKKT_HEADER_INSTALL} DESTINATION include/hykkt)


