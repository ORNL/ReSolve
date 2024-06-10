# this has to be done this way because lusol.f90 depends upon lusol_precision.f90
add_library(lusol_precision OBJECT lusol_precision.f90)
add_library(lusol SHARED lusol.f90)
# i've yet to verify the necessity of this
# \/ PRIVATE ${FORTRAN_RUNTIME_LIBRARIES}
target_link_libraries(lusol PUBLIC lusol_precision)

install(FILES lusol.hpp DESTINATION include/resolve/)
