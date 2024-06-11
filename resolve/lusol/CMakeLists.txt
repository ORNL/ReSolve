add_library(lusol_lib SHARED lusol.f90 lusol_precision.f90)

install(FILES lusol.hpp DESTINATION include/resolve/)
