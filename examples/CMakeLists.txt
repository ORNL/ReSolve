#[[

@brief Build ReSolve library

@author Slaven Peles <peless@ornl.gov>

]]

# Build example with KLU factorization and KLU refactorization
add_executable(klu_klu.exe r_KLU_KLU.cpp)
target_link_libraries(klu_klu.exe PRIVATE ReSolve)

# Build example with KLU factorization and GLU refactorization
add_executable(klu_glu.exe r_KLU_GLU.cpp)
target_link_libraries(klu_glu.exe PRIVATE ReSolve)

# Build example with KLU factorization and Rf refactorization
add_executable(klu_rf.exe r_KLU_rf.cpp)
target_link_libraries(klu_rf.exe PRIVATE ReSolve)

# Build example with KLU factorization, Rf refactorization, and FGMRES iterative refinement
add_executable(klu_rf_fgmres.exe r_KLU_rf_FGMRES.cpp)
target_link_libraries(klu_rf_fgmres.exe PRIVATE ReSolve)

set(installable_executables klu_klu.exe klu_glu.exe klu_rf.exe klu_rf_fgmres.exe)
install(TARGETS ${installable_executables} 
        RUNTIME DESTINATION bin)
# SP: I want to use a generator expression here, but I am not sure what is the right one
set_property(TARGET ${installable_executables} APPEND PROPERTY INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib) #$<TARGET_FILE_DIR:ReSolve>)
