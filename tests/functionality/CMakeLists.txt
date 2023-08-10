#[[

@brief Build ReSolve functionality tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build KLU+KLU test
add_executable(klu_klu_test.exe testKLU.cpp)
target_link_libraries(klu_klu_test.exe PRIVATE ReSolve)

# Build KLU+Rf test
add_executable(klu_rf_test.exe testKLU_Rf.cpp)
target_link_libraries(klu_rf_test.exe PRIVATE ReSolve)

# Build KLU+Rf+fgmres test
add_executable(klu_rf_fgmres_test.exe testKLU_Rf_FGMRES.cpp)
target_link_libraries(klu_rf_fgmres_test.exe PRIVATE ReSolve)



# Build KLU+GLU test
add_executable(klu_glu_test.exe testKLU_GLU.cpp)
target_link_libraries(klu_glu_test.exe PRIVATE ReSolve)

# Install tests
set(installable_tests 
        klu_klu_test.exe
        klu_rf_test.exe
        klu_rf_fgmres_test.exe
        klu_glu_test.exe)

install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/functionality)

# Install directory with data files
install(DIRECTORY data DESTINATION bin/resolve/tests/functionality)

set(test_data_dir ${CMAKE_SOURCE_DIR}/tests/functionality/)

add_test(NAME klu_klu_test COMMAND $<TARGET_FILE:klu_klu_test.exe> "${test_data_dir}")
add_test(NAME klu_rf_test  COMMAND $<TARGET_FILE:klu_rf_test.exe>  "${test_data_dir}")
add_test(NAME klu_rf_fgmres_test COMMAND $<TARGET_FILE:klu_rf_fgmres_test.exe> "${test_data_dir}")
add_test(NAME klu_glu_test COMMAND $<TARGET_FILE:klu_glu_test.exe> "${test_data_dir}")
