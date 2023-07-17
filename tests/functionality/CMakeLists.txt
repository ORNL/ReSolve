#[[

@brief Build ReSolve functionality tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build KLU+KLU test
add_executable(klu_klu_test.exe testKLU.cpp)
target_link_libraries(klu_klu_test.exe PRIVATE ReSolve resolve_matrix)

# Install tests
set(installable_tests klu_klu_test.exe)
install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests)
# Install directory with data files
install(DIRECTORY data DESTINATION bin/resolve/tests)


add_test(NAME klu_klu_test COMMAND $<TARGET_FILE:klu_klu_test.exe> "${CMAKE_SOURCE_DIR}/tests/functionality/")