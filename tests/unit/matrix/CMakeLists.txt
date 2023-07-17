#[[

@brief Build ReSolve matrix unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build KLU+KLU test
add_executable(runMatrixTests.exe runMatrixTests.cpp)
target_link_libraries(runMatrixTests.exe PRIVATE ReSolve resolve_matrix)

# Install tests
set(installable_tests runMatrixTests.exe)
install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/unit)
# Install directory with data files
install(DIRECTORY data DESTINATION bin/resolve/tests/unit)


add_test(NAME matrix_test COMMAND $<TARGET_FILE:runMatrixTests.exe> "${CMAKE_SOURCE_DIR}/tests/unit/")