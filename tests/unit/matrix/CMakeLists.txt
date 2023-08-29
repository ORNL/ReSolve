#[[

@brief Build ReSolve matrix unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build matrix tests
add_executable(runMatrixIoTests.exe runMatrixIoTests.cpp)
target_link_libraries(runMatrixIoTests.exe PRIVATE ReSolve resolve_matrix)

# Build matrix tests
add_executable(runMatrixHandlerTests.exe runMatrixHandlerTests.cpp)
target_link_libraries(runMatrixHandlerTests.exe PRIVATE ReSolve resolve_matrix)

# Install tests
set(installable_tests runMatrixIoTests.exe runMatrixHandlerTests.exe)
install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/unit)
# Install directory with data files
install(DIRECTORY data DESTINATION bin/resolve/tests/unit)


add_test(NAME matrix_test         COMMAND $<TARGET_FILE:runMatrixIoTests.exe> "${CMAKE_SOURCE_DIR}/tests/unit/")
add_test(NAME matrix_handler_test COMMAND $<TARGET_FILE:runMatrixHandlerTests.exe>)