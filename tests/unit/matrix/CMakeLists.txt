#[[

@brief Build ReSolve matrix unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build matrix I/O tests
add_executable(runMatrixIoTests.exe runMatrixIoTests.cpp)
target_link_libraries(runMatrixIoTests.exe PRIVATE ReSolve resolve_matrix)

# Build matrix handler tests
add_executable(runMatrixHandlerTests.exe runMatrixHandlerTests.cpp)
target_link_libraries(runMatrixHandlerTests.exe PRIVATE ReSolve resolve_matrix)

# Build matrix factorization tests
add_executable(runMatrixFactorizationTests.exe runMatrixFactorizationTests.cpp)
target_link_libraries(runMatrixFactorizationTests.exe PRIVATE ReSolve resolve_matrix)

# Build LUSOL-related tests
add_executable(runLUSOLTests.exe runLUSOLTests.cpp)
target_link_libraries(runLUSOLTests.exe PRIVATE ReSolve resolve_matrix)

# Install tests
set(installable_tests runMatrixIoTests.exe runMatrixHandlerTests.exe runMatrixFactorizationTests.exe runLUSOLTests.exe)
install(TARGETS ${installable_tests}
        RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME matrix_test               COMMAND $<TARGET_FILE:runMatrixIoTests.exe>)
add_test(NAME matrix_handler_test       COMMAND $<TARGET_FILE:runMatrixHandlerTests.exe>)
add_test(NAME matrix_factorization_test COMMAND $<TARGET_FILE:runMatrixFactorizationTests.exe>)
add_test(NAME lusol_test                COMMAND $<TARGET_FILE:runLUSOLTests.exe>)
