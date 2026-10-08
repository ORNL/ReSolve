#[[

@brief Build ReSolve preconditioner unit tests

]]

add_executable(
  runPreconditionerUserMatrixTests.exe runPreconditionerUserMatrixTests.cpp
)
add_executable(
  runPreconditionerIdentityTests.exe runPreconditionerIdentityTests.cpp
)
add_executable(runPreconditionerLUTests.exe runPreconditionerLUTests.cpp)

target_link_libraries(runPreconditionerUserMatrixTests.exe PRIVATE ReSolve)
target_link_libraries(runPreconditionerIdentityTests.exe PRIVATE ReSolve)
target_link_libraries(runPreconditionerLUTests.exe PRIVATE ReSolve)

install(TARGETS runPreconditionerUserMatrixTests.exe
        RUNTIME DESTINATION bin/resolve/tests/unit
)
install(TARGETS runPreconditionerIdentityTests.exe
        RUNTIME DESTINATION bin/resolve/tests/unit
)
install(TARGETS runPreconditionerLUTests.exe
        RUNTIME DESTINATION bin/resolve/tests/unit
)

add_test(NAME preconditioner_user_matrix_test
         COMMAND $<TARGET_FILE:runPreconditionerUserMatrixTests.exe>
)
add_test(NAME preconditioner_identity_test
         COMMAND $<TARGET_FILE:runPreconditionerIdentityTests.exe>
)
add_test(NAME preconditioner_lu_test
         COMMAND $<TARGET_FILE:runPreconditionerLUTests.exe>
)
