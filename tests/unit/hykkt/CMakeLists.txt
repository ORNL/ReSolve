#[[

@brief Build ReSolve matrix unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build matrix I/O tests
add_executable(runHykktHelloWorldTests.exe runHykktHelloWorld.cpp)
target_link_libraries(runHykktHelloWorldTests.exe PRIVATE resolve_hykkt)

# Install tests
set(installable_tests runHykktHelloWorldTests.exe)

install(TARGETS ${installable_tests}
        RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME hello_world_test          COMMAND $<TARGET_FILE:runHykktHelloWorldTests.exe>)




# add_executable(runHykktPermTests.exe runHykktPerm.cpp)
# target_link_libraries(runHykktPermTests.exe PRIVATE ReSolve)

# add_test(NAME hykkt_perm_test   COMMAND $<TARGET_FILE:runHykktPermTests.exe>)
