#[[

@brief Build HyKKT module unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build HyKKT tests

add_executable(runHykktPermutationTests.exe runHykktPermutationTests.cpp)
target_link_libraries(runHykktPermutationTests.exe PRIVATE resolve_hykkt)

# Install tests
set(installable_tests runHykktPermutationTests.exe)

install(TARGETS ${installable_tests}
        RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME hykkt_perm_test   COMMAND $<TARGET_FILE:runHykktPermutationTests.exe>)

