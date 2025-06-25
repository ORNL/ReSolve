#[[

@brief Build HyKKT module unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build HyKKT tests

add_executable(runHykktPermutationTests.exe runHykktPermutationTests.cpp)
add_executable(runHykktRuizScalingTests.exe runHykktRuizScalingTests.cpp)
target_link_libraries(runHykktPermutationTests.exe PRIVATE resolve_hykkt resolve_matrix)
target_link_libraries(runHykktRuizScalingTests.exe PRIVATE resolve_hykkt resolve_hykkt_ruiz resolve_matrix)


# Install tests
set(installable_tests runHykktPermutationTests.exe
                      runHykktRuizScalingTests.exe)

install(TARGETS ${installable_tests}
        RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME hykkt_perm_test   COMMAND $<TARGET_FILE:runHykktPermutationTests.exe>)
add_test(NAME hykkt_ruiz_test   COMMAND $<TARGET_FILE:runHykktRuizScalingTests.exe>)