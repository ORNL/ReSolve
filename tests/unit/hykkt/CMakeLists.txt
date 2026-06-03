#[[

@brief Build HyKKT module unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build HyKKT tests

add_executable(runHykktPermutationTests.exe runHykktPermutationTests.cpp)
add_executable(runHykktRuizScalingTests.exe runHykktRuizScalingTests.cpp)
add_executable(runHykktSpGEMMTests.exe runHykktSpGEMMTests.cpp)
target_link_libraries(
  runHykktPermutationTests.exe PRIVATE resolve_hykkt resolve_matrix
)
target_link_libraries(
  runHykktRuizScalingTests.exe PRIVATE resolve_hykkt resolve_hykkt_ruiz
                                       resolve_matrix
)
target_link_libraries(
  runHykktSpGEMMTests.exe PRIVATE resolve_hykkt resolve_hykkt_spgemm
                                  resolve_matrix
)

add_executable(runHykktCholeskyTests.exe runHykktCholeskyTests.cpp)
target_link_libraries(
  runHykktCholeskyTests.exe PRIVATE resolve_hykkt_chol resolve_matrix
                                    resolve_vector
)

# Install tests
set(installable_tests runHykktPermutationTests.exe runHykktRuizScalingTests.exe
                      runHykktCholeskyTests.exe runHykktSpGEMMTests.exe
)

install(TARGETS ${installable_tests} RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME hykkt_perm_test
         COMMAND $<TARGET_FILE:runHykktPermutationTests.exe>
)
add_test(NAME hykkt_ruiz_test
         COMMAND $<TARGET_FILE:runHykktRuizScalingTests.exe>
)
add_test(NAME hykkt_chol_test COMMAND $<TARGET_FILE:runHykktCholeskyTests.exe>)
add_test(NAME hykkt_spgemm_test COMMAND $<TARGET_FILE:runHykktSpGEMMTests.exe>)
