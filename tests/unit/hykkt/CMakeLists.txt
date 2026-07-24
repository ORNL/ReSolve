#[[

@brief Build HyKKT module unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build HyKKT tests

add_executable(runHykktPermutationTests.exe runHykktPermutationTests.cpp)
target_link_libraries(
  runHykktPermutationTests.exe PRIVATE resolve_hykkt resolve_matrix
)

add_executable(runHykktRuizScalingTests.exe runHykktRuizScalingTests.cpp)
target_link_libraries(
  runHykktRuizScalingTests.exe PRIVATE resolve_hykkt resolve_hykkt_ruiz
                                       resolve_matrix
)

add_executable(runHykktSpGEMMTests.exe runHykktSpGEMMTests.cpp)
target_link_libraries(
  runHykktSpGEMMTests.exe PRIVATE resolve_hykkt resolve_hykkt_spgemm
                                  resolve_matrix
)

add_executable(runHykktCholeskyTests.exe runHykktCholeskyTests.cpp)
target_link_libraries(
  runHykktCholeskyTests.exe PRIVATE resolve_hykkt_chol resolve_matrix
                                    resolve_vector
)

if(RESOLVE_USE_CUDSS)
  add_executable(runHykktCholeskyCuDssTests.exe runHykktCholeskyCuDssTests.cpp)
  target_link_libraries(
    runHykktCholeskyCuDssTests.exe PRIVATE resolve_hykkt_chol_cudss
                                           resolve_matrix resolve_vector
  )
endif()

add_executable(runHykktSCCGTests.exe runHykktSCCGTests.cpp)
target_link_libraries(
  runHykktSCCGTests.exe
  PRIVATE resolve_hykkt resolve_hykkt_sccg resolve_hykkt_chol resolve_matrix
          resolve_vector
)
target_compile_definitions(
  runHykktSCCGTests.exe PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
) # For file paths

add_executable(runHykktSolverTests.exe runHykktSolverTests.cpp)
target_link_libraries(
  runHykktSolverTests.exe
  PRIVATE resolve_hykkt
          resolve_hykkt_ruiz
          resolve_hykkt_chol
          resolve_hykkt_spgemm
          resolve_hykkt_sccg
          resolve_hykkt_solver
          resolve_vector
          resolve_matrix
)
target_compile_definitions(
  runHykktSolverTests.exe PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
) # For file paths

# Install tests
set(installable_tests
    runHykktPermutationTests.exe runHykktRuizScalingTests.exe
    runHykktCholeskyTests.exe runHykktSpGEMMTests.exe runHykktSolverTests.exe
)
if(RESOLVE_USE_CUDSS)
  list(APPEND installable_tests runHykktCholeskyCuDssTests.exe)
endif()

install(TARGETS ${installable_tests} RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME hykkt_perm_test
         COMMAND $<TARGET_FILE:runHykktPermutationTests.exe>
)
add_test(NAME hykkt_ruiz_test
         COMMAND $<TARGET_FILE:runHykktRuizScalingTests.exe>
)
add_test(NAME hykkt_chol_test COMMAND $<TARGET_FILE:runHykktCholeskyTests.exe>)
add_test(NAME hykkt_spgemm_test COMMAND $<TARGET_FILE:runHykktSpGEMMTests.exe>)
add_test(NAME hykkt_sccg_test COMMAND $<TARGET_FILE:runHykktSCCGTests.exe>)
add_test(NAME hykkt_solver_test COMMAND $<TARGET_FILE:runHykktSolverTests.exe>)
if(RESOLVE_USE_CUDSS)
  add_test(NAME hykkt_chol_cudss_test
           COMMAND $<TARGET_FILE:runHykktCholeskyCuDssTests.exe>
  )
endif()
