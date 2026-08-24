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

if (RESOLVE_USE_GPU)
  add_executable(runHykktMBPCGTests.exe runHykktMBPCGTests.cpp)
  target_link_libraries(
    runHykktMBPCGTests.exe
    PRIVATE ReSolve resolve_hykkt resolve_hykkt_mbpcg resolve_hykkt_chol resolve_matrix
            resolve_vector resolve_preconditioner_ichol0
  )
  target_compile_definitions(
    runHykktMBPCGTests.exe PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  ) # For file paths
endif()

add_executable(runHykktCGTests.exe runHykktCGTests.cpp)
target_link_libraries(
  runHykktCGTests.exe
  PRIVATE ReSolve resolve_hykkt resolve_hykkt_cg resolve_hykkt_chol resolve_matrix
          resolve_vector resolve_preconditioner_ichol0
)
target_compile_definitions(
  runHykktCGTests.exe PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
) # For file paths

add_executable(runHykktSolverTests.exe runHykktSolverTests.cpp)
target_link_libraries(
  runHykktSolverTests.exe
  PRIVATE resolve_hykkt
          resolve_hykkt_ruiz
          resolve_hykkt_chol
          resolve_hykkt_spgemm
          resolve_hykkt_sccg
          resolve_hykkt_mbpcg
          resolve_hykkt_cg
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
    runHykktCholeskyTests.exe runHykktSpGEMMTests.exe runHykktSCCGTests.exe
    runHykktCGTests.exe runHykktSolverTests.exe
)
if(RESOLVE_USE_CUDSS)
  list(APPEND installable_tests runHykktCholeskyCuDssTests.exe)
endif()
if(RESOLVE_USE_GPU)
  list(APPEND installable_tests runHykktMBPCGTests.exe)
endif()

install(TARGETS ${installable_tests} RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME hykkt_perm_test
         COMMAND $<TARGET_FILE:runHykktPermutationTests.exe>
)
add_test(NAME hykkt_ruiz_test
         COMMAND $<TARGET_FILE:runHykktRuizScalingTests.exe>
)
add_test(NAME hykkt_chol_test COMMAND $<TARGET_FILE:runHykktCholeskyTests.exe>)
if(RESOLVE_USE_CUDSS)
  add_test(NAME hykkt_chol_cudss_test
           COMMAND $<TARGET_FILE:runHykktCholeskyCuDssTests.exe>
  )
endif()
add_test(NAME hykkt_spgemm_test COMMAND $<TARGET_FILE:runHykktSpGEMMTests.exe>)
add_test(NAME hykkt_sccg_test COMMAND $<TARGET_FILE:runHykktSCCGTests.exe>)
if(RESOLVE_USE_GPU)
  add_test(NAME hykkt_mbpcg_test COMMAND $<TARGET_FILE:runHykktMBPCGTests.exe>)
endif()
add_test(NAME hykkt_cg_test COMMAND $<TARGET_FILE:runHykktCGTests.exe>)
add_test(NAME hykkt_solver_test COMMAND $<TARGET_FILE:runHykktSolverTests.exe>)
