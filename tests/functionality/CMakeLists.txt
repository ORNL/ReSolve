#[[

@brief Build Re::Solve functionality tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build basic version test
add_executable(version.exe testVersion.cpp) 
target_link_libraries(version.exe PRIVATE ReSolve)

# Build test for Krylov solvers
add_executable(rand_gmres_test.exe testRandGmres.cpp)
target_link_libraries(rand_gmres_test.exe PRIVATE ReSolve)

add_executable(sys_rand_gmres_test.exe testSysGmres.cpp)
target_link_libraries(sys_rand_gmres_test.exe PRIVATE ReSolve)

if(RESOLVE_USE_KLU)
  # Build KLU+KLU test
  add_executable(klu_klu_test.exe testKlu.cpp)
  target_link_libraries(klu_klu_test.exe PRIVATE ReSolve)
  if(RESOLVE_USE_GPU)
    # Build KLU+Rf tests
    add_executable(klu_rf_test.exe testRefactor.cpp)
    target_link_libraries(klu_rf_test.exe PRIVATE ReSolve)

    # System solver test with cusolver rf and iterative refinement
    add_executable(sys_refactor_test.exe testSysRefactor.cpp)
    target_link_libraries(sys_refactor_test.exe PRIVATE ReSolve)
  endif(RESOLVE_USE_GPU)

  if(RESOLVE_USE_CUDA)
    # System solver test with GLU (to be subsumed in testSysRefactor.cpp)
    add_executable(sys_glu_test.exe testSysGLU.cpp)
    target_link_libraries(sys_glu_test.exe PRIVATE ReSolve)

  endif(RESOLVE_USE_CUDA)
endif(RESOLVE_USE_KLU)

if(RESOLVE_USE_LUSOL)
  add_executable(lusol_test.exe testLUSOL.cpp)
  target_link_libraries(lusol_test.exe PRIVATE ReSolve)
endif()


# Install tests
set(installable_tests version.exe)
list(APPEND installable_tests rand_gmres_test.exe)
list(APPEND installable_tests sys_rand_gmres_test.exe)

if(RESOLVE_USE_KLU)
  list(APPEND installable_tests klu_klu_test.exe)
  if(RESOLVE_USE_GPU)
    list(APPEND installable_tests klu_rf_test.exe
                                  sys_refactor_test.exe)
  endif(RESOLVE_USE_GPU)

  if(RESOLVE_USE_CUDA)
    list(APPEND installable_tests sys_glu_test.exe)
  endif(RESOLVE_USE_CUDA)
endif()

if(RESOLVE_USE_LUSOL)
  list(APPEND installable_tests lusol_test.exe)
endif()

install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/functionality)

# Install directory with data files
install(DIRECTORY data DESTINATION bin/resolve/tests/functionality)

# Set the matrices for the tests.
set(test_data_dir ${CMAKE_SOURCE_DIR}/tests/functionality)
set(sym_matrix_path /data/matrix_ACTIVSg200_AC_renumbered_add9_)
set(sym_rhs_path /data/rhs_ACTIVSg200_AC_renumbered_add9_ones_)
set(asym_matrix_path /data/A_asymmetric)
set(asym_rhs_path /data/b_asymmetric)


add_test(NAME version COMMAND $<TARGET_FILE:version.exe>)

if(RESOLVE_USE_LUSOL)
  add_test(NAME lusol_test COMMAND $<TARGET_FILE:lusol_test.exe> "${test_data_dir}")
endif()

# Krylov solvers tests (FGMRES)
add_test(NAME sys_rand_count_fgmres_cgs2_test     COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "randgmres" "-g" "cgs2" "-s" "count")
add_test(NAME sys_rand_count_fgmres_mgs_test      COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "randgmres" "-g" "mgs" "-s" "count")
add_test(NAME sys_rand_count_fgmres_mgs2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "randgmres" "-g" "mgs_two_sync" "-s" "count")
add_test(NAME sys_rand_count_fgmres_mgspm_test    COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "randgmres" "-g" "mgs_pm" "-s" "count")
add_test(NAME sys_rand_fwht_fgmres_cgs2_test      COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "randgmres" "-g" "cgs2" "-s" "fwht")
add_test(NAME sys_rand_fwht_fgmres_mgs_test       COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "randgmres" "-g" "mgs" "-s" "fwht")
add_test(NAME sys_rand_fwht_fgmres_mgs2sync_test  COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "randgmres" "-g" "mgs_two_sync" "-s" "fwht")
add_test(NAME sys_rand_fwht_fgmres_mgspm_test     COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "randgmres" "-g" "mgs_pm" "-s" "fwht")
add_test(NAME sys_fgmres_cgs2_test     COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "fgmres" "-g" "cgs2")
add_test(NAME sys_fgmres_mgs_test      COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "fgmres" "-g" "mgs")
add_test(NAME sys_fgmres_mgs2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "fgmres" "-g" "mgs_two_sync")
add_test(NAME sys_fgmres_mgspm_test    COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "fgmres" "-g" "mgs_pm")

# Krylov solvers tests (GMRES)
add_test(NAME sys_rand_count_gmres_cgs2_test     COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "randgmres" "-g" "cgs2" "-s" "count")
add_test(NAME sys_rand_count_gmres_mgs_test      COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "randgmres" "-g" "mgs" "-s" "count")
add_test(NAME sys_rand_count_gmres_mgs2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "randgmres" "-g" "mgs_two_sync" "-s" "count")
add_test(NAME sys_rand_count_gmres_mgspm_test    COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "randgmres" "-g" "mgs_pm" "-s" "count")
add_test(NAME sys_rand_fwht_gmres_cgs2_test      COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "randgmres" "-g" "cgs2" "-s" "fwht")
add_test(NAME sys_rand_fwht_gmres_mgs_test       COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "randgmres" "-g" "mgs" "-s" "fwht")
add_test(NAME sys_rand_fwht_gmres_mgs2sync_test  COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "randgmres" "-g" "mgs_two_sync" "-s" "fwht")
add_test(NAME sys_rand_fwht_gmres_mgspm_test     COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "randgmres" "-g" "mgs_pm" "-s" "fwht")
add_test(NAME sys_gmres_cgs2_test     COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "fgmres" "-g" "cgs2")
add_test(NAME sys_gmres_mgs_test      COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "fgmres" "-g" "mgs")
add_test(NAME sys_gmres_mgs2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "fgmres" "-g" "mgs_two_sync")
add_test(NAME sys_gmres_mgspm_test    COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "fgmres" "-g" "mgs_pm")

# Randomized GMRES solvers tested without SystemSolver class
add_test(NAME rand_gmres_test    COMMAND $<TARGET_FILE:rand_gmres_test.exe>)

if(RESOLVE_USE_KLU)
  # Using KLU as refactorization solver
  add_test(NAME klu_klu_test    COMMAND $<TARGET_FILE:klu_klu_test.exe> "-d" "${test_data_dir}" "-m" "${sym_matrix_path}" "-r" "${sym_rhs_path}") 
  add_test(NAME klu_klu_ir_test COMMAND $<TARGET_FILE:klu_klu_test.exe> "-d" "${test_data_dir}" "-m" "${sym_matrix_path}" "-r" "${sym_rhs_path}" "-i")
  add_test(NAME klu_klu_asym_test COMMAND $<TARGET_FILE:klu_klu_test.exe> "-d" "${test_data_dir}" "-m" "${asym_matrix_path}" "-r" "${asym_rhs_path}")
  add_test(NAME klu_klu_asym_ir_test COMMAND $<TARGET_FILE:klu_klu_test.exe> "-d" "${test_data_dir}" "-m" "${asym_matrix_path}" "-r" "${asym_rhs_path}" "-i") 

  # CUDA-SDK specific tests
  if(RESOLVE_USE_CUDA)
    add_test(NAME klu_rf_cuda_test    COMMAND $<TARGET_FILE:klu_rf_test.exe> "-d" "${test_data_dir}")
    add_test(NAME klu_rf_ir_cuda_test COMMAND $<TARGET_FILE:klu_rf_test.exe> "-d" "${test_data_dir}" "-i")
    add_test(NAME klu_glu_cuda_test   COMMAND $<TARGET_FILE:klu_rf_test.exe> "-d" "${test_data_dir}" "-s" "glu")
    add_test(NAME sys_refactor_cuda_test COMMAND $<TARGET_FILE:sys_refactor_test.exe> "${test_data_dir}" "-d" "${test_data_dir}" "-m" "${sym_matrix_path}" "-r" "${sym_rhs_path}")
    add_test(NAME sys_refactor_cuda_asym_test COMMAND $<TARGET_FILE:sys_refactor_test.exe> "${test_data_dir}" "-d" "${test_data_dir}" "-m" "${asym_matrix_path}" "-r" "${asym_rhs_path}") 
    add_test(NAME sys_glu_test COMMAND $<TARGET_FILE:sys_glu_test.exe> "${test_data_dir}" "-d" "${test_data_dir}" "-m" "${sym_matrix_path}" "-r" "${sym_rhs_path}")
    add_test(NAME sys_glu_asym_test COMMAND $<TARGET_FILE:sys_glu_test.exe> "${test_data_dir}" "-d" "${test_data_dir}" "-m" "${asym_matrix_path}" "-r" "${asym_rhs_path}") 
  endif(RESOLVE_USE_CUDA)

  # ROCm specific tests
  if(RESOLVE_USE_HIP)
    add_test(NAME rocsolver_rf_test       COMMAND $<TARGET_FILE:klu_rf_test.exe> "-d" "${test_data_dir}")
    add_test(NAME rocsolver_rf_m1_test    COMMAND $<TARGET_FILE:klu_rf_test.exe> "-d" "${test_data_dir}" "-m" "rocsparse_trisolve")
    add_test(NAME rocsolver_rf_ir_test    COMMAND $<TARGET_FILE:klu_rf_test.exe> "-d" "${test_data_dir}" "-i")
    add_test(NAME rocsolver_rf_ir_m1_test COMMAND $<TARGET_FILE:klu_rf_test.exe> "-d" "${test_data_dir}" "-i" "-m" "rocsparse_trisolve")
    add_test(NAME sys_refactor_hip_test COMMAND $<TARGET_FILE:sys_refactor_test.exe> "${test_data_dir}" "-d" "${test_data_dir}" "-m" "${sym_matrix_path}" "-r" "${sym_rhs_path}")
    add_test(NAME sys_refactor_hip_asym_test COMMAND $<TARGET_FILE:sys_refactor_test.exe> "${test_data_dir}" "-d" "${test_data_dir}" "-m" "${asym_matrix_path}" "-r" "${asym_rhs_path}")
  endif(RESOLVE_USE_HIP)
endif(RESOLVE_USE_KLU)
