#[[

@brief Build Re::Solve examples

@author Slaven Peles <peless@ornl.gov>

]]

if(RESOLVE_USE_KLU)

  # Create KLU+CUDA examples
  if(RESOLVE_USE_CUDA)

    # Build example where matrix is factorized once, refactorized once and then the preconditioner is REUSED
    add_executable(klu_rf_fgmres_reuse_refactorization.exe r_KLU_rf_FGMRES_reuse_factorization.cpp)
    target_link_libraries(klu_rf_fgmres_reuse_refactorization.exe PRIVATE ReSolve)

    # Build example where matrix data is updated
    add_executable(klu_glu_values_update.exe r_KLU_GLU_matrix_values_update.cpp)
    target_link_libraries(klu_glu_values_update.exe PRIVATE ReSolve)

    # Example in which factorization is redone if solution is bad
    add_executable(klu_cusolverrf_check_redo.exe r_KLU_cusolverrf_redo_factorization.cpp)
    target_link_libraries(klu_cusolverrf_check_redo.exe PRIVATE ReSolve)

  endif(RESOLVE_USE_CUDA)

  # Create KLU+HIP examples
  if(RESOLVE_USE_HIP)

    # Example in which factorization is redone if solution is bad
    add_executable(klu_rocsolverrf_check_redo.exe r_KLU_rocsolverrf_redo_factorization.cpp)
    target_link_libraries(klu_rocsolverrf_check_redo.exe PRIVATE ReSolve)

  endif(RESOLVE_USE_HIP)

endif(RESOLVE_USE_KLU)


set(installable_executables "")

# Install all examples in bin directory
list(APPEND installable_executables  rand_gmres.exe)

if(RESOLVE_USE_KLU)

  if(RESOLVE_USE_CUDA)
    list(APPEND installable_executables klu_glu_values_update.exe
                                        klu_cusolverrf_check_redo.exe
                                        klu_rf_fgmres_reuse_refactorization.exe)
  endif(RESOLVE_USE_CUDA)

  if(RESOLVE_USE_HIP)
    list(APPEND installable_executables klu_rocsolverrf_check_redo.exe)
  endif(RESOLVE_USE_HIP)
endif(RESOLVE_USE_KLU)

install(TARGETS ${installable_executables}
        RUNTIME DESTINATION bin)
