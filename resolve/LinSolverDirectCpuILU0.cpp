/**
 * @file LinSolverDirectCpuILU0.cpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Contains definition of a class for incomplete LU factorization on CPU
 * 
 * 
 */
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#include "LinSolverDirectCpuILU0.hpp"

namespace ReSolve 
{
  LinSolverDirectCpuILU0::LinSolverDirectCpuILU0(LinAlgWorkspaceCpu* workspace)
    : workspace_(workspace)
  {
  }

  LinSolverDirectCpuILU0::~LinSolverDirectCpuILU0()
  {
  }

  int LinSolverDirectCpuILU0::setup(matrix::Sparse* /* A */,
                                    matrix::Sparse*,
                                    matrix::Sparse*,
                                    index_type*,
                                    index_type*,
                                    vector_type* )
  {
    int error_sum = 1;
    return error_sum;
  }

  int LinSolverDirectCpuILU0::reset(matrix::Sparse* /* A */)
  {
    int error_sum = 1;
    return error_sum;
  }

  // solution is returned in RHS
  int LinSolverDirectCpuILU0::solve(vector_type* /* rhs */)
  {
    int error_sum = 1;
    return error_sum;
  }

  int LinSolverDirectCpuILU0::solve(vector_type* /* rhs */, vector_type* /* x */)
  {
    int error_sum = 1;
    return error_sum;
  }

}// namespace resolve
