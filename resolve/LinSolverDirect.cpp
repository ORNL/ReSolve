/**
 * @file LinSolverIterative.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Implementation of direct solver base class.
 * 
 */
#include <resolve/matrix/Sparse.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#include <resolve/LinSolverDirect.hpp>


namespace ReSolve 
{
  using out = io::Logger;

  LinSolverDirect::LinSolverDirect()
  {
    L_ = nullptr;
    U_ = nullptr;
    P_ = nullptr;
    Q_ = nullptr;
  }

  LinSolverDirect::~LinSolverDirect()
  {
  }

  int LinSolverDirect::setup(matrix::Sparse* A,
                             matrix::Sparse* /* L */,
                             matrix::Sparse* /* U */,
                             index_type*     /* P */,
                             index_type*     /* Q */,
                             vector_type*  /* rhs */)
  {
    if (A == nullptr) {
      return 1;
    }
    A_ = A;
    return 0;
  }

  int LinSolverDirect::analyze()
  {
    return 1;
  } //the same as symbolic factorization

  int LinSolverDirect::factorize()
  {
    return 1;
  }

  int LinSolverDirect::refactorize()
  {
    return 1;
  }

  matrix::Sparse* LinSolverDirect::getLFactor()
  {
    return nullptr;
  } 
  
  matrix::Sparse* LinSolverDirect::getUFactor()
  {
    return nullptr;
  } 
  
  index_type*  LinSolverDirect::getPOrdering()
  {
    return nullptr;
  } 
  
  index_type*  LinSolverDirect::getQOrdering()
  {
    return nullptr;
  }

} // namespace ReSolve
