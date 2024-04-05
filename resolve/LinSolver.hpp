/**
 * @file LinSolverIterative.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Declaration of linear solver base class.
 * 
 */
#pragma once

#include <string>
#include "Common.hpp"

namespace ReSolve 
{
  // Forward declaration of vector::Vector class
  namespace vector
  {
    class Vector;
  }

  // Forward declaration of VectorHandler class
  class VectorHandler;

  // Forward declaration of matrix::Sparse class
  namespace matrix
  {
    class Sparse;
  }

  // Forward declaration of MatrixHandler class
  class MatrixHandler;
  
  /**
   * @brief Base class for all linear solvers.
   * 
   */
  class LinSolver 
  {
    protected:
      using vector_type = vector::Vector;

    public:
      LinSolver();
      virtual ~LinSolver();

      real_type evaluateResidual();
        
    protected:  
      matrix::Sparse* A_{nullptr};
      real_type* rhs_{nullptr};
      real_type* sol_{nullptr};

      MatrixHandler* matrix_handler_{nullptr};
      VectorHandler* vector_handler_{nullptr};
  };

} // namespace ReSolve
