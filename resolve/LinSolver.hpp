/**
 * @file LinSolverIterative.hpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Declaration of linear solver base class.
 * 
 */
#pragma once

#include <map>
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

  class SolverParameters;
  
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

      virtual int setCliParam(const std::string /* id */, const std::string /* value */)
      {
        return 1;
      }
        
      virtual int getCliParam(const std::string /* id */, std::string& /* value */)
      {
        return 1;
      }
        
      virtual int getCliParam(const std::string /* id */, index_type& /* value */)
      {
        return 1;
      }
        
      virtual int getCliParam(const std::string /* id */, real_type& /* value */)
      {
        return 1;
      }
        
      virtual int getCliParam(const std::string /* id */, bool& /* value */)
      {
        return 1;
      }
        
      virtual int printCliParam(const std::string /* id */)
      {
        return 1;
      }
        
    protected:
      int getParamId(std::string id);

      matrix::Sparse* A_{nullptr};

      MatrixHandler* matrix_handler_{nullptr};
      VectorHandler* vector_handler_{nullptr};

      std::map<std::string, int> params_list_;
  };

} // namespace ReSolve
