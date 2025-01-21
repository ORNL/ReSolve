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

      virtual int setCliParam(const std::string /* id */, const std::string /* value */) = 0;        
      virtual std::string getCliParamString(const std::string /* id */) const = 0;        
      virtual index_type getCliParamInt(const std::string /* id */) const = 0;
      virtual real_type getCliParamReal(const std::string /* id */) const = 0;
      virtual bool getCliParamBool(const std::string /* id */) const = 0;
      virtual int printCliParam(const std::string /* id */) const = 0;
        
    protected:
      int getParamId(std::string id) const;

      matrix::Sparse* A_{nullptr};

      MatrixHandler* matrix_handler_{nullptr};
      VectorHandler* vector_handler_{nullptr};

      std::map<std::string, int> params_list_;
  };

} // namespace ReSolve
