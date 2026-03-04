/**
 * @file   PreconditionerUserMatrix.hpp
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Declaration of preconditioner class with user matrix
 *
 */

#pragma once

#include <string>

#include "Common.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/Preconditioner.hpp>
#include <resolve/matrix/MatrixHandler.hpp>

namespace ReSolve
{

  namespace matrix
  {
    class Sparse;
  } // namespace matrix

  namespace vector
  {
    class Vector;
  } // namespace vector

  // Forward declaration of MatrixHandler class
  class MatrixHandler;

  /**
   * @brief Class allows for a user specified matrix to be used for preconditioning
   *
   */
  class PreconditionerUserMatrix : public Preconditioner
  {
  public:
    using vector_type = vector::Vector;
    using matrix_type = matrix::Sparse;

    PreconditionerUserMatrix(MatrixHandler* matrix_handler);
    ~PreconditionerUserMatrix();

    int setup(matrix_type* A) override;
    int setPrecMatrix(matrix_type* B);
    int apply(vector_type* rhs, vector_type* x) override;

    matrix_type* getPrecMatrix();

  private:
    void setMemorySpace();

    matrix_type*        B_{nullptr};
    MatrixHandler*      matrix_handler_{nullptr};
    memory::MemorySpace memspace_;
  };
} // namespace ReSolve
