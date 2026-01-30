/**
 * @file   Preconditioner.hpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Declaration of preconditioner base class.
 *
 */
#pragma once

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

  /**
   * @class Preconditioner
   *
   * @brief Interface for preconditioner.
   */
  class Preconditioner
  {
  public:
    using vector_type = vector::Vector;
    using matrix_type = matrix::Sparse;

    Preconditioner();
    virtual ~Preconditioner();

    virtual int setup(matrix_type* A)                   = 0;
    virtual int apply(vector_type* rhs, vector_type* x) = 0;
    virtual int reset(matrix_type* /* A */);
  };
} // namespace ReSolve
