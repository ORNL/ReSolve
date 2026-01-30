/**
 * @file   PreconditionerILU0.cpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Declaration of preconditioner ILU0 class.
 *
 */

#include <resolve/Preconditioner.hpp>

namespace ReSolve
{
  // Forward declaration of workspace
  class LinSolverDirect;

  namespace matrix
  {
    class Sparse;
  } // namespace matrix

  namespace vector
  {
    class Vector;
  } // namespace vector

  class PreconditionerLU : public Preconditioner
  {
  public:
    using vector_type = vector::Vector;
    using matrix_type = matrix::Sparse;

    PreconditionerLU(LinSolverDirect* solver);
    ~PreconditionerLU();

    int setup(matrix_type* A) override;
    int apply(vector_type* rhs, vector_type* x) override;
    int reset(matrix_type* A) override;

  private:
    LinSolverDirect* solver_{nullptr};
  };
} // namespace ReSolve
