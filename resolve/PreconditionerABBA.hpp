/**
 * @file   PreconditionerABBA.hpp
 * @author Jeffery Zhang (jefferyz@vt.edu)
 * @brief  Declaration of left and right preconditioner class
 */

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
   * Allows user to switch between right preconditioning (AB) and
   * left preconditioning (BA)
   *
   * @author Jeffery Zhang (jefferyz@vt.edu)
   *
   */
  class PreconditionerABBA : public Preconditioner
  {
  public:
    using vector_type = vector::Vector;
    using matrix_type = matrix::Sparse;

    PreconditionerABBA(matrix_type* A, MatrixHandler* matrix_handler);
    ~PreconditionerABBA();

    int          apply(vector_type* rhs, vector_type* x) override; // Applies preconditioning
    int          setup(matrix_type*) override;
    std::string  getSide() override;
    matrix_type* getPrecMatrix() override;  // Used to get the preconditioning matrix for calculation of initial residual for BAGMRES
    int          setSide(std::string side); // Changes the preconditioning side for BAGMRES

  private:
    void setMemorySpace();

    matrix_type*        A_{nullptr};
    matrix_type*        B_{nullptr};
    std::string         side_ = "right"; // Defaults to ABGMRES
    MatrixHandler*      matrix_handler_{nullptr};
    memory::MemorySpace memspace_;
  };
} // namespace ReSolve
